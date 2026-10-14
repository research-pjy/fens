"""
Prediction Aggregator Models (f_lambda) for FENS.

Paper Reference:
  - Section 2.2 (Equation 1):
      pi(x) := f_lambda(pi_1(x), ..., pi_M(x))
  - Section 3.6 ("Dissecting FENS", Figure 8 & Table 9)
  - Appendix B.3 (FENS Aggregator Architectures, Table 8)
  - Appendix B.4 (Aggregation Rules: Averaging, Weighted Averaging, Linear,
    Neural Network, Polychotomous Voting, and Mixture-of-Experts).

Input convention across all logit aggregators:
  - `concat_logits`: Tensor of shape `(batch_size, M * C)` representing
    `z = concat(z_1, ..., z_M)` where `z_i in R^C` are the raw logits from
    client model `pi_i`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UniformAveragingAggregator(nn.Module):
    """
    Static Uniform Averaging (`FedEns`, Guha et al. [11], Appendix B.4):
      f(z) = (1 / M) * sum_{i=1}^M z_i
    Contains 0 trainable parameters.
    """

    def __init__(self, num_clients: int, num_classes: int) -> None:
        super().__init__()
        self.num_clients = num_clients
        self.num_classes = num_classes

    def forward(
        self, concat_logits: torch.Tensor, images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b = concat_logits.size(0)
        # Reshape (B, M * C) -> (B, M, C) and average over M
        stacked = concat_logits.view(b, self.num_clients, self.num_classes)
        return stacked.mean(dim=1)


class WeightedAveragingAggregator(nn.Module):
    """
    Static Class-Count Weighted Averaging (FedKD [10], Appendix B.4):
      f(z) = sum_{i=1}^M lambda_i * z_i (coordinate-wise)
      where lambda_i^c = n_i^c / sum_{j=1}^M n_j^c
      and n_i^c is the number of training samples of class c held by client i.
    Contains 0 trainable parameters (weights are registered as a buffer).
    """

    def __init__(
        self,
        num_clients: int,
        num_classes: int,
        class_counts_matrix: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.num_clients = num_clients
        self.num_classes = num_classes

        if class_counts_matrix is not None:
            counts = torch.as_tensor(class_counts_matrix, dtype=torch.float32)
            col_sums = counts.sum(dim=0, keepdim=True).clamp(min=1e-8)
            weights = counts / col_sums
        else:
            weights = torch.full(
                (num_clients, num_classes), 1.0 / num_clients, dtype=torch.float32
            )
        self.register_buffer("class_weights", weights)

    def forward(
        self, concat_logits: torch.Tensor, images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b = concat_logits.size(0)
        stacked = concat_logits.view(b, self.num_clients, self.num_classes)
        # Multiply (B, M, C) by (1, M, C) and sum over clients M
        return (stacked * self.class_weights.unsqueeze(0)).sum(dim=1)


class ScalarLinearAggregator(nn.Module):
    """
    Trainable Scalar Linear Aggregator ("Linear" in Appendix B.4, Figure 8, Table 9):
      f(z) = sum_{i=1}^M w_i * z_i
      where w = [w_1, ..., w_M]^T in R^M is a single learnable scalar weight per client.
    Total trainable parameters: M.
    """

    def __init__(self, num_clients: int, num_classes: int) -> None:
        super().__init__()
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.client_weights = nn.Parameter(
            torch.full((num_clients,), 1.0 / num_clients, dtype=torch.float32)
        )

    def forward(
        self, concat_logits: torch.Tensor, images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b = concat_logits.size(0)
        stacked = concat_logits.view(b, self.num_clients, self.num_classes)
        w = self.client_weights.view(1, self.num_clients, 1)
        return (stacked * w).sum(dim=1)


class PerClassLinearAggregator(nn.Module):
    """
    Trainable Per-Client Per-Class Coordinate-Wise Weight Aggregator
    (Appendix B.3, Equation: f = sum_{i=1}^M lambda_i odot z_i):
      Learns weight vectors lambda_1, ..., lambda_M in R^C.
      Used in Table 8 for CIFAR-100, Fed-Heart-Disease, and Fed-Camelyon16,
      and ideal for inspecting learned client-class trust weights in the
      2-Client Generalist-vs-Specialist demonstration.
    Total trainable parameters: M * C.
    """

    def __init__(self, num_clients: int, num_classes: int) -> None:
        super().__init__()
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.lambda_weights = nn.Parameter(
            torch.full(
                (num_clients, num_classes), 1.0 / num_clients, dtype=torch.float32
            )
        )

    def forward(
        self, concat_logits: torch.Tensor, images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b = concat_logits.size(0)
        stacked = concat_logits.view(b, self.num_clients, self.num_classes)
        return (stacked * self.lambda_weights.unsqueeze(0)).sum(dim=1)


class MLPAggregator(nn.Module):
    """
    2-Layer Perceptron Aggregator with ReLU Activation ("Neural network (NN)" in
    Appendix B.3 & B.4, Table 8, Figure 8):
      f(z) = W_2^T sigma(W_1^T z)
      where z = concat(z_1, ..., z_M) in R^{M*C},
            W_1 in R^{(M*C) x k},
            W_2 in R^{k x C},
            sigma(x) = max(x, 0) is ReLU,
            and k = 40 is the hidden layer width (Table 8).

    For M=20, C=10, k=40:
      Input dim = 200, hidden dim = 40, output dim = 10 -> ~8,450 parameters (~33 KiB),
      which is >500x smaller than a single ResNet-8 model!
    """

    def __init__(
        self,
        num_clients: int,
        num_classes: int,
        hidden_dim: int = 40,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        in_features = num_clients * num_classes

        self.fc1 = nn.Linear(in_features, hidden_dim, bias=use_bias)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=use_bias)

    def forward(
        self, concat_logits: torch.Tensor, images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden = F.relu(self.fc1(concat_logits))
        out = self.fc2(hidden)
        return out


class PolychotomousVotingAggregator(nn.Module):
    """
    Polychotomous Voting Aggregator (Ben-Yashar & Paroush [2], Equation 3, Appendix B.4).

    How it works in FENS (Appendix B.4):
      1. Competency matrix P_i of shape (C, C) for each client model i in [M]:
         P_i[r, c] = Pr(client model i votes for label c | ground truth is r).
      2. In our distributed setting, each client j computes local confusion counts
         for every model i in [M] on its local dataset, and the server aggregates
         them to obtain the global competency matrix P_i for each model i.
      3. Given uniform priors p_prior(r) = 1/C and 0-1 benefit B(c|r) = 1(c == r),
         maximizing expected utility chooses the class r that maximizes the
         posterior log-likelihood of the observed committee votes:
           score(r) = sum_{i=1}^M log P_i[r, vote_i]
    """

    def __init__(
        self,
        num_clients: int,
        num_classes: int,
        smoothing: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.smoothing = smoothing

        # Register log-competency matrices log_P of shape (M, C_true, C_pred)
        init_prob = torch.full(
            (num_clients, num_classes, num_classes),
            1.0 / num_classes,
            dtype=torch.float32,
        )
        self.register_buffer("log_competency", torch.log(init_prob))

    @torch.no_grad()
    def fit_from_confusion_counts(self, confusion_counts: torch.Tensor) -> None:
        """
        Updates `log_competency` from federated confusion counts of shape (M, C_true, C_pred),
        where entry [i, r, c] is the total number of samples with true label r for which
        model i predicted class c across all clients.
        """
        smoothed = confusion_counts.float() + self.smoothing
        row_sums = smoothed.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        probs = smoothed / row_sums
        self.log_competency.copy_(torch.log(probs.clamp(min=1e-8)))

    def forward(
        self, concat_logits: torch.Tensor, images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b = concat_logits.size(0)
        stacked = concat_logits.view(b, self.num_clients, self.num_classes)
        # Hard vote c_i = argmax_c z_{i, c} for each client model i in [M]
        votes = stacked.argmax(dim=-1)  # shape: (B, M)

        # Gather log P_i[r, vote_i] for each r in [C]:
        # log_competency has shape (M, C_true, C_pred)
        # We want output scores of shape (B, C_true)
        scores = torch.zeros(
            (b, self.num_classes),
            device=concat_logits.device,
            dtype=concat_logits.dtype,
        )
        for i in range(self.num_clients):
            # log_competency[i] is (C_true, C_pred); index columns by votes[:, i] -> (C_true, B) -> (B, C_true)
            scores += self.log_competency[i][:, votes[:, i]].t()
        return scores


class MoEAggregator(nn.Module):
    """
    Mixture-of-Experts (MoE) Aggregator (Shazeer et al. [36], Appendix B.4):
      f(x, z) = sum_{i=1}^M G(x)_i * pi_i(x)
      where G: X -> [0, 1]^M is a CNN gating network trained via federated learning:
      "We use a simple CNN with two convolutional blocks comprising ReLU activation
      and max pooling layers followed by 2 FC layers with ReLU activation, which in
      turn is followed by the final classification head." (Appendix B.4, page 17).
    """

    def __init__(self, num_clients: int, num_classes: int) -> None:
        super().__init__()
        self.num_clients = num_clients
        self.num_classes = num_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
        )
        self.fc_head = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_clients),
        )

    def forward(
        self, concat_logits: torch.Tensor, images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if images is None:
            raise ValueError(
                "MoEAggregator requires raw input `images` (x in X) in addition to `concat_logits`."
            )
        b = concat_logits.size(0)
        stacked = concat_logits.view(b, self.num_clients, self.num_classes)

        feats = self.conv_block1(images)
        feats = self.conv_block2(feats)
        feats = torch.flatten(feats, 1)
        gate_logits = self.fc_head(feats)
        gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1)  # (B, M, 1)

        return (stacked * gate_weights).sum(dim=1)


def build_aggregator(
    aggregator_type: str,
    num_clients: int,
    num_classes: int = 10,
    hidden_dim: int = 40,
    class_counts_matrix: Optional[np.ndarray] = None,
) -> nn.Module:
    """Factory function for constructing any of the paper's 7 aggregator models."""
    agg = aggregator_type.lower()
    if agg in ("mlp", "nn"):
        return MLPAggregator(
            num_clients=num_clients, num_classes=num_classes, hidden_dim=hidden_dim
        )
    elif agg in ("per_class_linear", "per_class", "coordinate"):
        return PerClassLinearAggregator(
            num_clients=num_clients, num_classes=num_classes
        )
    elif agg in ("scalar_linear", "linear"):
        return ScalarLinearAggregator(
            num_clients=num_clients, num_classes=num_classes
        )
    elif agg in ("uniform", "averaging", "fedens"):
        return UniformAveragingAggregator(
            num_clients=num_clients, num_classes=num_classes
        )
    elif agg in ("weighted", "weighted_averaging"):
        return WeightedAveragingAggregator(
            num_clients=num_clients,
            num_classes=num_classes,
            class_counts_matrix=class_counts_matrix,
        )
    elif agg in ("voting", "polychotomous_voting"):
        return PolychotomousVotingAggregator(
            num_clients=num_clients, num_classes=num_classes
        )
    elif agg == "moe":
        return MoEAggregator(num_clients=num_clients, num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown aggregator_type '{aggregator_type}'. Choose from: "
            "'mlp', 'per_class_linear', 'scalar_linear', 'uniform', 'weighted', 'voting', 'moe'."
        )
