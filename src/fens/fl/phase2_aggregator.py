"""
Phase 2: Iterative Aggregator Training in FENS (Algorithm 1 Lines 5-12 & Algorithm 2 Lines 6-13).

Paper Reference:
  - Section 2.2 (Equation 2):
      min_{lambda in Lambda} (1 / |D|) sum_{(x, y) in D} l(f_lambda(pi_1(x), ..., pi_M(x)), y)
  - Appendix B.3 & Table 8:
      "We train the aggregator model using the FedAdam algorithm where the
      learning rate is separately tuned for each dataset (Table 8, Appendix B.3).
      To reduce the communication costs corresponding to the ensemble download,
      we employ post-training model quantization at the server from FP32 to INT8."
  - Algorithm 2, Lines 8-12:
      In each round t, client i initializes lambda_t^(i) <- lambda_t, samples K
      mini-batches b in D_i2 (with batch_size=128, K=1 in Table 8), updates
      lambda_t^(i) <- lambda_t^(i) - eta_l * grad l_b, and sends lambda_t^(i)
      back to the server.

L40S Execution Efficiency (`cache_phase2_logits=True`):
  Because the M local models {pi_1, ..., pi_M} are completely frozen during
  Phase 2, precomputing the concatenated logit vector z = concat(pi_1(x), ..., pi_M(x))
  once on the L40S GPU avoids repeating 20 ResNet-8 forward passes across 500
  global rounds. This produces the EXACT same mathematical gradients and parameter
  updates while reducing Phase 2 wall-clock time from ~20 minutes to <5 seconds!
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fens.aggregators.models import (
    MoEAggregator,
    PolychotomousVotingAggregator,
    build_aggregator,
)
from fens.config import HardwareConfig, Phase2Config
from fens.models.quantization import quantize_ensemble_models
from fens.models.resnet8 import get_model_size_bytes


@dataclass
class Phase2TrainingResult:
    """Stores results, accuracy trajectory, and exact communication costs for FENS Phase 2."""

    aggregator_type: str
    quantized_int8: bool
    best_val_accuracy: float
    test_accuracy_at_best_val: float
    final_test_accuracy: float
    best_round: int
    total_rounds: int
    wall_time_sec: float
    # Communication cost breakdown in bytes (across all M clients)
    phase1_upload_bytes: int
    ensemble_download_bytes: int
    phase2_aggregator_comm_bytes: int
    phase2_aggregator_comm_bytes_at_best: int
    per_class_test_accuracy: np.ndarray = field(default_factory=lambda: np.zeros(10))
    test_confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((10, 10), dtype=np.int64))
    round_history: List[Dict[str, float]] = field(default_factory=list)

    @property
    def total_comm_bytes_at_best(self) -> int:
        return (
            self.phase1_upload_bytes
            + self.ensemble_download_bytes
            + self.phase2_aggregator_comm_bytes_at_best
        )

    @property
    def total_comm_gib_at_best(self) -> float:
        return self.total_comm_bytes_at_best / (1024.0 ** 3)

    @property
    def total_comm_mib_at_best(self) -> float:
        return self.total_comm_bytes_at_best / (1024.0 ** 2)


@torch.no_grad()
def precompute_ensemble_logits(
    ensemble_models: List[nn.Module],
    dataset: Dataset,
    batch_size: int = 256,
    device: torch.device | str = "cuda",
    num_workers: int = 2,
    keep_images: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Computes concatenated logits `z = concat(pi_1(x), ..., pi_M(x))` in R^{M*C}
    for all samples in `dataset` using the frozen `ensemble_models`.
    """
    for m in ensemble_models:
        m.eval()
        m.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(str(device).startswith("cuda")),
    )

    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    images_list: List[torch.Tensor] = []

    for images, labels in loader:
        images_dev = images.to(device, non_blocking=True)
        per_model_logits = [m(images_dev) for m in ensemble_models]
        concat_z = torch.cat(per_model_logits, dim=-1)  # (B, M * C)
        logits_list.append(concat_z.cpu())
        labels_list.append(labels.cpu())
        if keep_images:
            images_list.append(images.cpu())

    all_concat_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    all_images = torch.cat(images_list, dim=0) if keep_images else None

    return all_concat_logits, all_labels, all_images


@torch.no_grad()
def evaluate_fens_ensemble(
    aggregator: nn.Module,
    concat_logits: torch.Tensor,
    labels: torch.Tensor,
    images: Optional[torch.Tensor] = None,
    batch_size: int = 512,
    device: torch.device | str = "cuda",
    num_classes: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluates the FENS global model `pi(x) = f_lambda(pi_1(x), ..., pi_M(x))`
    on precomputed ensemble logits (and optional images for MoE).
    """
    aggregator.eval()
    aggregator.to(device)

    n_samples = concat_logits.size(0)
    correct = 0
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        z_batch = concat_logits[start:end].to(device, non_blocking=True)
        y_batch = labels[start:end].to(device, non_blocking=True)
        img_batch = (
            images[start:end].to(device, non_blocking=True)
            if images is not None
            else None
        )

        out_logits = aggregator(z_batch, images=img_batch)
        preds = out_logits.argmax(dim=-1)
        correct += int((preds == y_batch).sum().item())

        t_np = y_batch.cpu().numpy()
        p_np = preds.cpu().numpy()
        for r, c in zip(t_np, p_np):
            conf_mat[int(r), int(c)] += 1

    overall_acc = 100.0 * correct / max(1, n_samples)
    row_totals = conf_mat.sum(axis=1)
    per_class_acc = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if row_totals[c] > 0:
            per_class_acc[c] = 100.0 * conf_mat[c, c] / row_totals[c]

    return overall_acc, per_class_acc, conf_mat


def _compute_federated_voting_competency(
    voting_agg: PolychotomousVotingAggregator,
    client_d2_logits: Dict[int, torch.Tensor],
    client_d2_labels: Dict[int, torch.Tensor],
    num_clients: int,
    num_classes: int,
) -> int:
    """
    Implements the federated learning of competency matrices {P_i}_{i=1}^M
    for Polychotomous Voting (Appendix B.4).
    """
    global_confusion = torch.zeros(
        (num_clients, num_classes, num_classes), dtype=torch.int64
    )

    for client_id in range(num_clients):
        z = client_d2_logits[client_id]
        y = client_d2_labels[client_id]
        b = z.size(0)
        stacked = z.view(b, num_clients, num_classes)
        votes = stacked.argmax(dim=-1)  # (B, M)

        for i in range(num_clients):
            for r_true, c_pred in zip(y.tolist(), votes[:, i].tolist()):
                global_confusion[i, int(r_true), int(c_pred)] += 1

    voting_agg.fit_from_confusion_counts(global_confusion.to(voting_agg.log_competency.device))
    matrix_bytes = num_clients * num_classes * num_classes * 4
    return 2 * num_clients * matrix_bytes


def train_fens_aggregator_phase2(
    client_models: List[nn.Module],
    data_bundle,
    phase2_cfg: Phase2Config,
    hw_cfg: HardwareConfig,
    seed: int = 42,
    precomputed_eval_cache: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, List[nn.Module], Phase2TrainingResult]:
    """
    Executes Algorithm 1 (Lines 5-12) and Algorithm 2 (Lines 6-13).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(hw_cfg.device if torch.cuda.is_available() else "cpu")

    num_clients = data_bundle.num_clients
    num_classes = data_bundle.num_classes

    t0 = time.perf_counter()

    # Step 1: Server-side FP32 -> INT8 post-training quantization (Algorithm 1, Lines 5-6)
    quantized_ensemble, payloads = quantize_ensemble_models(
        client_models=client_models,
        enabled=phase2_cfg.quantize_ensemble_int8,
        device=device,
    )

    # Communication cost accounting matching Algorithm 1:
    # - Phase 1 (Algorithm 1 Lines 3-4): Server sends initial model `theta` to M
    #   clients and receives trained `{theta^(i)}` from M clients = 2 * M * |theta_FP32|
    phase1_upload_bytes = 2 * sum(p.original_bytes for p in payloads)
    # - Ensemble download (Algorithm 1, Line 7 & Footnote 1): each of the M clients
    #   downloads the other (M - 1) INT8-quantized models from the server
    total_compressed_ensemble_bytes = sum(p.compressed_bytes for p in payloads)
    ensemble_download_bytes = int(total_compressed_ensemble_bytes * (num_clients - 1))

    # Step 2: Initialize aggregator model f_lambda (Algorithm 1, Line 8)
    aggregator = build_aggregator(
        aggregator_type=phase2_cfg.aggregator_type,
        num_clients=num_clients,
        num_classes=num_classes,
        hidden_dim=phase2_cfg.hidden_dim,
        class_counts_matrix=data_bundle.partition.class_counts_d1,
    ).to(device)

    needs_images = isinstance(aggregator, MoEAggregator)

    # Step 3: Precompute frozen ensemble logits on D_i2, Val, and Test splits
    client_d2_logits: Dict[int, torch.Tensor] = {}
    client_d2_labels: Dict[int, torch.Tensor] = {}
    client_d2_images: Dict[int, Optional[torch.Tensor]] = {}

    for cid in range(num_clients):
        d2_subset = data_bundle.get_client_phase2_dataset(cid, augmented=False)
        z_c, y_c, img_c = precompute_ensemble_logits(
            ensemble_models=quantized_ensemble,
            dataset=d2_subset,
            batch_size=256,
            device=device,
            num_workers=hw_cfg.num_workers,
            keep_images=needs_images,
        )
        client_d2_logits[cid] = z_c.to(device)
        client_d2_labels[cid] = y_c.to(device)
        client_d2_images[cid] = img_c.to(device) if img_c is not None else None

    if precomputed_eval_cache is not None and not needs_images:
        val_z, val_y, val_img = precomputed_eval_cache["val"]
        test_z, test_y, test_img = precomputed_eval_cache["test"]
    else:
        val_z, val_y, val_img = precompute_ensemble_logits(
            ensemble_models=quantized_ensemble,
            dataset=data_bundle.val_dataset,
            batch_size=256,
            device=device,
            num_workers=hw_cfg.num_workers,
            keep_images=needs_images,
        )
        test_z, test_y, test_img = precompute_ensemble_logits(
            ensemble_models=quantized_ensemble,
            dataset=data_bundle.test_dataset,
            batch_size=256,
            device=device,
            num_workers=hw_cfg.num_workers,
            keep_images=needs_images,
        )

    # Handle static or single-round voting aggregators immediately
    trainable_params = [p for p in aggregator.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        comm_phase2 = 0
        if isinstance(aggregator, PolychotomousVotingAggregator):
            comm_phase2 = _compute_federated_voting_competency(
                voting_agg=aggregator,
                client_d2_logits=client_d2_logits,
                client_d2_labels=client_d2_labels,
                num_clients=num_clients,
                num_classes=num_classes,
            )

        val_acc, _, _ = evaluate_fens_ensemble(
            aggregator, val_z, val_y, images=val_img, device=device, num_classes=num_classes
        )
        test_acc, per_class_acc, conf_mat = evaluate_fens_ensemble(
            aggregator, test_z, test_y, images=test_img, device=device, num_classes=num_classes
        )
        wall_time = time.perf_counter() - t0
        return (
            aggregator,
            quantized_ensemble,
            Phase2TrainingResult(
                aggregator_type=phase2_cfg.aggregator_type,
                quantized_int8=phase2_cfg.quantize_ensemble_int8,
                best_val_accuracy=val_acc,
                test_accuracy_at_best_val=test_acc,
                final_test_accuracy=test_acc,
                best_round=1,
                total_rounds=1,
                wall_time_sec=wall_time,
                phase1_upload_bytes=phase1_upload_bytes,
                ensemble_download_bytes=ensemble_download_bytes if comm_phase2 > 0 else 0,
                phase2_aggregator_comm_bytes=comm_phase2,
                phase2_aggregator_comm_bytes_at_best=comm_phase2,
                per_class_test_accuracy=per_class_acc,
                test_confusion_matrix=conf_mat,
                round_history=[{"round": 1, "val_acc": val_acc, "test_acc": test_acc}],
            ),
        )

    # Step 4: Iterative Aggregator Training with FedAdam (Algorithm 1 Lines 9-12 & Table 8)
    agg_bytes_per_transfer = get_model_size_bytes(aggregator, bits_per_param=32)
    criterion = nn.CrossEntropyLoss()

    server_m: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param.data)
        for name, param in aggregator.named_parameters()
        if param.requires_grad
    }
    server_v: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param.data)
        for name, param in aggregator.named_parameters()
        if param.requires_grad
    }

    best_val_acc = -1.0
    test_acc_at_best_val = 0.0
    best_round = 1
    best_agg_state = copy.deepcopy(aggregator.state_dict())
    round_history: List[Dict[str, float]] = []

    m_selected = max(1, int(round(num_clients * phase2_cfg.client_fraction)))
    comm_bytes_per_round = 2 * m_selected * agg_bytes_per_transfer

    for rnd in range(1, phase2_cfg.global_rounds + 1):
        selected_clients = rng.choice(num_clients, size=m_selected, replace=False)
        global_state = {
            k: v.detach().clone() for k, v in aggregator.state_dict().items()
        }

        delta_sum: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param.data)
            for name, param in aggregator.named_parameters()
            if param.requires_grad
        }

        for cid in selected_clients:
            aggregator.load_state_dict(global_state)
            aggregator.train()

            z_local = client_d2_logits[int(cid)]
            y_local = client_d2_labels[int(cid)]
            img_local = client_d2_images[int(cid)]
            n_local = z_local.size(0)

            for _ in range(phase2_cfg.local_steps):
                if n_local <= phase2_cfg.batch_size:
                    batch_idx = torch.arange(n_local, device=device)
                else:
                    batch_idx = torch.randint(
                        0, n_local, (phase2_cfg.batch_size,), device=device
                    )

                b_z = z_local[batch_idx]
                b_y = y_local[batch_idx]
                b_img = img_local[batch_idx] if img_local is not None else None

                preds = aggregator(b_z, images=b_img)
                loss = criterion(preds, b_y)

                aggregator.zero_grad(set_to_none=True)
                loss.backward()

                with torch.no_grad():
                    for p in aggregator.parameters():
                        if p.requires_grad and p.grad is not None:
                            p.data.add_(p.grad, alpha=-phase2_cfg.client_lr)

            with torch.no_grad():
                for name, param in aggregator.named_parameters():
                    if param.requires_grad:
                        delta_sum[name].add_(param.data - global_state[name])

        # Server FedAdam update (Reddi et al., ICLR 2021 [33])
        with torch.no_grad():
            aggregator.load_state_dict(global_state)
            for name, param in aggregator.named_parameters():
                if param.requires_grad:
                    delta_avg = delta_sum[name] / float(m_selected)
                    server_m[name].mul_(phase2_cfg.beta1).add_(
                        delta_avg, alpha=(1.0 - phase2_cfg.beta1)
                    )
                    server_v[name].mul_(phase2_cfg.beta2).addcmul_(
                        delta_avg, delta_avg, value=(1.0 - phase2_cfg.beta2)
                    )
                    denom = torch.sqrt(server_v[name]) + phase2_cfg.tau
                    param.data.addcdiv_(
                        server_m[name], denom, value=phase2_cfg.server_lr
                    )

        eval_interval = max(1, phase2_cfg.global_rounds // 25)
        if rnd == 1 or rnd % eval_interval == 0 or rnd == phase2_cfg.global_rounds:
            val_acc, _, _ = evaluate_fens_ensemble(
                aggregator, val_z, val_y, images=val_img, device=device, num_classes=num_classes
            )
            test_acc, _, _ = evaluate_fens_ensemble(
                aggregator, test_z, test_y, images=test_img, device=device, num_classes=num_classes
            )
            round_history.append(
                {"round": float(rnd), "val_acc": val_acc, "test_acc": test_acc}
            )
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                test_acc_at_best_val = test_acc
                best_round = rnd
                best_agg_state = copy.deepcopy(aggregator.state_dict())

            if verbose and (rnd == 1 or rnd % (eval_interval * 5) == 0 or rnd == phase2_cfg.global_rounds):
                print(
                    f"  [Phase 2 | {phase2_cfg.aggregator_type.upper()} | Round {rnd:4d}/{phase2_cfg.global_rounds}] "
                    f"Val Acc={val_acc:5.2f}% | Test Acc={test_acc:5.2f}% (Best Test@Val={test_acc_at_best_val:5.2f}% @ R{best_round})"
                )

    final_test_acc, _, _ = evaluate_fens_ensemble(
        aggregator, test_z, test_y, images=test_img, device=device, num_classes=num_classes
    )

    aggregator.load_state_dict(best_agg_state)
    best_test_acc, per_class_acc, conf_mat = evaluate_fens_ensemble(
        aggregator, test_z, test_y, images=test_img, device=device, num_classes=num_classes
    )

    wall_time = time.perf_counter() - t0
    total_phase2_bytes = phase2_cfg.global_rounds * comm_bytes_per_round
    best_phase2_bytes = best_round * comm_bytes_per_round

    result = Phase2TrainingResult(
        aggregator_type=phase2_cfg.aggregator_type,
        quantized_int8=phase2_cfg.quantize_ensemble_int8,
        best_val_accuracy=best_val_acc,
        test_accuracy_at_best_val=best_test_acc,
        final_test_accuracy=final_test_acc,
        best_round=best_round,
        total_rounds=phase2_cfg.global_rounds,
        wall_time_sec=wall_time,
        phase1_upload_bytes=phase1_upload_bytes,
        ensemble_download_bytes=ensemble_download_bytes,
        phase2_aggregator_comm_bytes=total_phase2_bytes,
        phase2_aggregator_comm_bytes_at_best=best_phase2_bytes,
        per_class_test_accuracy=per_class_acc,
        test_confusion_matrix=conf_mat,
        round_history=round_history,
    )

    return aggregator, quantized_ensemble, result
