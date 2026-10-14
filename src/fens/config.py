"""
Configuration dataclasses matching the exact hyperparameters reported in:
"Revisiting Ensembling in One-Shot Federated Learning" (NeurIPS 2024),
Section 3.1, Appendix B.1, Table 6, and Table 8.

Hardware-specific execution parameters (e.g., BF16, DataLoader workers, logit
caching on the L40S GPU) are strictly separated into `HardwareConfig` so they
never alter the underlying scientific methodology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class HardwareConfig:
    """
    Execution & resource settings for the `rama` L40S GPU node.
    Kept strictly separate from the scientific hyperparameters.
    """

    device: str = "cuda"
    use_bf16: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    # Caching frozen Phase-1 client logits in RAM/VRAM during Phase-2 aggregator
    # training produces the exact same mathematical updates while speeding up
    # 500 rounds of aggregator training from ~20 minutes to <10 seconds.
    cache_phase2_logits: bool = True


@dataclass
class Phase1Config:
    """
    Phase 1: One-Shot Local Client Training hyperparameters (Section 3.1 & Appendix B.1).

    Paper specifications:
    - CIFAR-10 / CIFAR-100: 500 local epochs, SGD (lr=0.0025, momentum=0.9,
      weight_decay=1e-4), CosineAnnealingLR, batch_size=16.
    - SVHN: 50 local epochs, SGD (lr=0.01, momentum=0.9, weight_decay=1e-4),
      CosineAnnealingLR, batch_size=16.
    - FENS data split: Each client uses 90% of local data (D_i1) for Phase 1
      local training and reserves 10% (D_i2) for Phase 2 aggregator training.
      Standard OFL/FL baselines use 100% of local data.
    """

    local_epochs: int = 500
    learning_rate: float = 0.0025
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 16
    fens_train_fraction: float = 0.90  # 90% D_i1, 10% D_i2


@dataclass
class Phase2Config:
    """
    Phase 2: Iterative Aggregator Training hyperparameters (Appendix B.3, Table 8).

    Paper specifications (Table 8):
    - CIFAR-10: MLP aggregator (k=40), FedAdam with client lr (eta_l) = 1.0,
      server lr (eta_s) = 0.001, batch_size = 128, local_steps = 1, rounds = 500.
    - SVHN: MLP aggregator (k=40), FedAdam with client lr (eta_l) = 0.1,
      server lr (eta_s) = 0.01, batch_size = 128, local_steps = 1, rounds = 500.
    - CIFAR-100: Per-class-weight aggregator, FedAdam with eta_l = 1.0,
      eta_s = 0.003, batch_size = 128, local_steps = 1, rounds = 1000.
    """

    aggregator_type: str = "mlp"  # Options: "mlp", "per_class_linear", "scalar_linear", "moe", "voting", "uniform", "weighted"
    hidden_dim: int = 40          # Parameter `k` in Table 8
    client_lr: float = 1.0        # eta_l in Table 8
    server_lr: float = 0.001      # eta_s in Table 8
    batch_size: int = 128         # Batch size in Table 8
    local_steps: int = 1          # K local mini-batch steps per round (Algorithm 2, Line 9)
    global_rounds: int = 500      # Global communication rounds T (Table 8)
    quantize_ensemble_int8: bool = True  # Algorithm 1, Line 5-6 (FP32 -> INT8)
    client_fraction: float = 1.0  # Full client participation (S_t = [M])
    # FedAdam server optimizer hyperparameters (Reddi et al., ICLR 2021)
    beta1: float = 0.9
    beta2: float = 0.99
    tau: float = 1e-3


@dataclass
class IterativeFLConfig:
    """
    Iterative FL Baseline hyperparameters (Appendix B.2 & Table 6).

    Paper specifications:
    - Vision tasks run for 100 communication rounds with 2 local epochs per round
      and batch_size = 16.
    - Table 6 tuned learning rates for CIFAR-10:
      * FedAdam: eta_l = 0.01, eta_s = 0.01
      * FedYogi: eta_l = 0.01, eta_s = 0.01
      * FedAvg:  eta_l = 0.01 (for alpha in {0.01, 0.05}), 0.1 (for alpha = 0.1)
      * FedProx: eta_l = 0.01 (alpha in {0.01, 0.05}), 0.1 (alpha = 0.1), mu = 0.01
    """

    algorithm: str = "fedadam"  # "fedavg", "fedprox", "fedadam", "fedyogi"
    global_rounds: int = 100
    local_epochs: int = 2
    batch_size: int = 16
    client_lr: float = 0.01
    server_lr: float = 0.01
    prox_mu: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    tau: float = 1e-3


@dataclass
class DistillationConfig:
    """
    Knowledge Distillation hyperparameters for FedKD (One-Shot baseline, Appendix B.1)
    and Server-Side FENS Distillation (Section 3.7, Table 3).
    """

    distill_epochs: int = 30
    learning_rate: float = 1e-3
    batch_size: int = 128
    temperature: float = 3.0
    alpha_kd: float = 1.0  # Pure soft-label distillation on unlabeled auxiliary proxy data


@dataclass
class ExperimentConfig:
    """Master configuration for a FENS experiment run."""

    dataset: str = "cifar10"  # "cifar10" or "svhn"
    num_clients: int = 20
    num_classes: int = 10
    dirichlet_alpha: float = 0.05
    seed: int = 42
    model_arch: str = "resnet8"  # "resnet8" or "resnet8_ds" (downsized, Section 3.7)
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./outputs")
    checkpoint_dir: Path = Path("./checkpoints")

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    iterative_fl: IterativeFLConfig = field(default_factory=IterativeFLConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)

    @classmethod
    def for_dataset(
        cls,
        dataset: str = "cifar10",
        dirichlet_alpha: float = 0.05,
        num_clients: int = 20,
        seed: int = 42,
    ) -> "ExperimentConfig":
        """
        Factory method that returns the exact paper configuration for a dataset
        (Section 3.1, Table 6, and Table 8).
        """
        dataset = dataset.lower()
        cfg = cls(
            dataset=dataset,
            num_clients=num_clients,
            num_classes=10,
            dirichlet_alpha=dirichlet_alpha,
            seed=seed,
        )
        if dataset == "cifar10":
            cfg.phase1.local_epochs = 500
            cfg.phase1.learning_rate = 0.0025
            cfg.phase2.aggregator_type = "mlp"
            cfg.phase2.hidden_dim = 40
            cfg.phase2.client_lr = 1.0
            cfg.phase2.server_lr = 0.001
            cfg.phase2.global_rounds = 500
        elif dataset == "svhn":
            cfg.phase1.local_epochs = 50
            cfg.phase1.learning_rate = 0.01
            cfg.phase2.aggregator_type = "mlp"
            cfg.phase2.hidden_dim = 40
            cfg.phase2.client_lr = 0.1
            cfg.phase2.server_lr = 0.01
            cfg.phase2.global_rounds = 500
        else:
            raise ValueError(f"Unsupported core demo dataset: {dataset}. Choose 'cifar10' or 'svhn'.")

        # Set Table 6 tuned client learning rate for FedAvg / FedProx
        if dirichlet_alpha >= 0.1:
            cfg.iterative_fl.client_lr = 0.01 if cfg.iterative_fl.algorithm in ("fedadam", "fedyogi") else 0.1
        else:
            cfg.iterative_fl.client_lr = 0.01

        return cfg
