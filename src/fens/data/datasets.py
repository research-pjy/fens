"""
PyTorch Dataset and DataLoader wrappers for CIFAR-10, CIFAR-100 (auxiliary KD
dataset), and SVHN.

Follows Section 3.1 and Appendix B.1:
- Vision tasks use standard 32x32 RGB normalization.
- Phase 1 local training uses standard random crop and horizontal flip for CIFAR.
- Phase 2 aggregator training and evaluation use deterministic transforms so
  frozen client ensemble logits can also be cleanly cached on the L40S GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as T

from fens.data.partition import (
    ClientPartitionResult,
    partition_dirichlet,
    partition_two_client_skew,
    split_val_test,
)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)


def get_transforms(dataset_name: str) -> Tuple[T.Compose, T.Compose]:
    """Returns (train_transform, eval_transform) for CIFAR-10/100 or SVHN."""
    name = dataset_name.lower()
    if name in ("cifar10", "cifar100"):
        train_tf = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
        eval_tf = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    elif name == "svhn":
        train_tf = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.ToTensor(),
                T.Normalize(SVHN_MEAN, SVHN_STD),
            ]
        )
        eval_tf = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(SVHN_MEAN, SVHN_STD),
            ]
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_tf, eval_tf


def load_raw_datasets(
    dataset_name: str,
    data_dir: Path | str = "./data",
) -> Tuple[Dataset, Dataset, Dataset, np.ndarray, np.ndarray]:
    """
    Loads:
      1. `train_dataset_aug`: Training split with data augmentation (for Phase 1 & FL baselines)
      2. `train_dataset_eval`: Training split with deterministic eval transform (for Phase 2 D_i2)
      3. `test_dataset`: Test split with deterministic eval transform
      4. `train_labels`: NumPy array of integer training labels
      5. `test_labels`: NumPy array of integer test labels
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    name = dataset_name.lower()
    train_tf, eval_tf = get_transforms(name)

    if name == "cifar10":
        train_aug = torchvision.datasets.CIFAR10(
            root=str(data_path), train=True, download=True, transform=train_tf
        )
        train_eval = torchvision.datasets.CIFAR10(
            root=str(data_path), train=True, download=False, transform=eval_tf
        )
        test_ds = torchvision.datasets.CIFAR10(
            root=str(data_path), train=False, download=True, transform=eval_tf
        )
        train_labels = np.array(train_aug.targets, dtype=np.int64)
        test_labels = np.array(test_ds.targets, dtype=np.int64)

    elif name == "svhn":
        train_aug = torchvision.datasets.SVHN(
            root=str(data_path), split="train", download=True, transform=train_tf
        )
        train_eval = torchvision.datasets.SVHN(
            root=str(data_path), split="train", download=False, transform=eval_tf
        )
        test_ds = torchvision.datasets.SVHN(
            root=str(data_path), split="test", download=True, transform=eval_tf
        )
        train_labels = np.array(train_aug.labels, dtype=np.int64)
        test_labels = np.array(test_ds.labels, dtype=np.int64)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_aug, train_eval, test_ds, train_labels, test_labels


def load_auxiliary_kd_dataset(
    primary_dataset_name: str,
    data_dir: Path | str = "./data",
    max_samples: int = 10000,
    seed: int = 42,
) -> Dataset:
    """
    Loads the auxiliary public dataset used for server-side Knowledge Distillation
    in FedKD (Appendix B.1) and FENS Distillation (Section 3.7, Table 3):
      - When primary dataset is CIFAR-10 -> uses CIFAR-100 as auxiliary dataset.
      - When primary dataset is SVHN     -> uses CIFAR-10 as auxiliary dataset.
    """
    data_path = Path(data_dir)
    name = primary_dataset_name.lower()
    _, eval_tf = get_transforms(name)

    if name == "cifar10":
        aux_ds = torchvision.datasets.CIFAR100(
            root=str(data_path), train=True, download=True, transform=eval_tf
        )
    elif name == "svhn":
        aux_ds = torchvision.datasets.CIFAR10(
            root=str(data_path), train=True, download=True, transform=eval_tf
        )
    else:
        raise ValueError(f"No auxiliary KD dataset configured for {primary_dataset_name}")

    if max_samples is not None and max_samples < len(aux_ds):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(aux_ds), size=max_samples, replace=False)
        return Subset(aux_ds, indices.tolist())
    return aux_ds


@dataclass
class FederatedDataBundle:
    """Bundles all partitioned datasets and splits for an experiment run."""

    dataset_name: str
    num_clients: int
    num_classes: int
    partition: ClientPartitionResult
    train_dataset_aug: Dataset
    train_dataset_eval: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    def get_client_phase1_loader(
        self,
        client_id: int,
        batch_size: int = 16,
        use_full_100_percent: bool = False,
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        Returns the training DataLoader for client `client_id`.
        - For FENS (`use_full_100_percent=False`): uses the 90% split D_i1.
        - For OFL/FL baselines (`use_full_100_percent=True`): uses 100% of D_i.
        """
        indices = (
            self.partition.client_full_indices[client_id]
            if use_full_100_percent
            else self.partition.client_d1_indices[client_id]
        )
        subset = Subset(self.train_dataset_aug, indices.tolist())
        return DataLoader(
            subset,
            batch_size=min(batch_size, max(1, len(subset))),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    def get_client_phase2_dataset(self, client_id: int, augmented: bool = False) -> Subset:
        """
        Returns the 10% held-out dataset D_i2 reserved for Phase 2 iterative
        aggregator training at client `client_id`.
        """
        base_ds = self.train_dataset_aug if augmented else self.train_dataset_eval
        indices = self.partition.client_d2_indices[client_id]
        return Subset(base_ds, indices.tolist())

    def get_eval_loader(
        self,
        split: str = "test",
        batch_size: int = 256,
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> DataLoader:
        """Returns DataLoader for either the 50% 'val' or 50% 'test' split."""
        ds = self.val_dataset if split == "val" else self.test_dataset
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


def build_federated_data_bundle(
    dataset_name: str = "cifar10",
    data_dir: Path | str = "./data",
    num_clients: int = 20,
    dirichlet_alpha: float = 0.05,
    fens_train_fraction: float = 0.90,
    seed: int = 42,
    two_client_skew: bool = False,
    specialist_classes: Sequence[int] = (0, 1),
    generalist_samples_per_class: int = 1000,
    specialist_samples_per_class: int = 4000,
) -> FederatedDataBundle:
    """
    Constructs a complete `FederatedDataBundle` for either:
      1. Standard M-client Dirichlet partitioning (`two_client_skew=False`), or
      2. Custom 2-Client Generalist vs. Specialist partitioning (`two_client_skew=True`).
    """
    train_aug, train_eval, test_ds, train_labels, test_labels = load_raw_datasets(
        dataset_name, data_dir=data_dir
    )
    num_classes = 10

    if two_client_skew:
        partition = partition_two_client_skew(
            labels=train_labels,
            specialist_classes=specialist_classes,
            num_classes=num_classes,
            generalist_samples_per_class=generalist_samples_per_class,
            specialist_samples_per_class=specialist_samples_per_class,
            fens_train_fraction=fens_train_fraction,
            seed=seed,
        )
        num_clients = 2
    else:
        partition = partition_dirichlet(
            labels=train_labels,
            num_clients=num_clients,
            alpha=dirichlet_alpha,
            num_classes=num_classes,
            fens_train_fraction=fens_train_fraction,
            seed=seed,
        )

    val_indices, test_indices = split_val_test(test_labels, val_fraction=0.50, seed=seed)
    val_subset = Subset(test_ds, val_indices.tolist())
    test_subset = Subset(test_ds, test_indices.tolist())

    return FederatedDataBundle(
        dataset_name=dataset_name,
        num_clients=num_clients,
        num_classes=num_classes,
        partition=partition,
        train_dataset_aug=train_aug,
        train_dataset_eval=train_eval,
        val_dataset=val_subset,
        test_dataset=test_subset,
    )
