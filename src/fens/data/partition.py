"""
Non-IID Federated Data Partitioning for FENS.

Implements:
1. `partition_dirichlet`: Standard Dirichlet distribution Dir_M(alpha) partitioning
   across M clients (Section 3.1 & Appendix A, Figure 9), including the FENS
   90% / 10% local split (Algorithm 2, Line 2: D_i1 for local training, D_i2 for
   iterative aggregator training) and 100% full local data for OFL/FL baselines.
2. `partition_two_client_skew`: Custom 2-client experiment where Client 0 is a
   "Generalist" (well-balanced across all 10 classes) and Client 1 is a
   "Specialist" (contains data from only 1 or 2 classes, e.g., classes {0, 1}).
3. `split_val_test`: 50-50% stratified/seeded split of the test set into
   validation and testing sets (Section 3.1, page 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class ClientPartitionResult:
    """
    Stores the data indices and class distributions for all M clients.

    Attributes:
        num_clients: Number of federated clients (M).
        num_classes: Number of classification labels (C).
        client_full_indices: Dict mapping client_id -> 100% of local data indices D_i
            (used by OFL and Iterative FL baselines, Section 3.1).
        client_d1_indices: Dict mapping client_id -> 90% of local data indices D_i1
            (used by FENS Phase 1 one-shot local training, Algorithm 2 Line 2).
        client_d2_indices: Dict mapping client_id -> 10% of local data indices D_i2
            (used by FENS Phase 2 iterative aggregator training, Algorithm 2 Line 2).
        class_counts_full: Array of shape (M, C) with sample counts per class in D_i.
        class_counts_d1: Array of shape (M, C) with sample counts per class in D_i1.
        class_counts_d2: Array of shape (M, C) with sample counts per class in D_i2.
        description: Human-readable summary of the partitioning scheme.
    """

    num_clients: int
    num_classes: int
    client_full_indices: Dict[int, np.ndarray]
    client_d1_indices: Dict[int, np.ndarray]
    client_d2_indices: Dict[int, np.ndarray]
    class_counts_full: np.ndarray
    class_counts_d1: np.ndarray
    class_counts_d2: np.ndarray
    description: str


def _compute_class_counts(
    labels: np.ndarray,
    client_indices: Dict[int, np.ndarray],
    num_clients: int,
    num_classes: int,
) -> np.ndarray:
    """Computes an (M, C) matrix where entry (i, c) is the count of class c at client i."""
    counts = np.zeros((num_clients, num_classes), dtype=np.int64)
    for client_id, idxs in client_indices.items():
        if len(idxs) > 0:
            client_labels = labels[idxs]
            for c in range(num_classes):
                counts[client_id, c] = int(np.sum(client_labels == c))
    return counts


def _split_client_d1_d2(
    client_full_indices: Dict[int, np.ndarray],
    fens_train_fraction: float,
    rng: np.random.Generator,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Implements Algorithm 2, Line 2:
    `Split D_i randomly into 90% D_i1 and 10% D_i2`
    """
    client_d1: Dict[int, np.ndarray] = {}
    client_d2: Dict[int, np.ndarray] = {}

    for client_id, idxs in client_full_indices.items():
        shuffled = rng.permutation(idxs)
        n_total = len(shuffled)
        n_d1 = max(1, int(round(n_total * fens_train_fraction)))
        if n_d1 >= n_total and n_total > 1:
            n_d1 = n_total - 1
        client_d1[client_id] = shuffled[:n_d1]
        client_d2[client_id] = shuffled[n_d1:]

    return client_d1, client_d2


def partition_dirichlet(
    labels: Sequence[int] | np.ndarray,
    num_clients: int = 20,
    alpha: float = 0.05,
    num_classes: int = 10,
    fens_train_fraction: float = 0.90,
    min_samples_per_client: int = 20,
    seed: int = 42,
) -> ClientPartitionResult:
    """
    Partitions a dataset across `num_clients` using a Dirichlet distribution Dir_M(alpha),
    matching Section 3.1 and Figure 9 of the FENS paper.

    Lower values of `alpha` (e.g., 0.01) produce extreme statistical heterogeneity,
    whereas higher values (e.g., 1.0 or 100.0) approach IID distributions.

    Args:
        labels: Ground-truth integer class labels for the training dataset.
        num_clients: Number of clients M (default 20 for vision tasks).
        alpha: Dirichlet concentration parameter (e.g., 0.01, 0.05, 0.1).
        num_classes: Number of classes C (default 10 for CIFAR-10 / SVHN).
        fens_train_fraction: Fraction of D_i used for Phase 1 local training (default 0.90).
        min_samples_per_client: Minimum total samples required per client so both
            90% (D_i1) and 10% (D_i2) splits are non-empty.
        seed: Random seed for reproducible partitioning.

    Returns:
        ClientPartitionResult containing full, D_i1 (90%), and D_i2 (10%) splits.
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)

    current_min_size = 0
    attempt = 0
    client_lists: List[List[int]] = [[] for _ in range(num_clients)]

    while current_min_size < min_samples_per_client:
        attempt += 1
        client_lists = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx_c = np.where(labels_arr == c)[0]
            rng.shuffle(idx_c)

            # Sample class proportions across M clients from Dir_M(alpha)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            # Balance proportions slightly to avoid giving everything to already overloaded clients
            max_per_client = len(labels_arr) / num_clients
            proportions = np.array(
                [
                    p * (len(c_list) < max_per_client)
                    for p, c_list in zip(proportions, client_lists)
                ]
            )
            if proportions.sum() == 0:
                proportions = np.ones(num_clients) / num_clients
            else:
                proportions = proportions / proportions.sum()

            split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            splits = np.split(idx_c, split_points)
            for client_id, split in enumerate(splits):
                client_lists[client_id].extend(split.tolist())

        current_min_size = min(len(c_list) for c_list in client_lists)
        if attempt > 200 and current_min_size >= 2:
            break

    client_full_indices = {
        i: np.array(client_lists[i], dtype=np.int64) for i in range(num_clients)
    }
    client_d1, client_d2 = _split_client_d1_d2(
        client_full_indices, fens_train_fraction, rng
    )

    return ClientPartitionResult(
        num_clients=num_clients,
        num_classes=num_classes,
        client_full_indices=client_full_indices,
        client_d1_indices=client_d1,
        client_d2_indices=client_d2,
        class_counts_full=_compute_class_counts(
            labels_arr, client_full_indices, num_clients, num_classes
        ),
        class_counts_d1=_compute_class_counts(
            labels_arr, client_d1, num_clients, num_classes
        ),
        class_counts_d2=_compute_class_counts(
            labels_arr, client_d2, num_clients, num_classes
        ),
        description=f"Dirichlet Dir_{num_clients}(alpha={alpha})",
    )


def partition_two_client_skew(
    labels: Sequence[int] | np.ndarray,
    specialist_classes: Sequence[int] = (0, 1),
    num_classes: int = 10,
    generalist_samples_per_class: int = 1000,
    specialist_samples_per_class: int = 4000,
    fens_train_fraction: float = 0.90,
    seed: int = 42,
) -> ClientPartitionResult:
    """
    Creates the custom 2-Client Skewed Partition requested for the demonstration:
      - Client 0 ("Generalist / All-Class Balanced"): Receives a strictly uniform,
        well-balanced distribution across all `num_classes` classes
        (e.g., 1,000 samples per class = 10,000 samples total on CIFAR-10).
      - Client 1 ("Specialist / 1-2 Class Skewed"): Receives samples ONLY from
        `specialist_classes` (e.g., classes {0, 1}, with up to 4,000 samples per
        specialist class = 8,000 samples total, and ZERO samples from classes 2..9).

    Why this experiment is illuminating:
      - Client 0 learns a balanced decision boundary across all 10 classes.
      - Client 1 becomes a strong specialist on `specialist_classes`, but because it
        never sees classes outside `specialist_classes` during local cross-entropy
        training, it outputs heavily positive logits for `specialist_classes` and
        heavily negative/uncalibrated logits for all other classes.
      - Standard One-Shot methods (1-round FedAvg, Uniform Ensemble `FedEns`) blindly
        average Client 0 and Client 1, causing Client 1's extreme logit bias to
        corrupt predictions on classes 2..9.
      - FENS (Phase 2) trains a lightweight aggregator (Per-Class Weights or MLP)
        over the 10% held-out splits (D_0,2 and D_1,2), automatically suppressing
        Client 1 on unseen classes 2..9 while combining Client 0 and Client 1 on
        `specialist_classes`!

    Args:
        labels: Ground-truth integer class labels for the training dataset.
        specialist_classes: Tuple/list of 1 or 2 class IDs assigned to Client 1.
        num_classes: Total number of classes in the dataset (default 10).
        generalist_samples_per_class: Number of samples per class given to Client 0.
        specialist_samples_per_class: Number of samples per specialist class given to Client 1.
        fens_train_fraction: Fraction used for Phase 1 local training (default 0.90).
        seed: Random seed for reproducibility.

    Returns:
        ClientPartitionResult for M=2 clients (Client 0 = Generalist, Client 1 = Specialist).
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    specialist_set = set(int(c) for c in specialist_classes)

    client0_indices: List[int] = []
    client1_indices: List[int] = []

    for c in range(num_classes):
        idx_c = np.where(labels_arr == c)[0]
        rng.shuffle(idx_c)

        n_c0 = min(generalist_samples_per_class, len(idx_c))
        if c in specialist_set:
            # Reserve enough samples so Client 0 gets balanced samples and Client 1
            # gets disjoint specialist samples from the remaining pool
            if n_c0 + specialist_samples_per_class > len(idx_c):
                # In CIFAR-10 each class has 5,000 training images:
                # e.g., 1,000 for Client 0 and up to 4,000 for Client 1
                n_c0 = min(generalist_samples_per_class, len(idx_c) // 2)
            c0_slice = idx_c[:n_c0]
            remaining = idx_c[n_c0:]
            n_c1 = min(specialist_samples_per_class, len(remaining))
            c1_slice = remaining[:n_c1]
            client0_indices.extend(c0_slice.tolist())
            client1_indices.extend(c1_slice.tolist())
        else:
            c0_slice = idx_c[:n_c0]
            client0_indices.extend(c0_slice.tolist())

    client_full_indices = {
        0: np.array(client0_indices, dtype=np.int64),
        1: np.array(client1_indices, dtype=np.int64),
    }
    client_d1, client_d2 = _split_client_d1_d2(
        client_full_indices, fens_train_fraction, rng
    )

    spec_str = ", ".join(str(c) for c in sorted(specialist_set))
    return ClientPartitionResult(
        num_clients=2,
        num_classes=num_classes,
        client_full_indices=client_full_indices,
        client_d1_indices=client_d1,
        client_d2_indices=client_d2,
        class_counts_full=_compute_class_counts(
            labels_arr, client_full_indices, 2, num_classes
        ),
        class_counts_d1=_compute_class_counts(
            labels_arr, client_d1, 2, num_classes
        ),
        class_counts_d2=_compute_class_counts(
            labels_arr, client_d2, 2, num_classes
        ),
        description=f"2-Client Skew (Client 0: All {num_classes} Classes Balanced; Client 1: Only Classes {{{spec_str}}})",
    )


def split_val_test(
    test_labels: Sequence[int] | np.ndarray,
    val_fraction: float = 0.50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the dataset's test set 50-50% into validation and test splits
    (Section 3.1, page 5: "The testing set of each dataset is split (50-50%)
    for validation and testing. We use the validation set to tune hyperparameters
    and always report the accuracy on the testing split.").
    """
    labels_arr = np.asarray(test_labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    val_indices: List[int] = []
    test_indices: List[int] = []

    unique_classes = np.unique(labels_arr)
    for c in unique_classes:
        idx_c = np.where(labels_arr == c)[0]
        rng.shuffle(idx_c)
        n_val = int(round(len(idx_c) * val_fraction))
        val_indices.extend(idx_c[:n_val].tolist())
        test_indices.extend(idx_c[n_val:].tolist())

    return np.array(val_indices, dtype=np.int64), np.array(test_indices, dtype=np.int64)
