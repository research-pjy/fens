"""
Data loading and non-IID client partitioning utilities for FENS.
"""

from fens.data.partition import (
    ClientPartitionResult,
    partition_dirichlet,
    partition_two_client_skew,
    split_val_test,
)

__all__ = [
    "ClientPartitionResult",
    "partition_dirichlet",
    "partition_two_client_skew",
    "split_val_test",
]
