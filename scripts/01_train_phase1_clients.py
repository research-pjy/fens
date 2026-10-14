#!/usr/bin/env python3
"""
Step 1 of the Two-Step Demo Workflow:
Phase 1 One-Shot Local Client Training & Checkpointing.

Trains M=20 client models (ResNet-8) locally on their 90% D_i1 Dirichlet splits
and saves checkpoints to `checkpoints/phase1/<dataset>_alpha<alpha>_seed<seed>/`
so that Step 2 (`scripts/02_run_phase2_fens_and_baselines.py`) can be executed
live during your demonstration in seconds!

Also generates Figure 9 (`outputs/<dataset>/fig9_dirichlet_bubbles.png`) showing
how the Dirichlet parameter alpha controls data heterogeneity across clients.

Usage on `rama` (L40S):
    conda activate qwen-vl

    # Full paper settings (CIFAR-10, 500 epochs, alpha in {0.01, 0.05, 0.1}):
    PYTHONPATH=src python scripts/01_train_phase1_clients.py --dataset cifar10 --alphas 0.01 0.05 0.1

    # Fast demo run (e.g., 50 epochs on CIFAR-10 or SVHN for quick verification):
    PYTHONPATH=src python scripts/01_train_phase1_clients.py --dataset cifar10 --alphas 0.01 0.05 0.1 --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np

from fens.config import ExperimentConfig
from fens.data.datasets import build_federated_data_bundle
from fens.fl.phase1_local import save_phase1_checkpoints, train_all_clients_phase1
from fens.utils import (
    inspect_execution_environment,
    plot_dirichlet_bubbles,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1: Train and checkpoint Phase 1 local client models for FENS."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "svhn"],
        help="Dataset to partition and train on (default: cifar10).",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1],
        help="Dirichlet heterogeneity levels alpha (default: 0.01 0.05 0.1).",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=20,
        help="Number of federated clients M (default: 20).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override local training epochs (default: paper setting, 500 for CIFAR-10, 50 for SVHN).",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet8",
        choices=["resnet8", "resnet8_ds", "resnet8_light"],
        help="Client model architecture (default: resnet8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible partitioning and training (default: 42).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to store/load torchvision datasets.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/phase1",
        help="Directory to save Phase 1 client checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save partition visualizations.",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable BF16 mixed precision on L40S.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_report = inspect_execution_environment(ROOT_DIR)
    print(env_report.format_banner())

    class_counts_by_alpha: Dict[float, np.ndarray] = {}

    for alpha in args.alphas:
        set_global_seed(args.seed)
        cfg = ExperimentConfig.for_dataset(
            dataset=args.dataset,
            dirichlet_alpha=alpha,
            num_clients=args.num_clients,
            seed=args.seed,
        )
        cfg.model_arch = args.arch
        cfg.hardware.use_bf16 = not args.no_bf16
        if args.epochs is not None:
            cfg.phase1.local_epochs = args.epochs

        print("\n" + "=" * 72)
        print(
            f"  STEP 1 (PHASE 1 LOCAL TRAINING): dataset={args.dataset.upper()} | "
            f"alpha={alpha} | M={args.num_clients} clients | "
            f"epochs={cfg.phase1.local_epochs} | arch={cfg.model_arch}"
        )
        print("=" * 72)

        bundle = build_federated_data_bundle(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            num_clients=args.num_clients,
            dirichlet_alpha=alpha,
            fens_train_fraction=cfg.phase1.fens_train_fraction,
            seed=args.seed,
        )
        class_counts_by_alpha[alpha] = bundle.partition.class_counts_d1

        t_start = time.perf_counter()
        initial_model, client_models, client_stats = train_all_clients_phase1(
            data_bundle=bundle,
            model_arch=cfg.model_arch,
            phase1_cfg=cfg.phase1,
            hw_cfg=cfg.hardware,
            use_full_100_percent=False,  # 90% D_i1 for FENS Phase 1, 10% D_i2 reserved for Phase 2
            seed=args.seed,
            verbose=True,
        )
        elapsed = time.perf_counter() - t_start

        ckpt_subdir = (
            Path(args.checkpoint_dir)
            / f"{args.dataset}_{cfg.model_arch}_M{args.num_clients}_alpha{alpha}_seed{args.seed}"
        )
        save_phase1_checkpoints(
            save_dir=ckpt_subdir,
            initial_model=initial_model,
            client_models=client_models,
            client_stats=client_stats,
            metadata={
                "dataset": args.dataset,
                "alpha": alpha,
                "num_clients": args.num_clients,
                "arch": cfg.model_arch,
                "local_epochs": cfg.phase1.local_epochs,
                "seed": args.seed,
                "total_wall_time_sec": elapsed,
            },
        )
        mean_client_acc = float(np.mean([s.test_accuracy for s in client_stats]))
        max_client_acc = float(np.max([s.test_accuracy for s in client_stats]))
        print(
            f"\n  [Saved Phase 1 Checkpoints] -> {ckpt_subdir}\n"
            f"  Mean Local Client Test Acc: {mean_client_acc:.2f}% | "
            f"Best Single Client Test Acc: {max_client_acc:.2f}% | "
            f"Total Time: {elapsed / 60.0:.2f} min"
        )

    # Save Figure 9 Dirichlet bubble chart
    fig9_path = (
        Path(args.output_dir) / args.dataset / "fig9_dirichlet_heterogeneity.png"
    )
    plot_dirichlet_bubbles(
        class_counts_by_alpha=class_counts_by_alpha,
        save_path=fig9_path,
        title=f"Dirichlet Client Class Distributions on {args.dataset.upper()} (M={args.num_clients})",
    )
    print(f"\n[Saved Figure 9 Partition Visualization] -> {fig9_path}")


if __name__ == "__main__":
    main()
