#!/usr/bin/env python3
"""
Step 3: Custom 2-Client Skewed Demonstration
("Balanced All-Class Generalist vs. 1-2 Class Specialist").

Experiment Design:
  - Client 0 ("Generalist"): Receives a balanced, uniform distribution across
    all 10 classes (e.g., 1,000 samples per class = 10,000 samples total).
  - Client 1 ("Specialist"): Receives data exclusively from 1 or 2 classes
    (default: classes {0, 1}, with 4,000 samples per class = 8,000 samples total,
    and ZERO samples from classes 2..9).

What this demonstrates:
  1. Client 0 Solo learns a good generalist decision boundary across all 10 classes.
  2. Client 1 Solo achieves near-perfect accuracy on its specialist classes {0, 1},
     but 0% accuracy on classes 2..9 because it never saw them during local training.
     More importantly, Client 1 outputs huge positive logits for classes {0, 1} on
     EVERY test input.
  3. Standard One-Shot methods (`FedAvg 1-Round` and `FedEns Uniform Averaging`)
     blindly average Client 0 and Client 1. As a result, Client 1's extreme logit
     bias towards {0, 1} overwhelms Client 0's predictions on classes 2..9, causing
     severe accuracy degradation on non-specialist classes!
  4. FENS (Phase 2 Iterative Aggregator Training) trains a lightweight level-1
     generalizer (Per-Client Per-Class Weights `lambda in R^{2x10}` or 2-Layer MLP)
     via FedAdam on the 10% held-out splits (D_{0,2} and D_{1,2}). FENS automatically
     learns to assign near-zero/negative weight to Client 1 on classes 2..9 while
     combining both Client 0 and Client 1 on the specialist classes {0, 1}!

Usage on `rama` (L40S):
    conda activate qwen-vl

    # Train Phase 1 (2 clients) + Run Phase 2 analysis & generate 4-panel plot:
    PYTHONPATH=src python scripts/03_run_two_client_skew_demo.py --specialist-classes 0 1 --epochs 100

    # Instant live demo reload (skips Phase 1 training by loading saved checkpoints):
    PYTHONPATH=src python scripts/03_run_two_client_skew_demo.py --specialist-classes 0 1 --load-phase1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np
import torch

from fens.aggregators.models import PerClassLinearAggregator
from fens.config import ExperimentConfig
from fens.data.datasets import build_federated_data_bundle
from fens.fl.baselines import evaluate_one_shot_fedavg
from fens.fl.phase1_local import (
    evaluate_single_model,
    load_phase1_checkpoints,
    save_phase1_checkpoints,
    train_all_clients_phase1,
)
from fens.fl.phase2_aggregator import train_fens_aggregator_phase2
from fens.utils import (
    inspect_execution_environment,
    plot_two_client_skew_diagnostics,
    set_global_seed,
)


CIFAR10_CLASS_NAMES = (
    "0:airplane",
    "1:automobile",
    "2:bird",
    "3:cat",
    "4:deer",
    "5:dog",
    "6:frog",
    "7:horse",
    "8:ship",
    "9:truck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Custom 2-Client Skew Experiment: Balanced Generalist vs. 1-2 Class Specialist."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "svhn"],
        help="Dataset to run on (default: cifar10).",
    )
    parser.add_argument(
        "--specialist-classes",
        type=int,
        nargs="+",
        default=[0, 1],
        help="1 or 2 class IDs assigned exclusively to Client 1 (default: 0 1).",
    )
    parser.add_argument(
        "--generalist-samples-per-class",
        type=int,
        default=1000,
        help="Samples per class assigned to Client 0 across all 10 classes (default: 1000).",
    )
    parser.add_argument(
        "--specialist-samples-per-class",
        type=int,
        default=4000,
        help="Samples per specialist class assigned to Client 1 (default: 4000).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Phase 1 local training epochs for the 2 clients (default: 100; paper full is 500).",
    )
    parser.add_argument(
        "--phase2-rounds",
        type=int,
        default=500,
        help="Phase 2 FedAdam aggregator training rounds (default: 500).",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet8",
        choices=["resnet8", "resnet8_ds", "resnet8_light"],
        help="Client model architecture (default: resnet8).",
    )
    parser.add_argument(
        "--load-phase1",
        action="store_true",
        help="Load existing Phase 1 checkpoints for the 2 clients instead of training from scratch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/two_client_skew",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/two_client_skew",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    env_report = inspect_execution_environment(ROOT_DIR)
    print(env_report.format_banner())

    spec_tag = "_".join(str(c) for c in sorted(args.specialist_classes))
    ckpt_dir = Path(args.checkpoint_dir) / f"{args.dataset}_{args.arch}_spec_{spec_tag}_seed{args.seed}"
    out_dir = Path(args.output_dir) / f"{args.dataset}_{args.arch}_spec_{spec_tag}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig.for_dataset(
        dataset=args.dataset, dirichlet_alpha=0.05, num_clients=2, seed=args.seed
    )
    cfg.model_arch = args.arch
    cfg.phase1.local_epochs = args.epochs
    cfg.phase2.global_rounds = args.phase2_rounds

    bundle = build_federated_data_bundle(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        num_clients=2,
        fens_train_fraction=cfg.phase1.fens_train_fraction,
        seed=args.seed,
        two_client_skew=True,
        specialist_classes=args.specialist_classes,
        generalist_samples_per_class=args.generalist_samples_per_class,
        specialist_samples_per_class=args.specialist_samples_per_class,
    )

    print("\n" + "=" * 72)
    print(f"  CUSTOM 2-CLIENT SKEW EXPERIMENT: {bundle.partition.description}")
    print("=" * 72)
    print("  Client 0 (90% D_0,1 class counts):", bundle.partition.class_counts_d1[0].tolist())
    print("  Client 1 (90% D_1,1 class counts):", bundle.partition.class_counts_d1[1].tolist())
    print("  Client 0 (10% D_0,2 class counts):", bundle.partition.class_counts_d2[0].tolist())
    print("  Client 1 (10% D_1,2 class counts):", bundle.partition.class_counts_d2[1].tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.load_phase1 and (ckpt_dir / "phase1_summary.json").exists():
        print(f"\n[Loading Saved Phase 1 Client Models from {ckpt_dir}...]")
        _, client_models, _ = load_phase1_checkpoints(
            ckpt_dir, model_arch=cfg.model_arch, num_classes=cfg.num_classes, device=device
        )
    else:
        print(f"\n[Training Phase 1 Local Models for Client 0 (Generalist) & Client 1 (Specialist)...]")
        initial_model, client_models, client_stats = train_all_clients_phase1(
            data_bundle=bundle,
            model_arch=cfg.model_arch,
            phase1_cfg=cfg.phase1,
            hw_cfg=cfg.hardware,
            use_full_100_percent=False,
            seed=args.seed,
            verbose=True,
        )
        save_phase1_checkpoints(
            save_dir=ckpt_dir,
            initial_model=initial_model,
            client_models=client_models,
            client_stats=client_stats,
            metadata={
                "specialist_classes": args.specialist_classes,
                "epochs": args.epochs,
            },
        )
        print(f"[Saved 2-Client Checkpoints] -> {ckpt_dir}")

    test_loader = bundle.get_eval_loader(
        split="test", batch_size=256, num_workers=cfg.hardware.num_workers, pin_memory=cfg.hardware.pin_memory
    )

    method_overall_acc: Dict[str, float] = {}
    method_per_class_acc: Dict[str, np.ndarray] = {}

    # 1. Evaluate Client 0 Solo (Balanced Generalist)
    c0_acc, c0_per_class, _ = evaluate_single_model(
        client_models[0], test_loader, device=device, num_classes=cfg.num_classes
    )
    method_overall_acc["Client 0 (Generalist)"] = c0_acc
    method_per_class_acc["Client 0 (Generalist)"] = c0_per_class

    # 2. Evaluate Client 1 Solo (1-2 Class Specialist)
    c1_acc, c1_per_class, _ = evaluate_single_model(
        client_models[1], test_loader, device=device, num_classes=cfg.num_classes
    )
    method_overall_acc["Client 1 (Specialist)"] = c1_acc
    method_per_class_acc["Client 1 (Specialist)"] = c1_per_class

    # 3. Evaluate One-Shot FedAvg (1-Round Parameter Averaging)
    _, _, fedavg_acc, fedavg_per_class, _, _ = evaluate_one_shot_fedavg(
        client_models=client_models, data_bundle=bundle, hw_cfg=cfg.hardware
    )
    method_overall_acc["FedAvg (1-Round)"] = fedavg_acc
    method_per_class_acc["FedAvg (1-Round)"] = fedavg_per_class

    # 4. Evaluate FedEns (Uniform Logit Averaging)
    cfg.phase2.aggregator_type = "uniform"
    _, _, fedens_res = train_fens_aggregator_phase2(
        client_models=client_models,
        data_bundle=bundle,
        phase2_cfg=cfg.phase2,
        hw_cfg=cfg.hardware,
        seed=args.seed,
        verbose=False,
    )
    method_overall_acc["FedEns (Uniform Avg)"] = fedens_res.test_accuracy_at_best_val
    method_per_class_acc["FedEns (Uniform Avg)"] = fedens_res.per_class_test_accuracy

    # 5. Evaluate Static Class-Count Weighted Averaging
    cfg.phase2.aggregator_type = "weighted"
    _, _, weighted_res = train_fens_aggregator_phase2(
        client_models=client_models,
        data_bundle=bundle,
        phase2_cfg=cfg.phase2,
        hw_cfg=cfg.hardware,
        seed=args.seed,
        verbose=False,
    )
    method_overall_acc["W. Avg (Class Counts)"] = weighted_res.test_accuracy_at_best_val
    method_per_class_acc["W. Avg (Class Counts)"] = weighted_res.per_class_test_accuracy

    # 6. Evaluate FENS with Per-Client Per-Class Weight Aggregator
    print("\n[Phase 2 | Training FENS Per-Client Per-Class Weight Aggregator (lambda in R^{2x10})...]")
    cfg.phase2.aggregator_type = "per_class_linear"
    cfg.phase2.client_lr = 0.5
    cfg.phase2.server_lr = 0.01
    per_class_agg, _, fens_pc_res = train_fens_aggregator_phase2(
        client_models=client_models,
        data_bundle=bundle,
        phase2_cfg=cfg.phase2,
        hw_cfg=cfg.hardware,
        seed=args.seed,
        verbose=True,
    )
    method_overall_acc["FENS (Per-Class Agg)"] = fens_pc_res.test_accuracy_at_best_val
    method_per_class_acc["FENS (Per-Class Agg)"] = fens_pc_res.per_class_test_accuracy

    assert isinstance(per_class_agg, PerClassLinearAggregator)
    learned_weights = per_class_agg.lambda_weights.detach().cpu().numpy()

    # 7. Evaluate FENS with 2-Layer ReLU MLP Aggregator (k=40)
    print("\n[Phase 2 | Training FENS 2-Layer ReLU MLP Aggregator (k=40)...]")
    cfg.phase2.aggregator_type = "mlp"
    cfg.phase2.client_lr = 1.0
    cfg.phase2.server_lr = 0.001
    _, _, fens_mlp_res = train_fens_aggregator_phase2(
        client_models=client_models,
        data_bundle=bundle,
        phase2_cfg=cfg.phase2,
        hw_cfg=cfg.hardware,
        seed=args.seed,
        verbose=True,
    )
    method_overall_acc["FENS (MLP Agg)"] = fens_mlp_res.test_accuracy_at_best_val
    method_per_class_acc["FENS (MLP Agg)"] = fens_mlp_res.per_class_test_accuracy

    # Compute specialist vs non-specialist group accuracies
    spec_set = set(int(c) for c in args.specialist_classes)
    spec_idx = sorted(spec_set)
    non_spec_idx = [c for c in range(cfg.num_classes) if c not in spec_set]

    md_lines = [
        f"# 2-Client Skew Analysis: Generalist (All 10 Classes) vs. Specialist (Classes {spec_idx})",
        "",
        "## 1. Overall vs. Class-Subset Test Accuracy",
        "",
        "| Method | Overall Accuracy (All 10 Classes) | Specialist Classes "
        f"{spec_idx} Acc | Other Classes {non_spec_idx} Acc |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for m_name, overall_val in method_overall_acc.items():
        pc = method_per_class_acc[m_name]
        spec_acc = float(np.mean(pc[spec_idx])) if spec_idx else 0.0
        other_acc = float(np.mean(pc[non_spec_idx])) if non_spec_idx else 0.0
        md_lines.append(
            f"| **{m_name}** | **{overall_val:.2f}%** | {spec_acc:.2f}% | {other_acc:.2f}% |"
        )

    md_lines.extend(
        [
            "",
            "## 2. Learned FENS Per-Client Per-Class Aggregator Weights ($\\lambda_{i, c}$)",
            "",
            "| Class | Client 0 Weight (Generalist) | Client 1 Weight (Specialist) | Role |",
            "| :--- | ---: | ---: | :--- |",
        ]
    )
    for c in range(cfg.num_classes):
        c_label = CIFAR10_CLASS_NAMES[c] if args.dataset == "cifar10" else f"Digit {c}"
        role = "Specialist + Generalist Class" if c in spec_set else "Generalist Only (Unseen by Client 1)"
        md_lines.append(
            f"| **{c_label}** | `{learned_weights[0, c]:+.3f}` | `{learned_weights[1, c]:+.3f}` | {role} |"
        )

    report_md = "\n".join(md_lines) + "\n"
    print("\n" + report_md)

    report_path = out_dir / "two_client_skew_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    json_path = out_dir / "two_client_skew_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "specialist_classes": spec_idx,
                "overall_accuracy": method_overall_acc,
                "per_class_accuracy": {k: v.tolist() for k, v in method_per_class_acc.items()},
                "learned_per_class_weights": learned_weights.tolist(),
            },
            f,
            indent=2,
        )

    plot_path = out_dir / "two_client_skew_diagnostics.png"
    class_labels = (
        CIFAR10_CLASS_NAMES
        if args.dataset == "cifar10"
        else tuple(f"Digit {i}" for i in range(10))
    )
    plot_two_client_skew_diagnostics(
        class_counts=bundle.partition.class_counts_d1,
        specialist_classes=spec_idx,
        method_overall_acc=method_overall_acc,
        method_per_class_acc=method_per_class_acc,
        learned_per_class_weights=learned_weights,
        save_path=plot_path,
        class_names=class_labels,
    )
    print(f"[Saved 4-Panel Diagnostic Figure] -> {plot_path}")
    print(f"[Saved Detailed Report]           -> {report_path}")


if __name__ == "__main__":
    main()
