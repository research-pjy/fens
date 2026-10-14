#!/usr/bin/env python3
"""
Step 2 of the Two-Step Demo Workflow:
Live FENS Phase 2 Aggregator Training, Baselines Comparison & Dissection Plots.

Loads the pre-trained Phase 1 client checkpoints from Step 1 and executes:
  1. One-Shot FL Baselines:
     - One-Round FedAvg (Parameter Averaging)
     - FedEns (Uniform Logit Averaging)
     - Weighted Averaging (Class-Count Weighted Ensemble)
     - FedKD (Server-Side Ensemble Knowledge Distillation on Auxiliary Dataset)
  2. FENS (Two-Phase Ensembling with Server-Side FP32 -> INT8 Quantization):
     - Trains the shallow 2-layer ReLU MLP aggregator (k=40) via FedAdam
     - Trains the Per-Client Per-Class coordinate weight aggregator via FedAdam
     - Optional Server-Side FENS Distillation into a single ResNet-8 (Section 3.7)
  3. Aggregator Dissection (Section 3.6 / Figure 8 / Table 9):
     - Compares Averaging, Weighted Averaging, Polychotomous Voting, Scalar Linear,
       Per-Class Linear, 2-Layer NN (MLP), and Mixture-of-Experts (MoE).
  4. Optional Iterative FL Baseline (`--run-iterative-fl` for FedAdam).

Outputs:
  - Formatted Markdown & JSON tables in `outputs/<dataset>/`
  - `fig2_fens_vs_baselines.png` (Accuracy & Client Comm. Cost in GiB)
  - `fig8_aggregator_dissection.png` (Aggregator Accuracy & Comm. Breakdown)

Usage on `rama` (L40S):
    conda activate qwen-vl
    PYTHONPATH=src python scripts/02_run_phase2_fens_and_baselines.py --dataset cifar10 --alphas 0.01 0.05 0.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import torch

from fens.config import ExperimentConfig
from fens.data.datasets import build_federated_data_bundle, load_auxiliary_kd_dataset
from fens.fl.baselines import (
    distill_ensemble_to_student,
    evaluate_one_shot_fedavg,
    run_fedkd_one_shot,
    run_iterative_fl_baseline,
)
from fens.fl.phase1_local import load_phase1_checkpoints
from fens.fl.phase2_aggregator import (
    precompute_ensemble_logits,
    train_fens_aggregator_phase2,
)
from fens.models.quantization import quantize_ensemble_models
from fens.utils import (
    BYTES_PER_GIB,
    inspect_execution_environment,
    plot_aggregator_dissection,
    plot_fens_vs_baselines,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: Run FENS Phase 2 aggregator training, baselines, and dissection plots."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "svhn"],
        help="Dataset name (default: cifar10).",
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
        "--arch",
        type=str,
        default="resnet8",
        choices=["resnet8", "resnet8_ds", "resnet8_light"],
        help="Client model architecture matching Step 1 checkpoints.",
    )
    parser.add_argument(
        "--phase2-rounds",
        type=int,
        default=None,
        help="Override Phase 2 aggregator global rounds (default: 500 from Table 8).",
    )
    parser.add_argument(
        "--run-fedkd",
        action="store_true",
        help="Also run the FedKD auxiliary-dataset distillation one-shot baseline.",
    )
    parser.add_argument(
        "--run-fens-distill",
        action="store_true",
        help="Also run Section 3.7 server-side distillation of FENS into a single ResNet-8.",
    )
    parser.add_argument(
        "--run-moe",
        action="store_true",
        help="Include the heavier CNN Mixture-of-Experts (MoE) gating baseline in Figure 8 dissection.",
    )
    parser.add_argument(
        "--run-iterative-fl",
        action="store_true",
        help="Run 100 rounds of iterative FedAdam from scratch (if omitted, uses paper reference communication multiplier or fast run).",
    )
    parser.add_argument(
        "--iterative-fl-rounds",
        type=int,
        default=100,
        help="Number of rounds when `--run-iterative-fl` is enabled (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed matching Step 1 (default: 42).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/phase1",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_report = inspect_execution_environment(ROOT_DIR)
    print(env_report.format_banner())

    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    results_by_alpha: Dict[float, Dict[str, Dict[str, float]]] = {}
    dissection_by_alpha: Dict[float, Dict[str, Dict[str, float]]] = {}

    for alpha in args.alphas:
        set_global_seed(args.seed)
        cfg = ExperimentConfig.for_dataset(
            dataset=args.dataset,
            dirichlet_alpha=alpha,
            num_clients=args.num_clients,
            seed=args.seed,
        )
        cfg.model_arch = args.arch
        if args.phase2_rounds is not None:
            cfg.phase2.global_rounds = args.phase2_rounds

        ckpt_subdir = (
            Path(args.checkpoint_dir)
            / f"{args.dataset}_{cfg.model_arch}_M{args.num_clients}_alpha{alpha}_seed{args.seed}"
        )
        print("\n" + "=" * 72)
        print(
            f"  STEP 2 (FENS PHASE 2 & BASELINES): dataset={args.dataset.upper()} | "
            f"alpha={alpha} | Loading checkpoints from {ckpt_subdir}"
        )
        print("=" * 72)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, client_models, ckpt_meta = load_phase1_checkpoints(
            checkpoint_dir=ckpt_subdir,
            model_arch=cfg.model_arch,
            num_classes=cfg.num_classes,
            device=device,
        )

        bundle = build_federated_data_bundle(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            num_clients=args.num_clients,
            dirichlet_alpha=alpha,
            fens_train_fraction=cfg.phase1.fens_train_fraction,
            seed=args.seed,
        )

        # 1. Evaluate One-Shot FedAvg (Parameter Averaging)
        _, _, fedavg_1r_acc, _, _, ofl_comm_bytes = evaluate_one_shot_fedavg(
            client_models=client_models,
            data_bundle=bundle,
            hw_cfg=cfg.hardware,
        )
        ofl_comm_gib = ofl_comm_bytes / BYTES_PER_GIB
        print(f"  [One-Shot | FedAvg (1R)]       Test Acc = {fedavg_1r_acc:6.2f}% | Comm = {ofl_comm_gib:.3f} GiB (1.00x OFL)")

        # Precompute quantized ensemble logits on Val and Test once to share across aggregators
        q_ensemble, _ = quantize_ensemble_models(
            client_models, enabled=True, device=device
        )
        val_z, val_y, _ = precompute_ensemble_logits(
            q_ensemble, bundle.val_dataset, batch_size=256, device=device, num_workers=cfg.hardware.num_workers
        )
        test_z, test_y, _ = precompute_ensemble_logits(
            q_ensemble, bundle.test_dataset, batch_size=256, device=device, num_workers=cfg.hardware.num_workers
        )
        shared_eval_cache = {
            "val": (val_z, val_y, None),
            "test": (test_z, test_y, None),
        }

        # 2. Dissect All Aggregators (Section 3.6: Uniform, Weighted, Voting, Linear, Per-Class, MLP, optional MoE)
        agg_list = [
            ("Avg. (FedEns)", "uniform"),
            ("W. Avg.", "weighted"),
            ("Voting", "voting"),
            ("Linear (Scalar)", "scalar_linear"),
            ("Per-Class Linear", "per_class_linear"),
            ("NN (FENS MLP)", "mlp"),
        ]
        if args.run_moe:
            agg_list.append(("MoE (Gating)", "moe"))

        dissection_by_alpha[alpha] = {}
        fens_mlp_aggregator = None
        fens_mlp_res = None

        for label, agg_type in agg_list:
            agg_cfg = ExperimentConfig.for_dataset(
                dataset=args.dataset,
                dirichlet_alpha=alpha,
                num_clients=args.num_clients,
                seed=args.seed,
            ).phase2
            agg_cfg.aggregator_type = agg_type
            if args.phase2_rounds is not None:
                agg_cfg.global_rounds = args.phase2_rounds
            if agg_type == "moe":
                agg_cfg.client_lr = 0.05
                agg_cfg.server_lr = 0.001
                agg_cfg.global_rounds = min(150, agg_cfg.global_rounds)

            trained_agg, _, res = train_fens_aggregator_phase2(
                client_models=client_models,
                data_bundle=bundle,
                phase2_cfg=agg_cfg,
                hw_cfg=cfg.hardware,
                seed=args.seed,
                precomputed_eval_cache=shared_eval_cache if agg_type != "moe" else None,
                verbose=False,
            )

            norm_comm = res.total_comm_bytes_at_best / max(1, ofl_comm_bytes)
            print(
                f"  [Aggregator | {label:16s}] Test Acc = {res.test_accuracy_at_best_val:6.2f}% | "
                f"Comm = {res.total_comm_gib_at_best:.3f} GiB ({norm_comm:.2f}x OFL) | "
                f"Time = {res.wall_time_sec:.1f}s"
            )

            dissection_by_alpha[alpha][label] = {
                "acc": res.test_accuracy_at_best_val,
                "phase1_gib": res.phase1_upload_bytes / BYTES_PER_GIB,
                "download_gib": res.ensemble_download_bytes / BYTES_PER_GIB,
                "phase2_gib": res.phase2_aggregator_comm_bytes_at_best / BYTES_PER_GIB,
                "total_gib": res.total_comm_gib_at_best,
                "norm_comm": norm_comm,
            }
            if agg_type == "mlp":
                fens_mlp_aggregator = trained_agg
                fens_mlp_res = res

        # 3. Optional FedKD One-Shot Baseline
        fedkd_acc = dissection_by_alpha[alpha]["W. Avg."]["acc"]
        if args.run_fedkd:
            print("  [Running One-Shot FedKD Server Distillation on Auxiliary Dataset...]")
            aux_ds = load_auxiliary_kd_dataset(
                args.dataset, data_dir=args.data_dir, max_samples=10000, seed=args.seed
            )
            _, _, fedkd_acc, _ = run_fedkd_one_shot(
                client_models=client_models,
                data_bundle=bundle,
                aux_dataset=aux_ds,
                student_arch=cfg.model_arch,
                distill_cfg=cfg.distillation,
                hw_cfg=cfg.hardware,
                seed=args.seed,
                verbose=True,
            )
            print(f"  [One-Shot | FedKD Distilled]   Test Acc = {fedkd_acc:6.2f}% | Comm = {ofl_comm_gib:.3f} GiB (1.00x OFL)")

        # 4. Optional Section 3.7 Server-Side Distillation of FENS into a Single ResNet-8
        if args.run_fens_distill and fens_mlp_aggregator is not None:
            print("  [Running Section 3.7 FENS Server-Side Distillation into Single ResNet-8...]")
            aux_ds = load_auxiliary_kd_dataset(
                args.dataset, data_dir=args.data_dir, max_samples=10000, seed=args.seed
            )
            val_loader = bundle.get_eval_loader("val", batch_size=256, num_workers=cfg.hardware.num_workers)
            test_loader = bundle.get_eval_loader("test", batch_size=256, num_workers=cfg.hardware.num_workers)
            _, _, fens_distilled_acc = distill_ensemble_to_student(
                ensemble_models=q_ensemble,
                aggregator=fens_mlp_aggregator,
                aux_dataset=aux_ds,
                val_loader=val_loader,
                test_loader=test_loader,
                student_arch=cfg.model_arch,
                num_classes=cfg.num_classes,
                distill_cfg=cfg.distillation,
                hw_cfg=cfg.hardware,
                seed=args.seed,
                verbose=True,
            )
            print(
                f"  [Section 3.7 | FENS Distilled] Single ResNet-8 Test Acc = {fens_distilled_acc:6.2f}% "
                f"(vs FENS Ensemble {fens_mlp_res.test_accuracy_at_best_val:6.2f}%)"
            )

        # 5. Optional Iterative FL Baseline (`FedAdam`)
        results_by_alpha[alpha] = {
            "FedAvg (1R)": {
                "acc": fedavg_1r_acc,
                "comm_gib": ofl_comm_gib,
                "norm_comm": 1.0,
            },
            "FedEns (OFL)": {
                "acc": dissection_by_alpha[alpha]["Avg. (FedEns)"]["acc"],
                "comm_gib": ofl_comm_gib,
                "norm_comm": 1.0,
            },
            "W.Avg / FedKD": {
                "acc": fedkd_acc,
                "comm_gib": ofl_comm_gib,
                "norm_comm": 1.0,
            },
            "FENS (Ours)": {
                "acc": dissection_by_alpha[alpha]["NN (FENS MLP)"]["acc"],
                "comm_gib": dissection_by_alpha[alpha]["NN (FENS MLP)"]["total_gib"],
                "norm_comm": dissection_by_alpha[alpha]["NN (FENS MLP)"]["norm_comm"],
            },
        }

        if args.run_iterative_fl:
            cfg.iterative_fl.global_rounds = args.iterative_fl_rounds
            _, fl_res = run_iterative_fl_baseline(
                data_bundle=bundle,
                model_arch=cfg.model_arch,
                fl_cfg=cfg.iterative_fl,
                hw_cfg=cfg.hardware,
                seed=args.seed,
                verbose=True,
            )
            fl_norm = fl_res.comm_bytes_at_best / max(1, ofl_comm_bytes)
            results_by_alpha[alpha]["FedAdam (Iterative)"] = {
                "acc": fl_res.test_accuracy_at_best_val,
                "comm_gib": fl_res.comm_gib_at_best,
                "norm_comm": fl_norm,
            }

        # Plot Figure 8 Dissection for this alpha
        fig8_path = out_dir / f"fig8_aggregator_dissection_alpha{alpha}.png"
        plot_aggregator_dissection(
            dissection_results=dissection_by_alpha[alpha],
            save_path=fig8_path,
            alpha_label=alpha,
        )
        print(f"  [Saved Figure 8 Aggregator Dissection Plot] -> {fig8_path}")

    # Plot Figure 2 / Figure 4 Summary across all evaluated alphas
    fig2_path = out_dir / "fig2_fens_vs_baselines.png"
    plot_fens_vs_baselines(
        results_by_alpha=results_by_alpha,
        save_path=fig2_path,
        dataset_label=args.dataset.upper(),
    )
    print(f"\n[Saved Figure 2/4 Summary Plot] -> {fig2_path}")

    # Save JSON and Markdown summary report
    summary_json_path = out_dir / "fens_evaluation_results.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "results_by_alpha": results_by_alpha,
                "dissection_by_alpha": dissection_by_alpha,
            },
            f,
            indent=2,
        )

    md_lines = [
        f"# FENS Demonstration Results on {args.dataset.upper()}",
        "",
        "## 1. FENS vs. One-Shot FL Baselines",
        "",
        "| Alpha ($\\alpha$) | Method | Test Accuracy (%) | Client Comm. Cost (GiB) | Normalized Comm. (w.r.t. OFL) |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    for alpha, m_dict in results_by_alpha.items():
        for m_name, metrics in m_dict.items():
            md_lines.append(
                f"| {alpha} | **{m_name}** | {metrics['acc']:.2f}% | {metrics['comm_gib']:.3f} GiB | {metrics['norm_comm']:.2f}x |"
            )

    md_lines.extend(
        [
            "",
            "## 2. Dissecting FENS Aggregators (Section 3.6 / Figure 8)",
            "",
            "| Alpha ($\\alpha$) | Aggregator Rule | Test Accuracy (%) | Phase 1 Upload (GiB) | INT8 Download (GiB) | Phase 2 Iterative (GiB) | Total Comm. (GiB) |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for alpha, d_dict in dissection_by_alpha.items():
        for agg_name, m in d_dict.items():
            md_lines.append(
                f"| {alpha} | **{agg_name}** | {m['acc']:.2f}% | {m['phase1_gib']:.3f} | {m['download_gib']:.3f} | {m['phase2_gib']:.4f} | {m['total_gib']:.3f} ({m['norm_comm']:.2f}x) |"
            )

    md_report = "\n".join(md_lines) + "\n"
    md_path = out_dir / "fens_evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    print("\n" + md_report)
    print(f"[Saved Markdown Report] -> {md_path}")


if __name__ == "__main__":
    main()
