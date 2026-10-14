"""
Publication-Quality Matplotlib Plotting Utilities for FENS.

Generates figures matching the paper's key visual comparisons:
  1. `plot_dirichlet_bubbles`: Recreates Figure 9 (Dirichlet alpha heterogeneity
     bubble plot showing class sample counts per client).
  2. `plot_fens_vs_baselines`: Recreates Figure 2 & Figure 4 (Test Accuracy [%]
     and Client Communication Cost [GiB, log scale] with normalized `w.r.t. OFL`
     bar labels).
  3. `plot_aggregator_dissection`: Recreates Figure 8 (Dissecting FENS across
     Averaging, Weighted Averaging, Voting, Linear, NN, and MoE, including the
     stacked horizontal bar breakdown of Phase 1 upload, Ensemble download, and
     Aggregator training).
  4. `plot_two_client_skew_diagnostics`: Custom 4-panel visualization for the
     2-Client Generalist-vs-Specialist experiment (data distribution, overall &
     group accuracy, per-class accuracy across all 10 classes, and learned FENS
     per-client per-class weight heatmap).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Clean academic color palette inspired by the NeurIPS 2024 FENS plots
PALETTE = {
    "ofl": "#4C78A8",        # Steel Blue
    "fedavg_1r": "#72B7B2",  # Teal
    "fedens": "#9ECAE1",     # Light Blue
    "fedkd": "#3182BD",      # Deep Blue
    "fens": "#F58518",       # Vibrant Orange (FENS signature color in Figs 2, 3, 4, 7)
    "fens_mlp": "#E6550D",   # Rich Orange-Red
    "fedadam": "#54A24B",    # Forest Green (Iterative FL signature color in Figs 2, 4, 7)
    "moe": "#EECA3B",        # Warm Gold
    "linear": "#B279A2",     # Muted Purple
    "voting": "#FF9DA6",     # Coral Pink
}


def plot_dirichlet_bubbles(
    class_counts_by_alpha: Dict[float, np.ndarray],
    save_path: Path | str,
    title: str = "Effect of Dirichlet Concentration α on Client Class Distributions",
) -> Path:
    """
    Recreates Figure 9 (Appendix A, page 14):
    Visualizes the number of samples of each class (y-axis) in each client node
    (x-axis) across one or more values of `alpha`. Dot size is proportional to
    sample count.
    """
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    alphas = list(class_counts_by_alpha.keys())
    n_panels = len(alphas)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.2 * n_panels, 3.8), sharey=True, squeeze=False
    )
    cmap = plt.get_cmap("tab10")

    for idx, alpha in enumerate(alphas):
        ax = axes[0, idx]
        counts = class_counts_by_alpha[alpha]  # shape (M, C)
        num_clients, num_classes = counts.shape
        max_count = max(1.0, float(counts.max()))

        for client_id in range(num_clients):
            for c in range(num_classes):
                cnt = counts[client_id, c]
                if cnt > 0:
                    size = 220.0 * (cnt / max_count)
                    ax.scatter(
                        client_id,
                        c,
                        s=max(4.0, size),
                        color=cmap(c % 10),
                        alpha=0.78,
                        edgecolors="none",
                    )

        ax.set_title(f"$\\alpha = {alpha}$", fontsize=12, fontweight="bold")
        ax.set_xlabel("Client Node Identifier", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Class Label", fontsize=10)
        ax.set_yticks(range(num_classes))
        ax.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fens_vs_baselines(
    results_by_alpha: Dict[float, Dict[str, Dict[str, float]]],
    save_path: Path | str,
    dataset_label: str = "CIFAR-10",
) -> Path:
    """
    Recreates Figure 2 & Figure 4:
      - Top row: Test Accuracy [%] across heterogeneity levels `alpha`.
      - Bottom row: Total Client Communication Cost [GiB, log scale] with bar
        annotations showing normalized communication cost w.r.t. One-Shot FL.

    `results_by_alpha` format:
      {
          0.01: {
              "FedAvg (1R)": {"acc": ..., "comm_gib": ..., "norm_comm": 1.0},
              "FedEns (OFL)": {"acc": ..., "comm_gib": ..., "norm_comm": 1.0},
              "FedKD (OFL)": {"acc": ..., "comm_gib": ..., "norm_comm": 1.0},
              "FENS": {"acc": ..., "comm_gib": ..., "norm_comm": 4.2},
              "FedAdam (FL)": {"acc": ..., "comm_gib": ..., "norm_comm": 84.0},
          },
          ...
      }
    """
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    alphas = sorted(results_by_alpha.keys())
    methods = list(next(iter(results_by_alpha.values())).keys())
    n_methods = len(methods)
    x = np.arange(len(alphas))
    width = min(0.16, 0.78 / max(1, n_methods))

    method_colors = [
        PALETTE["fedavg_1r"],
        PALETTE["fedens"],
        PALETTE["fedkd"],
        PALETTE["fens"],
        PALETTE["fedadam"],
    ]

    fig, (ax_acc, ax_comm) = plt.subplots(2, 1, figsize=(8.5, 6.8), sharex=True)

    for m_idx, method in enumerate(methods):
        offset = (m_idx - (n_methods - 1) / 2.0) * width
        accs = [results_by_alpha[a][method]["acc"] for a in alphas]
        comms = [max(1e-3, results_by_alpha[a][method]["comm_gib"]) for a in alphas]
        norms = [results_by_alpha[a][method]["norm_comm"] for a in alphas]

        color = (
            PALETTE["fens"]
            if "FENS" in method
            else PALETTE["fedadam"]
            if "FedAdam" in method or "Iterative" in method
            else method_colors[m_idx % len(method_colors)]
        )

        bars_acc = ax_acc.bar(
            x + offset,
            accs,
            width=width * 0.92,
            label=method,
            color=color,
            edgecolor="#222222",
            linewidth=0.6,
        )
        for bar, acc_val in zip(bars_acc, accs):
            ax_acc.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1.2,
                f"{acc_val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold" if "FENS" in method else "normal",
            )

        bars_comm = ax_comm.bar(
            x + offset,
            comms,
            width=width * 0.92,
            label=method,
            color=color,
            edgecolor="#222222",
            linewidth=0.6,
        )
        for bar, norm_val in zip(bars_comm, norms):
            label_txt = f"{norm_val:.1f}x" if norm_val < 10 else f"{norm_val:.0f}x"
            ax_comm.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() * 1.12,
                label_txt,
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold" if "FENS" in method else "normal",
            )

    ax_acc.set_ylabel("Test Accuracy [%]", fontsize=11, fontweight="bold")
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(axis="y", linestyle="--", alpha=0.4)
    ax_acc.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=min(5, n_methods),
        frameon=True,
        fontsize=9.5,
    )
    ax_acc.set_title(
        f"FENS vs. One-Shot & Iterative FL on {dataset_label}",
        fontsize=12.5,
        fontweight="bold",
        pad=32,
    )

    ax_comm.set_yscale("log")
    ax_comm.set_ylabel("Client Comm. [GiB]", fontsize=11, fontweight="bold")
    ax_comm.set_xlabel("Data Heterogeneity ($\\alpha$)", fontsize=11, fontweight="bold")
    ax_comm.set_xticks(x)
    ax_comm.set_xticklabels([str(a) for a in alphas], fontsize=10.5)
    ax_comm.grid(axis="y", linestyle="--", alpha=0.4)
    max_comm = max(
        results_by_alpha[a][m]["comm_gib"] for a in alphas for m in methods
    )
    ax_comm.set_ylim(top=max(10.0, max_comm * 4.0))

    fig.text(
        0.5,
        -0.01,
        "Bottom bar labels indicate normalized communication cost w.r.t. One-Shot FL (1.0x)",
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_aggregator_dissection(
    dissection_results: Dict[str, Dict[str, float]],
    save_path: Path | str,
    alpha_label: float = 0.05,
) -> Path:
    """
    Recreates Figure 8 (Section 3.6 "Dissecting FENS"):
      - Left panel: Test Accuracy [%] across aggregation rules
      - Right panel: Horizontal stacked bar chart breaking down total client
        communication cost [GiB] into:
          (1) One-shot local training upload,
          (2) Quantized ensemble download, and
          (3) Iterative aggregator training.

    `dissection_results` format:
      {
          "Avg.": {"acc": ..., "phase1_gib": ..., "download_gib": ..., "phase2_gib": ...},
          "W. Avg.": {...},
          "Voting": {...},
          "Linear": {...},
          "Per-Class": {...},
          "NN (MLP)": {...},
          "MoE": {...},
      }
    """
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(dissection_results.keys())
    accs = [dissection_results[k]["acc"] for k in names]
    p1 = np.array([dissection_results[k]["phase1_gib"] for k in names])
    dl = np.array([dissection_results[k]["download_gib"] for k in names])
    p2 = np.array([dissection_results[k]["phase2_gib"] for k in names])

    fig, (ax_acc, ax_comm) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    bar_colors = [
        PALETTE["fens"] if "NN" in n or "MLP" in n or "Per-Class" in n else PALETTE["ofl"]
        for n in names
    ]
    bars = ax_acc.bar(
        names, accs, color=bar_colors, edgecolor="#222222", linewidth=0.7
    )
    for b, val in zip(bars, accs):
        ax_acc.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + 1.2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax_acc.set_ylabel("Test Accuracy [%]", fontsize=11, fontweight="bold")
    ax_acc.set_ylim(0, 100)
    ax_acc.set_title(
        f"(a) Aggregator Test Accuracy ($\\alpha={alpha_label}$)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_acc.tick_params(axis="x", rotation=22)
    ax_acc.grid(axis="y", linestyle="--", alpha=0.4)

    y_pos = np.arange(len(names))
    ax_comm.barh(
        y_pos,
        p1,
        label="One-shot local training (Upload)",
        color="#8DA0CB",
        edgecolor="#222222",
        height=0.6,
    )
    ax_comm.barh(
        y_pos,
        dl,
        left=p1,
        label="Ensemble download (INT8)",
        color="#66C2A5",
        edgecolor="#222222",
        height=0.6,
    )
    ax_comm.barh(
        y_pos,
        p2,
        left=p1 + dl,
        label="Aggregator training (Iterative)",
        color="#FC8D62",
        edgecolor="#222222",
        height=0.6,
    )

    ax_comm.set_yticks(y_pos)
    ax_comm.set_yticklabels(names, fontsize=10)
    ax_comm.set_xlabel("Client Comm. Cost [GiB]", fontsize=11, fontweight="bold")
    ax_comm.set_title(
        f"(b) Communication Breakdown ($\\alpha={alpha_label}$)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_comm.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax_comm.grid(axis="x", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Dissecting FENS Aggregation Rules (Section 3.6 / Figure 8)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_two_client_skew_diagnostics(
    class_counts: np.ndarray,
    specialist_classes: Sequence[int],
    method_overall_acc: Dict[str, float],
    method_per_class_acc: Dict[str, np.ndarray],
    learned_per_class_weights: np.ndarray,
    save_path: Path | str,
    class_names: Sequence[str] = (
        "0:airpl",
        "1:auto",
        "2:bird",
        "3:cat",
        "4:deer",
        "5:dog",
        "6:frog",
        "7:horse",
        "8:ship",
        "9:truck",
    ),
) -> Path:
    """
    Generates a 4-panel diagnostic figure for the Custom 2-Client Skew Experiment
    (Client 0 = Balanced All-Class Generalist vs. Client 1 = 1-2 Class Specialist):

      - Panel (a): Grouped bar chart of training sample counts per class for
        Client 0 (Balanced Generalist) vs. Client 1 (Specialist).
      - Panel (b): Overall Test Accuracy, Specialist-Class Accuracy, and
        Non-Specialist-Class Accuracy across Client 0 Solo, Client 1 Solo,
        One-Shot FedAvg, Uniform Ensemble (`FedEns`), and FENS.
      - Panel (c): Per-Class Test Accuracy across all 10 classes (0..9),
        showing how Uniform Ensemble and 1-Round FedAvg collapse on classes 2..9
        due to Client 1's extreme logit bias, whereas FENS recovers full accuracy.
      - Panel (d): Heatmap of the learned FENS Per-Client Per-Class weights
        `lambda in R^{2 x 10}`, demonstrating how FENS automatically suppresses
        Client 1 on unseen classes 2..9 while leveraging both clients on the
        specialist classes!
    """
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_classes = class_counts.shape[1]
    classes = np.arange(num_classes)
    spec_set = set(int(c) for c in specialist_classes)
    non_spec_list = [c for c in range(num_classes) if c not in spec_set]
    spec_list = sorted(spec_set)

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
    ax_dist, ax_summary = axes[0, 0], axes[0, 1]
    ax_per_class, ax_weights = axes[1, 0], axes[1, 1]

    # Panel (a): Client 0 vs Client 1 sample distribution
    w = 0.38
    ax_dist.bar(
        classes - w / 2,
        class_counts[0],
        width=w,
        label="Client 0 (All-Class Generalist)",
        color="#4C78A8",
        edgecolor="#222222",
    )
    ax_dist.bar(
        classes + w / 2,
        class_counts[1],
        width=w,
        label=f"Client 1 (Specialist: Classes {spec_list})",
        color="#E45756",
        edgecolor="#222222",
    )
    ax_dist.set_xticks(classes)
    ax_dist.set_xticklabels(class_names, rotation=25, fontsize=8.5)
    ax_dist.set_ylabel("Training Samples", fontsize=10.5, fontweight="bold")
    ax_dist.set_title(
        "(a) Client Training Data Distribution (90% $D_{i1}$ Split)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_dist.legend(fontsize=9, frameon=True)
    ax_dist.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel (b): Overall vs Specialist vs Non-Specialist Accuracy
    methods = list(method_overall_acc.keys())
    m_x = np.arange(len(methods))
    bw = 0.25

    overall_vals = [method_overall_acc[m] for m in methods]
    spec_vals = [
        float(np.mean(method_per_class_acc[m][spec_list])) if spec_list else 0.0
        for m in methods
    ]
    non_spec_vals = [
        float(np.mean(method_per_class_acc[m][non_spec_list])) if non_spec_list else 0.0
        for m in methods
    ]

    ax_summary.bar(
        m_x - bw,
        overall_vals,
        width=bw,
        label="Overall Acc (All 10 Classes)",
        color="#2B5C8F",
        edgecolor="#222222",
    )
    ax_summary.bar(
        m_x,
        spec_vals,
        width=bw,
        label=f"Specialist Classes {spec_list}",
        color="#54A24B",
        edgecolor="#222222",
    )
    ax_summary.bar(
        m_x + bw,
        non_spec_vals,
        width=bw,
        label="Other Classes (Unseen by Client 1)",
        color="#F58518",
        edgecolor="#222222",
    )
    for idx, val in enumerate(overall_vals):
        ax_summary.text(
            m_x[idx] - bw,
            val + 1.2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )
    ax_summary.set_xticks(m_x)
    ax_summary.set_xticklabels(methods, rotation=18, fontsize=9)
    ax_summary.set_ylim(0, 108)
    ax_summary.set_ylabel("Test Accuracy [%]", fontsize=10.5, fontweight="bold")
    ax_summary.set_title(
        "(b) Why Naive Ensembling Fails & How FENS Recovers",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_summary.legend(fontsize=8.5, frameon=True)
    ax_summary.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel (c): Per-Class Accuracy curve across all 10 classes
    line_styles = [
        ("Client 0 (Generalist)", "#4C78A8", "--", "o"),
        ("Client 1 (Specialist)", "#E45756", ":", "s"),
        ("FedAvg (1-Round)", "#72B7B2", "-.", "^"),
        ("FedEns (Uniform Avg)", "#9D755D", "-.", "v"),
        ("FENS (Per-Class Agg)", "#54A24B", "-", "D"),
        ("FENS (MLP Agg)", "#F58518", "-", "*"),
    ]
    for label, color, ls, marker in line_styles:
        if label in method_per_class_acc:
            ax_per_class.plot(
                classes,
                method_per_class_acc[label],
                label=label,
                color=color,
                linestyle=ls,
                marker=marker,
                linewidth=2.0,
                markersize=6,
            )

    for sc in spec_list:
        ax_per_class.axvspan(sc - 0.35, sc + 0.35, color="#FFE6CC", alpha=0.55)
    ax_per_class.set_xticks(classes)
    ax_per_class.set_xticklabels(class_names, rotation=25, fontsize=8.5)
    ax_per_class.set_ylim(-2, 105)
    ax_per_class.set_ylabel("Per-Class Test Accuracy [%]", fontsize=10.5, fontweight="bold")
    ax_per_class.set_title(
        "(c) Per-Class Accuracy (Shaded = Client 1 Specialist Classes)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_per_class.legend(fontsize=8.5, loc="lower left", frameon=True)
    ax_per_class.grid(True, linestyle="--", alpha=0.4)

    # Panel (d): Heatmap of learned FENS Per-Client Per-Class weights lambda (2 x 10)
    im = ax_weights.imshow(
        learned_per_class_weights, cmap="RdYlGn", aspect="auto"
    )
    ax_weights.set_yticks([0, 1])
    ax_weights.set_yticklabels(
        ["Client 0\n(Generalist)", "Client 1\n(Specialist)"],
        fontsize=9.5,
        fontweight="bold",
    )
    ax_weights.set_xticks(classes)
    ax_weights.set_xticklabels(class_names, rotation=25, fontsize=8.5)
    ax_weights.set_title(
        "(d) Learned FENS Per-Class Weights $\\lambda_{i, c}$ (Phase 2 FedAdam)",
        fontsize=11.5,
        fontweight="bold",
    )
    for r in range(learned_per_class_weights.shape[0]):
        for c in range(learned_per_class_weights.shape[1]):
            val = learned_per_class_weights[r, c]
            ax_weights.text(
                c,
                r,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8.5,
                fontweight="bold",
            )
    fig.colorbar(im, ax=ax_weights, fraction=0.046, pad=0.04)

    fig.suptitle(
        "FENS 2-Client Skew Study: Balanced Generalist vs. 1–2 Class Specialist",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path
