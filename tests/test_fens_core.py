"""
Unit test suite for the FENS (`src/fens`) package.

Works out-of-the-box with both standard Python `unittest` (no extra packages
needed on `rama`) and `pytest`:

    PYTHONPATH=src python3 -m unittest discover -s tests -v
    # or:
    PYTHONPATH=src pytest -v
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np

from fens.config import ExperimentConfig
from fens.data.partition import (
    partition_dirichlet,
    partition_two_client_skew,
    split_val_test,
)
from fens.utils.comm_cost import (
    compute_fens_comm_breakdown,
    compute_iterative_fl_comm,
    compute_one_shot_comm,
)
from fens.utils.plotting import (
    plot_aggregator_dissection,
    plot_dirichlet_bubbles,
    plot_fens_vs_baselines,
    plot_two_client_skew_diagnostics,
)


class TestFensCore(unittest.TestCase):
    """Tests configuration, partitioning, communication cost math, and plotting."""

    def test_experiment_config_factory(self) -> None:
        cfg_c10 = ExperimentConfig.for_dataset("cifar10", dirichlet_alpha=0.05, num_clients=20)
        self.assertEqual(cfg_c10.phase1.local_epochs, 500)
        self.assertAlmostEqual(cfg_c10.phase1.learning_rate, 0.0025)
        self.assertEqual(cfg_c10.phase2.hidden_dim, 40)
        self.assertAlmostEqual(cfg_c10.phase2.client_lr, 1.0)
        self.assertAlmostEqual(cfg_c10.phase2.server_lr, 0.001)

        cfg_svhn = ExperimentConfig.for_dataset("svhn", dirichlet_alpha=0.01, num_clients=20)
        self.assertEqual(cfg_svhn.phase1.local_epochs, 50)
        self.assertAlmostEqual(cfg_svhn.phase1.learning_rate, 0.01)
        self.assertAlmostEqual(cfg_svhn.phase2.client_lr, 0.1)
        self.assertAlmostEqual(cfg_svhn.phase2.server_lr, 0.01)

    def test_dirichlet_partition_invariants(self) -> None:
        labels = np.repeat(np.arange(10), 500)
        part = partition_dirichlet(
            labels=labels,
            num_clients=20,
            alpha=0.05,
            num_classes=10,
            fens_train_fraction=0.90,
            seed=42,
        )
        self.assertEqual(part.num_clients, 20)
        self.assertEqual(part.class_counts_full.shape, (20, 10))
        self.assertEqual(part.class_counts_d1.shape, (20, 10))
        self.assertEqual(part.class_counts_d2.shape, (20, 10))

        all_full = []
        for cid in range(20):
            full_i = part.client_full_indices[cid]
            d1_i = part.client_d1_indices[cid]
            d2_i = part.client_d2_indices[cid]
            self.assertTrue(len(d1_i) > 0 and len(d2_i) > 0)
            self.assertEqual(len(d1_i) + len(d2_i), len(full_i))
            self.assertEqual(len(set(d1_i).intersection(set(d2_i))), 0)
            all_full.extend(full_i.tolist())

        self.assertEqual(len(all_full), len(labels))
        self.assertEqual(len(set(all_full)), len(labels))

    def test_two_client_skew_partition(self) -> None:
        labels = np.repeat(np.arange(10), 500)
        part = partition_two_client_skew(
            labels=labels,
            specialist_classes=(0, 1),
            num_classes=10,
            generalist_samples_per_class=150,
            specialist_samples_per_class=300,
            fens_train_fraction=0.90,
            seed=42,
        )
        self.assertEqual(part.num_clients, 2)
        self.assertTrue(np.all(part.class_counts_full[0] == 150))
        self.assertEqual(part.class_counts_full[1, 0], 300)
        self.assertEqual(part.class_counts_full[1, 1], 300)
        self.assertTrue(np.all(part.class_counts_full[1, 2:] == 0))

        c0_set = set(part.client_full_indices[0].tolist())
        c1_set = set(part.client_full_indices[1].tolist())
        self.assertEqual(len(c0_set.intersection(c1_set)), 0)

    def test_val_test_split(self) -> None:
        test_labels = np.repeat(np.arange(10), 100)
        val_idx, test_idx = split_val_test(test_labels, val_fraction=0.50, seed=42)
        self.assertEqual(len(val_idx), 500)
        self.assertEqual(len(test_idx), 500)
        self.assertEqual(len(set(val_idx).intersection(set(test_idx))), 0)

    def test_communication_cost_ratios_match_paper(self) -> None:
        resnet8_fp32_bytes = int(18.85 * (1024 ** 2))
        resnet8_int8_bytes = resnet8_fp32_bytes // 4
        mlp_agg_bytes = 8450 * 4

        ofl_cost = compute_one_shot_comm(20, resnet8_fp32_bytes)
        self.assertAlmostEqual(ofl_cost.normalized_wrt_ofl, 1.0, places=5)

        fens_cost = compute_fens_comm_breakdown(
            num_clients=20,
            client_model_fp32_bytes=resnet8_fp32_bytes,
            client_model_compressed_bytes=resnet8_int8_bytes,
            aggregator_fp32_bytes=mlp_agg_bytes,
            aggregator_rounds=300,
            clients_per_round=20,
        )
        # Paper Section 3.2: "The clients in FENS expend 3.6 - 4.3x more than one-shot algorithms"
        self.assertTrue(3.4 <= fens_cost.normalized_wrt_ofl <= 4.5)

        fl_cost = compute_iterative_fl_comm(
            num_clients=20,
            client_model_fp32_bytes=resnet8_fp32_bytes,
            rounds_to_best=84,
            clients_per_round=20,
        )
        self.assertAlmostEqual(fl_cost.normalized_wrt_ofl, 84.0, places=5)

    def test_plotting_utilities_generate_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            counts = {
                0.01: np.random.randint(0, 500, size=(20, 10)),
                0.05: np.random.randint(10, 300, size=(20, 10)),
                0.1: np.random.randint(50, 200, size=(20, 10)),
            }
            p1 = plot_dirichlet_bubbles(counts, tmp / "bubbles.png")
            self.assertTrue(p1.exists() and p1.stat().st_size > 1000)

            res_alpha = {
                0.01: {
                    "FedAvg (1R)": {"acc": 10.35, "comm_gib": 0.368, "norm_comm": 1.0},
                    "FedKD (OFL)": {"acc": 18.59, "comm_gib": 0.368, "norm_comm": 1.0},
                    "FENS (Ours)": {"acc": 44.20, "comm_gib": 1.58, "norm_comm": 4.3},
                    "FedAdam (FL)": {"acc": 39.32, "comm_gib": 17.3, "norm_comm": 47.0},
                },
                0.05: {
                    "FedAvg (1R)": {"acc": 13.56, "comm_gib": 0.368, "norm_comm": 1.0},
                    "FedKD (OFL)": {"acc": 38.84, "comm_gib": 0.368, "norm_comm": 1.0},
                    "FENS (Ours)": {"acc": 68.22, "comm_gib": 1.55, "norm_comm": 4.2},
                    "FedAdam (FL)": {"acc": 68.74, "comm_gib": 30.9, "norm_comm": 84.0},
                },
            }
            p2 = plot_fens_vs_baselines(res_alpha, tmp / "fens_vs_baselines.png")
            self.assertTrue(p2.exists() and p2.stat().st_size > 1000)

            dissect = {
                "Avg.": {"acc": 39.56, "phase1_gib": 0.37, "download_gib": 0.0, "phase2_gib": 0.0},
                "W. Avg.": {"acc": 37.72, "phase1_gib": 0.37, "download_gib": 0.0, "phase2_gib": 0.0},
                "Voting": {"acc": 52.28, "phase1_gib": 0.37, "download_gib": 1.10, "phase2_gib": 0.01},
                "Linear": {"acc": 56.94, "phase1_gib": 0.37, "download_gib": 1.10, "phase2_gib": 0.01},
                "NN (MLP)": {"acc": 68.22, "phase1_gib": 0.37, "download_gib": 1.10, "phase2_gib": 0.08},
            }
            p3 = plot_aggregator_dissection(dissect, tmp / "dissect.png")
            self.assertTrue(p3.exists() and p3.stat().st_size > 1000)

            p4 = plot_two_client_skew_diagnostics(
                class_counts=np.array([[900] * 10, [3600, 3600] + [0] * 8]),
                specialist_classes=[0, 1],
                method_overall_acc={
                    "Client 0 (Generalist)": 74.0,
                    "Client 1 (Specialist)": 19.5,
                    "FedAvg (1-Round)": 38.0,
                    "FedEns (Uniform Avg)": 49.0,
                    "FENS (Per-Class Agg)": 78.5,
                    "FENS (MLP Agg)": 79.8,
                },
                method_per_class_acc={
                    "Client 0 (Generalist)": np.array([78, 82, 68, 60, 72, 65, 80, 76, 81, 78], dtype=float),
                    "Client 1 (Specialist)": np.array([97, 98, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
                    "FedAvg (1-Round)": np.array([92, 94, 20, 15, 25, 18, 30, 28, 32, 26], dtype=float),
                    "FedEns (Uniform Avg)": np.array([95, 96, 35, 28, 40, 32, 45, 42, 39, 38], dtype=float),
                    "FENS (Per-Class Agg)": np.array([94, 96, 70, 62, 74, 67, 81, 78, 83, 80], dtype=float),
                    "FENS (MLP Agg)": np.array([95, 97, 72, 64, 75, 69, 82, 79, 84, 81], dtype=float),
                },
                learned_per_class_weights=np.array(
                    [
                        [0.45, 0.42, 0.98, 0.96, 0.99, 0.97, 1.01, 0.98, 0.95, 0.97],
                        [0.62, 0.65, 0.01, -0.02, 0.00, 0.01, -0.01, 0.02, 0.01, 0.00],
                    ]
                ),
                save_path=tmp / "two_client.png",
            )
            self.assertTrue(p4.exists() and p4.stat().st_size > 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
