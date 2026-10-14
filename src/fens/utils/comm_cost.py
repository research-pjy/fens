"""
Communication Cost Accounting for One-Shot FL, FENS, and Iterative FL.

Paper Reference:
  - Algorithm 1, Lines 3-4 (Phase 1 / One-Shot FL):
    Line 3: Server sends initial model `theta` to all M clients (M * |theta_FP32|)
    Line 4: Server receives locally trained `{theta^(i)}` from all M clients (M * |theta_FP32|)
    -> Total 1-round OFL communication = 2 * M * |theta_FP32| (1.0x OFL).
  - Algorithm 1, Lines 5-7 & Footnote 1 (INT8 Ensemble Download):
    Server quantizes all M client models from FP32 to INT8 (4x reduction:
    |theta_INT8| = 0.25 * |theta_FP32|) and sends the other (M - 1) models to
    each of the M clients:
    -> Download cost = M * (M - 1) * |theta_INT8| = ((M - 1) / 8) * OFL
       (For M=20, 19/8 = 2.375x OFL).
  - Algorithm 1, Lines 9-12 (Phase 2 Iterative Aggregator Training):
    Over T rounds, M clients download and upload the lightweight aggregator `f_lambda`:
    -> Phase 2 cost = 2 * M * T * |lambda_FP32| (~0.3x - 0.9x OFL).
    -> Total FENS communication = 1.0 + 2.375 + (~0.3 to 0.9) = 3.7x - 4.3x OFL,
       matching the bar labels (3.7, 3.8, 4.1, 4.2, 4.3) in Figures 2, 3, and 4!
  - Iterative FL (`FedAdam` / `FedAvg`, Appendix B.2):
    Each communication round transfers 2 * M * |theta_FP32| = 1.0x OFL.
    Converging in R in [47, 96] rounds (within the 100-round vision budget)
    yields 47x - 96x OFL, matching Figures 2 and 4!
"""

from __future__ import annotations

from dataclasses import dataclass


BYTES_PER_MIB = 1024.0 ** 2
BYTES_PER_GIB = 1024.0 ** 3


@dataclass
class CommCostBreakdown:
    """
    Stores communication cost in bytes, MiB, GiB, and normalized ratio w.r.t. OFL.
    """

    method_name: str
    phase1_upload_bytes: int
    ensemble_download_bytes: int
    phase2_iterative_bytes: int
    reference_ofl_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.phase1_upload_bytes
            + self.ensemble_download_bytes
            + self.phase2_iterative_bytes
        )

    @property
    def total_mib(self) -> float:
        return self.total_bytes / BYTES_PER_MIB

    @property
    def total_gib(self) -> float:
        return self.total_bytes / BYTES_PER_GIB

    @property
    def phase1_gib(self) -> float:
        return self.phase1_upload_bytes / BYTES_PER_GIB

    @property
    def download_gib(self) -> float:
        return self.ensemble_download_bytes / BYTES_PER_GIB

    @property
    def phase2_gib(self) -> float:
        return self.phase2_iterative_bytes / BYTES_PER_GIB

    @property
    def normalized_wrt_ofl(self) -> float:
        return self.total_bytes / max(1, self.reference_ofl_bytes)


def compute_one_shot_comm(
    num_clients: int,
    client_model_fp32_bytes: int,
    method_name: str = "One-Shot FL",
) -> CommCostBreakdown:
    """
    Computes communication cost for 1 round of One-Shot FL (Algorithm 1, Lines 3-4:
    initial model broadcast + trained local model upload across all M clients =
    2 * M * |theta_FP32|).
    """
    ofl_bytes = int(2 * num_clients * client_model_fp32_bytes)
    return CommCostBreakdown(
        method_name=method_name,
        phase1_upload_bytes=ofl_bytes,
        ensemble_download_bytes=0,
        phase2_iterative_bytes=0,
        reference_ofl_bytes=ofl_bytes,
    )


def compute_fens_comm_breakdown(
    num_clients: int,
    client_model_fp32_bytes: int,
    client_model_compressed_bytes: int,
    aggregator_fp32_bytes: int,
    aggregator_rounds: int,
    clients_per_round: int,
    method_name: str = "FENS",
    exclude_own_model_on_download: bool = True,
) -> CommCostBreakdown:
    """
    Computes the 3-component communication cost of FENS (matching Figure 8 right):
      1. One-shot local training (Algorithm 1, Lines 3-4): 2 * M * |theta_FP32|
      2. Ensemble download (INT8 quantized, Algorithm 1 Lines 6-7 & Footnote 1):
         M * (M - 1) * |theta_INT8|
      3. Iterative aggregator training (Algorithm 1, Lines 9-12):
         2 * clients_per_round * aggregator_rounds * |lambda_FP32|
    """
    ofl_bytes = int(2 * num_clients * client_model_fp32_bytes)
    models_downloaded_per_client = (
        (num_clients - 1) if exclude_own_model_on_download else num_clients
    )
    download_bytes = int(
        num_clients * models_downloaded_per_client * client_model_compressed_bytes
    )
    phase2_bytes = int(
        2 * clients_per_round * aggregator_rounds * aggregator_fp32_bytes
    )
    return CommCostBreakdown(
        method_name=method_name,
        phase1_upload_bytes=ofl_bytes,
        ensemble_download_bytes=download_bytes,
        phase2_iterative_bytes=phase2_bytes,
        reference_ofl_bytes=ofl_bytes,
    )


def compute_iterative_fl_comm(
    num_clients: int,
    client_model_fp32_bytes: int,
    rounds_to_best: int,
    clients_per_round: int,
    method_name: str = "FedAdam (Iterative)",
) -> CommCostBreakdown:
    """
    Computes communication cost for Iterative FL (e.g., FedAdam, FedAvg):
    In each round, every participating client downloads and uploads the full
    FP32 model (`2 * clients_per_round * rounds_to_best * |theta_FP32|`).
    With full participation (`clients_per_round == num_clients`), R rounds
    equals exactly R * OFL_cost.
    """
    ofl_bytes = int(2 * num_clients * client_model_fp32_bytes)
    iter_bytes = int(2 * clients_per_round * rounds_to_best * client_model_fp32_bytes)
    return CommCostBreakdown(
        method_name=method_name,
        phase1_upload_bytes=0,
        ensemble_download_bytes=0,
        phase2_iterative_bytes=iter_bytes,
        reference_ofl_bytes=ofl_bytes,
    )
