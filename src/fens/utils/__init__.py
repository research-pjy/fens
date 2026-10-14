"""
Utility modules for hardware inspection on `rama` (L40S), communication cost
accounting, reproducibility seeding, and publication-quality plotting.
"""

from fens.utils.comm_cost import (
    CommCostBreakdown,
    compute_fens_comm_breakdown,
    compute_iterative_fl_comm,
    compute_one_shot_comm,
)
from fens.utils.hardware import EnvironmentReport, inspect_execution_environment, set_global_seed
from fens.utils.plotting import (
    plot_aggregator_dissection,
    plot_dirichlet_bubbles,
    plot_fens_vs_baselines,
    plot_two_client_skew_diagnostics,
)

__all__ = [
    "EnvironmentReport",
    "inspect_execution_environment",
    "set_global_seed",
    "CommCostBreakdown",
    "compute_one_shot_comm",
    "compute_fens_comm_breakdown",
    "compute_iterative_fl_comm",
    "plot_dirichlet_bubbles",
    "plot_fens_vs_baselines",
    "plot_aggregator_dissection",
    "plot_two_client_skew_diagnostics",
]
