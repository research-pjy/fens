"""
Federated Learning training engines for FENS (Phase 1 & Phase 2), One-Shot FL
baselines, Iterative FL baselines, and Server-Side Knowledge Distillation.
"""

from fens.fl.phase1_local import (
    ClientTrainingStats,
    evaluate_single_model,
    load_phase1_checkpoints,
    save_phase1_checkpoints,
    train_all_clients_phase1,
    train_single_client_phase1,
)
from fens.fl.phase2_aggregator import (
    Phase2TrainingResult,
    evaluate_fens_ensemble,
    precompute_ensemble_logits,
    train_fens_aggregator_phase2,
)
from fens.fl.baselines import (
    IterativeFLResult,
    distill_ensemble_to_student,
    evaluate_one_shot_fedavg,
    run_fedkd_one_shot,
    run_iterative_fl_baseline,
)

__all__ = [
    "ClientTrainingStats",
    "train_single_client_phase1",
    "train_all_clients_phase1",
    "save_phase1_checkpoints",
    "load_phase1_checkpoints",
    "evaluate_single_model",
    "Phase2TrainingResult",
    "precompute_ensemble_logits",
    "train_fens_aggregator_phase2",
    "evaluate_fens_ensemble",
    "IterativeFLResult",
    "evaluate_one_shot_fedavg",
    "run_fedkd_one_shot",
    "run_iterative_fl_baseline",
    "distill_ensemble_to_student",
]
