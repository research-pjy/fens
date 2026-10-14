"""
One-Shot FL Baselines, Iterative FL Baselines, and Server-Side Knowledge Distillation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fens.aggregators.models import WeightedAveragingAggregator
from fens.config import DistillationConfig, HardwareConfig, IterativeFLConfig
from fens.fl.phase1_local import evaluate_single_model
from fens.models.resnet8 import build_client_model, get_model_size_bytes


def average_client_parameters(
    client_models: List[nn.Module],
    client_weights: Optional[Sequence[float]] = None,
) -> nn.Module:
    """
    Computes the weighted parameter average of `client_models`:
      theta_global = sum_{i=1}^M w_i * theta^(i)
    """
    num_clients = len(client_models)
    if client_weights is None:
        weights = [1.0 / num_clients] * num_clients
    else:
        total_w = float(sum(client_weights))
        weights = [float(w) / max(1e-12, total_w) for w in client_weights]

    avg_model = copy.deepcopy(client_models[0])
    avg_state = avg_model.state_dict()

    with torch.no_grad():
        for key, tensor in avg_state.items():
            if tensor.is_floating_point():
                acc = torch.zeros_like(tensor, dtype=torch.float32)
                for w, m in zip(weights, client_models):
                    acc.add_(m.state_dict()[key].to(acc.device, dtype=torch.float32), alpha=w)
                avg_state[key].copy_(acc.to(tensor.dtype))
            else:
                avg_state[key].copy_(client_models[0].state_dict()[key])

    avg_model.load_state_dict(avg_state)
    return avg_model


def evaluate_one_shot_fedavg(
    client_models: List[nn.Module],
    data_bundle,
    hw_cfg: HardwareConfig,
    use_sample_weights: bool = True,
) -> Tuple[nn.Module, float, float, np.ndarray, np.ndarray, int]:
    """
    Evaluates One-Shot FedAvg (1 communication round of parameter averaging).
    """
    device = torch.device(hw_cfg.device if torch.cuda.is_available() else "cpu")
    if use_sample_weights:
        weights = [
            float(len(data_bundle.partition.client_d1_indices[i]))
            for i in range(len(client_models))
        ]
    else:
        weights = [1.0] * len(client_models)

    fedavg_model = average_client_parameters(client_models, client_weights=weights)
    val_loader = data_bundle.get_eval_loader(
        split="val", batch_size=256, num_workers=hw_cfg.num_workers, pin_memory=hw_cfg.pin_memory
    )
    test_loader = data_bundle.get_eval_loader(
        split="test", batch_size=256, num_workers=hw_cfg.num_workers, pin_memory=hw_cfg.pin_memory
    )

    val_acc, _, _ = evaluate_single_model(
        fedavg_model, val_loader, device=device, num_classes=data_bundle.num_classes
    )
    test_acc, per_class_acc, conf_mat = evaluate_single_model(
        fedavg_model, test_loader, device=device, num_classes=data_bundle.num_classes
    )
    # 1 round of OFL = initial model broadcast + trained model upload across all M clients
    comm_bytes = 2 * sum(get_model_size_bytes(m, bits_per_param=32) for m in client_models)

    return fedavg_model, val_acc, test_acc, per_class_acc, conf_mat, comm_bytes


def distill_ensemble_to_student(
    ensemble_models: List[nn.Module],
    aggregator: nn.Module,
    aux_dataset: Dataset,
    val_loader: DataLoader,
    test_loader: DataLoader,
    student_arch: str,
    num_classes: int,
    distill_cfg: DistillationConfig,
    hw_cfg: HardwareConfig,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[nn.Module, float, float]:
    """
    Server-Side Knowledge Distillation (Hinton et al. [15]) for FedKD and FENS Distilled.
    """
    torch.manual_seed(seed)
    device = torch.device(hw_cfg.device if torch.cuda.is_available() else "cpu")
    use_bf16 = (
        hw_cfg.use_bf16
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    for m in ensemble_models:
        m.eval()
        m.to(device)
    aggregator.eval()
    aggregator.to(device)

    aux_eval_loader = DataLoader(
        aux_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=hw_cfg.num_workers,
        pin_memory=hw_cfg.pin_memory,
    )
    teacher_logits_list: List[torch.Tensor] = []
    aux_images_list: List[torch.Tensor] = []

    with torch.no_grad():
        for images, _ in aux_eval_loader:
            images_dev = images.to(device, non_blocking=True)
            per_model = [m(images_dev) for m in ensemble_models]
            concat_z = torch.cat(per_model, dim=-1)
            t_logits = aggregator(concat_z, images=images_dev)
            teacher_logits_list.append(t_logits.cpu())
            aux_images_list.append(images.cpu())

    all_teacher_logits = torch.cat(teacher_logits_list, dim=0)
    all_aux_images = torch.cat(aux_images_list, dim=0)

    kd_dataset = torch.utils.data.TensorDataset(all_aux_images, all_teacher_logits)
    kd_loader = DataLoader(
        kd_dataset,
        batch_size=distill_cfg.batch_size,
        shuffle=True,
        num_workers=hw_cfg.num_workers,
        pin_memory=hw_cfg.pin_memory,
    )

    student = build_client_model(arch=student_arch, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=distill_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, distill_cfg.distill_epochs)
    )
    temp = distill_cfg.temperature

    best_val_acc = -1.0
    best_test_acc = 0.0
    best_state = copy.deepcopy(student.state_dict())

    for epoch in range(1, distill_cfg.distill_epochs + 1):
        student.train()
        for images, t_logits in kd_loader:
            images = images.to(device, non_blocking=True)
            t_logits = t_logits.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    s_logits = student(images)
                    loss = F.kl_div(
                        F.log_softmax(s_logits / temp, dim=-1),
                        F.softmax(t_logits / temp, dim=-1),
                        reduction="batchmean",
                    ) * (temp * temp)
            else:
                s_logits = student(images)
                loss = F.kl_div(
                    F.log_softmax(s_logits / temp, dim=-1),
                    F.softmax(t_logits / temp, dim=-1),
                    reduction="batchmean",
                ) * (temp * temp)

            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == distill_cfg.distill_epochs:
            val_acc, _, _ = evaluate_single_model(
                student, val_loader, device=device, num_classes=num_classes
            )
            test_acc, _, _ = evaluate_single_model(
                student, test_loader, device=device, num_classes=num_classes
            )
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_state = copy.deepcopy(student.state_dict())
            if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == distill_cfg.distill_epochs):
                print(
                    f"  [Server KD | Epoch {epoch:2d}/{distill_cfg.distill_epochs}] "
                    f"Val Acc={val_acc:5.2f}% | Test Acc={test_acc:5.2f}%"
                )

    student.load_state_dict(best_state)
    return student, best_val_acc, best_test_acc


def run_fedkd_one_shot(
    client_models: List[nn.Module],
    data_bundle,
    aux_dataset: Dataset,
    student_arch: str,
    distill_cfg: DistillationConfig,
    hw_cfg: HardwareConfig,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[nn.Module, float, float, int]:
    """
    Runs the One-Shot FedKD baseline (Gong et al., AAAI 2022 [10]; Appendix B.1 & B.4).
    """
    weighted_agg = WeightedAveragingAggregator(
        num_clients=data_bundle.num_clients,
        num_classes=data_bundle.num_classes,
        class_counts_matrix=data_bundle.partition.class_counts_d1,
    )
    val_loader = data_bundle.get_eval_loader(
        split="val", batch_size=256, num_workers=hw_cfg.num_workers, pin_memory=hw_cfg.pin_memory
    )
    test_loader = data_bundle.get_eval_loader(
        split="test", batch_size=256, num_workers=hw_cfg.num_workers, pin_memory=hw_cfg.pin_memory
    )

    student, val_acc, test_acc = distill_ensemble_to_student(
        ensemble_models=client_models,
        aggregator=weighted_agg,
        aux_dataset=aux_dataset,
        val_loader=val_loader,
        test_loader=test_loader,
        student_arch=student_arch,
        num_classes=data_bundle.num_classes,
        distill_cfg=distill_cfg,
        hw_cfg=hw_cfg,
        seed=seed,
        verbose=verbose,
    )
    comm_bytes = 2 * sum(get_model_size_bytes(m, bits_per_param=32) for m in client_models)
    return student, val_acc, test_acc, comm_bytes


@dataclass
class IterativeFLResult:
    """Stores results and communication cost metrics for an Iterative FL baseline."""

    algorithm: str
    best_val_accuracy: float
    test_accuracy_at_best_val: float
    final_test_accuracy: float
    best_round: int
    total_rounds: int
    wall_time_sec: float
    comm_bytes_at_best: int
    total_comm_bytes: int
    round_history: List[Dict[str, float]] = field(default_factory=list)

    @property
    def comm_gib_at_best(self) -> float:
        return self.comm_bytes_at_best / (1024.0 ** 3)


def run_iterative_fl_baseline(
    data_bundle,
    model_arch: str,
    fl_cfg: IterativeFLConfig,
    hw_cfg: HardwareConfig,
    seed: int = 42,
    target_accuracy_to_match: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, IterativeFLResult]:
    """
    Runs an Iterative Federated Learning baseline (`FedAdam`, `FedYogi`, `FedAvg`,
    or `FedProx`) across `fl_cfg.global_rounds` (Appendix B.2, Table 6).
    """
    torch.manual_seed(seed)
    device = torch.device(hw_cfg.device if torch.cuda.is_available() else "cpu")
    use_bf16 = (
        hw_cfg.use_bf16
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    global_model = build_client_model(
        arch=model_arch, num_classes=data_bundle.num_classes
    ).to(device)
    local_model = copy.deepcopy(global_model).to(device)

    model_bytes = get_model_size_bytes(global_model, bits_per_param=32)
    num_clients = data_bundle.num_clients
    comm_bytes_per_round = 2 * num_clients * model_bytes

    client_loaders = [
        data_bundle.get_client_phase1_loader(
            client_id=cid,
            batch_size=fl_cfg.batch_size,
            use_full_100_percent=True,
            num_workers=hw_cfg.num_workers,
            pin_memory=hw_cfg.pin_memory,
        )
        for cid in range(num_clients)
    ]
    client_sample_counts = [
        float(len(data_bundle.partition.client_full_indices[cid]))
        for cid in range(num_clients)
    ]
    total_samples = sum(client_sample_counts)
    client_weights = [c / max(1.0, total_samples) for c in client_sample_counts]

    val_loader = data_bundle.get_eval_loader(
        split="val", batch_size=256, num_workers=hw_cfg.num_workers, pin_memory=hw_cfg.pin_memory
    )
    test_loader = data_bundle.get_eval_loader(
        split="test", batch_size=256, num_workers=hw_cfg.num_workers, pin_memory=hw_cfg.pin_memory
    )

    algo = fl_cfg.algorithm.lower()
    server_m: Dict[str, torch.Tensor] = {
        k: torch.zeros_like(p.data) for k, p in global_model.named_parameters()
    }
    server_v: Dict[str, torch.Tensor] = {
        k: torch.zeros_like(p.data) for k, p in global_model.named_parameters()
    }

    criterion = nn.CrossEntropyLoss()
    best_val_acc = -1.0
    test_acc_at_best_val = 0.0
    best_round = 1
    best_state = copy.deepcopy(global_model.state_dict())
    round_history: List[Dict[str, float]] = []

    t0 = time.perf_counter()

    for rnd in range(1, fl_cfg.global_rounds + 1):
        global_state = {
            k: v.detach().clone() for k, v in global_model.state_dict().items()
        }
        global_params_ref = {
            k: p.detach().clone() for k, p in global_model.named_parameters()
        }

        avg_state_accum: Dict[str, torch.Tensor] = {
            k: torch.zeros_like(v, dtype=torch.float32)
            for k, v in global_state.items()
            if v.is_floating_point()
        }

        for cid in range(num_clients):
            local_model.load_state_dict(global_state)
            local_model.train()
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=fl_cfg.client_lr,
                momentum=0.9,
                weight_decay=1e-4,
            )

            for _ in range(fl_cfg.local_epochs):
                for images, labels in client_loaders[cid]:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    if use_bf16:
                        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = local_model(images)
                            loss = criterion(logits, labels)
                    else:
                        logits = local_model(images)
                        loss = criterion(logits, labels)

                    if algo == "fedprox":
                        prox_term = torch.tensor(0.0, device=device)
                        for name, p in local_model.named_parameters():
                            prox_term = prox_term + torch.sum(
                                (p - global_params_ref[name]) ** 2
                            )
                        loss = loss + 0.5 * fl_cfg.prox_mu * prox_term

                    loss.backward()
                    optimizer.step()

            w_i = client_weights[cid]
            with torch.no_grad():
                for k, v in local_model.state_dict().items():
                    if k in avg_state_accum:
                        avg_state_accum[k].add_(v.to(torch.float32), alpha=w_i)

        with torch.no_grad():
            if algo in ("fedavg", "fedprox"):
                new_state = copy.deepcopy(global_state)
                for k, v in avg_state_accum.items():
                    new_state[k].copy_(v.to(global_state[k].dtype))
                global_model.load_state_dict(new_state)

            elif algo in ("fedadam", "fedyogi"):
                new_state = copy.deepcopy(global_state)
                for k, v in avg_state_accum.items():
                    if k not in server_m:
                        new_state[k].copy_(v.to(global_state[k].dtype))
                global_model.load_state_dict(new_state)

                for name, param in global_model.named_parameters():
                    delta = avg_state_accum[name].to(param.dtype) - global_state[name]
                    server_m[name].mul_(fl_cfg.beta1).add_(
                        delta, alpha=(1.0 - fl_cfg.beta1)
                    )
                    delta_sq = delta * delta
                    if algo == "fedadam":
                        server_v[name].mul_(fl_cfg.beta2).add_(
                            delta_sq, alpha=(1.0 - fl_cfg.beta2)
                        )
                    else:
                        sign = torch.sign(server_v[name] - delta_sq)
                        server_v[name].sub_(
                            (1.0 - fl_cfg.beta2) * delta_sq * sign
                        )
                    denom = torch.sqrt(server_v[name].clamp(min=1e-12)) + fl_cfg.tau
                    param.data.addcdiv_(
                        server_m[name], denom, value=fl_cfg.server_lr
                    )
            else:
                raise ValueError(f"Unknown Iterative FL algorithm: {fl_cfg.algorithm}")

        val_acc, _, _ = evaluate_single_model(
            global_model, val_loader, device=device, num_classes=data_bundle.num_classes
        )
        test_acc, _, _ = evaluate_single_model(
            global_model, test_loader, device=device, num_classes=data_bundle.num_classes
        )
        round_history.append(
            {"round": float(rnd), "val_acc": val_acc, "test_acc": test_acc}
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            test_acc_at_best_val = test_acc
            best_round = rnd
            best_state = copy.deepcopy(global_model.state_dict())

        if verbose and (rnd == 1 or rnd % 10 == 0 or rnd == fl_cfg.global_rounds):
            print(
                f"  [Iterative FL | {algo.upper()} | Round {rnd:3d}/{fl_cfg.global_rounds}] "
                f"Val Acc={val_acc:5.2f}% | Test Acc={test_acc:5.2f}% "
                f"(Best Test@Val={test_acc_at_best_val:5.2f}% @ R{best_round})"
            )

    final_test_acc = round_history[-1]["test_acc"] if round_history else 0.0
    global_model.load_state_dict(best_state)
    wall_time = time.perf_counter() - t0

    result = IterativeFLResult(
        algorithm=fl_cfg.algorithm,
        best_val_accuracy=best_val_acc,
        test_accuracy_at_best_val=test_acc_at_best_val,
        final_test_accuracy=final_test_acc,
        best_round=best_round,
        total_rounds=fl_cfg.global_rounds,
        wall_time_sec=wall_time,
        comm_bytes_at_best=best_round * comm_bytes_per_round,
        total_comm_bytes=fl_cfg.global_rounds * comm_bytes_per_round,
        round_history=round_history,
    )
    return global_model, result
