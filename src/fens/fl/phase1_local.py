"""
Phase 1: One-Shot Local Client Training (Algorithm 1 Lines 2-4 & Algorithm 2 Lines 2-5).

Paper Reference:
  - Section 2.2 (Step 1): "Each client i does local training to compute theta^(i),
    identical to OFL, and sends it to the server."
  - Section 3.1 (FENS): "For the CIFAR-10 and CIFAR-100 datasets, each client
    conducts local training for 500 epochs utilizing SGD as the local optimizer
    with an initial learning rate of 0.0025. For the SVHN and AG-News datasets,
    local training extends to 50 and 20 epochs respectively with a learning rate
    of 0.01. The learning rate is decayed using Cosine Annealing across all
    datasets."
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fens.config import HardwareConfig, Phase1Config
from fens.models.resnet8 import build_client_model, get_model_size_bytes


@dataclass
class ClientTrainingStats:
    """Stores training and evaluation metrics for a single client in Phase 1."""

    client_id: int
    num_samples: int
    epochs: int
    final_train_loss: float
    val_accuracy: float
    test_accuracy: float
    wall_time_sec: float
    upload_bytes: int


@torch.no_grad()
def evaluate_single_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cuda",
    num_classes: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluates a single model on `loader`, returning:
      - `overall_accuracy` (percentage in [0, 100])
      - `per_class_accuracy` (array of shape (C,) in [0, 100])
      - `confusion_matrix` (array of shape (C, C) where [r, c] is true r predicted as c)
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

        t_np = labels.cpu().numpy()
        p_np = preds.cpu().numpy()
        for r, c in zip(t_np, p_np):
            conf_mat[int(r), int(c)] += 1

    overall_acc = 100.0 * correct / max(1, total)
    row_totals = conf_mat.sum(axis=1)
    per_class_acc = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if row_totals[c] > 0:
            per_class_acc[c] = 100.0 * conf_mat[c, c] / row_totals[c]

    return overall_acc, per_class_acc, conf_mat


def train_single_client_phase1(
    client_id: int,
    initial_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    phase1_cfg: Phase1Config,
    hw_cfg: HardwareConfig,
    epochs_override: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, ClientTrainingStats]:
    """
    Implements Algorithm 2, Lines 3-5 for a single client `i`:
      - Receives initial model parameters `theta` from the server
      - Trains `theta` locally on `D_i1` (90% split for FENS) using SGD + CosineAnnealingLR
      - Returns converged model `theta^(i)` and training statistics
    """
    device = torch.device(hw_cfg.device if torch.cuda.is_available() else "cpu")
    use_bf16 = (
        hw_cfg.use_bf16
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    model = copy.deepcopy(initial_model).to(device)
    model.train()

    epochs = epochs_override if epochs_override is not None else phase1_cfg.local_epochs
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=phase1_cfg.learning_rate,
        momentum=phase1_cfg.momentum,
        weight_decay=phase1_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    criterion = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    final_loss = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        final_loss = running_loss / max(1, n_batches)

    wall_time = time.perf_counter() - t0
    val_acc, _, _ = evaluate_single_model(model, val_loader, device=device)
    test_acc, _, _ = evaluate_single_model(model, test_loader, device=device)

    num_samples = len(train_loader.dataset)  # type: ignore[arg-type]
    upload_bytes = get_model_size_bytes(model, bits_per_param=32)

    stats = ClientTrainingStats(
        client_id=client_id,
        num_samples=num_samples,
        epochs=epochs,
        final_train_loss=final_loss,
        val_accuracy=val_acc,
        test_accuracy=test_acc,
        wall_time_sec=wall_time,
        upload_bytes=upload_bytes,
    )

    if verbose:
        print(
            f"  [Phase 1 | Client {client_id:02d}] Samples={num_samples:5d} | "
            f"Epochs={epochs:3d} | Loss={final_loss:.4f} | "
            f"Val Acc={val_acc:5.2f}% | Test Acc={test_acc:5.2f}% | "
            f"Time={wall_time:.1f}s"
        )

    model.cpu()
    return model, stats


def train_all_clients_phase1(
    data_bundle,
    model_arch: str,
    phase1_cfg: Phase1Config,
    hw_cfg: HardwareConfig,
    use_full_100_percent: bool = False,
    epochs_override: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[ nn.Module, List[nn.Module], List[ClientTrainingStats]]:
    """
    Runs Phase 1 local training across all M clients starting from a common
    server-initialized model `theta` (Algorithm 1, Lines 2-4).

    Returns:
      - `initial_server_model`: The initial global model `theta` before local training
      - `client_models`: List of M locally converged client models `[theta^(0), ..., theta^(M-1)]`
      - `client_stats`: List of M `ClientTrainingStats` records
    """
    torch.manual_seed(seed)
    initial_server_model = build_client_model(
        arch=model_arch, num_classes=data_bundle.num_classes
    )

    val_loader = data_bundle.get_eval_loader(
        split="val",
        batch_size=256,
        num_workers=hw_cfg.num_workers,
        pin_memory=hw_cfg.pin_memory,
    )
    test_loader = data_bundle.get_eval_loader(
        split="test",
        batch_size=256,
        num_workers=hw_cfg.num_workers,
        pin_memory=hw_cfg.pin_memory,
    )

    client_models: List[nn.Module] = []
    client_stats: List[ClientTrainingStats] = []

    for client_id in range(data_bundle.num_clients):
        train_loader = data_bundle.get_client_phase1_loader(
            client_id=client_id,
            batch_size=phase1_cfg.batch_size,
            use_full_100_percent=use_full_100_percent,
            num_workers=hw_cfg.num_workers,
            pin_memory=hw_cfg.pin_memory,
        )
        c_model, c_stat = train_single_client_phase1(
            client_id=client_id,
            initial_model=initial_server_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            phase1_cfg=phase1_cfg,
            hw_cfg=hw_cfg,
            epochs_override=epochs_override,
            verbose=verbose,
        )
        client_models.append(c_model)
        client_stats.append(c_stat)

    return initial_server_model, client_models, client_stats


def save_phase1_checkpoints(
    save_dir: Path | str,
    initial_model: nn.Module,
    client_models: List[nn.Module],
    client_stats: List[ClientTrainingStats],
    metadata: Optional[Dict] = None,
) -> Path:
    """
    Saves Phase 1 client checkpoints to disk so Step 2 (Phase 2 aggregator
    training and live demos) can be run instantaneously without re-training
    local models.
    """
    out_path = Path(save_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    torch.save(initial_model.state_dict(), out_path / "initial_server_model.pt")
    for idx, model in enumerate(client_models):
        torch.save(model.state_dict(), out_path / f"client_{idx:02d}.pt")

    summary = {
        "num_clients": len(client_models),
        "metadata": metadata or {},
        "client_stats": [asdict(s) for s in client_stats],
    }
    with open(out_path / "phase1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return out_path


def load_phase1_checkpoints(
    checkpoint_dir: Path | str,
    model_arch: str = "resnet8",
    num_classes: int = 10,
    device: torch.device | str = "cpu",
) -> Tuple[nn.Module, List[nn.Module], Dict]:
    """Loads saved Phase 1 client models and summary metadata from `checkpoint_dir`."""
    ckpt_path = Path(checkpoint_dir)
    summary_file = ckpt_path / "phase1_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(
            f"Phase 1 checkpoint summary not found at {summary_file}. "
            "Run Step 1 (`scripts/01_train_phase1_clients.py`) first."
        )

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    num_clients = int(summary["num_clients"])
    initial_model = build_client_model(arch=model_arch, num_classes=num_classes)
    init_pt = ckpt_path / "initial_server_model.pt"
    if init_pt.exists():
        initial_model.load_state_dict(
            torch.load(init_pt, map_location=device, weights_only=True)
        )
    initial_model.to(device)

    client_models: List[nn.Module] = []
    for idx in range(num_clients):
        m = build_client_model(arch=model_arch, num_classes=num_classes)
        c_pt = ckpt_path / f"client_{idx:02d}.pt"
        m.load_state_dict(torch.load(c_pt, map_location=device, weights_only=True))
        m.to(device)
        m.eval()
        client_models.append(m)

    return initial_model, client_models, summary
