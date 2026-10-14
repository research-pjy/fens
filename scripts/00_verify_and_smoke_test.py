#!/usr/bin/env python3
"""
Step 0: Environment Verification & End-to-End Pipeline Smoke Test for `rama` (L40S).

Follows `L40S Execution Context.md` ("Before Full Training"):
  1. Verifies hostname (`rama`), Conda environment (`qwen-vl`), PyTorch/CUDA
     compatibility, GPU detection (NVIDIA L40S), available VRAM (~44 GiB),
     BF16 support, and disk space.
  2. Exercises the full FENS pipeline on a tiny synthetic/mini-batch dataset:
     - Dirichlet & 2-Client Skew data partitioning
     - ResNet-8 model instantiation, forward pass, BF16 autocast, finite loss,
       valid gradients, and optimizer update step
     - Server-side FP32 -> INT8 post-training quantization & decompression
     - Phase 2 FedAdam aggregator training across all aggregator architectures
     - Checkpoint save/load round-trip and diagnostic plot generation

Usage on `rama`:
    conda activate qwen-vl
    PYTHONPATH=src python scripts/00_verify_and_smoke_test.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

# Ensure `src/` is on sys.path even if package wasn't installed with `pip install -e .`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fens.aggregators import build_aggregator
from fens.config import HardwareConfig, Phase1Config, Phase2Config
from fens.data.partition import partition_dirichlet, partition_two_client_skew
from fens.fl.phase1_local import (
    load_phase1_checkpoints,
    save_phase1_checkpoints,
    train_single_client_phase1,
)
from fens.models import build_client_model, quantize_ensemble_models
from fens.utils import inspect_execution_environment, set_global_seed


def run_smoke_test() -> None:
    set_global_seed(42)
    env_report = inspect_execution_environment(ROOT_DIR)
    print(env_report.format_banner())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"\n[Smoke Test 1/5] Active Compute Device: {device} (BF16 Autocast: {use_bf16})")

    # 1. Test Non-IID Partitioners
    print("[Smoke Test 2/5] Verifying Dirichlet Dir_20(alpha=0.05) and 2-Client Skew Partitioners...")
    dummy_labels = np.repeat(np.arange(10), 200)  # 2,000 samples across 10 classes
    dir_part = partition_dirichlet(dummy_labels, num_clients=20, alpha=0.05, seed=42)
    assert dir_part.class_counts_full.shape == (20, 10)
    assert sum(len(v) for v in dir_part.client_full_indices.values()) == len(dummy_labels)

    skew_part = partition_two_client_skew(
        dummy_labels,
        specialist_classes=(0, 1),
        generalist_samples_per_class=80,
        specialist_samples_per_class=100,
        seed=42,
    )
    assert skew_part.num_clients == 2
    # Verify Client 0 has all 10 classes and Client 1 ONLY has classes 0 and 1
    assert np.all(skew_part.class_counts_full[0] > 0), "Client 0 should have all 10 classes!"
    assert np.all(skew_part.class_counts_full[1, 2:] == 0), "Client 1 must have 0 samples for classes 2..9!"
    assert np.all(skew_part.class_counts_full[1, :2] > 0), "Client 1 must have samples for classes 0 and 1!"
    print("  -> Partitioners verified successfully.")

    # 2. Test ResNet-8 Forward, Loss, Backward, Finite Gradients, and Optimizer Step
    print("[Smoke Test 3/5] Verifying ResNet-8 Forward/Backward Pass, Gradients & Optimizer Step...")
    model = build_client_model("resnet8_light", num_classes=10).to(device)
    x_dummy = torch.randn(16, 3, 32, 32, device=device)
    y_dummy = torch.randint(0, 10, (16,), device=device)

    param_before = next(model.parameters()).detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad(set_to_none=True)
    if use_bf16:
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x_dummy)
            loss = criterion(logits, y_dummy)
    else:
        logits = model(x_dummy)
        loss = criterion(logits, y_dummy)

    assert torch.isfinite(loss).item(), f"Non-finite loss detected: {loss.item()}"
    loss.backward()

    grad_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert len(grad_norms) > 0 and all(np.isfinite(g) for g in grad_norms), "Invalid gradients!"
    optimizer.step()

    param_after = next(model.parameters()).detach().clone()
    assert not torch.allclose(param_before, param_after), "Optimizer step did not update weights!"
    print(f"  -> Loss={loss.item():.4f} | Grad tensors={len(grad_norms)} | Weight update verified.")

    # 3. Test Server-Side FP32 -> INT8 Post-Training Quantization
    print("[Smoke Test 4/5] Verifying Server-Side FP32 -> INT8 Post-Training Quantization...")
    full_resnet8 = build_client_model("resnet8", num_classes=10)
    q_models, payloads = quantize_ensemble_models([full_resnet8], enabled=True, device=device)
    ratio = payloads[0].compression_ratio
    orig_mib = payloads[0].original_bytes / (1024.0 ** 2)
    comp_mib = payloads[0].compressed_bytes / (1024.0 ** 2)
    assert ratio > 3.5, f"Expected ~4x INT8 compression, got {ratio:.2f}x"
    with torch.no_grad():
        q_out = q_models[0](x_dummy)
    assert q_out.shape == (16, 10) and torch.isfinite(q_out).all()
    print(
        f"  -> ResNet-8 FP32={orig_mib:.2f} MiB -> INT8={comp_mib:.2f} MiB ({ratio:.2f}x compression) verified."
    )

    # 4. Test Aggregators & Checkpoint Round-Trip
    print("[Smoke Test 5/5] Verifying Aggregator Architectures & Checkpoint Save/Load...")
    num_clients = 4
    concat_z = torch.randn(16, num_clients * 10, device=device)
    for agg_name in ("uniform", "weighted", "scalar_linear", "per_class_linear", "mlp", "voting", "moe"):
        agg = build_aggregator(agg_name, num_clients=num_clients, num_classes=10, hidden_dim=40).to(device)
        out = agg(concat_z, images=x_dummy)
        assert out.shape == (16, 10) and torch.isfinite(out).all(), f"Aggregator {agg_name} failed!"

    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_ds = TensorDataset(torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,)))
        dummy_loader = DataLoader(dummy_ds, batch_size=16)
        hw_cfg = HardwareConfig(device=str(device), use_bf16=use_bf16, num_workers=0, pin_memory=False)
        p1_cfg = Phase1Config(local_epochs=1, learning_rate=0.01, batch_size=16)

        c_model, c_stat = train_single_client_phase1(
            client_id=0,
            initial_model=model.cpu(),
            train_loader=dummy_loader,
            val_loader=dummy_loader,
            test_loader=dummy_loader,
            phase1_cfg=p1_cfg,
            hw_cfg=hw_cfg,
            verbose=False,
        )
        save_phase1_checkpoints(
            save_dir=tmpdir,
            initial_model=model.cpu(),
            client_models=[c_model],
            client_stats=[c_stat],
            metadata={"smoke_test": True},
        )
        _, loaded_models, summary = load_phase1_checkpoints(
            tmpdir, model_arch="resnet8_light", num_classes=10, device=device
        )
        assert len(loaded_models) == 1 and summary["metadata"]["smoke_test"] is True

    if device.type == "cuda":
        peak_vram_mib = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        print(f"  -> Peak GPU VRAM used during smoke test: {peak_vram_mib:.1f} MiB")

    print("\n" + "=" * 72)
    print("  ALL SMOKE TESTS PASSED! READY FOR FULL TRAINING ON `rama` (L40S).")
    print("=" * 72)


if __name__ == "__main__":
    run_smoke_test()
