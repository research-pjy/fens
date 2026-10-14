"""
Server-Side FP32 -> INT8 Post-Training Model Quantization for FENS.

Paper Reference:
  - Algorithm 1, Lines 5-6:
      if quantize then
          theta^(i) <- quantization_alg(theta^(i), FP32, INT8) for all i in [M]
  - Appendix B.3 (Model quantization in FENS):
      "Clients in FENS incur a critical cost of downloading the ensemble model
      from the server to initiate the aggregator training. To reduce the
      communication burden on the clients, the server employs post-training model
      quantization of all received client local models from FP32 to INT8, reducing
      the download costs by 4x... The quantization results in a drop of ~1-2% test
      accuracy for every client model compared to the corresponding non-quantized
      model. However, the subsequent aggregator training phase in FENS provides
      resilience to this drop in the accuracy of client models in the ensemble. In
      fact, we observe that the final accuracy achieved after aggregator training
      is slightly higher when using the quantized models as compared to unquantized
      models due to the regularising effect of quantization on generated logits."

Implementation Note for NVIDIA L40S (`rama`):
  PyTorch's eager CPU dynamic quantization (`torch.ao.quantization.quantize_dynamic`)
  only targets `nn.Linear` layers on CPU and does not run `nn.Conv2d` on CUDA GPUs.
  To faithfully implement 8-bit post-training quantization across ALL Conv2d and
  Linear layers while still executing ensemble inference at full speed on the L40S
  CUDA GPU, we implement symmetric per-channel INT8 weight quantization:
    1. On the server, each FP32 weight tensor W is quantized to `int8` in [-127, 127]
       with per-output-channel scale s_c = max(|W_c|) / 127.0, reducing the
       transmitted state payload by 4x.
    2. When reconstructed on the client (on CUDA/L40S), W_hat = int8_W * s_c
       preserves the exact INT8 quantization rounding/regularization effect on
       generated logits while running native CUDA kernels!
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


@dataclass
class QuantizedStatePayload:
    """
    Represents the compressed INT8 state dictionary transmitted from the server
    to clients in Algorithm 1, Line 7.
    """

    int8_tensors: Dict[str, torch.Tensor]
    scales: Dict[str, torch.Tensor]
    non_quantized_tensors: Dict[str, torch.Tensor]
    original_bytes: int
    compressed_bytes: int

    @property
    def compression_ratio(self) -> float:
        return self.original_bytes / max(1, self.compressed_bytes)


def compress_state_dict_to_int8(
    state_dict: Dict[str, torch.Tensor],
) -> QuantizedStatePayload:
    """
    Compresses a model's FP32 state_dict into INT8 weights + scales.
    Quantizes all >=2D floating-point weight tensors (`Conv2d`, `Linear`) using
    symmetric per-channel INT8 quantization (`[-127, 127]`).
    """
    int8_tensors: Dict[str, torch.Tensor] = {}
    scales: Dict[str, torch.Tensor] = {}
    non_quantized: Dict[str, torch.Tensor] = {}

    original_bytes = 0
    compressed_bytes = 0

    for name, tensor in state_dict.items():
        t_cpu = tensor.detach().cpu()
        t_bytes = t_cpu.numel() * t_cpu.element_size()
        original_bytes += t_bytes

        # Quantize Conv2d and Linear weight matrices (ndim >= 2, floating point)
        if t_cpu.is_floating_point() and t_cpu.ndim >= 2 and t_cpu.numel() >= 64:
            # Per-output-channel scale along dim 0
            flat = t_cpu.view(t_cpu.size(0), -1)
            max_val = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale = max_val / 127.0

            # Reshape scale to broadcast along all dimensions of t_cpu
            scale_shape = [t_cpu.size(0)] + [1] * (t_cpu.ndim - 1)
            scale_broad = scale.view(*scale_shape)

            q_int8 = torch.clamp(torch.round(t_cpu / scale_broad), -127, 127).to(
                torch.int8
            )
            int8_tensors[name] = q_int8
            scales[name] = scale.squeeze(1).to(torch.float16)

            # 1 byte per weight + 2 bytes per output channel scale
            compressed_bytes += q_int8.numel() * 1 + scales[name].numel() * 2
        else:
            # Small 1D tensors (BatchNorm running stats / biases)
            if t_cpu.is_floating_point():
                t_half = t_cpu.to(torch.float16)
                non_quantized[name] = t_half
                compressed_bytes += t_half.numel() * 2
            else:
                non_quantized[name] = t_cpu
                compressed_bytes += t_bytes

    return QuantizedStatePayload(
        int8_tensors=int8_tensors,
        scales=scales,
        non_quantized_tensors=non_quantized,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
    )


def decompress_int8_state_dict(
    payload: QuantizedStatePayload,
    target_device: torch.device | str = "cpu",
    target_dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Reconstructs a state_dict from a `QuantizedStatePayload` on the client side,
    retaining the exact INT8 quantization rounding behavior while allowing fast
    execution on `target_device` (e.g., L40S CUDA GPU).
    """
    reconstructed: Dict[str, torch.Tensor] = {}

    for name, q_int8 in payload.int8_tensors.items():
        scale = payload.scales[name].to(device=target_device, dtype=target_dtype)
        scale_shape = [q_int8.size(0)] + [1] * (q_int8.ndim - 1)
        scale_broad = scale.view(*scale_shape)
        dequant = q_int8.to(device=target_device, dtype=target_dtype) * scale_broad
        reconstructed[name] = dequant

    for name, tensor in payload.non_quantized_tensors.items():
        if tensor.is_floating_point():
            reconstructed[name] = tensor.to(device=target_device, dtype=target_dtype)
        else:
            reconstructed[name] = tensor.to(device=target_device)

    return reconstructed


def quantize_client_model_int8(
    model: nn.Module,
    device: torch.device | str = "cpu",
) -> Tuple[nn.Module, QuantizedStatePayload]:
    """
    Quantizes a single client model from FP32 to INT8 and returns:
      1. A copy of `model` loaded with the INT8-quantized parameters on `device`
      2. The `QuantizedStatePayload` containing byte-size metrics
    """
    q_model = copy.deepcopy(model)
    payload = compress_state_dict_to_int8(q_model.state_dict())
    dequant_state = decompress_int8_state_dict(payload, target_device=device)
    q_model.load_state_dict(dequant_state)
    q_model.to(device)
    q_model.eval()
    return q_model, payload


def quantize_ensemble_models(
    client_models: List[nn.Module],
    enabled: bool = True,
    device: torch.device | str = "cpu",
) -> Tuple[List[nn.Module], List[QuantizedStatePayload]]:
    """
    Implements Algorithm 1, Lines 5-6 across all M client models in the ensemble.
    """
    processed_models: List[nn.Module] = []
    payloads: List[QuantizedStatePayload] = []

    for model in client_models:
        if enabled:
            q_model, payload = quantize_client_model_int8(model, device=device)
            processed_models.append(q_model)
            payloads.append(payload)
        else:
            m_copy = copy.deepcopy(model).to(device)
            m_copy.eval()
            raw_bytes = sum(
                t.numel() * t.element_size() for t in m_copy.state_dict().values()
            )
            processed_models.append(m_copy)
            payloads.append(
                QuantizedStatePayload(
                    int8_tensors={},
                    scales={},
                    non_quantized_tensors={},
                    original_bytes=raw_bytes,
                    compressed_bytes=raw_bytes,
                )
            )

    return processed_models, payloads
