"""
Client neural network architectures and server-side FP32 -> INT8 post-training
quantization utilities for FENS.
"""

from fens.models.resnet8 import ResNet8, build_client_model, get_model_size_bytes, get_model_size_mib
from fens.models.quantization import (
    QuantizedStatePayload,
    compress_state_dict_to_int8,
    decompress_int8_state_dict,
    quantize_client_model_int8,
    quantize_ensemble_models,
)

__all__ = [
    "ResNet8",
    "build_client_model",
    "get_model_size_bytes",
    "get_model_size_mib",
    "QuantizedStatePayload",
    "compress_state_dict_to_int8",
    "decompress_int8_state_dict",
    "quantize_client_model_int8",
    "quantize_ensemble_models",
]
