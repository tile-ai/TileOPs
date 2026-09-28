"""Quantization and dequantization kernels and their call records."""

from .call_spec import QuantizeCall
from .dequant_call import DequantizeCall

__all__ = [
    "DequantizeCall",
    "QuantizeCall",
]
