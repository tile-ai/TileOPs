"""Quantization workloads and reference computations."""

from workloads.quantization.fp8_quant import (
    FP8QuantWorkload,
)
from workloads.quantization.int8_dequant import (
    INT8DequantPerBlockWorkload,
    INT8DequantPerChannelWorkload,
    INT8DequantPerTensorWorkload,
    int8_dequant_per_block,
    int8_dequant_per_channel,
    int8_dequant_per_tensor,
)
from workloads.quantization.quantize import (
    FP8QuantPerBlockWorkload,
    INT4QuantPerGroupWorkload,
    INT8QuantPerBlockWorkload,
    INT8QuantPerChannelWorkload,
    INT8QuantPerTensorWorkload,
    SmoothQuantWorkload,
    fp8_quant_per_block,
    int4_quant_per_group,
    int8_quant_per_block,
    int8_quant_per_channel,
    int8_quant_per_tensor,
    smooth_quant,
)

__all__ = [
    "FP8QuantPerBlockWorkload",
    "FP8QuantWorkload",
    "INT4QuantPerGroupWorkload",
    "INT8DequantPerBlockWorkload",
    "INT8DequantPerChannelWorkload",
    "INT8DequantPerTensorWorkload",
    "INT8QuantPerBlockWorkload",
    "INT8QuantPerChannelWorkload",
    "INT8QuantPerTensorWorkload",
    "SmoothQuantWorkload",
    "fp8_quant_per_block",
    "int4_quant_per_group",
    "int8_dequant_per_block",
    "int8_dequant_per_channel",
    "int8_dequant_per_tensor",
    "int8_quant_per_block",
    "int8_quant_per_channel",
    "int8_quant_per_tensor",
    "smooth_quant",
]
