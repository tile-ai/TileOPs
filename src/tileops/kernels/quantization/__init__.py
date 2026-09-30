"""Quantization and dequantization kernels and their call records."""

from tileops.kernels.quantization.call_spec import (
    FP8QuantPerBlockFwdInterface,
    INT4QuantPerGroupFwdInterface,
    INT8DequantPerBlockFwdInterface,
    INT8DequantPerChannelFwdInterface,
    INT8DequantPerTensorFwdInterface,
    INT8QuantPerBlockFwdInterface,
    INT8QuantPerChannelFwdInterface,
    INT8QuantPerTensorFwdInterface,
    QuantizeCall,
    SmoothQuantFwdInterface,
)
from tileops.kernels.quantization.dequant_call import DequantizeCall
from tileops.kernels.quantization.fp8_quant_per_block import (
    FP8QuantPerBlockFwdKernel,
    FP8QuantPerBlockUnalignedFwdKernel,
)
from tileops.kernels.quantization.int4_quant_per_group import (
    INT4QuantPerGroupFwdKernel,
    INT4QuantPerGroupRowFwdKernel,
)
from tileops.kernels.quantization.int8_dequant import (
    INT8DequantPerBlockFwdKernel,
    INT8DequantPerBlockSmallFwdKernel,
    INT8DequantPerChannelFwdKernel,
    INT8DequantPerTensorFwdKernel,
    INT8DequantPerTensorSmallFwdKernel,
)
from tileops.kernels.quantization.int8_quant_per_block import (
    INT8QuantPerBlockFwdKernel,
    INT8QuantPerBlockShiftedFwdKernel,
)
from tileops.kernels.quantization.int8_quant_per_channel import (
    INT8QuantPerChannelFwdKernel,
    SmoothQuantFwdKernel,
)
from tileops.kernels.quantization.int8_quant_per_tensor import INT8QuantPerTensorFwdKernel

__all__ = [
    "DequantizeCall",
    "FP8QuantPerBlockFwdInterface",
    "FP8QuantPerBlockFwdKernel",
    "FP8QuantPerBlockUnalignedFwdKernel",
    "INT4QuantPerGroupFwdInterface",
    "INT4QuantPerGroupFwdKernel",
    "INT4QuantPerGroupRowFwdKernel",
    "INT8DequantPerBlockFwdInterface",
    "INT8DequantPerBlockFwdKernel",
    "INT8DequantPerBlockSmallFwdKernel",
    "INT8DequantPerChannelFwdInterface",
    "INT8DequantPerChannelFwdKernel",
    "INT8DequantPerTensorFwdInterface",
    "INT8DequantPerTensorFwdKernel",
    "INT8DequantPerTensorSmallFwdKernel",
    "INT8QuantPerBlockFwdInterface",
    "INT8QuantPerBlockFwdKernel",
    "INT8QuantPerBlockShiftedFwdKernel",
    "INT8QuantPerChannelFwdInterface",
    "INT8QuantPerChannelFwdKernel",
    "INT8QuantPerTensorFwdInterface",
    "INT8QuantPerTensorFwdKernel",
    "QuantizeCall",
    "SmoothQuantFwdInterface",
    "SmoothQuantFwdKernel",
]
