"""Quantization and dequantization kernels and their call records."""

from tileops.kernels.quantization.call_spec import (
    INT8DequantFwdInterface,
    INT8QuantPerChannelFwdInterface,
    INT8QuantPerTensorFwdInterface,
    QuantizeCall,
)
from tileops.kernels.quantization.dequant_call import DequantizeCall
from tileops.kernels.quantization.int8_dequant import (
    INT8DequantPerChannelFwdKernel,
    INT8DequantPerTensorFwdKernel,
    INT8DequantPerTensorSmallFwdKernel,
)
from tileops.kernels.quantization.int8_quant_per_channel import INT8QuantPerChannelFwdKernel
from tileops.kernels.quantization.int8_quant_per_tensor import INT8QuantPerTensorFwdKernel

__all__ = [
    "DequantizeCall",
    "INT8DequantFwdInterface",
    "INT8DequantPerChannelFwdKernel",
    "INT8DequantPerTensorFwdKernel",
    "INT8DequantPerTensorSmallFwdKernel",
    "INT8QuantPerChannelFwdInterface",
    "INT8QuantPerChannelFwdKernel",
    "INT8QuantPerTensorFwdInterface",
    "INT8QuantPerTensorFwdKernel",
    "QuantizeCall",
]
