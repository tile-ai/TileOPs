"""Call workloads of the quantization family."""

from workloads.quantization.fp8_quant import FP8QuantWorkload
from workloads.quantization.int8_dequant import (
    INT8DequantPerBlockWorkload,
    INT8DequantPerChannelWorkload,
    INT8DequantPerTensorWorkload,
)
from workloads.quantization.quantize import (
    FP8QuantPerBlockWorkload,
    INT4QuantPerGroupWorkload,
    INT8QuantPerBlockWorkload,
    INT8QuantPerChannelWorkload,
    INT8QuantPerTensorWorkload,
    SmoothQuantWorkload,
)

WORKLOADS = {
    "FP8QuantPerTokenGroupFwdOp": FP8QuantWorkload.from_call,
    "FP8QuantPerBlockFwdOp": FP8QuantPerBlockWorkload.from_call,
    "INT4QuantPerGroupFwdOp": INT4QuantPerGroupWorkload.from_call,
    "INT8DequantPerTensorFwdOp": INT8DequantPerTensorWorkload.from_call,
    "INT8DequantPerChannelFwdOp": INT8DequantPerChannelWorkload.from_call,
    "INT8DequantPerBlockFwdOp": INT8DequantPerBlockWorkload.from_call,
    "INT8QuantPerTensorFwdOp": INT8QuantPerTensorWorkload.from_call,
    "INT8QuantPerChannelFwdOp": INT8QuantPerChannelWorkload.from_call,
    "INT8QuantPerTokenGroupFwdOp": INT8QuantPerBlockWorkload.from_call,
    "SmoothQuantFwdOp": SmoothQuantWorkload.from_call,
}
