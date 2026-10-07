"""Case factories of the quantization family."""

from benchmarks._cases import Entry
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

ENTRIES = {
    "FP8QuantFwdOp": Entry(FP8QuantWorkload.from_call),
    "FP8QuantPerBlockFwdOp": Entry(FP8QuantPerBlockWorkload.from_call),
    "INT4QuantPerGroupFwdOp": Entry(INT4QuantPerGroupWorkload.from_call),
    "INT8DequantPerTensorFwdOp": Entry(INT8DequantPerTensorWorkload.from_call),
    "INT8DequantPerChannelFwdOp": Entry(INT8DequantPerChannelWorkload.from_call),
    "INT8DequantPerBlockFwdOp": Entry(INT8DequantPerBlockWorkload.from_call),
    "INT8QuantPerTensorFwdOp": Entry(INT8QuantPerTensorWorkload.from_call),
    "INT8QuantPerChannelFwdOp": Entry(INT8QuantPerChannelWorkload.from_call),
    "INT8QuantPerBlockFwdOp": Entry(INT8QuantPerBlockWorkload.from_call),
    "SmoothQuantFwdOp": Entry(SmoothQuantWorkload.from_call),
}
