"""Fused gated elementwise ops: y = activation(gate) * value."""

from tileops.kernels.elementwise import (
    GeluAndMulFwdKernel,
    GeluTanhAndMulFwdKernel,
    SiluAndMulFwdKernel,
)

from ._base import FusedGatedOp


class SiluAndMulFwdOp(FusedGatedOp):
    """SiLU-and-Mul: y = silu(gate) * value."""

    _op_name = "silu_and_mul"
    kernel_cls = SiluAndMulFwdKernel


class GeluAndMulFwdOp(FusedGatedOp):
    """GELU-and-Mul: y = gelu(gate) * value (exact GELU)."""

    _op_name = "gelu_and_mul"
    kernel_cls = GeluAndMulFwdKernel


class GeluTanhAndMulFwdOp(FusedGatedOp):
    """GELU-Tanh-and-Mul: y = gelu_tanh(gate) * value (tanh approximation)."""

    _op_name = "gelu_tanh_and_mul"
    kernel_cls = GeluTanhAndMulFwdKernel
    FLOPS_PER_ELEM = 10
