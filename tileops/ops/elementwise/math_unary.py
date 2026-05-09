"""Unary math elementwise ops (exp/log/sqrt/abs/neg/round/etc.)."""

from typing import Dict, Optional

import torch

from tileops.kernels.elementwise import (
    AbsFwdKernel,
    CeilFwdKernel,
    CosFwdKernel,
    ErfFwdKernel,
    ExpFwdKernel,
    Expm1FwdKernel,
    FloorFwdKernel,
    Log1pFwdKernel,
    LogFwdKernel,
    NegFwdKernel,
    ReciprocalFwdKernel,
    RoundFwdKernel,
    RsqrtFwdKernel,
    SignFwdKernel,
    SinFwdKernel,
    SqrtFwdKernel,
    TruncFwdKernel,
)
from tileops.kernels.kernel_base import Kernel

from ._base import _MANIFEST_INT_DTYPES, UnaryOp, _IntIdentityUnaryOp


class ExpFwdOp(UnaryOp):
    """Element-wise exp(x)."""

    _op_name = "exp"
    kernel_cls = ExpFwdKernel


class LogFwdOp(UnaryOp):
    """Element-wise log(x)."""

    _op_name = "log"
    kernel_cls = LogFwdKernel


class SqrtFwdOp(UnaryOp):
    """Element-wise sqrt(x)."""

    _op_name = "sqrt"
    kernel_cls = SqrtFwdKernel


class RsqrtFwdOp(UnaryOp):
    """Element-wise 1/sqrt(x)."""

    _op_name = "rsqrt"
    kernel_cls = RsqrtFwdKernel


class AbsFwdOp(_IntIdentityUnaryOp):
    """Element-wise |x|."""

    _op_name = "abs"
    kernel_cls = AbsFwdKernel
    _int_handler = staticmethod(torch.abs)


class NegFwdOp(_IntIdentityUnaryOp):
    """Element-wise -x."""

    _op_name = "neg"
    kernel_cls = NegFwdKernel
    _int_handler = staticmethod(torch.neg)


class ReciprocalFwdOp(UnaryOp):
    """Element-wise 1/x.

    Mirrors ``torch.reciprocal`` int-input promotion: integral dtypes
    (uint8 / int8 / int16 / int32 / int64) are cast to float32 before the
    float kernel runs, and the op's ``output_dtype`` is float32 in that
    case. Floating inputs (float16 / bfloat16 / float32) follow the
    standard same-dtype path.
    """

    _op_name = "reciprocal"
    kernel_cls = ReciprocalFwdKernel

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if dtype in _MANIFEST_INT_DTYPES:
            # Build the kernel against the promoted compute dtype (float32)
            # so the float-only ReciprocalFwdKernel can run, then restore
            # the user-declared dtype on ``self.dtype`` so metadata and
            # ``eval_roofline`` reflect the real I/O contract: integer
            # input bytes + float32 output bytes. ``self.output_dtype``
            # stays float32 (set by the kernel) per the manifest's
            # ``promote_int_to_float`` contract.
            super().__init__(
                N_total, torch.float32, strategy=strategy,
                kernel_map=kernel_map, tune=tune,
            )
            self.dtype = dtype
        else:
            super().__init__(
                N_total, dtype, strategy=strategy,
                kernel_map=kernel_map, tune=tune,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if self.dtype in _MANIFEST_INT_DTYPES:
            self._validate_input(input)
            promoted = input.to(torch.float32)
            wrapped = type(self)._wrapped
            if wrapped is not None:
                return wrapped(promoted, self._instance_key)
            return self._eager_forward(promoted)
        return super().forward(input)


class SignFwdOp(_IntIdentityUnaryOp):
    """Element-wise sign(x): -1, 0, or +1."""

    _op_name = "sign"
    kernel_cls = SignFwdKernel
    # Manifest: flops = "2 * N" (two compares + selects per element).
    FLOPS_PER_ELEM = 2
    _int_handler = staticmethod(torch.sign)


class SinFwdOp(UnaryOp):
    """Element-wise sin(x)."""

    _op_name = "sin"
    kernel_cls = SinFwdKernel


class CosFwdOp(UnaryOp):
    """Element-wise cos(x)."""

    _op_name = "cos"
    kernel_cls = CosFwdKernel


class FloorFwdOp(_IntIdentityUnaryOp):
    """Element-wise floor(x)."""

    _op_name = "floor"
    kernel_cls = FloorFwdKernel


class CeilFwdOp(_IntIdentityUnaryOp):
    """Element-wise ceil(x)."""

    _op_name = "ceil"
    kernel_cls = CeilFwdKernel


class RoundFwdOp(_IntIdentityUnaryOp):
    """Element-wise round(x) to ``decimals`` decimal places.

    The underlying kernel performs banker's round-to-nearest-integer, matching
    ``torch.round`` for ``decimals=0``. Non-zero ``decimals`` is supported at
    the op layer via the standard decomposition:
    ``round(x, decimals=k) == round(x * 10**k) / 10**k``.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        strategy: Kernel strategy override.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    _op_name = "round"
    kernel_cls = RoundFwdKernel

    def forward(  # noqa: A002
        self, input: torch.Tensor, decimals: int = 0,
    ) -> torch.Tensor:
        if decimals == 0:
            return super().forward(input)
        # Non-zero decimals path still owes the same input contract as the
        # ``decimals=0`` fast path (UnaryOp.forward). Run the shared validator
        # before any fp32 arithmetic so a CPU tensor / wrong dtype / wrong
        # numel cannot silently bypass the checks.
        self._validate_input(input)
        # Integer dtypes are no-ops regardless of decimals (rounding an int
        # produces the same int). Match the float-path identity contract.
        if self.dtype in _MANIFEST_INT_DTYPES:
            return input.clone()
        # Run through fp32 so low-precision inputs (fp16/bf16) cannot overflow
        # when ``torch.round`` internally scales by ``10**decimals`` — e.g.
        # ``100 * 10**4 = 1e6`` exceeds fp16 max (~65504). The single down-cast
        # at the end restores the op's contract dtype. The manifest's
        # ``kernel_map`` continues to describe the round-to-nearest-integer
        # kernel that handles the ``decimals=0`` fast path above.
        return torch.round(input.float(), decimals=decimals).to(self.dtype)


class TruncFwdOp(_IntIdentityUnaryOp):
    """Element-wise trunc(x)."""

    _op_name = "trunc"
    kernel_cls = TruncFwdKernel


class ErfFwdOp(UnaryOp):
    """Element-wise erf(x)."""

    _op_name = "erf"
    kernel_cls = ErfFwdKernel


class Log1pFwdOp(UnaryOp):
    """Element-wise log(1 + x)."""

    _op_name = "log1p"
    kernel_cls = Log1pFwdKernel
    # Manifest: flops = "2 * N" (1 add + 1 log).
    FLOPS_PER_ELEM = 2


class Expm1FwdOp(UnaryOp):
    """Element-wise exp(x) - 1."""

    _op_name = "expm1"
    kernel_cls = Expm1FwdKernel
    # Manifest: flops = "2 * N" (1 exp + 1 sub).
    FLOPS_PER_ELEM = 2
