"""Unary math elementwise ops (exp/log/sqrt/abs/neg/round/etc.)."""

from typing import Dict, Optional

import torch

from tileops.backend import Target
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

from ._base import (
    _MANIFEST_INT_DTYPES,
    UnaryOp,
    _IntIdentityUnaryOp,
)


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

    Mirrors ``torch.reciprocal`` int-input promotion: the manifest declares the
    output as ``promote_int_to_float(input)``, and ``ReciprocalFwdKernel.specialize``
    names float32 as the compute type for an integral input. The semantic dtype
    keys the specialization and drives roofline accounting — integer input bytes,
    float32 output bytes — while the kernel is built for the type it computes in.
    Floating inputs follow the standard same-dtype path.
    """

    _op_name = "reciprocal"
    kernel_cls = ReciprocalFwdKernel


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


class _RoundDecimalsCall:
    """In-tree stand-in for ``round(x, decimals=k)`` with ``k != 0``.

    ``round(x, decimals=k) == round(x * 10**k) / 10**k``, which the shipped
    round-to-nearest-integer kernel does not do. Only the in-tree path builds one:
    with a target selected, ``decimals`` is handed over as the manifest param it is
    and the backend serves every value of it. Not a ``Kernel``, so ``autotune`` walks
    past it — there is nothing to tune.
    """

    def __init__(self, decimals: int):
        self._decimals = decimals

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        # Integer dtypes are no-ops regardless of decimals (rounding an int
        # produces the same int). Match the float-path identity contract.
        if input.dtype in _MANIFEST_INT_DTYPES:
            return input.clone()
        # Run through fp32 so low-precision inputs (fp16/bf16) cannot overflow
        # when ``torch.round`` internally scales by ``10**decimals`` — e.g.
        # ``100 * 10**4 = 1e6`` exceeds fp16 max (~65504). The single down-cast
        # at the end restores the op's contract dtype.
        return torch.round(input.float(), decimals=self._decimals).to(input.dtype)


class RoundFwdOp(_IntIdentityUnaryOp):
    """Element-wise round(x) to ``decimals`` decimal places.

    The shipped kernel performs banker's round-to-nearest-integer, matching
    ``torch.round`` for ``decimals=0``. ``decimals`` is a manifest param, so it is
    fixed for the instance and handed to whichever kernel serves the op; in-tree, a
    non-zero value selects ``_RoundDecimalsCall``.

    Args:
        decimals: Number of decimal places to round to (manifest
            ``params.decimals``, default 0).
        target: Which set of kernels serves this op.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    _op_name = "round"
    kernel_cls = RoundFwdKernel

    def __init__(
        self,
        *,
        decimals: int = 0,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.decimals = int(decimals)
        super().__init__(target=target, kernel_map=kernel_map, tune=tune)

    def _build(self, dtype: torch.dtype, n_total: int):
        if self.decimals != 0:
            return _RoundDecimalsCall(self.decimals)
        return super()._build(dtype, n_total)


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
