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
    UnaryOp,
    _IntIdentityUnaryOp,
)


class ExpFwdOp(UnaryOp):
    """Element-wise exp(x)."""

    kernel_types = {"exp": ExpFwdKernel}


class LogFwdOp(UnaryOp):
    """Element-wise log(x)."""

    kernel_types = {"log": LogFwdKernel}


class SqrtFwdOp(UnaryOp):
    """Element-wise sqrt(x)."""

    kernel_types = {"sqrt": SqrtFwdKernel}


class RsqrtFwdOp(UnaryOp):
    """Element-wise 1/sqrt(x)."""

    kernel_types = {"rsqrt": RsqrtFwdKernel}


class AbsFwdOp(_IntIdentityUnaryOp):
    """Element-wise |x|."""

    kernel_types = {"abs": AbsFwdKernel}
    _int_handler = staticmethod(torch.abs)


class NegFwdOp(_IntIdentityUnaryOp):
    """Element-wise -x."""

    kernel_types = {"neg": NegFwdKernel}
    _int_handler = staticmethod(torch.neg)


class ReciprocalFwdOp(UnaryOp):
    """Element-wise 1/x.

    Mirrors ``torch.reciprocal`` int-input promotion: an integral input gives a
    float32 output, and ``ReciprocalFwdKernel.specialize`` names float32 as the
    compute type for it. Floating inputs keep their dtype.
    """

    kernel_types = {"reciprocal": ReciprocalFwdKernel}


class SignFwdOp(_IntIdentityUnaryOp):
    """Element-wise sign(x): -1, 0, or +1."""

    kernel_types = {"sign": SignFwdKernel}
    _int_handler = staticmethod(torch.sign)


class SinFwdOp(UnaryOp):
    """Element-wise sin(x)."""

    kernel_types = {"sin": SinFwdKernel}


class CosFwdOp(UnaryOp):
    """Element-wise cos(x)."""

    kernel_types = {"cos": CosFwdKernel}


class FloorFwdOp(_IntIdentityUnaryOp):
    """Element-wise floor(x)."""

    kernel_types = {"floor": FloorFwdKernel}


class CeilFwdOp(_IntIdentityUnaryOp):
    """Element-wise ceil(x)."""

    kernel_types = {"ceil": CeilFwdKernel}


class _RoundDecimalsCall:
    """In-tree stand-in for ``round(x, decimals=k)`` with ``k != 0``.

    ``round(x, decimals=k) == round(x * 10**k) / 10**k``, which the shipped
    round-to-nearest-integer kernel does not do. Only the in-tree path builds one:
    with a target selected, ``decimals`` is handed over as the manifest param it is
    and the backend serves every value of it. Not a ``Kernel``, so ``autotune`` walks
    past it — there is nothing to tune.
    """

    def __init__(self, decimals: int):
        """Build the op. Shapes and dtype are taken from the first call."""
        self._decimals = decimals

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
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

    """

    kernel_types = {"round": RoundFwdKernel}

    def __init__(
        self,
        *,
        decimals: int = 0,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            decimals: Number of decimal places to round to (manifest
                ``params.decimals``, default 0).
            target: Which set of kernels serves this op.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.
        """
        self.decimals = decimals
        super().__init__(target=target, kernel_map=kernel_map, tune=tune)

    def _build(self, dtype: torch.dtype, n_total: int):
        if self.decimals != 0:
            return _RoundDecimalsCall(self.decimals)
        return super()._build(dtype, n_total)


class TruncFwdOp(_IntIdentityUnaryOp):
    """Element-wise trunc(x)."""

    kernel_types = {"trunc": TruncFwdKernel}


class ErfFwdOp(UnaryOp):
    """Element-wise erf(x).

    On float16 and bfloat16 the error function is evaluated as a polynomial that
    saturates to exactly +/-1; its worst case over the real line is 1.7e-5, an
    order below half a float16 ulp at 1.0. float32 keeps `erff`.
    """

    kernel_types = {"erf": ErfFwdKernel}


class Log1pFwdOp(UnaryOp):
    """Element-wise log(1 + x)."""

    kernel_types = {"log1p": Log1pFwdKernel}


class Expm1FwdOp(UnaryOp):
    """Element-wise exp(x) - 1."""

    kernel_types = {"expm1": Expm1FwdKernel}
