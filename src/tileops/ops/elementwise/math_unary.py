"""Unary math elementwise ops (exp/log/sqrt/abs/neg/round/etc.)."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import (
    AbsFwdKernel,
    AbsIntFwdKernel,
    CeilFwdKernel,
    CosFwdKernel,
    ErfFwdKernel,
    ExpFwdKernel,
    Expm1FwdKernel,
    FloorFwdKernel,
    IntIdentityFwdKernel,
    Log1pFwdKernel,
    LogFwdKernel,
    NegFwdKernel,
    NegIntFwdKernel,
    ReciprocalFwdKernel,
    RoundDecimalsFwdKernel,
    RoundFwdKernel,
    RsqrtFwdKernel,
    SignFwdKernel,
    SignIntFwdKernel,
    SinFwdKernel,
    SqrtFwdKernel,
    TruncFwdKernel,
)
from tileops.kernels.elementwise.call_spec import (
    ReciprocalFwdInterface,
    RoundCall,
    RoundFwdInterface,
)
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.elementwise._base import ELEMENTWISE, UnaryOp


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


class AbsFwdOp(UnaryOp):
    """Element-wise |x|."""

    kernel_types = {"abs": AbsFwdKernel, "abs_int": AbsIntFwdKernel}


class NegFwdOp(UnaryOp):
    """Element-wise -x."""

    kernel_types = {"neg": NegFwdKernel, "neg_int": NegIntFwdKernel}


class ReciprocalFwdOp(UnaryOp):
    """Element-wise 1/x.

    Mirrors ``torch.reciprocal`` int-input promotion: an integral input gives a
    float32 output, and ``ReciprocalFwdKernel.specialize`` names float32 as the
    compute type for it. Floating inputs keep their dtype.
    """

    kernel_types = {"reciprocal": ReciprocalFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: ReciprocalFwdInterface
    }


class SignFwdOp(UnaryOp):
    """Element-wise sign(x): -1, 0, or +1."""

    kernel_types = {"sign": SignFwdKernel, "sign_int": SignIntFwdKernel}


class SinFwdOp(UnaryOp):
    """Element-wise sin(x)."""

    kernel_types = {"sin": SinFwdKernel}


class CosFwdOp(UnaryOp):
    """Element-wise cos(x)."""

    kernel_types = {"cos": CosFwdKernel}


class FloorFwdOp(UnaryOp):
    """Element-wise floor(x)."""

    kernel_types = {"floor": FloorFwdKernel, "floor_int": IntIdentityFwdKernel}


class CeilFwdOp(UnaryOp):
    """Element-wise ceil(x)."""

    kernel_types = {"ceil": CeilFwdKernel, "ceil_int": IntIdentityFwdKernel}


class RoundFwdOp(UnaryOp):
    """Element-wise round(x) to ``decimals`` decimal places.

    The shipped kernel performs banker's round-to-nearest-integer, matching
    ``torch.round`` for ``decimals=0``. ``decimals`` is a manifest param, so it is
    fixed for the instance and handed to whichever kernel serves the op.

    """

    kernel_types = {
        "round": RoundFwdKernel,
        "round_int": IntIdentityFwdKernel,
        "round_decimals": RoundDecimalsFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {ELEMENTWISE: RoundFwdInterface}

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

    def _call_spec(self, input: torch.Tensor) -> RoundCall:
        return RoundCall(
            device=input.device,
            n_total=input.numel(),
            dtype=input.dtype,
            decimals=self.decimals,
        )


class TruncFwdOp(UnaryOp):
    """Element-wise trunc(x)."""

    kernel_types = {"trunc": TruncFwdKernel, "trunc_int": IntIdentityFwdKernel}


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
