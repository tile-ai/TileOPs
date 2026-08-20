"""Elementwise op package.

Re-exports every public symbol of the package module so that
``from tileops.ops.elementwise import <Symbol>`` continues to work.

Concrete ops are organised one cluster per leaf module
(``arithmetic.py``, ``activations.py``, ``clamp.py``, ...). Umbrella
template classes (``UnaryOp`` / ``BinaryOp`` / ``FusedGatedOp``) and the
shared registration / broadcast infrastructure live in ``_base.py``.

Concrete ops register their ``torch.library.custom_op`` wrappers at
package import time via the registration loops at the bottom of this
module.
"""

import torch as _torch

from ._base import (
    BinaryOp,
    FusedGatedOp,
    UnaryOp,
    _register_binary_custom_op,
    _register_fused_gated_custom_op,
    _register_unary_custom_op,
    _register_unary_inplace_custom_op,
)
from .activations import (
    EluFwdOp,
    GeluAndMulFwdOp,
    GeluFwdOp,
    GeluTanhAndMulFwdOp,
    HardsigmoidFwdOp,
    HardswishFwdOp,
    HardtanhFwdOp,
    LeakyReluFwdOp,
    MishFwdOp,
    ReluFwdOp,
    SeluFwdOp,
    SigmoidFwdOp,
    SiluAndMulFwdOp,
    SiluFwdOp,
    SoftplusFwdOp,
    TanhFwdOp,
)
from .alibi import AlibiFwdOp
from .arithmetic import (
    AddFwdOp,
    DivFwdOp,
    FloorDivideFwdOp,
    LerpFwdOp,
    LerpTensorFwdOp,
    MaximumFwdOp,
    MinimumFwdOp,
    MulFwdOp,
    PowFwdOp,
    RemainderFwdOp,
    SubFwdOp,
)
from .bitwise import (
    BitwiseAndFwdOp,
    BitwiseNotFwdOp,
    BitwiseOrFwdOp,
    BitwiseXorFwdOp,
)
from .clamp import ClampFwdOp, ClampScalarFwdOp
from .comparison import (
    EqFwdOp,
    GeFwdOp,
    GtFwdOp,
    IsfiniteFwdOp,
    IsinfFwdOp,
    IsnanFwdOp,
    LeFwdOp,
    LtFwdOp,
    NeFwdOp,
)
from .logical import LogicalAndFwdOp, LogicalNotFwdOp, LogicalOrFwdOp
from .masked_fill import MaskedFillFwdOp, MaskedFillScalarFwdOp
from .math_unary import (
    AbsFwdOp,
    CeilFwdOp,
    CosFwdOp,
    ErfFwdOp,
    ExpFwdOp,
    Expm1FwdOp,
    FloorFwdOp,
    Log1pFwdOp,
    LogFwdOp,
    NegFwdOp,
    ReciprocalFwdOp,
    RoundFwdOp,
    RsqrtFwdOp,
    SignFwdOp,
    SinFwdOp,
    SqrtFwdOp,
    TruncFwdOp,
)
from .nan_to_num import NanToNumFwdOp
from .prelu import PreluFwdOp
from .sinusoidal import SinusoidalFwdOp
from .where import WhereFwdOp

__all__ = [
    "AbsFwdOp",
    "AddFwdOp",
    "AlibiFwdOp",
    "BinaryOp",
    "BitwiseAndFwdOp",
    "BitwiseNotFwdOp",
    "BitwiseOrFwdOp",
    "BitwiseXorFwdOp",
    "CeilFwdOp",
    "ClampFwdOp",
    "ClampScalarFwdOp",
    "CosFwdOp",
    "DivFwdOp",
    "EluFwdOp",
    "EqFwdOp",
    "ErfFwdOp",
    "ExpFwdOp",
    "Expm1FwdOp",
    "FloorDivideFwdOp",
    "FloorFwdOp",
    "FusedGatedOp",
    "GeFwdOp",
    "GeluAndMulFwdOp",
    "GeluFwdOp",
    "GeluTanhAndMulFwdOp",
    "GtFwdOp",
    "HardsigmoidFwdOp",
    "HardswishFwdOp",
    "HardtanhFwdOp",
    "IsfiniteFwdOp",
    "IsinfFwdOp",
    "IsnanFwdOp",
    "LeFwdOp",
    "LeakyReluFwdOp",
    "LerpFwdOp",
    "LerpTensorFwdOp",
    "Log1pFwdOp",
    "LogFwdOp",
    "LogicalAndFwdOp",
    "LogicalNotFwdOp",
    "LogicalOrFwdOp",
    "LtFwdOp",
    "MaskedFillFwdOp",
    "MaskedFillScalarFwdOp",
    "MaximumFwdOp",
    "MinimumFwdOp",
    "MishFwdOp",
    "MulFwdOp",
    "NanToNumFwdOp",
    "NeFwdOp",
    "NegFwdOp",
    "PowFwdOp",
    "PreluFwdOp",
    "ReciprocalFwdOp",
    "ReluFwdOp",
    "RemainderFwdOp",
    "RoundFwdOp",
    "RsqrtFwdOp",
    "SeluFwdOp",
    "SigmoidFwdOp",
    "SignFwdOp",
    "SiluAndMulFwdOp",
    "SiluFwdOp",
    "SinFwdOp",
    "SinusoidalFwdOp",
    "SoftplusFwdOp",
    "SqrtFwdOp",
    "SubFwdOp",
    "TanhFwdOp",
    "TruncFwdOp",
    "UnaryOp",
    "WhereFwdOp",
]


# torch.compile registration for the concrete elementwise ops.
#
# ``AlibiFwdOp`` and ``SinusoidalFwdOp`` are intentionally excluded: they
# have zero tensor inputs (output is fully derived from ``__init__``
# params), so they bypass the custom-op wrapper and run eager-only.

# --- Unary ops: float-preserving output (1 + 17 + 8 + 1 = 27 ops) ---
for _cls in [
    ReluFwdOp,
    # math (17)
    ExpFwdOp,
    LogFwdOp,
    SqrtFwdOp,
    RsqrtFwdOp,
    AbsFwdOp,
    NegFwdOp,
    ReciprocalFwdOp,
    SignFwdOp,
    SinFwdOp,
    CosFwdOp,
    FloorFwdOp,
    CeilFwdOp,
    RoundFwdOp,
    TruncFwdOp,
    ErfFwdOp,
    Log1pFwdOp,
    Expm1FwdOp,
    # activations (8)
    GeluFwdOp,
    SiluFwdOp,
    SigmoidFwdOp,
    TanhFwdOp,
    HardswishFwdOp,
    HardsigmoidFwdOp,
    MishFwdOp,
    SeluFwdOp,
    # bitwise (1) -- output same dtype as input
    BitwiseNotFwdOp,
]:
    _register_unary_custom_op(_cls)

# --- Unary ops: bool output (4 ops) ---
for _cls in [LogicalNotFwdOp, IsnanFwdOp, IsinfFwdOp, IsfiniteFwdOp]:
    _register_unary_custom_op(_cls)

# --- Binary ops (arithmetic 10 + bitwise 3 + comparison 6 + logical 2 = 21) ---
# Output dtype comes from each op's manifest entry, so comparison and logical
# ops need no separate registration group.
for _cls in [
    AddFwdOp,
    SubFwdOp,
    MulFwdOp,
    DivFwdOp,
    RemainderFwdOp,
    PowFwdOp,
    FloorDivideFwdOp,
    LerpFwdOp,
    MaximumFwdOp,
    MinimumFwdOp,
    BitwiseAndFwdOp,
    BitwiseOrFwdOp,
    BitwiseXorFwdOp,
    EqFwdOp,
    NeFwdOp,
    GtFwdOp,
    LtFwdOp,
    GeFwdOp,
    LeFwdOp,
    LogicalAndFwdOp,
    LogicalOrFwdOp,
]:
    _register_binary_custom_op(_cls)

# --- Fused gated ops (3 ops) ---
for _cls in [SiluAndMulFwdOp, GeluAndMulFwdOp, GeluTanhAndMulFwdOp]:
    _register_fused_gated_custom_op(_cls)

# --- Independent unary-like ops (6 ops: x -> y with baked params) ---
# ClampScalarFwdOp is the scalar-bound clamp (single-tensor input + min/max
# baked into __init__). The Tensor-bound ClampFwdOp registers its own
# multi-input custom_op below.
for _cls in [
    LeakyReluFwdOp,
    EluFwdOp,
    HardtanhFwdOp,
    SoftplusFwdOp,
    ClampScalarFwdOp,
    NanToNumFwdOp,
]:
    _register_unary_custom_op(_cls)

# --- Inplace companions for activations declaring ``inplace`` ---
# Each leaf below has ``inplace`` in its manifest signature. Register a
# parallel ``_wrapped_inplace`` custom op with ``mutates_args=("x",)``
# so ``forward(input)`` with ``self.inplace=True`` traces correctly
# under ``torch.compile``.
for _cls in [
    ReluFwdOp,
    SiluFwdOp,
    HardswishFwdOp,
    HardsigmoidFwdOp,
    MishFwdOp,
    SeluFwdOp,
    LeakyReluFwdOp,
    EluFwdOp,
    HardtanhFwdOp,
]:
    _register_unary_inplace_custom_op(_cls)

# Clean up loop variable
del _cls
