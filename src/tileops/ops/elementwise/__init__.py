"""Elementwise op package.

Re-exports every public symbol of the package module so that
``from tileops.ops.elementwise import <Symbol>`` continues to work.

Concrete ops are organised one cluster per leaf module
(``arithmetic.py``, ``activations.py``, ``clamp.py``, ...). Umbrella
template classes (``UnaryOp`` / ``BinaryOp`` / ``FusedGatedOp``) live in
``_base.py``. Each op's compile-boundary operators are generated from its
manifest signature when the class is created.
"""

from tileops.ops.elementwise._base import BinaryOp, FusedGatedOp, UnaryOp
from tileops.ops.elementwise.activations import (
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
from tileops.ops.elementwise.alibi import AlibiFwdOp
from tileops.ops.elementwise.arithmetic import (
    AddFwdOp,
    DivFwdOp,
    FloorDivideFwdOp,
    LerpScalarFwdOp,
    LerpTensorFwdOp,
    MaximumFwdOp,
    MinimumFwdOp,
    MulFwdOp,
    PowFwdOp,
    RemainderFwdOp,
    SubFwdOp,
)
from tileops.ops.elementwise.bitwise import (
    BitwiseAndFwdOp,
    BitwiseNotFwdOp,
    BitwiseOrFwdOp,
    BitwiseXorFwdOp,
)
from tileops.ops.elementwise.clamp import ClampScalarFwdOp, ClampTensorFwdOp
from tileops.ops.elementwise.comparison import (
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
from tileops.ops.elementwise.dropout import DropoutFwdOp
from tileops.ops.elementwise.logical import LogicalAndFwdOp, LogicalNotFwdOp, LogicalOrFwdOp
from tileops.ops.elementwise.masked_fill import MaskedFillScalarFwdOp, MaskedFillTensorFwdOp
from tileops.ops.elementwise.math_unary import (
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
from tileops.ops.elementwise.nan_to_num import NanToNumFwdOp
from tileops.ops.elementwise.prelu import PreluFwdOp
from tileops.ops.elementwise.sinusoidal import SinusoidalFwdOp
from tileops.ops.elementwise.where import WhereFwdOp

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
    "ClampTensorFwdOp",
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
    "LerpScalarFwdOp",
    "LerpTensorFwdOp",
    "Log1pFwdOp",
    "LogFwdOp",
    "LogicalAndFwdOp",
    "LogicalNotFwdOp",
    "LogicalOrFwdOp",
    "LtFwdOp",
    "MaskedFillTensorFwdOp",
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
    "DropoutFwdOp",
]
