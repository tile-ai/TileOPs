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

from ._base import BinaryOp, FusedGatedOp, UnaryOp
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


# ``AlibiFwdOp`` and ``SinusoidalFwdOp`` register no operator: they have zero tensor
# inputs, so there is nothing for a traced graph to hand over, and they run eager-only.
