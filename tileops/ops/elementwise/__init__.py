"""Elementwise op package.

Re-exports every public symbol previously provided by the monolithic
``tileops/ops/elementwise.py`` module so that
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
    _register_clamp_max_custom_op,
    _register_clamp_min_custom_op,
    _register_clamp_tensor_custom_op,
    _register_fused_gated_custom_op,
    _register_lerp_tensor_custom_op,
    _register_masked_fill_custom_op,
    _register_masked_fill_tensor_value_custom_op,
    _register_prelu_custom_op,
    _register_unary_custom_op,
    _register_unary_inplace_custom_op,
    _register_where_custom_op,
    coalesce_broadcast_dims,
)
from .activations import (
    EluFwdOp,
    GeluFwdOp,
    HardsigmoidFwdOp,
    HardswishFwdOp,
    HardtanhFwdOp,
    LeakyReluFwdOp,
    MishFwdOp,
    ReluFwdOp,
    SeluFwdOp,
    SigmoidFwdOp,
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
from .clamp import ClampFwdOp, ClampMaxFwdOp, ClampMinFwdOp, ClampScalarFwdOp
from .comparison import EqFwdOp, GeFwdOp, GtFwdOp, LeFwdOp, LtFwdOp, NeFwdOp
from .fused_gated import GeluAndMulFwdOp, GeluTanhAndMulFwdOp, SiluAndMulFwdOp
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
from .predicates import IsfiniteFwdOp, IsinfFwdOp, IsnanFwdOp
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
    "ClampMaxFwdOp",
    "ClampMinFwdOp",
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
    "coalesce_broadcast_dims",
]


# ---------------------------------------------------------------------------
# torch.compile registration for the concrete elementwise ops.
#
# ``AlibiFwdOp`` and ``SinusoidalFwdOp`` are intentionally excluded: they
# have zero tensor inputs (output is fully derived from ``__init__``
# params), so they bypass the custom-op wrapper and run eager-only.
# ---------------------------------------------------------------------------

# --- Unary ops: float-preserving output (1 + 17 + 8 + 1 = 27 ops) ---
for _cls in [
    ReluFwdOp,
    # math (17)
    ExpFwdOp, LogFwdOp, SqrtFwdOp, RsqrtFwdOp, AbsFwdOp, NegFwdOp, ReciprocalFwdOp, SignFwdOp,
    SinFwdOp, CosFwdOp, FloorFwdOp, CeilFwdOp, RoundFwdOp, TruncFwdOp, ErfFwdOp, Log1pFwdOp, Expm1FwdOp,
    # activations (8)
    GeluFwdOp, SiluFwdOp, SigmoidFwdOp, TanhFwdOp, HardswishFwdOp, HardsigmoidFwdOp, MishFwdOp, SeluFwdOp,
    # bitwise (1) -- output same dtype as input
    BitwiseNotFwdOp,
]:
    _register_unary_custom_op(_cls)

# --- Unary ops: bool output (4 ops) ---
for _cls in [LogicalNotFwdOp, IsnanFwdOp, IsinfFwdOp, IsfiniteFwdOp]:
    _register_unary_custom_op(_cls, output_dtype_override=_torch.bool)

# --- Binary ops: same-dtype output (10 + 3 = 13 ops) ---
for _cls in [
    # arithmetic (10)
    AddFwdOp, SubFwdOp, MulFwdOp, DivFwdOp, RemainderFwdOp, PowFwdOp, FloorDivideFwdOp,
    LerpFwdOp, MaximumFwdOp, MinimumFwdOp,
    # bitwise (3)
    BitwiseAndFwdOp, BitwiseOrFwdOp, BitwiseXorFwdOp,
]:
    _register_binary_custom_op(_cls)

# --- Binary ops: bool output (comparison 6 + logical 2 = 8 ops) ---
for _cls in [
    EqFwdOp, NeFwdOp, GtFwdOp, LtFwdOp, GeFwdOp, LeFwdOp,
    LogicalAndFwdOp, LogicalOrFwdOp,
]:
    _register_binary_custom_op(_cls, output_bool=True)

# --- Fused gated ops (3 ops) ---
for _cls in [SiluAndMulFwdOp, GeluAndMulFwdOp, GeluTanhAndMulFwdOp]:
    _register_fused_gated_custom_op(_cls)

# --- Independent unary-like ops (6 ops: x -> y with baked params) ---
# ClampScalarFwdOp is the scalar-bound clamp (single-tensor input + min/max
# baked into __init__). The Tensor-bound ClampFwdOp / ClampMinFwdOp /
# ClampMaxFwdOp variants register their own multi-input custom_ops below.
for _cls in [
    LeakyReluFwdOp, EluFwdOp, HardtanhFwdOp, SoftplusFwdOp, ClampScalarFwdOp,
    NanToNumFwdOp,
]:
    _register_unary_custom_op(_cls)

# --- Inplace companions for activations declaring ``inplace`` ---
# Each leaf below has ``inplace`` in its manifest signature. Register a
# parallel ``_wrapped_inplace`` custom op with ``mutates_args=("x",)``
# so ``forward(input)`` with ``self.inplace=True`` traces correctly
# under ``torch.compile``.
for _cls in [
    ReluFwdOp, SiluFwdOp, HardswishFwdOp, HardsigmoidFwdOp, MishFwdOp,
    SeluFwdOp, LeakyReluFwdOp, EluFwdOp, HardtanhFwdOp,
]:
    _register_unary_inplace_custom_op(_cls)

# --- PReLU op (1 op: x, weight -> y) ---
_register_prelu_custom_op(PreluFwdOp)

# --- Tensor-bound clamp variants (3 ops: multi-tensor inputs -> out) ---
# Registered under distinct custom_op namespaces from ClampScalarFwdOp:
# ``top::elementwise_clamp_tensor`` (Optional Tensor min/max),
# ``top::elementwise_clamp_min`` and ``top::elementwise_clamp_max`` for the
# single-bound variants. register_fake is broadcast-aware so
# torch.compile(fullgraph=True) traces correctly for both same-shape and
# broadcasting inputs.
_register_clamp_tensor_custom_op(ClampFwdOp)
_register_clamp_min_custom_op(ClampMinFwdOp)
_register_clamp_max_custom_op(ClampMaxFwdOp)

# --- MaskedFill variants (input, mask[, value] -> out) ---
# Both register broadcast-aware fake functions so torch.compile(fullgraph=True)
# works for same-shape and broadcasting inputs. The Tensor-value variant is
# registered under a distinct ``_tensor_value`` namespace to avoid colliding
# with the scalar variant's ``top::elementwise_masked_fill``.
_register_masked_fill_custom_op(MaskedFillScalarFwdOp)
_register_masked_fill_tensor_value_custom_op(MaskedFillFwdOp)

# --- Where op (1 op: cond, x, y -> out) ---
# The fake function is broadcast-aware so torch.compile(fullgraph=True)
# traces correctly for both same-shape and broadcasting inputs.
_register_where_custom_op(WhereFwdOp)

# --- Tensor-weight lerp (1 op: input, end, weight -> out) ---
# Registered under ``top::elementwise_lerp_tensor`` to avoid colliding with
# the scalar ``LerpFwdOp``'s ``top::elementwise_binary_lerp`` namespace. The fake
# function is broadcast-aware so torch.compile(fullgraph=True) traces
# correctly for both same-shape and broadcasting inputs.
_register_lerp_tensor_custom_op(LerpTensorFwdOp)

# Clean up loop variable
del _cls
