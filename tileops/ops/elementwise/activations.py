"""Activation elementwise ops (ReLU + parametric/param-free families)."""

from typing import Dict, Optional

import torch

from tileops.kernels.elementwise import (
    EluFwdKernel,
    GeluFwdKernel,
    GeluTanhFwdKernel,
    HardsigmoidFwdKernel,
    HardswishFwdKernel,
    HardtanhFwdKernel,
    LeakyReluFwdKernel,
    MishFwdKernel,
    ReluFwdKernel,
    SeluFwdKernel,
    SigmoidFwdKernel,
    SiluFwdKernel,
    SoftplusFwdKernel,
    TanhFwdKernel,
)
from tileops.kernels.kernel_base import Kernel

from ._base import (
    UnaryOp,
    _GeluApproximateBase,
    _ParametricActivationOp,
    _ParamFreeActivationOp,
    _validate_scalar_param_repr,
)


class ReluFwdOp(_ParamFreeActivationOp):
    """ReLU activation: y = max(x, 0)."""

    _op_name = "relu"
    kernel_cls = ReluFwdKernel
    # Manifest: flops = "N". Per roofline.md §1.3, one
    # compare-and-select counts as 1 FLOP per element.
    FLOPS_PER_ELEM = 1


class GeluFwdOp(_GeluApproximateBase):
    """Element-wise GELU honoring the manifest ``approximate`` contract.

    Args:
        N_total: Number of elements (flattened input).
        dtype: Torch dtype.
        approximate: Approximation mode. ``'none'`` (default) routes to
            the erf-based ``GeluFwdKernel``. ``'tanh'`` routes to
            ``GeluTanhFwdKernel`` (the fused tanh approximation
            ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``).
        strategy: Optional kernel strategy override.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune the kernel.
    """

    _op_name = "gelu"
    kernel_cls = GeluFwdKernel
    # Manifest: flops = "5 * N". Per roofline.md §1.3:
    # gelu(x) = x * 0.5 * (1 + erf(x/sqrt(2))) =
    # div + erf(transcendental) + add + mul-by-half + mul = 5 per elem.
    FLOPS_PER_ELEM = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        kernel_cls = (
            GeluTanhFwdKernel if self.approximate == "tanh" else GeluFwdKernel
        )
        return {self._op_name: kernel_cls}


class SiluFwdOp(_ParamFreeActivationOp):
    """Element-wise SiLU (Swish): y = x * sigmoid(x)."""

    _op_name = "silu"
    kernel_cls = SiluFwdKernel
    # Manifest: flops = "5 * N". Per roofline.md §1.3:
    # sigmoid = neg + exp + add + recip = 4; silu adds one mul = 5 per elem.
    FLOPS_PER_ELEM = 5


class SigmoidFwdOp(UnaryOp):
    """Element-wise sigmoid(x)."""

    _op_name = "sigmoid"
    kernel_cls = SigmoidFwdKernel
    # Manifest: flops = "4 * N" (sigmoid(x) = 1 / (1 + exp(-x)) ≈ 4 ops/elem).
    FLOPS_PER_ELEM = 4


class TanhFwdOp(UnaryOp):
    """Element-wise tanh(x)."""

    _op_name = "tanh"
    kernel_cls = TanhFwdKernel
    # Manifest: flops = "N". Per roofline.md §1.3, tanh is one
    # transcendental call = 1 FLOP per element.
    FLOPS_PER_ELEM = 1


class HardswishFwdOp(_ParamFreeActivationOp):
    """Element-wise HardSwish: y = x * clamp(x + 3, 0, 6) / 6."""

    _op_name = "hardswish"
    kernel_cls = HardswishFwdKernel
    # Manifest: flops = "4 * N". Per roofline.md §1.3:
    # hardswish(x) = x * relu6(x+3)/6 =
    # add + two-sided-clamp(1) + mul + div = 4 per elem.
    FLOPS_PER_ELEM = 4


class HardsigmoidFwdOp(_ParamFreeActivationOp):
    """Element-wise HardSigmoid: y = clamp(x + 3, 0, 6) / 6."""

    _op_name = "hardsigmoid"
    kernel_cls = HardsigmoidFwdKernel
    # Manifest: flops = "3 * N". Per roofline.md §1.3:
    # hardsigmoid(x) = relu6(x+3)/6 =
    # add + two-sided-clamp(1) + div = 3 per elem.
    FLOPS_PER_ELEM = 3


class MishFwdOp(_ParamFreeActivationOp):
    """Element-wise Mish: y = x * tanh(softplus(x))."""

    _op_name = "mish"
    kernel_cls = MishFwdKernel
    # Manifest: flops = "4 * N". Per roofline.md §1.3:
    # mish(x) = x * tanh(softplus(x));
    # softplus = exp + log1p = 2; tanh(transcendental) + final mul = 4 per elem.
    FLOPS_PER_ELEM = 4


class SeluFwdOp(_ParamFreeActivationOp):
    """Element-wise SELU activation."""

    _op_name = "selu"
    kernel_cls = SeluFwdKernel
    # Manifest: flops = "5 * N" (branch + exp/sub/mul + lambda mul).
    FLOPS_PER_ELEM = 5


class LeakyReluFwdOp(_ParametricActivationOp):
    """Leaky ReLU: y = x if x > 0 else negative_slope * x.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        negative_slope: Slope for negative inputs (default 0.01).
        inplace: When True, copy the result back into ``input`` and
            return ``input`` (preserving tensor identity). The kernel
            still computes into a fresh buffer; only the user-visible
            tensor is mutated, mirroring ``torch.nn.functional.leaky_relu``.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune the kernel.
    """

    _op_name = "leaky_relu"
    _wrapped = None
    # Manifest: flops = "2 * N". Per roofline.md §1.3:
    # compare-and-select(1) + mul = 2 per elem.
    FLOPS_PER_ELEM = 2

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        negative_slope: float = 0.01,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        inplace: bool = False,
    ):
        _validate_scalar_param_repr("negative_slope", negative_slope, dtype, self._op_name)
        self.negative_slope = negative_slope
        self.dispatch_kernel(kernel_map)
        kernel = self.kernel_map[self._op_name](
            N_total, dtype, negative_slope=negative_slope, tune=tune,
        )
        self._finalize_init(N_total, dtype, kernel, inplace=inplace)

    @property
    def default_kernel_map(self):
        return {"leaky_relu": LeakyReluFwdKernel}


class EluFwdOp(_ParametricActivationOp):
    """ELU: y = x if x > 0 else alpha * (exp(x) - 1).

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        alpha: Scale for the negative part (default 1.0).
        inplace: When True, copy the result back into ``input`` and
            return ``input`` (preserving tensor identity).
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune the kernel.
    """

    _op_name = "elu"
    _wrapped = None
    # Manifest: flops = "4 * N". Per roofline.md §1.3:
    # compare-and-select(1) + exp + sub + mul = 4 per elem.
    FLOPS_PER_ELEM = 4

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        alpha: float = 1.0,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        inplace: bool = False,
    ):
        _validate_scalar_param_repr("alpha", alpha, dtype, self._op_name)
        self.alpha = alpha
        self.dispatch_kernel(kernel_map)
        kernel = self.kernel_map[self._op_name](
            N_total, dtype, alpha=alpha, tune=tune,
        )
        self._finalize_init(N_total, dtype, kernel, inplace=inplace)

    @property
    def default_kernel_map(self):
        return {"elu": EluFwdKernel}


class HardtanhFwdOp(_ParametricActivationOp):
    """Hardtanh: y = clamp(x, min_val, max_val).

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        min_val: Lower bound (default -1.0).
        max_val: Upper bound (default 1.0).
        inplace: When True, copy the result back into ``input`` and
            return ``input`` (preserving tensor identity).
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune the kernel.
    """

    _op_name = "hardtanh"
    _wrapped = None
    # Manifest: flops = "N". Per roofline.md §1.3, two-sided clamp
    # collapses to 1 compare-and-select per output element.
    FLOPS_PER_ELEM = 1

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        min_val: float = -1.0,
        max_val: float = 1.0,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        inplace: bool = False,
    ):
        _validate_scalar_param_repr("min_val", min_val, dtype, self._op_name)
        _validate_scalar_param_repr("max_val", max_val, dtype, self._op_name)
        self.min_val = min_val
        self.max_val = max_val
        self.dispatch_kernel(kernel_map)
        kernel = self.kernel_map[self._op_name](
            N_total, dtype, min_val=min_val, max_val=max_val, tune=tune,
        )
        self._finalize_init(N_total, dtype, kernel, inplace=inplace)

    @property
    def default_kernel_map(self):
        return {"hardtanh": HardtanhFwdKernel}


class SoftplusFwdOp(_ParametricActivationOp):
    """Softplus: y = log(1 + exp(x*beta))/beta if x*beta <= threshold else x.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
        beta: Scaling factor (default 1.0).
        threshold: Linear regime threshold (default 20.0).
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune the kernel.
    """

    _op_name = "softplus"
    _wrapped = None
    # Manifest: flops = "5 * N". Per roofline.md §1.3:
    # mul-beta + threshold compare-and-select(1) + exp + log1p + div-by-beta
    # = 5 per elem.
    FLOPS_PER_ELEM = 5

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        beta: float = 1.0,
        threshold: float = 20.0,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        _validate_scalar_param_repr("beta", beta, dtype, self._op_name)
        _validate_scalar_param_repr("threshold", threshold, dtype, self._op_name)
        self.beta = beta
        self.threshold = threshold
        self.dispatch_kernel(kernel_map)
        kernel = self.kernel_map[self._op_name](
            N_total, dtype, beta=beta, threshold=threshold, tune=tune,
        )
        # Softplus does not expose ``inplace`` to callers; default to False.
        self._finalize_init(N_total, dtype, kernel, inplace=False)

    @property
    def default_kernel_map(self):
        return {"softplus": SoftplusFwdKernel}
