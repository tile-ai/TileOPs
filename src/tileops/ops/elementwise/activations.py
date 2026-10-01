"""Activation elementwise ops (ReLU + parametric/param-free families)."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import (
    EluFwdKernel,
    GeluAndMulFwdKernel,
    GeluFwdKernel,
    GeluTanhAndMulFwdKernel,
    GeluTanhFwdKernel,
    HardsigmoidFwdKernel,
    HardswishFwdKernel,
    HardtanhFwdKernel,
    LeakyReluFwdKernel,
    MishFwdKernel,
    ReluFwdKernel,
    SeluFwdKernel,
    SigmoidFwdKernel,
    SiluAndMulFwdKernel,
    SiluFwdKernel,
    SoftplusFwdKernel,
    TanhFwdKernel,
)
from tileops.kernels.elementwise.call_spec import (
    BoundedUnaryFwdInterface,
    BoundsCall,
    EluCall,
    EluFwdInterface,
    LeakyReluCall,
    LeakyReluFwdInterface,
    SoftplusCall,
    SoftplusFwdInterface,
)
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.elementwise._base import (
    ELEMENTWISE,
    FusedGatedOp,
    UnaryOp,
    _ParamFreeActivationOp,
    _UnaryActivationMixin,
)


class ReluFwdOp(_ParamFreeActivationOp):
    """ReLU activation: y = max(x, 0)."""

    kernel_types = {"relu": ReluFwdKernel}


class GeluFwdOp(UnaryOp):
    """Element-wise GELU honoring the manifest ``approximate`` contract.

    On float16 and bfloat16 the error function is evaluated as a polynomial that
    saturates to exactly +/-1; its worst case over the real line is 1.7e-5, an
    order below half a float16 ulp at 1.0. float32 keeps `erff`.
    """

    kernel_types = {"gelu": GeluFwdKernel, "gelu_tanh": GeluTanhFwdKernel}

    def __init__(
        self,
        *,
        approximate: str = "none",
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            approximate: ``'none'`` (default) evaluates the erf form; ``'tanh'`` the
                tanh approximation ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``.
            target: Which set of kernels serves this op.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune the kernel.
        """
        self.approximate = approximate
        super().__init__(target=target, kernel_map=kernel_map, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        key = "gelu_tanh" if self.approximate == "tanh" else "gelu"
        return {key: self.kernel_types[key]}


class SiluFwdOp(_ParamFreeActivationOp):
    """Element-wise SiLU (Swish): y = x * sigmoid(x)."""

    kernel_types = {"silu": SiluFwdKernel}


class SigmoidFwdOp(UnaryOp):
    """Element-wise sigmoid(x)."""

    kernel_types = {"sigmoid": SigmoidFwdKernel}


class TanhFwdOp(UnaryOp):
    """Element-wise tanh(x)."""

    kernel_types = {"tanh": TanhFwdKernel}


class HardswishFwdOp(_ParamFreeActivationOp):
    """Element-wise HardSwish: y = x * clamp(x + 3, 0, 6) / 6."""

    kernel_types = {"hardswish": HardswishFwdKernel}


class HardsigmoidFwdOp(_ParamFreeActivationOp):
    """Element-wise HardSigmoid: y = clamp(x + 3, 0, 6) / 6."""

    kernel_types = {"hardsigmoid": HardsigmoidFwdKernel}


class MishFwdOp(_ParamFreeActivationOp):
    """Element-wise Mish: y = x * tanh(softplus(x))."""

    kernel_types = {"mish": MishFwdKernel}


class SeluFwdOp(_ParamFreeActivationOp):
    """Element-wise SELU activation."""

    kernel_types = {"selu": SeluFwdKernel}


class LeakyReluFwdOp(_UnaryActivationMixin, UnaryOp):
    """Leaky ReLU: y = x if x > 0 else negative_slope * x."""

    kernel_types = {"leaky_relu": LeakyReluFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {ELEMENTWISE: LeakyReluFwdInterface}

    def __init__(
        self,
        *,
        negative_slope: float = 0.01,
        inplace: bool = False,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            negative_slope: Slope for negative inputs (default 0.01).
            inplace: When True, write the result into ``input`` and return ``input``.
            target: Which set of kernels serves this op.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune the kernel.
        """
        self.negative_slope = negative_slope
        self.inplace = inplace
        super().__init__(target=target, kernel_map=kernel_map, tune=tune)

    def _call_spec(self, input: torch.Tensor) -> LeakyReluCall:
        return LeakyReluCall(
            device=input.device,
            n_total=input.numel(),
            dtype=input.dtype,
            negative_slope=self.negative_slope,
        )


class EluFwdOp(_UnaryActivationMixin, UnaryOp):
    """ELU: y = x if x > 0 else alpha * (exp(x) - 1)."""

    kernel_types = {"elu": EluFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {ELEMENTWISE: EluFwdInterface}

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        inplace: bool = False,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            alpha: Scale for the negative part (default 1.0).
            inplace: When True, write the result into ``input`` and return ``input``.
            target: Which set of kernels serves this op.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune the kernel.
        """
        self.alpha = alpha
        self.inplace = inplace
        super().__init__(target=target, kernel_map=kernel_map, tune=tune)

    def _call_spec(self, input: torch.Tensor) -> EluCall:
        return EluCall(
            device=input.device, n_total=input.numel(), dtype=input.dtype, alpha=self.alpha
        )


class HardtanhFwdOp(_UnaryActivationMixin, UnaryOp):
    """Hardtanh: y = clamp(x, min_val, max_val)."""

    kernel_types = {"hardtanh": HardtanhFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: BoundedUnaryFwdInterface
    }

    def __init__(
        self,
        *,
        min_val: float = -1.0,
        max_val: float = 1.0,
        inplace: bool = False,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            min_val: Lower bound (default -1.0).
            max_val: Upper bound (default 1.0).
            inplace: When True, write the result into ``input`` and return ``input``.
            target: Which set of kernels serves this op.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune the kernel.
        """
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        super().__init__(target=target, kernel_map=kernel_map, tune=tune)

    def _call_spec(self, input: torch.Tensor) -> BoundsCall:
        return BoundsCall(
            device=input.device,
            n_total=input.numel(),
            dtype=input.dtype,
            min_val=self.min_val,
            max_val=self.max_val,
        )


class SoftplusFwdOp(UnaryOp):
    """Softplus: y = log(1 + exp(x*beta))/beta if x*beta <= threshold else x."""

    kernel_types = {"softplus": SoftplusFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {ELEMENTWISE: SoftplusFwdInterface}

    def __init__(
        self,
        *,
        beta: float = 1.0,
        threshold: float = 20.0,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            beta: Scaling factor (default 1.0).
            threshold: Linear regime threshold (default 20.0).
            target: Which set of kernels serves this op.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune the kernel.
        """
        self.beta = beta
        self.threshold = threshold
        super().__init__(target=target, kernel_map=kernel_map, tune=tune)

    def _call_spec(self, input: torch.Tensor) -> SoftplusCall:
        return SoftplusCall(
            device=input.device,
            n_total=input.numel(),
            dtype=input.dtype,
            beta=self.beta,
            threshold=self.threshold,
        )


class SiluAndMulFwdOp(FusedGatedOp):
    """SiLU-and-Mul: y = silu(gate) * value."""

    kernel_types = {"silu_and_mul": SiluAndMulFwdKernel}


class GeluAndMulFwdOp(FusedGatedOp):
    """GELU-and-Mul: y = gelu(gate) * value (exact GELU).

    On float16 and bfloat16 the error function is evaluated as a polynomial that
    saturates to exactly +/-1; its worst case over the real line is 1.7e-5, an
    order below half a float16 ulp at 1.0. float32 keeps `erff`.
    """

    kernel_types = {"gelu_and_mul": GeluAndMulFwdKernel}


class GeluTanhAndMulFwdOp(FusedGatedOp):
    """GELU-Tanh-and-Mul: y = gelu_tanh(gate) * value (tanh approximation)."""

    kernel_types = {"gelu_tanh_and_mul": GeluTanhAndMulFwdKernel}
