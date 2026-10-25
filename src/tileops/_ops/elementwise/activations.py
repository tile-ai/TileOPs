"""Activation elementwise ops (ReLU + parametric/param-free families)."""

from typing import ClassVar, Mapping

import torch

from tileops._kernels.elementwise import (
    ELUFwdKernel,
    GELUAndMulFwdKernel,
    GELUFwdKernel,
    GELUTanhAndMulFwdKernel,
    GELUTanhFwdKernel,
    HardsigmoidFwdKernel,
    HardswishFwdKernel,
    HardtanhFwdKernel,
    LeakyReLUFwdKernel,
    MishFwdKernel,
    ReLUFwdKernel,
    SELUFwdKernel,
    SigmoidFwdKernel,
    SiLUAndMulFwdKernel,
    SiLUFwdKernel,
    SoftplusFwdKernel,
    TanhFwdKernel,
)
from tileops._kernels.elementwise.call_spec import (
    BoundedUnaryFwdInterface,
    BoundsCall,
    ELUCall,
    ELUFwdInterface,
    GELUAndMulCall,
    GELUCall,
    LeakyReLUCall,
    LeakyReLUFwdInterface,
    SoftplusCall,
    SoftplusFwdInterface,
)
from tileops._kernels.kernel_base import KernelInterface
from tileops._ops.elementwise._base import (
    ELEMENTWISE,
    FusedGatedOp,
    UnaryOp,
    _ParamFreeActivationOp,
    _UnaryActivationMixin,
)
from tileops.backend import Target


class ReLUFwdOp(_ParamFreeActivationOp):
    """ReLU activation: y = max(x, 0); in place when ``inplace``."""

    kernel_types = {"relu": ReLUFwdKernel}


class GELUFwdOp(UnaryOp):
    """Element-wise GELU honoring the manifest ``approximate`` contract.

    On float16 and bfloat16 the error function is evaluated as a polynomial that
    saturates to exactly +/-1; its worst case over the real line is 1.7e-5, an
    order below half a float16 ulp at 1.0. float32 keeps `erff`.
    """

    kernel_types = {"gelu": GELUFwdKernel, "gelu_tanh": GELUTanhFwdKernel}

    def __init__(
        self,
        *,
        approximate: str = "none",
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            approximate: ``'none'`` (default) evaluates the erf form; ``'tanh'`` the
                tanh approximation ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``.
            target: Which set of kernels serves this op.
        """
        self.approximate = approximate
        super().__init__(target=target)

    def _call_spec(self, input: torch.Tensor) -> GELUCall:
        return GELUCall(
            device=input.device,
            n_total=input.numel(),
            dtype=input.dtype,
            approximate=self.approximate,
        )


class SiLUFwdOp(_ParamFreeActivationOp):
    """Element-wise SiLU (Swish): y = x * sigmoid(x); in place when ``inplace``."""

    kernel_types = {"silu": SiLUFwdKernel}


class SigmoidFwdOp(UnaryOp):
    """Element-wise sigmoid(x).

    A float16 or bfloat16 result equals ``torch.sigmoid``'s bit for bit at every input.
    """

    kernel_types = {"sigmoid": SigmoidFwdKernel}


class TanhFwdOp(UnaryOp):
    """Element-wise tanh(x)."""

    kernel_types = {"tanh": TanhFwdKernel}


class HardswishFwdOp(_ParamFreeActivationOp):
    """Element-wise HardSwish: y = x * clamp(x + 3, 0, 6) / 6; in place when ``inplace``."""

    kernel_types = {"hardswish": HardswishFwdKernel}


class HardsigmoidFwdOp(_ParamFreeActivationOp):
    """Element-wise HardSigmoid: y = clamp(x + 3, 0, 6) / 6; in place when ``inplace``."""

    kernel_types = {"hardsigmoid": HardsigmoidFwdKernel}


class MishFwdOp(_ParamFreeActivationOp):
    """Element-wise Mish: y = x * tanh(softplus(x)); in place when ``inplace``."""

    kernel_types = {"mish": MishFwdKernel}


class SELUFwdOp(_ParamFreeActivationOp):
    """Element-wise SELU activation; in place when ``inplace``."""

    kernel_types = {"selu": SELUFwdKernel}


class LeakyReLUFwdOp(_UnaryActivationMixin, UnaryOp):
    """Leaky ReLU: y = x if x > 0 else negative_slope * x; in place when ``inplace``."""

    kernel_types = {"leaky_relu": LeakyReLUFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {ELEMENTWISE: LeakyReLUFwdInterface}

    def __init__(
        self,
        *,
        negative_slope: float = 0.01,
        inplace: bool = False,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            negative_slope: Slope for negative inputs (default 0.01).
            inplace: When True, write the result into ``input`` and return ``input``.
            target: Which set of kernels serves this op.
        """
        self.negative_slope = negative_slope
        self.inplace = inplace
        super().__init__(target=target)

    def _call_spec(self, input: torch.Tensor) -> LeakyReLUCall:
        return LeakyReLUCall(
            device=input.device,
            n_total=input.numel(),
            dtype=input.dtype,
            negative_slope=self.negative_slope,
        )


class ELUFwdOp(_UnaryActivationMixin, UnaryOp):
    """ELU: y = x if x > 0 else alpha * (exp(x) - 1); in place when ``inplace``."""

    kernel_types = {"elu": ELUFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {ELEMENTWISE: ELUFwdInterface}

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        inplace: bool = False,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            alpha: Scale for the negative part (default 1.0).
            inplace: When True, write the result into ``input`` and return ``input``.
            target: Which set of kernels serves this op.
        """
        self.alpha = alpha
        self.inplace = inplace
        super().__init__(target=target)

    def _call_spec(self, input: torch.Tensor) -> ELUCall:
        return ELUCall(
            device=input.device, n_total=input.numel(), dtype=input.dtype, alpha=self.alpha
        )


class HardtanhFwdOp(_UnaryActivationMixin, UnaryOp):
    """Hardtanh: y = clamp(x, min_val, max_val); in place when ``inplace``."""

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
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            min_val: Lower bound (default -1.0).
            max_val: Upper bound (default 1.0).
            inplace: When True, write the result into ``input`` and return ``input``.
            target: Which set of kernels serves this op.
        """
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        super().__init__(target=target)

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
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            beta: Scaling factor (default 1.0).
            threshold: Linear regime threshold (default 20.0).
            target: Which set of kernels serves this op.
        """
        self.beta = beta
        self.threshold = threshold
        super().__init__(target=target)

    def _call_spec(self, input: torch.Tensor) -> SoftplusCall:
        return SoftplusCall(
            device=input.device,
            n_total=input.numel(),
            dtype=input.dtype,
            beta=self.beta,
            threshold=self.threshold,
        )


class SiLUAndMulFwdOp(FusedGatedOp):
    """SiLU-and-Mul: y = silu(gate) * value."""

    kernel_types = {"silu_and_mul": SiLUAndMulFwdKernel}


class GELUAndMulFwdOp(FusedGatedOp):
    """GELU-and-Mul: y = gelu(gate) * value, GELU in the form ``approximate`` names.

    On float16 and bfloat16 the exact form evaluates the error function as a polynomial
    that saturates to exactly +/-1; its worst case over the real line is 1.7e-5, an
    order below half a float16 ulp at 1.0. float32 keeps `erff`.
    """

    kernel_types = {
        "gelu_and_mul": GELUAndMulFwdKernel,
        "gelu_tanh_and_mul": GELUTanhAndMulFwdKernel,
    }

    def __init__(
        self,
        *,
        approximate: str = "none",
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            approximate: ``'none'`` (default) evaluates the erf form of GELU; ``'tanh'`` the
                tanh approximation ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
                the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.approximate = approximate
        super().__init__(target=target)

    def _call_spec(self, input: torch.Tensor) -> GELUAndMulCall:
        return GELUAndMulCall(
            device=input.device,
            m=input.shape[0],
            n=input.shape[1] // 2,
            dtype=input.dtype,
            approximate=self.approximate,
        )
