from typing import ClassVar, Dict, Optional, Sequence, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.norm import LayerNormKernel

from .._compile_boundary_codegen import OperatorSpec
from ..op_base import Op
from .norm_base import normalized_shape_to_n

__all__ = ["LayerNormFwdOp"]


class LayerNormFwdOp(Op):
    """Layer Normalization operator.

    Computes layer normalization over the trailing ``normalized_shape``
    axes:

    $$
    y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
    \\cdot w + b
    $$

    Mirrors `torch.nn.functional.layer_norm`. ``normalized_shape``
    is the only entry point (the manifest spec).

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    """

    # Manifest ``params.eps.default``, which PyTorch shares. The signature default and the
    # ``None`` normalization both read it, so the two cannot drift apart.
    DEFAULT_EPS = 1e-5

    compile_boundary: ClassVar[tuple[OperatorSpec, ...]] = (OperatorSpec(),)

    def __init__(
        self,
        normalized_shape: Sequence[int],
        eps: Optional[float] = DEFAULT_EPS,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            normalized_shape: Trailing-axis shape tuple over which the
                reduction runs (manifest ``params.normalized_shape``).
            eps: Epsilon for numerical stability (manifest ``params.eps``).
                ``None`` uses the PyTorch default ``1e-5``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dictionary.
            tune: If ``True``, autotune tile configurations.
        """
        self.N = normalized_shape_to_n(normalized_shape)
        self.normalized_shape = tuple(int(d) for d in normalized_shape)
        # The manifest type is ``float | None``: an explicit None means the same default,
        # so a backend is handed a number either way.
        self.eps = self.DEFAULT_EPS if eps is None else float(eps)
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._last_m: Optional[int] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"layer_norm": LayerNormKernel}

    def _infer_output_shapes(
        self,
        x_shape: Tuple[int, ...],
        weight_shape: Tuple[int, ...],
        bias_shape: Tuple[int, ...],
    ) -> Dict[str, Tuple[int, ...]]:
        """Manifest ``shape_rules``: ``output.shape == x.shape``."""
        return {"output": tuple(x_shape)}

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor with trailing shape equal to ``normalized_shape``.
            weight: Affine scale of shape ``normalized_shape``.
            bias: Affine shift of shape ``normalized_shape``.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: Dtypes or devices disagree, or shapes are incompatible with the
                configured ``normalized_shape``. Raised from inside the operator, by
                `_eager_forward`.
        """
        return self._wrapped(x, weight, bias, self._instance_key)

    def _eager_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo cannot follow.
        """
        self._validate_dtypes(x, weight, bias)
        self.dtype = x.dtype
        for name, t in (("weight", weight), ("bias", bias)):
            if t.device != x.device or t.dtype != x.dtype:
                raise ValueError(
                    f"{name} must be on {x.device} with dtype {x.dtype}, "
                    f"got {t.device} and {t.dtype}"
                )

        ns = self.normalized_shape
        k = len(ns)
        if x.ndim < k or tuple(x.shape[-k:]) != ns:
            raise ValueError(
                f"Expected x trailing shape {ns}, "
                f"got {tuple(x.shape[-k:]) if x.ndim >= k else tuple(x.shape)}"
            )
        if tuple(weight.shape) != ns:
            raise ValueError(f"Expected weight shape {ns}, got {tuple(weight.shape)}")
        if tuple(bias.shape) != ns:
            raise ValueError(f"Expected bias shape {ns}, got {tuple(bias.shape)}")

        # Handed over as the manifest declares it; the layout a kernel wants is its own business.
        x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        kernel = self.kernel_for("layer_norm", (x, weight, bias), x.dtype)
        self._last_m = x.numel() // self.N
        # What the manifest roofline resolves ``x`` through.
        self.x_shape = tuple(x.shape)
        return kernel(x, weight, bias)

    def entry_for(self, role: str, call: torch.dtype) -> Entry:
        """One implementation, built per dtype; the row width and epsilon are the op's."""
        return call, lambda: self.kernel_map["layer_norm"](self.N, self.eps, call, tune=self.tune)
