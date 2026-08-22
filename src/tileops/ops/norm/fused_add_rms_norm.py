from typing import ClassVar, Dict, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import FusedAddRMSNormKernel

from ..compile_boundary import get_instance
from ..op_base import Op

__all__ = ["FusedAddRMSNormFwdOp"]


class FusedAddRMSNormFwdOp(Op):
    """Fused residual addition and RMS Normalization operator.

    Computes the residual sum followed by RMS normalization in a single
    fused kernel:

    .. math::

        \\begin{aligned}
        r &= x + \\mathrm{residual} \\\\
        y &= \\frac{r}{\\sqrt{\\mathrm{mean}(r^2) + \\epsilon}} \\cdot w
        \\end{aligned}

    Returns dual outputs ``(y, residual_out)`` so downstream residual connections can
    reuse the pre-norm sum without recomputation.

    Supported dtypes:
        ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary leading dimensions (3-D+) via flatten/unflatten.
        Handles non-contiguous inputs and non-power-of-two hidden dims
        by padding to 256-element alignment.

    Args:
        eps: Epsilon for numerical stability (manifest ``params.eps``).
        target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
            in-tree kernels, or ``None`` to decide from the input device.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    #: The operator this op registers; a test asserts the graph holds nothing else.
    compile_op_names: ClassVar[Tuple[str, ...]] = ("tileops::norm_fused_add_rms_norm_fwd",)

    def __init__(
        self,
        eps: float = 1e-6,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.eps = eps
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._last_roofline_mn: Optional[tuple[int, int]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fused_add_rms_norm": FusedAddRMSNormKernel}

    def _infer_output_shapes(
        self,
        x_shape: Tuple[int, ...],
        residual_shape: Tuple[int, ...],
        weight_shape: Tuple[int, ...],
    ) -> Dict[str, Tuple[int, ...]]:
        """Manifest ``shape_rules``: both outputs have ``x``'s shape."""
        return {"output": tuple(x_shape), "residual_out": tuple(x_shape)}

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_mn is None or self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior "
                "forward() call to bind input shape and dtype"
            )
        M, N = self._last_roofline_mn
        elem_bytes = self.dtype.itemsize
        return (
            5 * M * N,
            (4 * M * N + N) * elem_bytes,
        )

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply fused residual addition and normalization.

        Args:
            x: Input tensor of shape ``(*leading, N)``.
            residual: Residual tensor of the same shape as *x*.
            weight: Affine scale of shape ``(N,)``.

        Returns:
            ``(y, residual_out)``, where *residual_out* is ``x + residual``, both of the
            same shape as *x*.

        Raises:
            ValueError: Dtypes or shapes disagree. Raised from inside the operator, by
                :meth:`_eager_forward`.
        """
        return _norm_fused_add_rms_norm_fwd(x, residual, weight, self._instance_key)

    def _eager_forward(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo cannot follow.
        """
        self._validate_dtypes(x, residual, weight)
        self.dtype = x.dtype
        for name, tensor in (
            ("residual", residual),
            ("weight", weight),
        ):
            if tensor.dtype != x.dtype:
                raise ValueError(f"Expected {name}.dtype {x.dtype}, got {tensor.dtype}")
        n = x.shape[-1]
        if residual.shape != x.shape:
            raise ValueError(
                f"Expected residual shape {tuple(x.shape)}, got {tuple(residual.shape)}"
            )
        if weight.ndim != 1 or weight.shape[0] != n:
            raise ValueError(f"Expected weight shape ({n},), got {tuple(weight.shape)}")

        # Handed over as the manifest declares it; the layout a kernel wants is its own business.
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        kernel = self.get_or_build_kernel(
            "fused_add_rms_norm",
            (x, residual, weight),
            key=(n, x.dtype),  # this instance's in-tree cache key
            build=lambda: self.kernel_map["fused_add_rms_norm"](
                n,
                self.eps,
                x.dtype,
                tune=self.tune,
            ),
        )
        self._last_roofline_mn = (x.numel() // n, n)
        y, residual_out = kernel(x, residual, weight)
        return y, residual_out


@torch.library.custom_op("tileops::norm_fused_add_rms_norm_fwd", mutates_args=())
def _norm_fused_add_rms_norm_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    instance_key: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return get_instance(instance_key)._eager_forward(x, residual, weight)


@_norm_fused_add_rms_norm_fwd.register_fake
def _norm_fused_add_rms_norm_fwd_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    instance_key: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(tuple(x.shape), tuple(residual.shape), tuple(weight.shape))
    # The manifest's shapes, not the kernel's: alignment padding is the kernel's business
    # and never reaches the op's return.
    return x.new_empty(shapes["output"]), x.new_empty(shapes["residual_out"])
