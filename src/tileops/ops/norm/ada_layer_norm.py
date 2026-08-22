from typing import ClassVar, Dict, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import AdaLayerNormKernel

from ..compile_boundary import get_instance
from ..op_base import Op

__all__ = ["AdaLayerNormFwdOp"]


class AdaLayerNormFwdOp(Op):
    """Adaptive Layer Normalization (AdaLN) operator.

    Applies layer normalization with per-token adaptive scale and shift:

    .. math::

        y = s \\cdot \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x]
            + \\epsilon}} + d

    where *s* (scale) and *d* (shift) are per-token tensors of shape
    ``(M, N)``, pre-computed by the caller from a conditioning signal.
    Linear projection from the conditioning input to scale/shift is the
    caller's responsibility.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary leading dimensions (3-D+) via flatten/unflatten.
        Handles non-contiguous inputs and non-power-of-two hidden dims.

    Args:
        eps: Epsilon for numerical stability (manifest ``params.eps``).
        target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
            in-tree kernels, or ``None`` to decide from the input device.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    #: The operator this op registers; a test asserts the graph holds nothing else.
    compile_op_names: ClassVar[Tuple[str, ...]] = ("tileops::norm_ada_layer_norm_fwd",)

    def __init__(
        self,
        eps: float = 1e-5,
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
        return {"ada_layer_norm": AdaLayerNormKernel}

    def _infer_output_shapes(
        self,
        x_shape: Tuple[int, ...],
        scale_shape: Tuple[int, ...],
        shift_shape: Tuple[int, ...],
    ) -> Dict[str, Tuple[int, ...]]:
        """Manifest ``shape_rules``: ``output.shape == x.shape``."""
        return {"output": tuple(x_shape)}

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_mn is None or self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior "
                "forward() call to bind input shape and dtype"
            )
        M, N = self._last_roofline_mn
        elem_bytes = self.dtype.itemsize
        return 5 * M * N, 4 * M * N * elem_bytes

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Apply adaptive layer normalization.

        Args:
            x: Tensor of shape ``(*leading, N)``.
            scale: Tensor of shape ``(*leading, N)``.
            shift: Tensor of shape ``(*leading, N)``.

        Returns:
            Tensor of the same shape as *x*.

        Raises:
            ValueError: Dtypes or shapes disagree. Raised from inside the operator, by
                :meth:`_eager_forward`.
        """
        return _norm_ada_layer_norm_fwd(x, scale, shift, self._instance_key)

    def _eager_forward(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo cannot follow.
        """
        self._validate_dtypes(x, scale, shift)
        self.dtype = x.dtype
        if scale.dtype != x.dtype:
            raise ValueError(f"Expected scale.dtype {x.dtype}, got {scale.dtype}")
        if scale.shape != x.shape:
            raise ValueError(f"Expected scale shape {tuple(x.shape)}, got {tuple(scale.shape)}")
        if shift.dtype != x.dtype:
            raise ValueError(f"Expected shift.dtype {x.dtype}, got {shift.dtype}")
        if shift.shape != x.shape:
            raise ValueError(f"Expected shift shape {tuple(x.shape)}, got {tuple(shift.shape)}")

        # Handed over as the manifest declares it; the layout a kernel wants is its own business.
        x = x.contiguous()
        scale = scale.contiguous()
        shift = shift.contiguous()
        n = x.shape[-1]
        kernel = self.get_or_build_kernel(
            "ada_layer_norm",
            (x, scale, shift),
            key=(n, x.dtype),  # this instance's in-tree cache key
            build=lambda: self.kernel_map["ada_layer_norm"](
                n,
                self.eps,
                x.dtype,
                has_gate=False,
                tune=self.tune,
            ),
        )
        self._last_roofline_mn = (x.numel() // n, n)
        return kernel(x, scale, shift)


@torch.library.custom_op("tileops::norm_ada_layer_norm_fwd", mutates_args=())
def _norm_ada_layer_norm_fwd(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(x, scale, shift)


@_norm_ada_layer_norm_fwd.register_fake
def _norm_ada_layer_norm_fwd_fake(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(tuple(x.shape), tuple(scale.shape), tuple(shift.shape))
    # ``new_empty``, not ``empty_like``: a non-contiguous input's strides must not reach the fake.
    return x.new_empty(shapes["output"])
