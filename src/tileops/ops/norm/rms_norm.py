"""RMS norm, and the shape an op takes once its compile boundary is in the op layer.

An op that wants ``torch.compile(op, fullgraph=True)`` writes these five:

1. ``forward`` — one call to the module-level operator, nothing else. This is as far as
   dynamo traces.
2. ``_eager_forward`` — what ``forward`` used to hold. It runs inside the operator, so
   validation, kernel construction and the launch are never traced.
3. ``_rms_norm_fwd`` — the operator, registered once at import time.
4. ``_rms_norm_fwd_fake`` — what the compiler is told about the output while tracing.
5. ``compile_op_names`` — the operator's name, which lets a test assert that the traced
   graph holds nothing else.

The sixth part comes from the base class: ``dispatch_kernel`` assigns the
``_instance_key`` that piece 1 passes across the boundary.
"""

from typing import ClassVar, Dict, Optional, Sequence, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import RMSNormKernel

from ..compile_boundary import get_instance
from ..op_base import Op
from .norm_base import normalized_shape_to_n

__all__ = ["RMSNormFwdOp"]


class RMSNormFwdOp(Op):
    """Standalone Root Mean Square (RMS) Norm operator.

    Mirrors :func:`torch.nn.functional.rms_norm`. Computes::

        y = x * rsqrt(mean(x ** 2, trailing_axes) + eps) * weight

    where the reduction runs over the trailing ``len(normalized_shape)``
    axes; ``normalized_shape`` is the only entry point (the manifest spec).

    Args:
        normalized_shape: Trailing-axis shape tuple over which the
            reduction runs (manifest ``params.normalized_shape``).
        eps: Epsilon for numerical stability (manifest ``params.eps``). ``None``
            selects the same default the signature carries. Normalized here, so a
            backend is handed the number rather than ``None``.
        target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
            in-tree kernels, or ``None`` to decide from the input device.
        kernel_map: Optional kernel override dictionary.
        tune: Whether to autotune (default ``False``).

    Example:
        >>> op = RMSNormFwdOp(normalized_shape=(4096,))
        >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
        >>> w = torch.randn(4096, dtype=torch.float16, device="cuda")
        >>> y = op(x, w)  # shape: (1024, 4096)
    """

    #: Manifest ``params.eps.default``. The signature default and the ``None``
    #: normalization both read it, so the two cannot drift apart.
    DEFAULT_EPS = 1.0e-6

    #: The operator this op registers on the compile boundary. Tests assert the traced
    #: graph holds nothing but this.
    compile_op_names: ClassVar[Tuple[str, ...]] = ("top::norm_rms_norm_fwd",)

    def __init__(
        self,
        normalized_shape: Sequence[int],
        eps: Optional[float] = DEFAULT_EPS,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.N = normalized_shape_to_n(normalized_shape)
        self.normalized_shape = tuple(int(d) for d in normalized_shape)
        # The manifest type is ``float | None``: an explicit None means the same default,
        # so a backend is handed a number either way.
        self.eps = self.DEFAULT_EPS if eps is None else float(eps)
        self.target = target
        self.tune = tune
        # Installs the kernel map, and assigns the ``_instance_key`` that ``forward``
        # passes across the boundary.
        self.dispatch_kernel(kernel_map)
        self._last_m: Optional[int] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"rms_norm": RMSNormKernel}

    def _infer_output_shapes(
        self,
        x_shape: Tuple[int, ...],
        weight_shape: Tuple[int, ...],
    ) -> Dict[str, Tuple[int, ...]]:
        """Manifest ``shape_rules``: ``output.shape == x.shape``."""
        return {"output": tuple(x_shape)}

    def eval_roofline(self) -> Tuple[int, int]:
        if self._last_m is None or self.dtype is None:
            raise RuntimeError(
                "RMSNormFwdOp.eval_roofline() requires a prior forward() "
                "call to bind the leading-dims product and the dtype."
            )
        m, n = self._last_m, self.N
        elem_bytes = self.dtype.itemsize
        return (4 * m * n, (2 * m * n + n) * elem_bytes)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization over the trailing ``normalized_shape``.

        Args:
            x: Input tensor whose trailing shape equals ``normalized_shape``.
            weight: Affine scale of shape ``normalized_shape``.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: Dtypes or devices disagree, or shapes are incompatible with the
                configured ``normalized_shape``. Raised from inside the operator, by
                :meth:`_eager_forward`.
        """
        return _rms_norm_fwd(x, weight, self._instance_key)

    def _eager_forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Validate, normalize, resolve the kernel and launch.

        Runs inside the operator, never under dynamo: kernel construction enters a
        TileLang builder, which is not traceable.
        """
        ns = self.normalized_shape
        k = len(ns)
        self._validate_dtypes(x, weight)
        self.dtype = x.dtype
        if weight.device != x.device or weight.dtype != x.dtype:
            raise ValueError(
                f"weight must be on {x.device} with dtype {x.dtype}, "
                f"got {weight.device} and {weight.dtype}"
            )
        if x.ndim < k or tuple(x.shape[-k:]) != ns:
            raise ValueError(
                f"Expected x trailing shape {ns}, "
                f"got {tuple(x.shape[-k:]) if x.ndim >= k else tuple(x.shape)}"
            )
        if tuple(weight.shape) != ns:
            raise ValueError(
                f"Expected weight shape {ns}, got {tuple(weight.shape)}"
            )

        # The op normalizes contiguity and hands over what the manifest declares; how a
        # kernel wants that laid out is its own business.
        x = x.contiguous()
        weight = weight.contiguous()
        kernel = self.get_or_build_kernel(
            "rms_norm",
            (x, weight),
            key=x.dtype,                      # this instance's in-tree cache key
            build=lambda: self.kernel_map["rms_norm"](
                self.N, self.eps, x.dtype, tune=self.tune,
            ),
        )
        self._last_m = x.numel() // self.N
        return kernel(x, weight)


# Both are module-level functions rather than methods, for three reasons: registration
# happens at import time and once per qualified name; the schema is read off the
# annotations, so ``self`` cannot appear in either signature; the instance is therefore
# recovered from a string key. Why that key is a string and why it is never reused:
# src/tileops/ops/compile_boundary.py.


@torch.library.custom_op("top::norm_rms_norm_fwd", mutates_args=())
def _rms_norm_fwd(
    x: torch.Tensor, weight: torch.Tensor, instance_key: str,
) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(x, weight)


@_rms_norm_fwd.register_fake
def _rms_norm_fwd_fake(
    x: torch.Tensor, weight: torch.Tensor, instance_key: str,
) -> torch.Tensor:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(tuple(x.shape), tuple(weight.shape))
    # ``new_empty``, not ``empty_like``: ``_eager_forward`` normalizes contiguity, so a
    # non-contiguous public input's strides must not survive into the fake. Dtype is the
    # manifest's ``same_as(x)``.
    return x.new_empty(shapes["output"])
