"""GroupNorm forward operator.

Wraps GroupNormKernel in the standard TileOPs Op interface.

User-facing API mirrors torch.nn.functional.group_norm:

    op = GroupNormFwdOp(num_groups=groups)
    y = op(x, weight, bias)   # affine
    y = op(x)                 # torch.nn.GroupNorm(affine=False)

Input tensors accept shape (N, C, *spatial); the kernel reshapes to
(N*num_groups, D) internally where D = (C/num_groups) * spatial_size.
"""

import math
from typing import ClassVar, Dict, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import GroupNormKernel, GroupNormNoAffineKernel

from ..compile_boundary import get_instance
from ..op_base import Op

__all__ = ["GroupNormFwdOp"]


class GroupNormFwdOp(Op):
    """Group Normalization forward operator.

    Computes group normalization over ``(C/num_groups, *spatial)`` slices:

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
            \\cdot w + b

    where the mean and variance are computed per group over
    ``(C/num_groups, *spatial)`` elements.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary spatial dimensions (1-D, 2-D, 3-D+).
        Handles non-contiguous inputs via explicit ``contiguous()`` call.
        The per-channel affine is applied inside the kernel, so the op does
        no post-kernel arithmetic.

    ``weight`` and ``bias`` are one switch: pass both for the affine form,
    pass neither for ``torch.nn.GroupNorm(affine=False)``. Passing one alone
    is an error — the manifest states the same in ``shape_rules``.

    Args:
        num_groups: Number of groups (manifest ``params.num_groups``).
            Must divide *C* evenly.
        eps: Epsilon for numerical stability (manifest ``params.eps``).
        target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
            in-tree kernels, or ``None`` to decide from the input device.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    #: The operator this op registers; a test asserts the graph holds nothing else.
    compile_op_names: ClassVar[Tuple[str, ...]] = ("top::norm_group_norm_fwd",)

    def __init__(
        self,
        num_groups: int,
        eps: float = 1e-5,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.num_groups = num_groups
        self.dtype: Optional[torch.dtype] = None
        self.eps = eps
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self.kernel: Optional[Kernel] = None
        self._last_roofline_spec: Optional[tuple[int, int, int, torch.dtype, bool]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "group_norm": GroupNormKernel,
            "group_norm_no_affine": GroupNormNoAffineKernel,
        }

    def _infer_output_shapes(
        self,
        x_shape: Tuple[int, ...],
        weight_shape: Optional[Tuple[int, ...]],
        bias_shape: Optional[Tuple[int, ...]],
    ) -> Dict[str, Tuple[int, ...]]:
        """Manifest ``shape_rules``: ``output.shape == x.shape``."""
        return {"output": tuple(x_shape)}

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError("GroupNormFwdOp.eval_roofline() requires a prior forward() call")
        N, C, spatial_size, dtype, affine = self._last_roofline_spec
        elem_bytes = dtype.itemsize
        return (
            (5 if affine else 3) * N * C * spatial_size,
            (2 * N * C * spatial_size + (2 * C if affine else 0)) * elem_bytes,
        )

    def _resolve_spec(self, x: torch.Tensor) -> Tuple[int, int, int, int, int, torch.dtype]:
        if x.ndim < 2:
            raise ValueError("x must have shape (N, C, *spatial)")
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"x.dtype must be float32, float16, or bfloat16, got {x.dtype}")
        N, C, *spatial = x.shape
        if C % self.num_groups != 0:
            raise ValueError(f"C={C} must be divisible by num_groups={self.num_groups}")
        spatial_size = math.prod(spatial)
        cpg = C // self.num_groups
        return N, C, spatial_size, cpg, cpg * spatial_size, x.dtype

    def _bind_spec(
        self,
        N: int,
        C: int,
        spatial_size: int,
        dtype: torch.dtype,
        affine: bool,
    ) -> None:
        """Bind what ``eval_roofline`` reads off the call that just ran."""
        self.dtype = dtype
        self._last_roofline_spec = (N, C, spatial_size, dtype, affine)

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply group normalization.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)``.
            weight: Affine scale of shape ``(C,)``, or ``None``.
            bias: Affine shift of shape ``(C,)``, or ``None``.
                ``weight`` and ``bias`` are one switch: give both or neither.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: Only one of ``weight`` / ``bias`` is given, dtypes disagree, or a
                shape is incompatible with *x*. Raised from inside the operator, by
                :meth:`_eager_forward`.
        """
        return _norm_group_norm_fwd(x, weight, bias, self._instance_key)

    def _eager_forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo cannot follow.
        """
        N, C, spatial_size, cpg, D, dtype = self._resolve_spec(x)
        # weight and bias are a single affine switch.
        if (weight is None) != (bias is None):
            given, missing = ("weight", "bias") if bias is None else ("bias", "weight")
            raise ValueError(
                f"weight and bias are one switch: got {given} without "
                f"{missing}. Pass both for the affine form, or neither for "
                f"torch.nn.GroupNorm(affine=False)"
            )
        affine = weight is not None
        if affine:
            for name, t in (("weight", weight), ("bias", bias)):
                if t.device != x.device:
                    raise ValueError(f"Expected {name} on {x.device}, got {t.device}")
                if t.dtype != dtype:
                    raise ValueError(f"Expected {name}.dtype {dtype}, got {t.dtype}")
                if t.ndim != 1 or t.shape[0] != C:
                    raise ValueError(f"Expected {name} shape ({C},), got {tuple(t.shape)}")

        self._bind_spec(N, C, spatial_size, dtype, affine)

        # Handed over as the manifest declares it; the layout a kernel wants is its own business.
        x = x.contiguous()
        if affine:
            weight = weight.contiguous()
            bias = bias.contiguous()
        # The affine pair picks the implementation, so it belongs in the key; both are
        # fetched under one name, which is what a target is asked to serve.
        slot = "group_norm" if affine else "group_norm_no_affine"
        kernel = self.get_or_build_kernel(
            "group_norm",
            (x, weight, bias),
            key=(D, cpg, dtype, affine),  # this instance's in-tree cache key
            build=lambda: (
                self.kernel_map[slot](D, self.eps, dtype, self.num_groups, cpg, tune=self.tune)
                if affine
                else self.kernel_map[slot](D, self.eps, dtype, tune=self.tune)
            ),
        )
        self.kernel = kernel

        # The affine kernel derives each element's channel from its position
        # in the row, so the per-channel affine is applied inside the kernel.
        return kernel(x, weight, bias)


@torch.library.custom_op("top::norm_group_norm_fwd", mutates_args=())
def _norm_group_norm_fwd(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    instance_key: str,
) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(x, weight, bias)


@_norm_group_norm_fwd.register_fake
def _norm_group_norm_fwd_fake(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    instance_key: str,
) -> torch.Tensor:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(
        tuple(x.shape),
        None if weight is None else tuple(weight.shape),
        None if bias is None else tuple(bias.shape),
    )
    # ``new_empty``, not ``empty_like``: a non-contiguous input's strides must not reach the fake.
    return x.new_empty(shapes["output"])
