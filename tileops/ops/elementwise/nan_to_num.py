"""NanToNum op: replace NaN, +Inf, -Inf with specified values."""

from typing import Dict, Optional

import torch

from tileops.kernels.elementwise import NanToNumFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import (
    KernelEntry,
    _PerDtypeKernels,
    _validate_scalar_param_repr,
    resolve_output_dtype,
)


class NanToNumFwdOp(_PerDtypeKernels, Op):
    """NanToNum: replace NaN, +Inf, -Inf with specified values.

    Args:
        N_total: Total number of elements (flattened).
        nan: Replacement for NaN (default 0.0).
        posinf: Replacement for +Inf. Manifest default ``None`` resolves
            to the largest finite value representable in the user-facing
            ``dtype`` (matches ``torch.nan_to_num``). Explicit values
            must also be representable in ``dtype`` end-to-end; values
            that fit only in the kernel's intermediate dtype (e.g. fp16
            for fp8_e5m2) are rejected so the post-cast cannot resurface
            them as Inf.
        neginf: Replacement for -Inf. Manifest default ``None`` resolves
            to the smallest (most negative) finite value representable
            in the user-facing ``dtype``.
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune the kernel.
    """

    _op_name = "nan_to_num"
    _wrapped = None

    def __init__(
        self,
        N_total: int,
        nan: float = 0.0,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.N_total = N_total
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf
        # Manifest input binding for the synthesized eval_roofline
        # (docs/design/roofline.md §4.4.3): the resolver locates the
        # input as ``self.input_shape`` since this op stores only the
        # flat element count, not the original-rank tensor.
        self.input_shape = (N_total,)
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._init_entries()

    def _build_entry(self, dtype: torch.dtype, *shape: int) -> KernelEntry:
        """Resolve the replacement values against *dtype*, then build.

        A ``None`` bound means "this dtype's largest finite value", so it
        cannot be resolved before the element type is known. Picking
        ``finfo(dtype).max`` keeps the replacement finite end-to-end and
        matches ``torch.nan_to_num``; forwarding ``+inf`` would resolve to
        fp16's 65504.0 and resurface as ``+Inf`` after an e5m2 post-cast.
        """
        _validate_scalar_param_repr("nan", self.nan, dtype, self._op_name)
        if self.posinf is None:
            posinf = torch.finfo(dtype).max
        else:
            _validate_scalar_param_repr(
                "posinf", self.posinf, dtype, self._op_name)
            posinf = self.posinf
        if self.neginf is None:
            neginf = torch.finfo(dtype).min
        else:
            _validate_scalar_param_repr(
                "neginf", self.neginf, dtype, self._op_name)
            neginf = self.neginf
        # Replacement values are positional; the kernel constructor's
        # parameter naming is encapsulated below the Op layer.
        kernel = self.kernel_map["nan_to_num"](
            self.N_total, dtype, self.nan, posinf, neginf, tune=self.tune,
        )

        return KernelEntry(
            kernel=kernel,
            compute_dtype=dtype,
            output_dtype=resolve_output_dtype(type(self).__name__, dtype),
        )

    @property
    def default_kernel_map(self):
        return {"nan_to_num": NanToNumFwdKernel}

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_shape = input.shape
        return self._entry(input.dtype).kernel(
            input.contiguous().reshape(-1)
        ).reshape(orig_shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        self._validate_dtypes(input)
        if input.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {input.numel()}")
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, self._instance_key)
        return self._eager_forward(input)
