"""NanToNum op: replace NaN, +Inf, -Inf with specified values."""

from typing import Dict, Optional

import torch

from tileops.kernels.elementwise import NanToNumFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import _OP_REGISTRY, _apply_fp8_post_cast, _validate_scalar_param_repr


class NanToNumFwdOp(Op):
    """NanToNum: replace NaN, +Inf, -Inf with specified values.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype.
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
        dtype: torch.dtype,
        nan: float = 0.0,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        # The manifest default ``None`` resolves to the *final*
        # user-facing dtype's max / min, not ``+/-inf``: the kernel runs
        # in ``output_dtype`` (fp16 for e5m2 to preserve Inf/NaN) and
        # _clamp_to_dtype_range targets that intermediate, so forwarding
        # ``+inf`` would resolve to fp16's 65504.0 and then surface as
        # ``+Inf`` after the e5m2 post-cast (e5m2 max is 57344.0).
        # Picking ``torch.finfo(dtype).max`` here keeps the replacement
        # value finite end-to-end and matches ``torch.nan_to_num``
        # semantics (replace Inf with the dtype's max finite value).
        _validate_scalar_param_repr("nan", nan, dtype, self._op_name)
        if posinf is None:
            kernel_posinf = torch.finfo(dtype).max
        else:
            _validate_scalar_param_repr("posinf", posinf, dtype, self._op_name)
            kernel_posinf = posinf
        if neginf is None:
            kernel_neginf = torch.finfo(dtype).min
        else:
            _validate_scalar_param_repr("neginf", neginf, dtype, self._op_name)
            kernel_neginf = neginf
        self.N_total = N_total
        self.dtype = dtype
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf
        self.dispatch_kernel(kernel_map)
        # Pass replacement values positionally; the kernel constructor's
        # internal parameter naming is encapsulated below the Op layer.
        self.kernel = self.kernel_map["nan_to_num"](
            N_total, dtype, nan, kernel_posinf, kernel_neginf, tune=tune,
        )
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"nan_to_num": NanToNumFwdKernel}

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        orig_shape = input.shape
        result = self.kernel(input.contiguous().reshape(-1)).reshape(orig_shape)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if not input.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if input.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {input.numel()}")
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, self._instance_key)
        return self._eager_forward(input)
