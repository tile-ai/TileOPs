"""The nan_to_num kernel."""

import tilelang.language as T
import torch

from ._base import ScalarParamUnaryKernel
from ._dtype import _clamp_to_dtype_range

__all__ = [
    "NanToNumFwdKernel",
]


class NanToNumFwdKernel(ScalarParamUnaryKernel):
    """NanToNum: replace NaN, +Inf, -Inf with specified values."""

    def __init__(
        self, N_total, dtype, nan_val=0.0, posinf_val=1e4, neginf_val=-1e4, config=None, tune=False
    ):
        self.nan_val = _clamp_to_dtype_range(nan_val, dtype)
        self.posinf_val = _clamp_to_dtype_range(posinf_val, dtype)
        self.neginf_val = _clamp_to_dtype_range(neginf_val, dtype)
        super().__init__(N_total, dtype, config=config, tune=tune)

    def _param_key(self):
        return f"nan={self.nan_val!r}|posinf={self.posinf_val!r}|neginf={self.neginf_val!r}"

    def _make_op_func(self):
        nan_val, posinf_val, neginf_val = self.nan_val, self.posinf_val, self.neginf_val
        # A clamp replaces the infinity tests only when the replacements are the
        # dtype's own ends; otherwise it would move finite values too.
        info = torch.finfo(self.dtype)
        clamps = posinf_val == info.max and neginf_val == info.min

        def op_func(x):
            wide = T.cast(x, "float32")
            nan_r = T.cast(nan_val, x.dtype)
            if clamps:
                bounded = T.min(
                    T.max(wide, T.cast(neginf_val, "float32")),
                    T.cast(posinf_val, "float32"),
                )
                return T.if_then_else(T.isnan(wide), nan_r, T.cast(bounded, x.dtype))
            pos_r = T.cast(posinf_val, x.dtype)
            neg_r = T.cast(neginf_val, x.dtype)
            # ``T.isinf`` lowers to this comparison plus a NaN test, which the
            # branch above has already taken.
            infinite = T.abs(wide) == T.cast(float("inf"), "float32")
            return T.if_then_else(
                T.isnan(wide),
                nan_r,
                T.if_then_else(
                    infinite,
                    T.if_then_else(wide > T.cast(0, "float32"), pos_r, neg_r),
                    x,
                ),
            )

        return op_func
