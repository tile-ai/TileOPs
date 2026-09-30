"""The nan_to_num kernel."""

import tilelang.language as T
import torch

from tileops.kernels.elementwise._base import ScalarParamUnaryKernel
from tileops.kernels.elementwise._dtype import clamp_to_dtype_range
from tileops.kernels.elementwise.call_spec import NanToNumCall, NanToNumFwdInterface
from tileops.kernels.kernel_base import Entry

__all__ = [
    "NanToNumFwdKernel",
]


def _cast(value: float, dtype: torch.dtype) -> float:
    """*value* as *dtype* holds it, through float32, as torch casts a Python scalar."""
    return torch.tensor(value, dtype=torch.float32).to(dtype).item()


class NanToNumFwdKernel(ScalarParamUnaryKernel, NanToNumFwdInterface):
    """NanToNum: replace NaN, +Inf, -Inf with specified values."""

    @classmethod
    def entry_for(cls, call: NanToNumCall) -> Entry:
        """Resolve the replacements against the element type, then build.

        A bound the call left unset stands for the largest finite value of the element
        type, which is what ``torch.nan_to_num`` writes; forwarding an infinity would
        write back the value the op was called to replace. A stated one is cast as torch
        casts it, through float32, so a value past the type's range becomes an infinity.
        """
        dtype = call.dtype
        posinf = torch.finfo(dtype).max if call.posinf is None else _cast(call.posinf, dtype)
        neginf = torch.finfo(dtype).min if call.neginf is None else _cast(call.neginf, dtype)
        nan = _cast(call.nan, dtype)
        return call, lambda: cls(call.n_total, dtype, nan, posinf, neginf)

    def __init__(
        self, N_total, dtype, nan_val=0.0, posinf_val=1e4, neginf_val=-1e4, config=None, tune=False
    ):
        self.nan_val = clamp_to_dtype_range(nan_val, dtype)
        self.posinf_val = clamp_to_dtype_range(posinf_val, dtype)
        self.neginf_val = clamp_to_dtype_range(neginf_val, dtype)
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
