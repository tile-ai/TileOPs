"""Independent RMS reductions over groups, with an affine weight over the full row."""

import math
import operator

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel, dynamic_extent_supply
from tileops._kernels.norm._config import rms_scale
from tileops._kernels.norm.call_spec import RMSNormCall, RMSNormFwdInterface
from tileops._kernels.warp_reduce import warp_reduce
from tileops._utils import WARP_LANES

__all__ = ["RMSNormGroupedKernel"]


@target_cache(maxsize=32)
def _rms_norm_grouped_kernel(n, group, eps, dtype, zero_centered, has_weight):
    groups = n // group
    itemsize = getattr(torch, dtype).itemsize

    @tilelang.jit(out_idx=[2])
    def build(threads):
        rows = T.dynamic("rows", "int64")
        # Whole thread chunks use the widest vector dividing each thread's share. Otherwise
        # the vector divides the group, so the final chunk masks whole vectors safely.
        vec = math.gcd(
            group // threads if group % threads == 0 else group, VECTOR_ACCESS_BYTES // itemsize
        )
        steps = -(-group // (threads * vec))
        warps = threads // WARP_LANES

        @T.prim_func
        def main(
            x: T.Tensor((rows, n), dtype),
            weight: T.Tensor((n if has_weight else 1,), dtype),
            y: T.Tensor((rows, n), dtype),
        ):
            with T.Kernel(rows * groups, threads=threads) as block:
                tx = T.get_thread_binding()
                row = T.cast(block, "int64") // groups
                start = block % groups * group
                held = T.alloc_local((steps * vec,), dtype)
                scale = T.alloc_local((steps * vec,), dtype)
                total = T.alloc_local((1,), "float32")
                partial = T.alloc_shared((warps,), "float32")
                for k in T.unroll(steps):
                    col = (k * threads + tx) * vec
                    for i in T.unroll(vec):
                        held[k * vec + i] = T.cast(0, dtype)
                    if col < group:
                        for i in T.vectorized(vec):
                            held[k * vec + i] = x[row, start + col + i]
                        if has_weight:
                            for i in T.vectorized(vec):
                                scale[k * vec + i] = weight[start + col + i]
                total[0] = T.float32(0)
                for i in T.unroll(steps * vec):
                    value = T.cast(held[i], "float32")
                    total[0] += value * value
                warp_reduce(total, operator.add)
                if warps > 1:
                    if tx % WARP_LANES == 0:
                        partial[tx // WARP_LANES] = total[0]
                    T.sync_threads()
                    total[0] = T.float32(0)
                    for w in T.unroll(warps):
                        total[0] += partial[w]
                inv = T.rsqrt(total[0] / T.float32(group) + T.float32(eps))
                for k in T.unroll(steps):
                    col = (k * threads + tx) * vec
                    if col < group:
                        for i in T.unroll(vec):
                            value = T.cast(held[k * vec + i], "float32") * inv
                            if has_weight:
                                value = value * rms_scale(
                                    T.cast(scale[k * vec + i], "float32"), zero_centered
                                )
                            held[k * vec + i] = T.cast(value, dtype)
                        for i in T.vectorized(vec):
                            y[row, start + col + i] = held[k * vec + i]

        return main

    return build


class RMSNormGroupedKernel(Kernel, RMSNormFwdInterface):
    """One CTA per group, holding its input in registers until its RMS is known."""

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def refusal(cls, call: RMSNormCall) -> str | None:
        if call.group_size in (None, call.n):
            return "serves only group_size shorter than the row"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: RMSNormCall) -> Entry:
        identity = (call.n, call.group_size, call.eps, call.dtype, call.zero_centered)
        return identity, lambda: cls(*identity)

    def __init__(self, n, group, eps, dtype, zero_centered, config=None):
        super().__init__()
        self.n, self.group, self.eps = n, group, eps
        self.dtype, self.zero_centered = dtype, zero_centered
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # Four warps expose independent loads while keeping the cross-warp sum short.
        return {"threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"threads": t} for t in (64, 128, 256)]

    @property
    def autotune_supply_prog(self):
        # Tune on at least two groups per SM.
        return dynamic_extent_supply(-(-2 * self.build_target.sm_count // (self.n // self.group)))

    def forward(self, x: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        self._require_devices(x=x, weight=weight)
        rows = x.reshape(-1, self.n)
        has_weight = weight is not None
        scale = weight.reshape(self.n) if has_weight else rows.new_empty(1)
        program = self.program(self._compile, has_weight)
        return program(rows, scale).reshape(x.shape)

    def _compile(self, cfg: dict, has_weight: bool):
        self.kernel = kernel = _rms_norm_grouped_kernel(
            self.n, self.group, self.eps, self.dtype_str, self.zero_centered, has_weight
        )
        return kernel(**cfg)
