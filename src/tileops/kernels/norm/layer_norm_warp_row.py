"""LayerNorm with one warp per row, for narrow rows the 256-element alignment would pad.

Each lane reads its share of the row in 16-byte vectors, and the mean and the centered
variance are summed across the warp by shuffles, so the row is never padded and no thread
waits on shared memory. The weight and bias are read with the row, before the reductions.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tvm import DataType

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.norm.call_spec import LayerNormCall, LayerNormFwdInterface
from tileops.kernels.tiling import ALIGNMENT
from tileops.utils import WARP_LANES

__all__ = ["LayerNormWarpRowKernel"]


@functools.lru_cache(maxsize=32)
def _layer_norm_warp_row_kernel(M, N, eps, dtype, has_weight, has_bias):
    vec = VECTOR_ACCESS_BYTES // (DataType(dtype).bits // 8)
    vectors = N // vec
    per_lane = -(-vectors // WARP_LANES)  # vectors a lane reads; the last may be past the row
    steps = WARP_LANES.bit_length() - 1

    @tilelang.jit(out_idx=[3])
    def _func(warps):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            weight: T.Tensor[(N if has_weight else 1,), dtype],
            bias: T.Tensor[(N if has_bias else 1,), dtype],
            y: T.Tensor[(M, N), dtype],
        ):
            with T.Kernel(T.ceildiv(M, warps), threads=warps * WARP_LANES) as pid:
                tx = T.get_thread_binding()
                lane = tx % WARP_LANES
                row = pid * warps + tx // WARP_LANES
                held = T.alloc_local([per_lane * vec], dtype)
                values = T.alloc_local([per_lane * vec], "float32")
                scale = T.alloc_local([per_lane * vec if has_weight else 1], dtype)
                shift = T.alloc_local([per_lane * vec if has_bias else 1], dtype)
                total = T.alloc_local([1], "float32")

                for k in T.unroll(per_lane):
                    if lane + k * WARP_LANES < vectors and row < M:
                        for i in T.vectorized(vec):
                            held[k * vec + i] = x[row, (lane + k * WARP_LANES) * vec + i]
                for k in T.unroll(per_lane):
                    if lane + k * WARP_LANES < vectors:
                        if has_weight:
                            for i in T.vectorized(vec):
                                scale[k * vec + i] = weight[(lane + k * WARP_LANES) * vec + i]
                        if has_bias:
                            for i in T.vectorized(vec):
                                shift[k * vec + i] = bias[(lane + k * WARP_LANES) * vec + i]

                total[0] = T.cast(0, "float32")
                for k in T.unroll(per_lane):
                    for i in T.unroll(vec):
                        values[k * vec + i] = T.if_then_else(
                            lane + k * WARP_LANES < vectors,
                            T.cast(held[k * vec + i], "float32"),
                            T.cast(0, "float32"),
                        )
                        total[0] += values[k * vec + i]
                for step in T.unroll(steps):
                    total[0] += T.shfl_xor(total[0], T.shift_left(1, step))
                mean = total[0] / float(N)

                total[0] = T.cast(0, "float32")
                for k in T.unroll(per_lane):
                    for i in T.unroll(vec):
                        centered = values[k * vec + i] - mean
                        total[0] += T.if_then_else(
                            lane + k * WARP_LANES < vectors,
                            centered * centered,
                            T.cast(0, "float32"),
                        )
                for step in T.unroll(steps):
                    total[0] += T.shfl_xor(total[0], T.shift_left(1, step))
                rstd = T.rsqrt(total[0] / float(N) + eps)

                for k in T.unroll(per_lane):
                    if lane + k * WARP_LANES < vectors and row < M:
                        for i in T.unroll(vec):
                            normed = (values[k * vec + i] - mean) * rstd
                            if has_weight:
                                normed = normed * T.cast(scale[k * vec + i], "float32")
                            if has_bias:
                                normed = normed + T.cast(shift[k * vec + i], "float32")
                            held[k * vec + i] = T.cast(normed, dtype)
                        for i in T.vectorized(vec):
                            y[row, (lane + k * WARP_LANES) * vec + i] = held[k * vec + i]

        return main

    return _func


class LayerNormWarpRowKernel(Kernel, LayerNormFwdInterface):
    """LayerNorm with one warp per row, for rows that split into 16-byte vectors, at most
    eight to a lane, but not into 256-element blocks.

    Supports SM80+ architectures.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    preferred_over = frozenset({"layer_norm"})

    # Vectors one lane holds at most. Past eight, the fp32 copy of the row costs a 4096-row
    # call more than the padded block it replaces.
    _MAX_LANE_VECTORS = 8

    # Warps, and so rows, in one CTA.
    _WARP_CANDIDATES = (1, 2, 4, 8)
    _DEFAULT_WARPS = 4

    @classmethod
    def applies(cls, call: LayerNormCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: LayerNormCall) -> Optional[str]:
        vec = VECTOR_ACCESS_BYTES // torch.empty((), dtype=call.dtype).element_size()
        if call.n % ALIGNMENT == 0:
            return f"serves rows that do not split into {ALIGNMENT}-element blocks"
        if call.n % vec:
            return f"reads the row in 16-byte vectors of {vec} elements"
        widest = WARP_LANES * cls._MAX_LANE_VECTORS * vec
        if call.n > widest:
            return f"holds a row of at most {widest} elements in one warp's registers"
        return None

    @classmethod
    def entry_for(cls, call: LayerNormCall) -> Entry:
        identity = (call.n, call.eps, call.dtype)
        return identity, lambda: cls(*identity)

    def __init__(
        self,
        N: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        """Build for a hidden size and dtype.

        The program for a given row count is resolved in ``forward``, memoized by
        ``_layer_norm_warp_row_kernel``.
        """
        super().__init__()
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self._tune_pending = tune  # tuning needs a program, so it waits for the first call
        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        return {"warps": self._DEFAULT_WARPS}

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"warps": warps} for warps in self._WARP_CANDIDATES]

    def forward(
        self, x: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Normalize ``x`` over its trailing ``N`` elements.

        Args:
            x: Input whose trailing axes multiply to ``N``, contiguous, on a CUDA device.
            weight: Affine scale holding ``N`` elements, contiguous, on the same device, or
                ``None`` to scale by one.
            bias: Affine shift holding ``N`` elements, contiguous, on the same device, or
                ``None`` to shift by zero.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        self._require_cuda(x=x, weight=weight, bias=bias)

        original_shape = x.shape
        rows = x.reshape(-1, self.N)
        has_weight, has_bias = weight is not None, bias is not None
        # An absent tensor is a one-element placeholder the program never reads.
        weight = weight.reshape(self.N) if has_weight else rows.new_empty(1)
        bias = bias.reshape(self.N) if has_bias else rows.new_empty(1)

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _layer_norm_warp_row_kernel(
            rows.shape[0], self.N, self.eps, self.dtype_str, has_weight, has_bias
        )
        if self._tune_pending:
            self._tune_pending = False
            self.autotune()

        y = self.kernel(self.config["warps"])(rows, weight, bias)
        return y.reshape(original_shape)
