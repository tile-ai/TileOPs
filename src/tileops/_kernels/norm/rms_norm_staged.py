"""RMS normalization of a launch that fits one wave, two rows a CTA staged in shared memory.

A CTA copies its two rows into shared memory by cp.async, one commit group a row, and
scales and writes the first row while the second is still in flight, so its stores overlap
its own loads. Without it, a launch whose rows are all resident at once issues most of its
loads before any row is ready to write.
"""

from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.norm.call_spec import LayerNormCall, RMSNormFwdInterface
from tileops._utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["RMSNormStagedKernel"]

_THREADS = 128
_ROWS = 2
# CTAs an SM holds: 64 registers a thread at 128 threads.
_CTAS_PER_SM = 8
# A served launch leaves less than one in this many of the wave's CTA slots free: with more
# free, the one-row bodies of ``RMSNormKernel`` overlap stores and loads as well or better.
_FREE_SLOT_SHARE = 16
# 16-byte vectors a thread copies a row, at least: below it the second row's copy is too
# short to cover the first row's scaling.
_MIN_VECTORS = 3


@target_cache(maxsize=32)
def _rms_norm_staged_kernel(M, N, eps, dtype, has_weight):
    vec = VECTOR_ACCESS_BYTES // getattr(torch, dtype).itemsize
    steps = N // (_THREADS * vec)
    warps = _THREADS // WARP_LANES

    @tilelang.jit(out_idx=[2], compile_flags=csrc_include("streaming_load.h"))
    def build():
        @T.macro
        def normalize(weight, y, tile, sums, piece, scale, total, tx, row, g, pending):
            """Scale row *g* of the tile once all but *pending* copy groups have landed."""
            T.ptx_wait_group(pending)
            total[0] = T.cast(0.0, "float32")
            for k in T.unroll(steps):
                for c in T.vectorized(vec):
                    piece[c] = tile[g, (k * _THREADS + tx) * vec + c]
                for c in T.unroll(vec):
                    total[0] += T.cast(piece[c], "float32") * T.cast(piece[c], "float32")
            for stage in T.unroll(WARP_SHUFFLE_STAGES):
                total[0] += T.shfl_xor(
                    total[0], T.int32(WARP_LANES // 2) >> stage, width=WARP_LANES
                )
            if tx % WARP_LANES == 0:
                sums[g, tx // WARP_LANES] = total[0]
            T.sync_threads()
            total[0] = sums[g, 0]
            for w in T.unroll(1, warps):
                total[0] += sums[g, w]
            total[0] = T.rsqrt(total[0] / float(N) + eps)
            for k in T.unroll(steps):
                at = (k * _THREADS + tx) * vec
                for c in T.vectorized(vec):
                    piece[c] = tile[g, at + c]
                if has_weight:
                    for c in T.vectorized(vec):
                        scale[c] = weight[at + c]
                for c in T.unroll(vec):
                    if has_weight:
                        piece[c] = T.cast(
                            T.cast(piece[c], "float32") * total[0] * T.cast(scale[c], "float32"),
                            dtype,
                        )
                    else:
                        piece[c] = T.cast(T.cast(piece[c], "float32") * total[0], dtype)
                if row < M:
                    for c in T.vectorized(vec):
                        y[row, at + c] = piece[c]

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            weight: T.Tensor[(N if has_weight else 1,), dtype],
            y: T.Tensor[(M, N), dtype],
        ):
            with T.Kernel(T.ceildiv(M, _ROWS), threads=_THREADS) as cta:
                T.annotate_min_blocks_per_sm(_CTAS_PER_SM)
                tx = T.get_thread_binding()
                tile = T.alloc_shared((_ROWS, N), dtype)
                sums = T.alloc_shared((_ROWS, warps), "float32")
                piece = T.alloc_local((vec,), dtype)
                scale = T.alloc_local((vec,), dtype)
                total = T.alloc_local((1,), "float32")
                for g in T.unroll(_ROWS):
                    for k in T.unroll(steps):
                        T.call_extern(
                            "handle",
                            "tileops::cp_async16",
                            T.address_of(tile[g, (k * _THREADS + tx) * vec]),
                            # A row past the last reads the last row and stores nothing.
                            T.address_of(
                                x[T.min(cta * _ROWS + g, M - 1), (k * _THREADS + tx) * vec]
                            ),
                        )
                    T.ptx_commit_group()
                normalize(weight, y, tile, sums, piece, scale, total, tx, cta * _ROWS, 0, 1)
                normalize(weight, y, tile, sums, piece, scale, total, tx, cta * _ROWS + 1, 1, 0)

        return main

    return build


class RMSNormStagedKernel(Kernel, RMSNormFwdInterface):
    """RMS normalization of a launch whose rows, two a CTA, fill one wave.

    Serves launches that leave less than ``1 / _FREE_SLOT_SHARE`` of one wave of two rows a
    CTA free and do not exceed it, of rows of at least ``_MIN_VECTORS`` whole vectors a thread.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    preferred_over = frozenset({"rms_norm"})

    @classmethod
    def refusal(cls, call: LayerNormCall) -> Optional[str]:
        vec = VECTOR_ACCESS_BYTES // call.dtype.itemsize
        wave = _CTAS_PER_SM * call.sm_count
        if call.n % (_THREADS * vec) or call.n // (_THREADS * vec) < _MIN_VECTORS:
            return f"a row of {call.n} elements is not {_MIN_VECTORS} or more whole thread chunks"
        rows = _ROWS * wave
        if not rows - rows // _FREE_SLOT_SHARE < call.rows <= rows:
            return f"{call.rows} rows do not fill one wave of {_ROWS} rows a CTA"
        props = torch.cuda.get_device_properties(call.device)
        # The SM keeps the difference from the per-block opt-in limit for each resident CTA.
        reserved = props.shared_memory_per_multiprocessor - props.shared_memory_per_block_optin
        footprint = _ROWS * (call.n * call.dtype.itemsize + _THREADS // WARP_LANES * 4) + reserved
        if _CTAS_PER_SM * footprint > props.shared_memory_per_multiprocessor:
            return (
                f"{_CTAS_PER_SM} CTAs of {_ROWS} rows of {call.n} do not fit an SM's shared memory"
            )
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: LayerNormCall) -> Entry:
        identity = (call.n, call.eps, call.dtype)
        return identity, lambda: cls(*identity)

    def __init__(self, N: int, eps: float, dtype: torch.dtype, config: Optional[dict] = None):
        """Build for a hidden size and dtype; ``forward`` resolves the program per row count."""
        super().__init__()
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, x: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
        """Normalize ``x`` over its trailing ``N`` elements.

        Args:
            x: Input whose trailing axes multiply to ``N``, contiguous, on a CUDA device.
            weight: Affine scale holding ``N`` elements, contiguous, on the same device,
                or ``None`` to scale by one.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: Either input is not on a device this kernel's ``devices`` names.
        """
        self._require_devices(x=x, weight=weight)
        rows = x.reshape(-1, self.N)
        has_weight = weight is not None
        # The prim_func keeps its weight parameter without one; it reads nothing from it.
        weight = weight.reshape(self.N) if has_weight else rows.new_empty(1)
        program = self.program(self._compile, rows.shape[0], has_weight)
        return program(rows, weight).reshape(x.shape)

    def _compile(self, cfg: dict, m: int, has_weight: bool):
        """The program for *m* rows, with or without a weight."""
        self.kernel = kernel = _rms_norm_staged_kernel(
            m, self.N, self.eps, self.dtype_str, has_weight
        )
        return kernel()
