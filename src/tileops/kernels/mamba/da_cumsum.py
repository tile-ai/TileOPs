"""
Mamba-2 dA_cumsum forward kernel.

Inputs:
  dt:       (batch, seq_len, n_heads)                  -- raw per-position dt (float32)
  A:        (n_heads,)                                  -- State Space Model (SSM) decay parameter (float32)
  dt_bias:  (n_heads,)                                  -- optional per-head dt bias (float32)

Outputs:
  dt_out:    (batch, n_heads, num_chunks, chunk_len)   -- dtype (bf16/fp16), processed dt after bias/softplus/clamp
  dA_cumsum: (batch, n_heads, num_chunks, chunk_len)   -- float32, inclusive prefix sum of dA = dt_val * A

For each (b, h, c, l), the kernel computes:

  dt_val            = dt[b, c*Q + l, h]
  if has_dt_bias:   dt_val += dt_bias[h]
  if dt_softplus:   dt_val = softplus(dt_val)   # with bypass for dt_val > 20
                    dt_val = clamp(dt_val, dt_min, dt_max)
  dt_out[b,h,c,l]  = dt_val (cast to dtype for storage efficiency)
  dA_cumsum[b,h,c,l] = sum_{i=0}^{l} dt_val[b,h,c,i] * A[h]  (computed from fp32 dt_val before casting dt_out)

This matches _chunk_cumsum_fwd_kernel in the Mamba-2 Triton reference
(mamba_ssm/ops/triton/ssd_chunk_state.py).

Alignment with Mamba-2 paper:
  In ssd_minimal_discrete, A already absorbs dt (A = dt * A_log), so A_cumsum = cumsum(A).
  Here dt and A are kept separate; dA = dt * A achieves the same result.
  Since A <= 0 in Mamba-2, dA_cumsum is monotonically non-increasing within each chunk,
  and exp(dA_cumsum[l] - dA_cumsum[s]) is a decaying factor in (0, 1] for s <= l.

Notation:
  B = batch, S = seq_len = C * Q, H = n_heads, C = num_chunks, Q = chunk_len
"""

import functools
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["DaCumsumFwdKernel"]

# Shared rows are chunk_len + _ROW_PAD wide. The pad keeps the head-major write off a
# single bank; keeping it a multiple of 4 keeps the chunk-major read-back vectorized.
_ROW_PAD = 4
# Two tiles have to fit the shared memory a block gets without opting in to more.
_MAX_SHARED_BYTES = 48 * 1024


_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}


def _shared_bytes(block_h: int, chunk_len: int) -> int:
    """Bytes a block's shared tiles take, bounding the run sums by a quarter row."""
    return block_h * ((chunk_len + _ROW_PAD) * 4 + chunk_len)


def _scan_groups(block_h: int, threads: int, chunk_len: int) -> int:
    """Threads that share one row's scan, each taking at least four of its positions.

    The row scan is two-level: every group reduces its own run, the runs are scanned
    against each other, and each position adds the run before it. Wider groups shorten
    that middle scan; runs below four positions stop paying for the split.
    """
    groups = 1
    while (
        groups * 2 <= threads // block_h
        and chunk_len % (groups * 2) == 0
        and chunk_len // (groups * 2) >= 4
    ):
        groups *= 2
    return groups


def _head_tile(n_heads: int, chunk_len: int) -> int:
    """Widest power-of-two head tile the heads and the shared budget allow, at most eight.

    Eight is where a warp's head-major write still lands one element per bank: a row
    stride of ``chunk_len + 4`` puts head ``h`` and position ``p`` on bank
    ``(4h + p) % 32``, which a ninth head starts repeating.
    """
    tile = 8
    while tile > 1 and (tile > n_heads or _shared_bytes(tile, chunk_len) > _MAX_SHARED_BYTES):
        tile //= 2
    return tile


def _thread_count(block_h: int, chunk_len: int) -> int:
    """Largest power of two up to 512 that every thread still has a tile element for."""
    threads = 32
    while threads * 2 <= min(512, block_h * chunk_len):
        threads *= 2
    return threads


@functools.lru_cache(maxsize=32)
def _da_cumsum_fwd_kernel(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    seq_len: int,
    dtype: str,
    dt_softplus: bool = False,
    has_dt_bias: bool = False,
    dt_min: float = 0.0,
    dt_max: float = float("inf"),
) -> Callable:
    """Build the chunk-local dA cumsum kernel: one CTA per (batch, chunk, head tile).

    ``dt`` is contiguous in H while both outputs are contiguous in chunk_len, so the
    tile is loaded with H on the fastest thread axis, transposed through shared memory,
    and written back with chunk_len on it. A tile that instead scans ``dt`` in place
    leaves one warp reading H floats apart, which costs eight sectors per useful one.

    The row scan is two-level. ``T.cumsum`` alone walks a row as one warp's worth of
    32-position segments chained by a carry, which on a 256-position row is 8 serial
    segments while the block's other warps idle. Here each group reduces its own run
    in registers, ``T.cumsum`` scans the per-run totals, and the write-out adds the
    run before it -- the same arithmetic against a far shorter dependence chain.

    One tile carries the row through both halves: the transpose fills it with the
    processed dt, and the group that holds a run stores that run to ``dt_out`` -- a
    run is contiguous there -- before scaling it by A and scanning it in place. A
    second tile for the dA values would be written and read once more for nothing.

    ``block_h`` (heads per CTA) and ``threads`` are supplied by the returned
    ``kernel_func``.
    """
    accum_dtype = "float"

    B = batch
    C = num_chunks
    Q = chunk_len
    H = n_heads
    S = seq_len
    row_stride = Q + _ROW_PAD

    @tilelang.jit(out_idx=[-2, -1])
    def kernel_func(block_h: int, threads: int):
        groups = _scan_groups(block_h, threads, Q)
        span = Q // groups
        # A run reaches dt_out in one store when it fits a 16-byte transaction.
        store_loop = T.vectorized if span * _DTYPE_BYTES[dtype] <= 16 else T.serial

        @T.prim_func
        def da_cumsum_fwd_main(
            dt: T.Tensor((B, S, H), accum_dtype),  # type: ignore
            A: T.Tensor((H,), accum_dtype),  # type: ignore
            dt_bias: T.Tensor((H,), accum_dtype),  # type: ignore
            dt_out: T.Tensor((B, H, C, Q), dtype),  # type: ignore
            dA_cumsum: T.Tensor((B, H, C, Q), accum_dtype),  # type: ignore
        ):
            with T.Kernel(B * C, T.ceildiv(H, block_h), threads=threads) as (bc, bh_tile):
                row_shared = T.alloc_shared((block_h, row_stride), accum_dtype)
                run_sum = T.alloc_shared((block_h, groups), accum_dtype)
                b = bc // C
                c = bc % C

                for pos, head in T.Parallel(Q, block_h):
                    bh = bh_tile * block_h + head
                    in_b = bh < H
                    safe_bh = T.min(bh, H - 1)

                    val = T.alloc_var(accum_dtype)
                    val = T.if_then_else(in_b, dt[b, c * Q + pos, safe_bh], T.float32(0.0))
                    if has_dt_bias:
                        val = val + T.if_then_else(in_b, dt_bias[safe_bh], T.float32(0.0))
                    if dt_softplus:
                        val = T.if_then_else(
                            val <= T.float32(20.0),
                            T.log(T.float32(1.0) + T.exp(val)),
                            val,
                        )
                    val = T.min(T.max(val, T.float32(dt_min)), T.float32(dt_max))
                    val = T.if_then_else(in_b, val, T.float32(0.0))

                    row_shared[head, pos] = val

                T.sync_threads()

                for head, g in T.Parallel(block_h, groups):
                    bh = bh_tile * block_h + head
                    run = T.alloc_local((span,), accum_dtype)
                    for j in T.serial(span):
                        run[j] = row_shared[head, g * span + j]
                    with T.If(bh < H), T.Then():
                        for j in store_loop(span):
                            dt_out[b, bh, c, g * span + j] = T.cast(run[j], dtype)
                    scale = T.if_then_else(bh < H, A[T.min(bh, H - 1)], T.float32(0.0))
                    for j in T.serial(span):
                        run[j] = run[j] * scale
                    if span > 1:
                        for j in T.serial(span - 1):
                            run[j + 1] = run[j + 1] + run[j]
                    for j in T.serial(span):
                        row_shared[head, g * span + j] = run[j]
                    run_sum[head, g] = run[span - 1]

                T.sync_threads()
                T.cumsum(run_sum, dim=1)
                T.sync_threads()

                for head, pos in T.Parallel(block_h, Q):
                    bh = bh_tile * block_h + head
                    with T.If(bh < H), T.Then():
                        carry = T.if_then_else(
                            pos >= span,
                            run_sum[head, T.max(pos // span - 1, 0)],
                            T.float32(0.0),
                        )
                        dA_cumsum[b, bh, c, pos] = row_shared[head, pos] + carry

        return da_cumsum_fwd_main

    return kernel_func


def _(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    seq_len: int,
    dtype: str,
    threads: int,
    dt_softplus: bool,
    has_dt_bias: bool,
    dt_min: float,
    dt_max: float,
    block_h: int,
    dt: torch.Tensor,
    A: torch.Tensor,
    dt_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.float16)
    dt_out = dt.new_empty((batch, n_heads, num_chunks, chunk_len), dtype=torch_dtype)
    dA_cumsum = dt.new_empty((batch, n_heads, num_chunks, chunk_len), dtype=torch.float32)
    return dt_out, dA_cumsum


class DaCumsumFwdKernel(Kernel):
    """Mamba-2 dA_cumsum forward kernel.

    Applies optional per-head bias, optional softplus activation, and clamping to
    raw dt values, then computes the chunk-local inclusive prefix sum of dA = dt * A.

    One block owns block_h heads of one (batch, chunk) tile. The tile is read with
    the head axis on the fastest thread axis, so the read of ``dt`` is coalesced on
    its own contiguous axis, transposed through shared memory, scanned with T.cumsum
    along chunk_len and written back coalesced on the outputs' contiguous axis.

    Inputs:
        dt      (batch, seq_len, n_heads) float32 — raw dt values.
        A       (n_heads,) float32 — State Space Model (SSM) decay parameters.
        dt_bias (n_heads,) float32 — per-head dt bias; required when has_dt_bias=True.

    Outputs:
        dt_out    (batch, n_heads, num_chunks, chunk_len) dtype — processed dt in target dtype.
        dA_cumsum (batch, n_heads, num_chunks, chunk_len) float32 — inclusive prefix sum.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    # This backend's own capability, which may be narrower than the manifest
    # union the op enforces.
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        seq_len: int,
        dtype: torch.dtype = torch.float32,
        dt_softplus: bool = False,
        has_dt_bias: bool = False,
        dt_min: float = 0.0,
        dt_max: float = float("inf"),
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.dt_softplus = dt_softplus
        self.has_dt_bias = has_dt_bias
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dtype = dtype
        self.kernel = _da_cumsum_fwd_kernel(
            batch,
            num_chunks,
            chunk_len,
            n_heads,
            seq_len,
            self.dtype_str,
            dt_softplus,
            has_dt_bias,
            dt_min,
            dt_max,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        block_h = _head_tile(self.n_heads, self.chunk_len)
        return {"block_h": block_h, "threads": _thread_count(block_h, self.chunk_len)}

    @property
    def autotune_configs(self) -> list[dict]:
        # The default leads, so a shape whose tile admits no swept pair still has one.
        # Sweep block_h ∈ {1, 2, 4, 8, 16} subject to:
        #   - block_h <= n_heads             (no more tile rows than heads)
        #   - shared tiles within _MAX_SHARED_BYTES
        # and, per block_h, every thread count the tile has an element for.
        valid = [self.default_config]
        for bh in [1, 2, 4, 8, 16]:
            if bh > self.n_heads:
                break
            if _shared_bytes(bh, self.chunk_len) > _MAX_SHARED_BYTES:
                break
            for threads in [128, 256, 512, 1024]:
                if threads > bh * self.chunk_len:
                    break
                candidate = {"block_h": bh, "threads": threads}
                if candidate not in valid:
                    valid.append(candidate)
        return valid

    def forward(
        self,
        dt: torch.Tensor,
        A: torch.Tensor,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the dA_cumsum forward pass.

        Args:
            dt: (batch, seq_len, n_heads) float32 — raw dt values.
            A:  (n_heads,) float32 — SSM decay parameters.
            dt_bias: (n_heads,) float32, optional — per-head dt bias.
                Required when the kernel was constructed with has_dt_bias=True.

        Returns:
            dt_out: (batch, n_heads, num_chunks, chunk_len) dtype — processed dt in target dtype.
            dA_cumsum: (batch, n_heads, num_chunks, chunk_len) float32 — inclusive prefix sum.
        """
        dt = dt.contiguous()
        A = A.contiguous()
        if self.has_dt_bias and dt_bias is None:
            raise ValueError("dt_bias is required when has_dt_bias=True")
        # The no-bias specialization does not read dt_bias. Reuse A as the
        # ABI placeholder instead of allocating/filling a dummy CUDA tensor.
        dt_bias = A if dt_bias is None else dt_bias.contiguous()

        return self.kernel(self.config["block_h"], self.config["threads"])(dt, A, dt_bias)
