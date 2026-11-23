"""Run one Mamba-2 SSD decode step and update its float32 state in place.

``dt`` is post-softplus and ``A`` is nonpositive, so the state update uses a
decay. B/C are shared by head group. The caller applies the skip connection,
gate, normalization, and projection after this kernel.
"""

import math
from collections.abc import Callable

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.mamba.call_spec import SSDDecodeCall, SSDDecodeFwdInterface

__all__ = ["SSDDecodeKernel"]


def _row_layout(d_state: int) -> tuple[int, int, int]:
    """How the lanes of a warp share one row of the state: (VEC, LANES, PIECES)."""
    # A lane moves VEC consecutive float32 state elements: 16 bytes when d_state allows it.
    vec = math.gcd(d_state, 4)
    # LANES lanes of one warp share a row: as many as cover d_state, at most the warp.
    lanes = min(32, 1 << math.ceil(math.log2(-(-d_state // vec))))
    # A lane covers a row in PIECES VEC-wide pieces, LANES * VEC elements apart.
    pieces = -(-d_state // (lanes * vec))
    return vec, lanes, pieces


@target_cache(maxsize=32)
def _ssd_decode_kernel(
    batch: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str = "float16",
) -> Callable:
    accum_dtype = "float"

    B = batch
    H = n_heads
    P = d_head
    N = d_state
    G = n_groups
    assert H % G == 0, f"n_heads ({H}) must be divisible by n_groups ({G})"
    HEADS_PER_GROUP = H // G
    VEC, LANES, PIECES = _row_layout(N)

    @tilelang.jit(out_idx=[-1])
    def kernel_func(warps: int, rows_per_thread: int, pieces_per_chunk: int):
        if not 0 < warps <= 32:
            raise ValueError(
                f"warps must be in [1, 32] for a CTA of at most 1024 threads; got {warps}"
            )
        if rows_per_thread <= 0:
            raise ValueError(f"rows_per_thread must be positive; got {rows_per_thread}")
        if pieces_per_chunk <= 0:
            raise ValueError(f"pieces_per_chunk must be positive; got {pieces_per_chunk}")
        # A chunk is `pieces` pieces of each of a thread's rows; registers hold one chunk.
        pieces = min(pieces_per_chunk, PIECES)
        chunks = -(-PIECES // pieces)
        threads = 32 * warps
        slots = threads // LANES
        rows = slots * rows_per_thread
        row_tiles = -(-P // rows)
        held = rows_per_thread * pieces * VEC

        @T.prim_func
        def main(
            A: T.Tensor((H, P, N), accum_dtype),  # type: ignore
            dt: T.Tensor((B, H, P), accum_dtype),  # type: ignore
            x: T.Tensor((B, H, P), dtype),  # type: ignore
            B_in: T.Tensor((B, G, N), dtype),  # type: ignore
            C_in: T.Tensor((B, G, N), dtype),  # type: ignore
            state: T.Tensor((B, H, P, N), accum_dtype),  # type: ignore  in-place
            y_out: T.Tensor((B, H, P), accum_dtype),  # type: ignore
        ):
            # A CTA owns `rows` rows of one (batch, head), and the row tile is the fastest
            # grid index, so neighbouring CTAs stream neighbouring state. Thread `slot` takes
            # rows p0 + slot * rows_per_thread + r; its `lane` takes VEC-wide pieces of each.
            with T.Kernel(row_tiles * B * H, threads=threads) as bid:
                bh = bid // row_tiles
                p0 = bid % row_tiles * rows
                b = bh // H
                h = bh % H
                g = h // HEADS_PER_GROUP
                tx = T.get_thread_binding()
                slot = tx // LANES
                lane = tx % LANES

                s_loc = T.alloc_local((held,), accum_dtype)
                a_loc = T.alloc_local((held,), accum_dtype)
                b_loc = T.alloc_local((pieces * VEC,), dtype)
                c_loc = T.alloc_local((pieces * VEC,), dtype)
                dt_loc = T.alloc_local((rows_per_thread,), accum_dtype)
                dtx = T.alloc_local((rows_per_thread,), accum_dtype)
                y_loc = T.alloc_local((rows_per_thread,), accum_dtype)

                # Each chunk issues all of its pieces' state and A loads before the first use;
                # y, dt and dt * x carry across chunks.
                for c in T.serial(chunks):
                    for j in T.unroll(pieces):
                        n0 = ((c * pieces + j) * LANES + lane) * VEC
                        if n0 < N:
                            for v in T.vectorized(VEC):
                                b_loc[j * VEC + v] = B_in[b, g, n0 + v]
                            for v in T.vectorized(VEC):
                                c_loc[j * VEC + v] = C_in[b, g, n0 + v]
                    for r in T.unroll(rows_per_thread):
                        p = p0 + slot * rows_per_thread + r
                        if c == 0:
                            y_loc[r] = T.float32(0.0)
                        if p < P:
                            if c == 0:
                                dt_loc[r] = dt[b, h, p]
                                dtx[r] = dt_loc[r] * T.cast(x[b, h, p], accum_dtype)
                            for j in T.unroll(pieces):
                                n0 = ((c * pieces + j) * LANES + lane) * VEC
                                if n0 < N:
                                    for v in T.vectorized(VEC):
                                        s_loc[(r * pieces + j) * VEC + v] = state[b, h, p, n0 + v]
                                    for v in T.vectorized(VEC):
                                        a_loc[(r * pieces + j) * VEC + v] = A[h, p, n0 + v]
                    for r in T.unroll(rows_per_thread):
                        p = p0 + slot * rows_per_thread + r
                        if p < P:
                            for j in T.unroll(pieces):
                                n0 = ((c * pieces + j) * LANES + lane) * VEC
                                if n0 < N:
                                    for v in T.unroll(VEC):
                                        i = (r * pieces + j) * VEC + v
                                        b_v = T.cast(b_loc[j * VEC + v], accum_dtype)
                                        c_v = T.cast(c_loc[j * VEC + v], accum_dtype)
                                        s_loc[i] = (
                                            T.exp(dt_loc[r] * a_loc[i]) * s_loc[i] + dtx[r] * b_v
                                        )
                                        y_loc[r] = y_loc[r] + s_loc[i] * c_v
                                    for v in T.vectorized(VEC):
                                        state[b, h, p, n0 + v] = s_loc[(r * pieces + j) * VEC + v]

                # The LANES lanes of a row sum their partial y; lane 0 writes it.
                for r in T.unroll(rows_per_thread):
                    for d in T.unroll(int(math.log2(LANES))):
                        y_loc[r] = y_loc[r] + T.shfl_xor(y_loc[r], LANES >> (d + 1), width=LANES)
                    p = p0 + slot * rows_per_thread + r
                    if lane == 0 and p < P:
                        y_out[b, h, p] = y_loc[r]

        return main

    return kernel_func


class SSDDecodeKernel(Kernel, SSDDecodeFwdInterface):
    """Mamba-2 SSD recurrent decode (step) kernel.

    Updates the SSM state in place and returns the current token's output:

      g                = h // (n_heads // n_groups)
      dA[b, h, p, n]   = exp(dt[b, h, p] * A[h, p, n])
      state[b,h,p,n]  <- dA[b,h,p,n] * state[b,h,p,n]
                         + dt[b,h,p] * B_in[b,g,n] * x[b,h,p]
      y_out[b, h, p]   = sum_n  state[b, h, p, n] * C_in[b, g, n]

    Matches the interface of ``selective_state_update`` in mamba_ssm.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: SSDDecodeCall) -> Entry:
        return call, lambda: cls(
            call.batch, call.n_heads, call.d_head, call.d_state, call.n_groups, call.dtype
        )

    def __init__(
        self,
        batch: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int = 1,
        dtype: torch.dtype = torch.float16,
        config: dict | None = None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.d_state = d_state
        self.kernel = _ssd_decode_kernel(
            batch,
            n_heads,
            d_head,
            d_state,
            n_groups,
            self.dtype_str,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # A thread puts up to 4 pieces of each of its 2 rows in flight before the first use,
        # 32 state and 32 A values at most; four warps a CTA keep it small enough that a
        # single decode step still spreads across every SM.
        return {"warps": 4, "rows_per_thread": 2, "pieces_per_chunk": 4}

    @property
    def autotune_configs(self) -> list[dict]:
        # At most 4 rows and 4 pieces a thread: its registers hold at most 16 * VEC state and
        # as many A values. A chunk never holds more pieces than a lane has in a row, so each
        # pieces_per_chunk is listed once per program it builds.
        row_pieces = _row_layout(self.d_state)[2]
        return [
            {"warps": warps, "rows_per_thread": rows, "pieces_per_chunk": pieces}
            for warps in [1, 2, 4, 8]
            for rows in [1, 2, 4]
            for pieces in sorted({min(p, row_pieces) for p in (1, 2, 4)})
        ]

    def forward(
        self,
        A: torch.Tensor,
        dt: torch.Tensor,
        x: torch.Tensor,
        B_in: torch.Tensor,
        C_in: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.program(self._compile)(A, dt, x, B_in, C_in, state)

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(cfg["warps"], cfg["rows_per_thread"], cfg["pieces_per_chunk"])
