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

    @tilelang.jit(out_idx=[-1])
    def kernel_func(
        block_p: int,
        block_n: int,
        threads: int,
    ):
        if not (block_n > 0 and (block_n & (block_n - 1)) == 0):
            raise ValueError(
                f"block_n must be a power of 2 for the log-depth tree reduction; got {block_n}"
            )
        if threads != block_p * block_n:
            raise ValueError(
                f"threads must be block_p * block_n ({block_p} * {block_n}) for the 2D "
                f"T.Parallel layout; got {threads}"
            )
        _log_n = int(math.log2(block_n))

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
            # Grid: axis-0 fuses (batch, head); axis-1 tiles d_head. Thread (pp, nn) owns
            # p = p0 + pp and n = n0 + nn of each n-chunk, so a pp row reads consecutive n.
            # Each thread keeps a partial y; a tree reduction in shared memory sums them per pp.
            with T.Kernel(B * H, T.ceildiv(P, block_p), threads=threads) as (bh, bp):
                b = bh // H
                h = bh % H
                g = h // HEADS_PER_GROUP
                p0 = bp * block_p

                y_frag = T.alloc_fragment((block_p, block_n), accum_dtype)
                T.clear(y_frag)

                # x and dt depend only on pp, so they load once before the n_blk loop.
                x_frag = T.alloc_fragment((block_p,), accum_dtype)
                dt_frag = T.alloc_fragment((block_p,), accum_dtype)
                for pp in T.Parallel(block_p):
                    p_idx = p0 + pp
                    valid_p = p_idx < P
                    x_frag[pp] = T.if_then_else(
                        valid_p,
                        T.cast(x[b, h, p_idx], accum_dtype),
                        T.float32(0.0),
                    )
                    dt_frag[pp] = T.if_then_else(
                        valid_p,
                        dt[b, h, p_idx],
                        T.float32(0.0),
                    )

                for n_blk in T.serial(T.ceildiv(N, block_n)):
                    n0 = n_blk * block_n
                    for pp, nn in T.Parallel(block_p, block_n):
                        p_idx = p0 + pp
                        n_idx = n0 + nn
                        valid = (p_idx < P) and (n_idx < N)

                        A_val = T.if_then_else(
                            valid,
                            A[h, p_idx, n_idx],
                            T.float32(0.0),
                        )
                        B_val = T.if_then_else(
                            valid,
                            T.cast(B_in[b, g, n_idx], accum_dtype),
                            T.float32(0.0),
                        )
                        C_val = T.if_then_else(
                            valid,
                            T.cast(C_in[b, g, n_idx], accum_dtype),
                            T.float32(0.0),
                        )
                        old_s = T.if_then_else(
                            valid,
                            state[b, h, p_idx, n_idx],
                            T.float32(0.0),
                        )

                        dA_val = T.exp(dt_frag[pp] * A_val)
                        new_s = dA_val * old_s + dt_frag[pp] * x_frag[pp] * B_val
                        if valid:
                            state[b, h, p_idx, n_idx] = new_s

                        y_frag[pp, nn] = y_frag[pp, nn] + new_s * C_val

                # Round r: threads with nn < block_n>>(r+1) add the value at
                # nn + block_n>>(r+1) into their cell.
                y_smem = T.alloc_shared((block_p, block_n), accum_dtype)
                for pp, nn in T.Parallel(block_p, block_n):
                    y_smem[pp, nn] = y_frag[pp, nn]

                T.sync_threads()

                for _d in range(_log_n):
                    _stride = block_n >> (_d + 1)
                    for pp, nn in T.Parallel(block_p, block_n):
                        if nn < _stride:
                            y_smem[pp, nn] = y_smem[pp, nn] + y_smem[pp, nn + _stride]
                    T.sync_threads()

                for pp, nn in T.Parallel(block_p, block_n):
                    if nn == 0:
                        p_idx = p0 + pp
                        if p_idx < P:
                            y_out[b, h, p_idx] = y_smem[pp, 0]

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
        # threads = block_p * block_n; block_n=32 gives one warp per pp-row.
        return {
            "block_p": 4,
            "block_n": 32,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # threads = block_p * block_n, at most 256; block_n is a power of 2.
        return [
            {"block_p": bp, "block_n": bn, "threads": bp * bn}
            for bn in [16, 32, 64, 128]
            for bp in [1, 2, 4, 8]
            if bp * bn <= 256
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
        return self.kernel(cfg["block_p"], cfg["block_n"], cfg["threads"])
