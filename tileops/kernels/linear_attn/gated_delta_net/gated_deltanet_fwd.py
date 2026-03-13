"""
Gated DeltaNet forward: (q, k, v, g, beta) -> output o.

Pipeline (3 stages):
  1. fused_prepare_compute_w_u: (k, v, g, beta) -> (Aw, Au, w, u)
  2. h_recurrence:  (k, g, w, u, S_0) -> (S, v_new)   [sequential over chunks]
  3. output_o:      (q, k, g, S, v_new) -> o            [parallel over chunks]

Splitting kernel2 into h_recurrence + output_o increases SM utilisation:
  - h_recurrence grid: (batch, head) — must be sequential (state dependency)
  - output_o grid:     (num_chunks, batch, head) — fully parallel
"""
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

from .fused_prepare_compute_w_u import fused_prepare_compute_w_u_tl

__all__ = ["GatedDeltaNetFwdKernel"]

_LOG2E = 1.4426950408889634


# =============================================================================
# Split kernel: h_recurrence  (sequential over chunks, state update only)
# =============================================================================


def _h_recurrence_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_v: int = 0,
):
    """State recurrence: (k, g, w, u, S_0) -> (S, v_new).

    Grid: (num_v_tiles, batch, head) — sequential over chunks, parallel over V tiles.
    Outputs per-chunk boundary states S and intermediate v_new for output_o.

    Args:
        block_v: V-tile size. 0 means no tiling (block_v = dim_v).
    """
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BV = dim_v if block_v <= 0 else block_v
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=128):
        @T.prim_func
        def h_recurrence_kernel(
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S_0: T.Tensor([batch, head, dim_k, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (vid, bid, hid):
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                u_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                k_scaled_s = T.alloc_shared([block_C, dim_k], dtype)

                ws_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                h_next_frag = T.alloc_fragment([dim_k, BV], accum_dtype)


                v_offset = vid * BV

                # Initialise h tile from S_0
                T.copy(S_0[bid, hid, :, v_offset : v_offset + BV], h_c,
                       disable_tma=True)
                for i, j in T.Parallel(dim_k, BV):
                    S[bid, hid, 0, i, v_offset + j] = h_c[i, j]

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    T.copy(k[bid, hid, t * block_C : (t + 1) * block_C, :], k_c,
                           disable_tma=True)
                    T.copy(g[bid, hid, t * block_C : (t + 1) * block_C], g_c,
                           disable_tma=True)
                    T.copy(w[bid, hid, t * block_C : (t + 1) * block_C, :], w_c,
                           disable_tma=True)
                    T.copy(u[bid, hid, t * block_C : (t + 1) * block_C,
                             v_offset : v_offset + BV], u_c, disable_tma=True)

                    # v_new_tile = u_tile - (w @ h_tile) * exp(g + g_last)
                    g_last_val = g_c[block_C - 1]
                    T.clear(ws_frag)
                    T.gemm(w_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, BV):
                        v_new_c[i, j] = u_c[i, j] - ws_frag[i, j] * T.exp2(
                            (g_c[i] + g_last_val) * _LOG2E)

                    # Store v_new tile
                    T.copy(v_new_c,
                           v_new[bid, hid, t * block_C : (t + 1) * block_C,
                                 v_offset : v_offset + BV],
                           disable_tma=True)

                    # h_tile_next = h_tile * exp(g_last) + k_scaled^T @ v_new_tile
                    g_last = g_c[block_C - 1]
                    for n, kk in T.Parallel(block_C, dim_k):
                        k_scaled_s[n, kk] = k_c[n, kk] * T.exp2(
                            (g_last - g_c[n]) * _LOG2E)
                    for i, j in T.Parallel(dim_k, BV):
                        h_next_frag[i, j] = h_c[i, j] * T.exp2(g_last * _LOG2E)
                    T.gemm(k_scaled_s, v_new_c, h_next_frag, transpose_A=True)
                    T.copy(h_next_frag, h_c)
                    for i, j in T.Parallel(dim_k, BV):
                        S[bid, hid, t + 1, i, v_offset + j] = h_c[i, j]

        return h_recurrence_kernel

    return _func


# =============================================================================
# Split kernel: output_o  (fully parallel over chunks)
# =============================================================================

def _output_o_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """Output projection: (q, k, g, S, v_new) -> o.

    Grid: (num_chunks, batch, head) — fully parallel across chunks.
    Each chunk reads h = S[t] (boundary state at start of chunk) and v_new[t].
    """
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=128):
        @T.prim_func
        def output_o_kernel(
            q: T.Tensor([batch, head, seq_len, dim_k], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            o: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(num_chunks, batch, head, threads=threads) as (tid, bid, hid):
                q_c = T.alloc_shared([block_C, dim_k], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                h_c = T.alloc_shared([dim_k, dim_v], dtype)
                v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)

                o_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)

                T.copy(q[bid, hid, tid * block_C : (tid + 1) * block_C, :], q_c,
                       disable_tma=True)
                T.copy(k[bid, hid, tid * block_C : (tid + 1) * block_C, :], k_c,
                       disable_tma=True)
                T.copy(g[bid, hid, tid * block_C : (tid + 1) * block_C], g_c,
                       disable_tma=True)
                T.copy(S[bid, hid, tid, :, :], h_c, disable_tma=True)
                T.copy(v_new[bid, hid, tid * block_C : (tid + 1) * block_C, :], v_new_c,
                       disable_tma=True)

                # o = (q @ h) * exp(g)
                T.clear(o_frag)
                T.gemm(q_c, h_c, o_frag)
                for i, j in T.Parallel(block_C, dim_v):
                    o_frag[i, j] = o_frag[i, j] * T.exp2(g_c[i] * _LOG2E)

                # attn = causal(q @ k^T) * Gamma
                T.clear(attn_frag)
                T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.if_then_else(
                        i >= j,
                        attn_frag[i, j] * T.exp2((g_c[i] - g_c[j]) * _LOG2E),
                        T.float32(0.0))

                # o += attn @ v_new
                T.gemm(attn, v_new_c, o_frag)
                T.copy(o_frag, o[bid, hid, tid * block_C : (tid + 1) * block_C, :],
                       disable_tma=True)

        return output_o_kernel

    return _func


def _chunk_local_cumsum(g: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Compute chunk-local cumulative sum of g along the sequence dimension."""
    B, H, S = g.shape
    return g.reshape(B, H, S // chunk_size, chunk_size).cumsum(-1).reshape(B, H, S)


@torch.library.custom_op("tileops::gated_deltanet_fwd_kernel", mutates_args=())
def _gated_deltanet_fwd_wrapped_kernel(
    batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int,
    dtype: str,
    fused_num_stages: int, fused_threads: int,
    h_num_stages: int, h_threads: int, h_block_v: int,
    o_threads: int,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    g: torch.Tensor, beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g_cum = _chunk_local_cumsum(g.float(), chunk_size).to(g.dtype)
    fused_fn = fused_prepare_compute_w_u_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
    )(fused_num_stages, fused_threads)
    h_fn = _h_recurrence_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
        block_v=h_block_v,
    )(h_num_stages, h_threads)
    o_fn = _output_o_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
    )(o_threads)
    S_0 = torch.zeros(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
    Aw, Au, w, u = fused_fn(k, v, g_cum, beta)
    S_buf, v_new = h_fn(k, g_cum, w, u, S_0)
    o = o_fn(q, k, g_cum, S_buf, v_new)
    return o, S_buf, Aw, Au


@_gated_deltanet_fwd_wrapped_kernel.register_fake
def _gated_deltanet_fwd_wrapped_kernel_fake(
    batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int,
    dtype: str,
    fused_num_stages: int, fused_threads: int,
    h_num_stages: int, h_threads: int, h_block_v: int,
    o_threads: int,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    g: torch.Tensor, beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_chunks = seq_len // chunk_size
    o = torch.empty(batch, head, seq_len, dim_v, dtype=q.dtype, device=q.device)
    S = torch.empty(batch, head, num_chunks + 1, dim_k, dim_v, dtype=q.dtype, device=q.device)
    Aw = torch.empty(batch, head, seq_len, chunk_size, dtype=q.dtype, device=q.device)
    Au = torch.empty_like(Aw)
    return o, S, Aw, Au


class GatedDeltaNetFwdKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        batch: int,
        head: int,
        seq_len: int,
        chunk_size: int,
        dim_k: int,
        dim_v: int,
        dtype: str = "float32",
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.batch = batch
        self.head = head
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.init_config(config, tune)
        # Cache JIT-compiled kernels
        self._fused_fn = fused_prepare_compute_w_u_tl(
            batch, head, seq_len, chunk_size, dim_k, dim_v, self.dtype_str,
        )(self.config["fused_num_stages"], self.config["fused_threads"])
        self._h_recurrence_fn = _h_recurrence_tl(
            batch, head, seq_len, chunk_size, dim_k, dim_v, self.dtype_str,
            block_v=self.config.get("h_block_v", 0),
        )(self.config["h_num_stages"], self.config["h_threads"])
        self._output_o_fn = _output_o_tl(
            batch, head, seq_len, chunk_size, dim_k, dim_v, self.dtype_str,
        )(self.config["o_threads"])
        torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                       "bfloat16": torch.bfloat16}[self.dtype_str]
        self._S_0 = torch.zeros(
            batch, head, dim_k, dim_v, dtype=torch_dtype, device="cuda",
        )

    @property
    def default_config(self) -> dict:
        # bv=32 V-tiling for chunk_size>=64; bv=16 has numerical issues in fp16.
        # Fall back to no tiling for smaller chunks.
        h_block_v = 32 if self.chunk_size >= 64 else 0
        return {
            "fused_num_stages": 2,
            "fused_threads": 256,
            "h_num_stages": 2,
            "h_threads": 256,
            "h_block_v": h_block_v,
            "o_threads": 256,
        }

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Autotune each sub-kernel independently and merge best configs."""
        from tilelang.autotuner import autotune as tl_autotune
        from tilelang.profiler import do_bench as _do_bench

        B, H, S, BC = self.batch, self.head, self.seq_len, self.chunk_size
        DK, DV, dt = self.dim_k, self.dim_v, self.dtype_str

        # --- Tune fused_prepare_compute_w_u ---
        fused_configs = [
            {"num_stages": ns, "threads": t}
            for ns in [1, 2] for t in [128, 256]
        ]
        print(f"Autotuning fused_prepare_compute_w_u ({len(fused_configs)} configs)...")
        fused_jit = fused_prepare_compute_w_u_tl(B, H, S, BC, DK, DV, dt)
        tuned_fused = tl_autotune(configs=fused_configs, warmup=warmup, rep=rep)(fused_jit)()
        fused_best = tuned_fused.config
        print(f"  Best: {fused_best}")

        # --- Tune h_recurrence (sweep block_v separately since it changes kernel shape) ---
        best_h_latency = float("inf")
        best_h_cfg = None
        bv_candidates = [bv for bv in [0, 32] if DV % (bv or DV) == 0]
        for bv in bv_candidates:
            h_configs = [
                {"num_stages": ns, "threads": t}
                for ns in [1, 2] for t in [128, 256]
            ]
            label = f"block_v={bv}" if bv else "block_v=0 (no tile)"
            print(f"Autotuning h_recurrence {label} ({len(h_configs)} configs)...")
            h_jit = _h_recurrence_tl(B, H, S, BC, DK, DV, dt, block_v=bv)
            tuned_h = tl_autotune(configs=h_configs, warmup=warmup, rep=rep)(h_jit)()
            if tuned_h.config is not None:
                lat = _do_bench(tuned_h, warmup=warmup, rep=rep)
                print(f"  Best: {tuned_h.config}, latency={lat:.4f} ms")
                if lat < best_h_latency:
                    best_h_latency = lat
                    best_h_cfg = {**tuned_h.config, "block_v": bv}
        h_best = best_h_cfg or {"num_stages": 2, "threads": 256, "block_v": 32}
        print(f"  Overall best h_recurrence: {h_best}")

        # --- Tune output_o ---
        o_configs = [{"threads": t} for t in [64, 128, 256]]
        print(f"Autotuning output_o ({len(o_configs)} configs)...")
        o_jit = _output_o_tl(B, H, S, BC, DK, DV, dt)
        tuned_o = tl_autotune(configs=o_configs, warmup=warmup, rep=rep)(o_jit)()
        o_best = tuned_o.config
        print(f"  Best: {o_best}")

        self.config = {
            "fused_num_stages": fused_best["num_stages"],
            "fused_threads": fused_best["threads"],
            "h_num_stages": h_best["num_stages"],
            "h_threads": h_best["threads"],
            "h_block_v": h_best.get("block_v", 32),
            "o_threads": o_best["threads"],
        }
        print(f"GatedDeltaNetFwdKernel autotuned config: {self.config}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _gated_deltanet_fwd_wrapped_kernel(
            self.batch, self.head, self.seq_len, self.chunk_size,
            self.dim_k, self.dim_v, self.dtype_str,
            self.config["fused_num_stages"], self.config["fused_threads"],
            self.config["h_num_stages"], self.config["h_threads"],
            self.config.get("h_block_v", 0),
            self.config["o_threads"],
            q, k, v, g, beta,
        )
