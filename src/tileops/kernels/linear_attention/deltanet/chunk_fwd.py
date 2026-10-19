"""
DeltaNet forward: (q, k, v, beta) -> output o.

Pipeline (3 stages):
  1. fused_prepare_compute_w_u: (k, v, beta) -> (Aw, Au, w, u)
  2. h_recurrence:  (k, w, u, S_0) -> (S, v_new)   [sequential over chunks]
  3. output_o:      (q, k, S, v_new) -> o            [parallel over chunks]

Where dim_k is below twice the chunk length, stage 1 also writes -(k^T w) and k^T u, stage 2 is
the state walk (-(k^T w), k^T u) -> S, one matmul a chunk, and stage 3 forms v_new = u - w h
from (q, k, w, u, S).

Unlike Gated DeltaNet (GDN), there is no gate parameter g:
  - No exp(g) scaling in any stage
  - State update: h_new = h + k^T @ v_new (no decay)
  - v_new = u - w @ h (no exp scaling)
  - Output: o = q @ h + causal_attn @ v_new (no Gamma weighting)
"""

import functools
import itertools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    DeltaNetChunkCall,
    DeltaNetChunkFwdInterface,
    head_count_refusal,
)
from tileops.kernels.linear_attention.deltanet.autotune import (
    OUTPUT_CONFIGS,
    PIPELINE_CONFIGS,
    default_h_block_v,
    default_h_threads,
    delta_rule_fwd_autotune_configs,
    h_block_v_candidates,
    tune_delta_rule_fwd,
)
from tileops.kernels.linear_attention.deltanet.chunk_bwd import (
    walk_block_v_options,
    walk_threads,
)
from tileops.kernels.linear_attention.deltanet.fused_prepare_compute_w_u import (
    fused_prepare_compute_w_u_tl,
)
from tileops.kernels.linear_attention.v_tile import min_gemm_n, resolve_block_v
from tileops.utils import get_shared_memory_optin, get_sm_count

__all__ = ["DeltaNetChunkFwdKernel"]


# Split kernel: h_recurrence  (sequential over chunks, state update only)


@functools.lru_cache(maxsize=32)
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
    """State recurrence: (k, w, u, S_0) -> (S, v_new).

    Grid: (num_v_tiles, batch, head) -- sequential over chunks, parallel over V tiles.
    Outputs per-chunk boundary states S and intermediate v_new for output_o.

    Args:
        block_v: V-tile size. 0 means no tiling (block_v = dim_v).
    """
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BV = resolve_block_v(dim_v, block_v)
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=128):
        if min_gemm_n(threads) > BV:
            raise ValueError(
                f"V-tile width {BV} (dim_v={dim_v}, block_v={block_v}) is below the "
                f"minimum T.gemm N extent ({min_gemm_n(threads)}) at {threads} threads"
            )

        @T.prim_func
        def h_recurrence_kernel(
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S_0: T.Tensor([batch, head, dim_k, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], accum_dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (vid, bid, hid):
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                u_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)

                ws_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                # Precise fp32 accumulator for running state (avoids
                # compounding quantization error over many chunks).
                h_fp32 = T.alloc_fragment([dim_k, BV], accum_dtype)

                v_offset = vid * BV

                # Initialise h tile from S_0 and promote to fp32
                T.copy(S_0[bid, hid, :, v_offset : v_offset + BV], h_c, disable_tma=True)
                for i, j in T.Parallel(dim_k, BV):
                    h_fp32[i, j] = T.cast(h_c[i, j], accum_dtype)
                for i, j in T.Parallel(dim_k, BV):
                    S[bid, hid, 0, i, v_offset + j] = h_fp32[i, j]

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    T.copy(k[bid, hid, t * block_C : (t + 1) * block_C, :], k_c, disable_tma=True)
                    T.copy(w[bid, hid, t * block_C : (t + 1) * block_C, :], w_c, disable_tma=True)
                    T.copy(
                        u[bid, hid, t * block_C : (t + 1) * block_C, v_offset : v_offset + BV],
                        u_c,
                        disable_tma=True,
                    )

                    # Cast precise state to dtype for gemm input
                    T.copy(h_fp32, h_c)

                    # v_new_tile = u_tile - w @ h_tile (no exp scaling)
                    T.clear(ws_frag)
                    T.gemm(w_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, BV):
                        v_new_c[i, j] = u_c[i, j] - ws_frag[i, j]

                    # Store v_new tile
                    T.copy(
                        v_new_c,
                        v_new[bid, hid, t * block_C : (t + 1) * block_C, v_offset : v_offset + BV],
                        disable_tma=True,
                    )

                    # h_tile_next = h_tile + k^T @ v_new_tile (fp32 accumulation)
                    T.gemm(k_c, v_new_c, h_fp32, transpose_A=True)
                    for i, j in T.Parallel(dim_k, BV):
                        S[bid, hid, t + 1, i, v_offset + j] = h_fp32[i, j]

        return h_recurrence_kernel

    return _func


# Split kernel: state_walk  (sequential over chunks, one matmul a chunk)


@functools.lru_cache(maxsize=32)
def _state_walk_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """Sequential walk of the state over chunks: (neg_kw, ku) -> S.

    Grid: (dim_v / block_v, batch, head). Each CTA carries one ``[dim_k, block_v]`` column tile
    of the state from the first chunk to the last. For ungated DeltaNet

        h_{c+1} = h_c + K_c^T U_c - (K_c^T W_c) h_c,

    because h_{c+1} = h_c + K_c^T v_new and v_new = U_c - W_c h_c. The column tiles are
    independent, and the w/u preparation writes ``neg_kw`` = -(K_c^T W_c) and ``ku`` = K_c^T U_c,
    so the chain is one matmul a chunk.

    Output: S[c], the FP32 state chunk c starts from, and S[num_chunks], the final state.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_v, num_stages, threads=128):
        if block_v < min_gemm_n(threads):
            raise ValueError(
                f"block_v ({block_v}) is below the minimum T.gemm N extent "
                f"({min_gemm_n(threads)}) at {threads} threads"
            )

        @T.prim_func
        def state_walk_kernel(
            neg_kw: T.Tensor([batch, head, num_chunks, dim_k, dim_k], dtype),
            ku: T.Tensor([batch, head, num_chunks, dim_k, dim_v], accum_dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], accum_dtype),
        ):
            with T.Kernel(dim_v // block_v, batch, head, threads=threads) as (vid, bid, hid):
                op_s = T.alloc_shared([dim_k, dim_k], dtype)
                local_s = T.alloc_shared([dim_k, block_v], accum_dtype)
                # Held transposed, so the matmul reads it K-major.
                h_s = T.alloc_shared([block_v, dim_k], dtype)
                h_frag = T.alloc_fragment([dim_k, block_v], accum_dtype)
                v0 = vid * block_v

                T.clear(h_frag)
                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    T.copy(neg_kw[bid, hid, t, :, :], op_s, disable_tma=True)
                    T.copy(ku[bid, hid, t, :, v0 : v0 + block_v], local_s, disable_tma=True)
                    for i, j in T.Parallel(dim_k, block_v):
                        h_s[j, i] = h_frag[i, j]
                    T.copy(h_frag, S[bid, hid, t, :, v0 : v0 + block_v], disable_tma=True)
                    for i, j in T.Parallel(dim_k, block_v):
                        h_frag[i, j] = h_frag[i, j] + local_s[i, j]
                    T.gemm(op_s, h_s, h_frag, transpose_B=True)
                T.copy(h_frag, S[bid, hid, num_chunks, :, v0 : v0 + block_v], disable_tma=True)

        return state_walk_kernel

    return _func


# Split kernel: output_o  (fully parallel over chunks)


@functools.lru_cache(maxsize=32)
def _output_o_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    late_loads: bool = False,
    from_walk: bool = False,
):
    """Output projection: (q, k, S, v_new) -> o.

    Grid: (num_chunks, batch, head) -- fully parallel across chunks.
    Each chunk reads h = S[t] (boundary state at start of chunk) and v_new[t].

    *from_walk* takes (q, k, w, u, S) instead and forms v_new = u - w h itself, since the state
    walk writes no v_new.

    *late_loads* loads k once h is spent and v_new once q and k are, so each product holds
    only its own operands; it costs the loads their overlap, so only a chunk that does not
    fit otherwise takes it. Under *from_walk* only k loads late.
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
        @T.macro
        def _output_body(q, k, w, u, S, v_new, o):
            with T.Kernel(num_chunks, batch, head, threads=threads) as (tid, bid, hid):
                q_c = T.alloc_shared([block_C, dim_k], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                h_c = T.alloc_shared([dim_k, dim_v], dtype)
                v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)

                o_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)

                r0 = tid * block_C
                T.copy(q[bid, hid, r0 : r0 + block_C, :], q_c, disable_tma=True)
                if not late_loads:
                    T.copy(k[bid, hid, r0 : r0 + block_C, :], k_c, disable_tma=True)
                T.copy(S[bid, hid, tid, :, :], h_c, disable_tma=True)
                if from_walk:
                    w_c = T.alloc_shared([block_C, dim_k], dtype)
                    ws_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                    T.copy(w[bid, hid, r0 : r0 + block_C, :], w_c, disable_tma=True)
                    T.copy(u[bid, hid, r0 : r0 + block_C, :], v_new_c, disable_tma=True)
                elif not late_loads:
                    T.copy(v_new[bid, hid, r0 : r0 + block_C, :], v_new_c, disable_tma=True)

                # o = q @ h (no exp(g) scaling)
                T.clear(o_frag)
                T.gemm(q_c, h_c, o_frag)

                if from_walk:
                    # v_new = u - w @ h
                    T.clear(ws_frag)
                    T.gemm(w_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, dim_v):
                        v_new_c[i, j] = v_new_c[i, j] - ws_frag[i, j]

                if late_loads:
                    T.copy(k[bid, hid, r0 : r0 + block_C, :], k_c, disable_tma=True)

                # attn = causal(q @ k^T) (no Gamma weighting)
                T.clear(attn_frag)
                T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.if_then_else(i >= j, attn_frag[i, j], T.float32(0.0))

                if late_loads and not from_walk:
                    T.copy(v_new[bid, hid, r0 : r0 + block_C, :], v_new_c, disable_tma=True)

                # o += attn @ v_new
                T.gemm(attn, v_new_c, o_frag)
                T.copy(o_frag, o[bid, hid, r0 : r0 + block_C, :], disable_tma=True)

        if from_walk:

            @T.prim_func
            def output_o_kernel(
                q: T.Tensor([batch, head, seq_len, dim_k], dtype),
                k: T.Tensor([batch, head, seq_len, dim_k], dtype),
                w: T.Tensor([batch, head, seq_len, dim_k], dtype),
                u: T.Tensor([batch, head, seq_len, dim_v], dtype),
                S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], accum_dtype),
                o: T.Tensor([batch, head, seq_len, dim_v], dtype),
            ):
                _output_body(q, k, w, u, S, None, o)

            return output_o_kernel

        @T.prim_func
        def output_o_kernel(
            q: T.Tensor([batch, head, seq_len, dim_k], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], accum_dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            o: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            _output_body(q, k, None, None, S, v_new, o)

        return output_o_kernel

    return _func


class DeltaNetChunkFwdKernel(Kernel, DeltaNetChunkFwdInterface):
    """DeltaNet chunked forward: the w/u preparation, a serial state walk, the output pass.

    Where dim_k is below twice the chunk length, the state pass is the one-matmul walk
    h <- h + k^T u - (k^T w) h over operands the preparation writes; otherwise it is the
    recurrence, and the preparation does not write those operands.
    """

    @staticmethod
    def _walks(chunk_size: int, dim_k: int) -> bool:
        """Whether a call runs the one-matmul state walk rather than the recurrence."""
        return dim_k < 2 * chunk_size

    @classmethod
    def refusal(cls, call: DeltaNetChunkCall) -> Optional[str]:
        """Why no program serves this call, or ``None``.

        Reads lower bounds, the state pass's over its narrowest V tile, so a call above them
        that TileLang still cannot place in the device's shared memory is built and fails at
        launch.
        """
        reason = head_count_refusal(call.heads)
        if reason is not None or not call.smem_budget:
            return reason
        c, k, v, elem = call.chunk_size, call.dim_k, call.dim_v, call.dtype.itemsize
        walk = cls._walks(c, k)
        if walk:
            state = cls._walk_shared_bytes(k, walk_block_v_options(v)[0], elem, 1)
        else:
            width = min((resolve_block_v(v, b) for b in h_block_v_candidates(v)), default=v)
            state = cls._state_live_bytes(c, k, width, elem)
        need = max(
            cls._fused_live_bytes(c, k, v, elem, walk),
            state,
            cls._output_live_bytes(c, k, v, elem, walk),
        )
        if need <= call.smem_budget:
            return None
        return (
            f"needs at least {need} bytes of shared memory per block at chunk {c}, head dims "
            f"{k} / {v} in {call.dtype}; the device gives {call.smem_budget}"
        )

    @staticmethod
    def _fused_live_bytes(c: int, k: int, v: int, elem: int, walk: bool) -> int:
        """Lower bound on the w/u preparation's shared memory: k, beta, S and P through the
        inverse, k_beta with k, beta and S, then S, beta, v and v_beta; for the walk's operands
        k stays to the end, beside S, beta and v scaled in place."""
        tail = c * k + c * c + c + c * v if walk else c * c + c + 2 * c * v
        return max(c * k + c + 2 * c * c, 2 * c * k + c + c * c, tail) * elem

    @staticmethod
    def _state_live_bytes(c: int, k: int, width: int, elem: int) -> int:
        """Lower bound on the recurrence's shared memory at one stage over a V tile: k, w, the
        tile of u and the cast state at w @ h."""
        return (2 * c * k + (c + k) * width) * elem

    @staticmethod
    def _state_shared_bytes(c: int, k: int, width: int, elem: int, stages: int) -> int:
        """Shared memory of the recurrence over a V tile: k, w and the tile of u once per
        stage, then the cast state and v_new."""
        return (stages * (2 * c * k + c * width) + (k + c) * width) * elem

    @staticmethod
    def _walk_shared_bytes(k: int, width: int, elem: int, stages: int) -> int:
        """Shared memory of the state walk over a V tile: the chunk operator and the FP32 k^T u
        tile per stage, then the state tile the matmul reads."""
        return stages * (k * k * elem + k * width * 4) + k * width * elem

    @staticmethod
    def _output_live_bytes(c: int, k: int, v: int, elem: int, walk: bool) -> int:
        """Lower bound on the output pass's shared memory: q with the state, q with k, then
        attn with v_new; fed by the walk, w with q, the state and v_new first."""
        if walk:
            return max(2 * c * k + k * v + c * v, 2 * c * k + c * v + c * c) * elem
        return max(c * k + k * v, 2 * c * k, c * c + c * v) * elem

    @staticmethod
    def _late_loads_for(
        budget: int, c: int, k: int, v: int, elem: int, walk: bool
    ) -> Tuple[bool, bool]:
        """Whether the w/u preparation and the output pass load late: only where their buffers,
        every one loaded up front, exceed *budget*."""
        fused = (2 * c * k + 2 * c * v + 2 * c * c + c) * elem
        output = ((3 if walk else 2) * c * k + k * v + c * v + c * c) * elem
        return fused > budget, output > budget

    @staticmethod
    def _deltanet_fwd_run(
        batch: int,
        head: int,
        seq_len: int,
        chunk_size: int,
        dim_k: int,
        dim_v: int,
        dtype: str,
        walk: bool,
        fused_num_stages: int,
        fused_threads: int,
        h_num_stages: int,
        h_threads: int,
        h_block_v: int,
        o_threads: int,
        fused_late_v: bool,
        o_late_loads: bool,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        args = (batch, head, seq_len, chunk_size, dim_k, dim_v, dtype)
        fused_fn = fused_prepare_compute_w_u_tl(*args, late_v=fused_late_v, walk_operands=walk)(
            fused_num_stages, fused_threads
        )
        o_fn = _output_o_tl(*args, late_loads=o_late_loads, from_walk=walk)(o_threads)
        if walk:
            Aw, Au, w, u, neg_kw, ku = fused_fn(k, v, beta)
            S_buf = _state_walk_tl(*args)(h_block_v, h_num_stages, h_threads)(neg_kw, ku)
            return o_fn(q, k, w, u, S_buf), S_buf, Aw, Au, w, u
        h_fn = _h_recurrence_tl(*args, block_v=h_block_v)(h_num_stages, h_threads)
        S_0 = torch.zeros(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
        Aw, Au, w, u = fused_fn(k, v, beta)
        S_buf, v_new = h_fn(k, w, u, S_0)
        return o_fn(q, k, S_buf, v_new), S_buf, Aw, Au, w, u

    supported_archs: list[int] = [80, 89, 90]

    @classmethod
    def entry_for(cls, call: DeltaNetChunkCall) -> Entry:
        """The device index is in the identity because the kernel compiles for the
        architecture it is built on."""
        arguments = (
            call.batch,
            call.heads,
            call.seq_len,
            call.chunk_size,
            call.dim_k,
            call.dim_v,
            cls.dtype_to_str(call.dtype),
        )
        index = call.device.index if call.device is not None else None
        return (*arguments, index), lambda: cls(*arguments)

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
    ):
        super().__init__()
        self.batch = batch
        self.head = head
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self._walk = self._walks(chunk_size, dim_k)
        self._fused_late_v, self._o_late_loads = self._late_loads_for(
            get_shared_memory_optin(self.device_index),
            chunk_size,
            dim_k,
            dim_v,
            getattr(torch, self.dtype_str).itemsize,
            self._walk,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        budget = get_shared_memory_optin(self.device_index)
        elem = getattr(torch, self.dtype_str).itemsize
        if self._walk:
            return self._walk_default_config_for(
                budget,
                get_sm_count(self.device_index),
                self.batch * self.head,
                self.dim_k,
                self.dim_v,
                elem,
            )
        return self._default_config_for(budget, self.chunk_size, self.dim_k, self.dim_v, elem)

    @classmethod
    def _default_config_for(
        cls, budget: int, chunk_size: int, dim_k: int, dim_v: int, elem: int
    ) -> dict:
        """The recurrence's config at *budget* bytes of shared memory per block."""
        c, k, v = chunk_size, dim_k, dim_v
        h_block_v = default_h_block_v(v, c)
        if cls._state_shared_bytes(c, k, resolve_block_v(v, h_block_v), elem, 1) > budget:
            h_block_v = min((b for b in h_block_v_candidates(v) if b), default=h_block_v)
        width = resolve_block_v(v, h_block_v)
        h_num_stages = 2 if cls._state_shared_bytes(c, k, width, elem, 2) <= budget else 1
        return {
            "fused_num_stages": 2,
            "fused_threads": 256,
            "h_num_stages": h_num_stages,
            "h_threads": default_h_threads(v, h_block_v),
            "h_block_v": h_block_v,
            "o_threads": 256,
        }

    @classmethod
    def _walk_default_config_for(
        cls, budget: int, sm_count: int, heads: int, dim_k: int, dim_v: int, elem: int
    ) -> dict:
        """The state walk's config at *budget* bytes of shared memory per block.

        The narrowest value tile whose CTAs fit one wave, among widths whose one stage fits
        *budget*; ``refusal`` vets the narrowest.
        """
        widths = [
            b
            for b in walk_block_v_options(dim_v)
            if cls._walk_shared_bytes(dim_k, b, elem, 1) <= budget
        ] or walk_block_v_options(dim_v)[:1]
        block_v = next((b for b in widths if heads * (dim_v // b) <= sm_count), widths[-1])
        two = cls._walk_shared_bytes(dim_k, block_v, elem, 2) <= budget
        return {
            "fused_num_stages": 2,
            "fused_threads": 256,
            "h_num_stages": 2 if two else 1,
            "h_threads": walk_threads(block_v),
            "h_block_v": block_v,
            "o_threads": 256,
        }

    def _walk_configs(self) -> list[dict]:
        """The state walk's candidates: every width, at one stage and, where they fit, two."""
        budget = get_shared_memory_optin(self.device_index)
        elem = getattr(torch, self.dtype_str).itemsize
        return [
            {"block_v": b, "num_stages": stages, "threads": walk_threads(b)}
            for b in walk_block_v_options(self.dim_v)
            for stages in (1, 2)
            if stages == 1 or self._walk_shared_bytes(self.dim_k, b, elem, 2) <= budget
        ]

    @property
    def autotune_configs(self) -> list[dict]:
        if not self._walk:
            return delta_rule_fwd_autotune_configs(self.dim_v)
        return [
            {
                "fused_num_stages": fused["num_stages"],
                "fused_threads": fused["threads"],
                "h_num_stages": walk["num_stages"],
                "h_threads": walk["threads"],
                "h_block_v": walk["block_v"],
                "o_threads": output["threads"],
            }
            for fused, walk, output in itertools.product(
                PIPELINE_CONFIGS, self._walk_configs(), OUTPUT_CONFIGS
            )
        ]

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Tune the three sub-kernels independently and merge the winners."""
        fused_builder = functools.partial(
            fused_prepare_compute_w_u_tl, late_v=self._fused_late_v, walk_operands=self._walk
        )
        o_builder = functools.partial(
            _output_o_tl, late_loads=self._o_late_loads, from_walk=self._walk
        )
        if not self._walk:
            self.config = tune_delta_rule_fwd(
                self,
                fused_builder=fused_builder,
                h_builder=_h_recurrence_tl,
                o_builder=o_builder,
                warmup=warmup,
                rep=rep,
            )
            return
        from tilelang.autotuner import autotune as tl_autotune

        shape = (
            self.batch,
            self.head,
            self.seq_len,
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            self.dtype_str,
        )
        default = self.default_config

        def tune(jit, configs, untuned):
            configs = self.tuning_candidates(configs, untuned)
            at = dict(configs=configs, warmup=warmup, rep=rep)
            dns = list(self._autotune_initial_kwargs(jit, configs[0]).keys())
            if dns:
                at["do_not_specialize"] = dns
            return self._call_autotuned_kernel(tl_autotune(**at)(jit), jit, configs[0]).config

        fused = tune(
            fused_builder(*shape),
            [dict(c) for c in PIPELINE_CONFIGS],
            {"num_stages": default["fused_num_stages"], "threads": default["fused_threads"]},
        )
        walk = tune(
            _state_walk_tl(*shape),
            self._walk_configs(),
            {
                "block_v": default["h_block_v"],
                "num_stages": default["h_num_stages"],
                "threads": default["h_threads"],
            },
        )
        output = tune(
            o_builder(*shape), [dict(c) for c in OUTPUT_CONFIGS], {"threads": default["o_threads"]}
        )
        self.config = {
            "fused_num_stages": fused["num_stages"],
            "fused_threads": fused["threads"],
            "h_num_stages": walk["num_stages"],
            "h_threads": walk["threads"],
            "h_block_v": walk["block_v"],
            "o_threads": output["threads"],
        }
        print(f"{self.__class__.__name__} autotuned config: {self.config}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._deltanet_fwd_run(
            self.batch,
            self.head,
            self.seq_len,
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            self.dtype_str,
            self._walk,
            self.config["fused_num_stages"],
            self.config["fused_threads"],
            self.config["h_num_stages"],
            self.config["h_threads"],
            self.config.get("h_block_v", 0),
            self.config["o_threads"],
            self._fused_late_v,
            self._o_late_loads,
            q,
            k,
            v,
            beta,
        )
