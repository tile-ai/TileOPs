"""
DeltaNet forward: (q, k, v, beta) -> output o.

Pipeline (3 stages):
  1. fused_prepare_compute_w_u: (k, v, beta) -> (Aw, Au, w, u)
  2. h_recurrence:  (k, w, u, S_0) -> (S, v_new)   [sequential over chunks]
  3. output_o:      (q, k, S, v_new) -> o            [parallel over chunks]

Unlike Gated DeltaNet (GDN), there is no gate parameter g:
  - No exp(g) scaling in any stage
  - State update: h_new = h + k^T @ v_new (no decay)
  - v_new = u - w @ h (no exp scaling)
  - Output: o = q @ h + causal_attn @ v_new (no Gamma weighting)
"""

import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    DeltaNetChunkCall,
    DeltaNetFwdInterface,
    head_count_refusal,
)
from tileops.kernels.linear_attention.deltanet.autotune import (
    default_h_block_v,
    default_h_threads,
    delta_rule_fwd_autotune_configs,
    h_block_v_candidates,
    tune_delta_rule_fwd,
)
from tileops.kernels.linear_attention.deltanet.fused_prepare_compute_w_u import (
    fused_prepare_compute_w_u_tl,
)
from tileops.kernels.linear_attention.v_tile import min_gemm_n, resolve_block_v
from tileops.utils import get_shared_memory_optin

__all__ = ["DeltaNetFwdKernel"]


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
):
    """Output projection: (q, k, S, v_new) -> o.

    Grid: (num_chunks, batch, head) -- fully parallel across chunks.
    Each chunk reads h = S[t] (boundary state at start of chunk) and v_new[t].

    *late_loads* loads k once h is spent and v_new once q and k are, so each product holds
    only its own operands; it costs the loads their overlap, so only a chunk that does not
    fit otherwise takes it.
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
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], accum_dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            o: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(num_chunks, batch, head, threads=threads) as (tid, bid, hid):
                q_c = T.alloc_shared([block_C, dim_k], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                h_c = T.alloc_shared([dim_k, dim_v], dtype)
                v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)

                o_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)

                T.copy(q[bid, hid, tid * block_C : (tid + 1) * block_C, :], q_c, disable_tma=True)
                if not late_loads:
                    T.copy(
                        k[bid, hid, tid * block_C : (tid + 1) * block_C, :], k_c, disable_tma=True
                    )
                T.copy(S[bid, hid, tid, :, :], h_c, disable_tma=True)
                if not late_loads:
                    T.copy(
                        v_new[bid, hid, tid * block_C : (tid + 1) * block_C, :],
                        v_new_c,
                        disable_tma=True,
                    )

                # o = q @ h (no exp(g) scaling)
                T.clear(o_frag)
                T.gemm(q_c, h_c, o_frag)

                if late_loads:
                    T.copy(
                        k[bid, hid, tid * block_C : (tid + 1) * block_C, :], k_c, disable_tma=True
                    )

                # attn = causal(q @ k^T) (no Gamma weighting)
                T.clear(attn_frag)
                T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.if_then_else(i >= j, attn_frag[i, j], T.float32(0.0))

                if late_loads:
                    T.copy(
                        v_new[bid, hid, tid * block_C : (tid + 1) * block_C, :],
                        v_new_c,
                        disable_tma=True,
                    )

                # o += attn @ v_new
                T.gemm(attn, v_new_c, o_frag)
                T.copy(
                    o_frag, o[bid, hid, tid * block_C : (tid + 1) * block_C, :], disable_tma=True
                )

        return output_o_kernel

    return _func


class DeltaNetFwdKernel(Kernel, DeltaNetFwdInterface):
    @classmethod
    def applies(cls, call: DeltaNetChunkCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: DeltaNetChunkCall) -> Optional[str]:
        """Why no program serves this call, or ``None``; reads lower bounds, the recurrence's
        over its narrowest V tile, so a call above them that TileLang still cannot place in the
        device's shared memory is built and fails at launch."""
        reason = head_count_refusal(call.heads)
        if reason is not None or not call.smem_budget:
            return reason
        c, k, v, elem = call.chunk_size, call.dim_k, call.dim_v, call.dtype.itemsize
        width = min((resolve_block_v(v, b) for b in h_block_v_candidates(v)), default=v)
        need = max(
            cls._fused_live_bytes(c, k, v, elem),
            cls._state_live_bytes(c, k, width, elem),
            cls._output_live_bytes(c, k, v, elem),
        )
        if need <= call.smem_budget:
            return None
        return (
            f"needs at least {need} bytes of shared memory per block at chunk {c}, head dims "
            f"{k} / {v} in {call.dtype}; the device gives {call.smem_budget}"
        )

    @staticmethod
    def _fused_live_bytes(c: int, k: int, v: int, elem: int) -> int:
        """Lower bound on the w/u preparation's shared memory: k, beta, S and P through the
        inverse, k_beta with k, beta and S, then S, beta, v and v_beta."""
        return max(c * k + c + 2 * c * c, 2 * c * k + c + c * c, c * c + c + 2 * c * v) * elem

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
    def _output_live_bytes(c: int, k: int, v: int, elem: int) -> int:
        """Lower bound on the output pass's shared memory: q with the state, q with k, then
        attn with v_new."""
        return max(c * k + k * v, 2 * c * k, c * c + c * v) * elem

    @staticmethod
    def _late_loads_for(budget: int, c: int, k: int, v: int, elem: int) -> Tuple[bool, bool]:
        """Whether the w/u preparation and the output pass load late: only where their buffers,
        every one loaded up front, exceed *budget*."""
        fused = (2 * c * k + 2 * c * v + 2 * c * c + c) * elem
        output = (2 * c * k + k * v + c * v + c * c) * elem
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
        fused_fn = fused_prepare_compute_w_u_tl(
            batch,
            head,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype,
            late_v=fused_late_v,
        )(fused_num_stages, fused_threads)
        h_fn = _h_recurrence_tl(
            batch,
            head,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype,
            block_v=h_block_v,
        )(h_num_stages, h_threads)
        o_fn = _output_o_tl(
            batch,
            head,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype,
            late_loads=o_late_loads,
        )(o_threads)
        S_0 = torch.zeros(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
        Aw, Au, w, u = fused_fn(k, v, beta)
        S_buf, v_new = h_fn(k, w, u, S_0)
        o = o_fn(q, k, S_buf, v_new)
        return o, S_buf, Aw, Au, w, u

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
        self._fused_late_v, self._o_late_loads = self._late_loads_for(
            get_shared_memory_optin(self.device_index),
            chunk_size,
            dim_k,
            dim_v,
            getattr(torch, self.dtype_str).itemsize,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return self._default_config_for(
            get_shared_memory_optin(self.device_index),
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            getattr(torch, self.dtype_str).itemsize,
        )

    @classmethod
    def _default_config_for(
        cls, budget: int, chunk_size: int, dim_k: int, dim_v: int, elem: int
    ) -> dict:
        """The config this kernel builds at *budget* bytes of shared memory per block."""
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

    @property
    def autotune_configs(self) -> list[dict]:
        return delta_rule_fwd_autotune_configs(self.dim_v)

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Tune the three sub-kernels independently and merge the winners."""
        self.config = tune_delta_rule_fwd(
            self,
            fused_builder=functools.partial(
                fused_prepare_compute_w_u_tl, late_v=self._fused_late_v
            ),
            h_builder=_h_recurrence_tl,
            o_builder=functools.partial(_output_o_tl, late_loads=self._o_late_loads),
            warmup=warmup,
            rep=rep,
        )

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
