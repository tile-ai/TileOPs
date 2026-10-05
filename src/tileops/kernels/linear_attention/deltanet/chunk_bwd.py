"""
DeltaNet backward: given dL/do, compute dL/d(q, k, v, beta).

Unlike Gated DeltaNet (GDN), there is no gate parameter g, so:
  - No dg output
  - No exp(g) scaling anywhere
  - Returns 4 gradients: (dq, dk, dv, dbeta)

Backward (split for SM utilisation):
  1. fused_prepare_compute_w_u: recompute w, u from forward
  2. bwd_parallel:    per-chunk gradients (grid: num_chunks x B x H)
  3. dh_recurrence_bwd: sequential dh propagation + corrections (grid: B x H)
  4. compute_w_u_bwd: dw, du -> dk_wu, dv, dbeta
  5. merge: dk = dk_parallel + dk_correction + dk_wu
"""

import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    DeltaNetBwdInterface,
    DeltaNetChunkCall,
    head_count_refusal,
)
from tileops.kernels.linear_attention.v_tile import min_gemm_n
from tileops.utils import get_shared_memory_optin

__all__ = [
    "DeltaNetBwdKernel",
]


# Split kernel: bwd_parallel (fully parallel over chunks)


@functools.lru_cache(maxsize=32)
def _bwd_parallel_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """Parallel per-chunk backward gradients.

    Grid: (num_chunks, batch, head) -- fully parallel across chunks.
    Computes everything that does NOT depend on dh_buf from other chunks.

    Outputs: dq, dk_partial, dw, du_partial, v_new, dh_local
    (6 outputs instead of 7 -- no dg_partial)
    """
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C

    @tilelang.jit(
        out_idx=[-6, -5, -4, -3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=256):
        if dim_v < min_gemm_n(threads):
            raise ValueError(
                f"dim_v ({dim_v}) is below the minimum T.gemm N extent "
                f"({min_gemm_n(threads)}) at {threads} threads"
            )

        @T.prim_func
        def bwd_parallel_kernel(
            do: T.Tensor([batch, head, seq_len, dim_v], dtype),
            q: T.Tensor([batch, head, seq_len, dim_k], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], accum_dtype),
            # Outputs
            dq: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dk_partial: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dw: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du_partial: T.Tensor([batch, head, seq_len, dim_v], dtype),
            v_new_out: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], accum_dtype),
        ):
            with T.Kernel(num_chunks, batch, head, threads=threads) as (tid, bid, hid):
                q_c = T.alloc_shared([block_C, dim_k], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                u_c = T.alloc_shared([block_C, dim_v], dtype)
                do_c = T.alloc_shared([block_C, dim_v], dtype)
                h_c = T.alloc_shared([dim_k, dim_v], dtype)
                v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                o_part = T.alloc_shared([block_C, dim_v], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)
                d_q_c = T.alloc_shared([block_C, dim_k], dtype)
                d_k_c = T.alloc_shared([block_C, dim_k], dtype)
                d_w_c = T.alloc_shared([block_C, dim_k], dtype)
                d_v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                d_attn = T.alloc_shared([block_C, block_C], dtype)
                dP = T.alloc_shared([block_C, dim_k], dtype)
                ws_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_v_new_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                d_attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_q_c_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                d_k_c_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dh_frag = T.alloc_fragment([dim_k, dim_v], accum_dtype)

                T.copy(q[bid, hid, tid * block_C : (tid + 1) * block_C, :], q_c, disable_tma=True)
                T.copy(k[bid, hid, tid * block_C : (tid + 1) * block_C, :], k_c, disable_tma=True)
                T.copy(w[bid, hid, tid * block_C : (tid + 1) * block_C, :], w_c, disable_tma=True)
                T.copy(u[bid, hid, tid * block_C : (tid + 1) * block_C, :], u_c, disable_tma=True)
                T.copy(do[bid, hid, tid * block_C : (tid + 1) * block_C, :], do_c, disable_tma=True)
                T.copy(S[bid, hid, tid, :, :], h_c, disable_tma=True)

                # Recompute forward: v_new_c, o_part, attn (no exp(g) scaling)
                T.clear(ws_frag)
                T.gemm(w_c, h_c, ws_frag)
                for i, j in T.Parallel(block_C, dim_v):
                    v_new_c[i, j] = u_c[i, j] - ws_frag[i, j]

                # Store v_new for recurrence kernel
                T.copy(
                    v_new_c,
                    v_new_out[bid, hid, tid * block_C : (tid + 1) * block_C, :],
                    disable_tma=True,
                )

                T.clear(ws_frag)
                T.gemm(q_c, h_c, ws_frag)
                for i, j in T.Parallel(block_C, dim_v):
                    o_part[i, j] = ws_frag[i, j]

                T.clear(attn_frag)
                T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.if_then_else(i >= j, attn_frag[i, j], T.float32(0.0))

                T.clear(dh_frag)

                # Step 2: d_v_new_c = attn^T @ do_c (partial du)
                T.clear(d_v_new_frag)
                T.gemm(attn, do_c, d_v_new_frag, transpose_A=True)
                T.copy(d_v_new_frag, d_v_new_c)

                # d_attn = do_c @ v_new_c^T (causal masked)
                T.clear(d_attn_frag)
                T.gemm(do_c, v_new_c, d_attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    d_attn[i, j] = T.if_then_else(i >= j, d_attn_frag[i, j], T.float32(0.0))

                # Step 3: dq from h, dh from q (no exp(g) scaling, no dg)
                T.clear(d_q_c_frag)
                # dq from inter-chunk: do * q @ h -> dq += do @ h^T
                T.gemm(do_c, h_c, d_q_c_frag, transpose_B=True)
                # dh from q: q^T @ do
                T.gemm(q_c, do_c, dh_frag, transpose_A=True)

                # Step 4: dq/dk from d_attn (no Gamma weighting)
                T.gemm(d_attn, k_c, d_q_c_frag)
                T.copy(d_q_c_frag, d_q_c)

                T.clear(d_k_c_frag)
                T.gemm(d_attn, q_c, d_k_c_frag, transpose_A=True)
                T.copy(d_k_c_frag, d_k_c)

                # Step 5: dh from w/v_new, dw (no exp scaling)
                T.clear(dP_frag)
                T.gemm(d_v_new_c, h_c, dP_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, dim_k):
                    dP[i, j] = -dP_frag[i, j]
                dh_sub_frag = T.alloc_fragment([dim_k, dim_v], accum_dtype)
                T.clear(dh_sub_frag)
                T.gemm(w_c, d_v_new_c, dh_sub_frag, transpose_A=True)
                for i, j in T.Parallel(dim_k, dim_v):
                    dh_frag[i, j] -= dh_sub_frag[i, j]
                # dw = -dP (since v_new = u - w @ h, dw = -(dv_new @ h^T))
                for i, j in T.Parallel(block_C, dim_k):
                    d_w_c[i, j] = dP[i, j]

                T.copy(
                    d_q_c, dq[bid, hid, tid * block_C : (tid + 1) * block_C, :], disable_tma=True
                )
                T.copy(
                    d_k_c,
                    dk_partial[bid, hid, tid * block_C : (tid + 1) * block_C, :],
                    disable_tma=True,
                )
                T.copy(
                    d_w_c, dw[bid, hid, tid * block_C : (tid + 1) * block_C, :], disable_tma=True
                )
                T.copy(
                    d_v_new_c,
                    du_partial[bid, hid, tid * block_C : (tid + 1) * block_C, :],
                    disable_tma=True,
                )
                # Store dh_local for recurrence kernel
                T.copy(dh_frag, dh_local[bid, hid, tid, :, :], disable_tma=True)

        return bwd_parallel_kernel

    return _func


# Split kernel: dh_recurrence_bwd (sequential backward over chunks)


@functools.lru_cache(maxsize=32)
def _dh_recurrence_bwd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """Sequential backward dh recurrence with corrections.

    Grid: (batch, head) -- sequential over chunks (backward).
    Reads dh_local from bwd_parallel, propagates dh backward, and computes
    corrections for dk, du that depend on dh_buf from other chunks.

    The correct recurrence for ungated DeltaNet is:
        dh_c = dh_local[c] + (I - W_c^T @ K_c) @ dh_{c+1}
    because h_{c+1} = h_c + k^T @ v_new and v_new = u - w @ h_c,
    so dh_{c+1}/dh_c = I - k^T @ w (not I).

    dw_corr is computed outside this kernel as a batched matmul for efficiency.

    Outputs: dk_correction, du_correction
    """
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        if dim_v < min_gemm_n(threads):
            raise ValueError(
                f"dim_v ({dim_v}) is below the minimum T.gemm N extent "
                f"({min_gemm_n(threads)}) at {threads} threads"
            )

        @T.prim_func
        def dh_recurrence_bwd_kernel(
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], accum_dtype),
            # Outputs
            dk_corr: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                dh_loc = T.alloc_shared([dim_k, dim_v], accum_dtype)
                dh_buf = T.alloc_shared([dim_k, dim_v], dtype)
                dh_fp32 = T.alloc_fragment([dim_k, dim_v], accum_dtype)
                du_corr_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                wk_dh_frag = T.alloc_fragment([dim_k, dim_v], accum_dtype)
                k_dh_shared = T.alloc_shared([block_C, dim_v], dtype)

                for i, j in T.Parallel(dim_k, dim_v):
                    dh_fp32[i, j] = T.float32(0.0)

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    t_bwd = num_chunks - 1 - t
                    T.copy(
                        k[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :],
                        k_c,
                        disable_tma=True,
                    )
                    T.copy(
                        w[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :],
                        w_c,
                        disable_tma=True,
                    )
                    T.copy(
                        v_new[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :],
                        v_new_c,
                        disable_tma=True,
                    )
                    T.copy(dh_local[bid, hid, t_bwd, :, :], dh_loc, disable_tma=True)

                    T.copy(dh_fp32, dh_buf)

                    # du_corr = k @ dh_buf; also save for wk_dh
                    T.clear(du_corr_frag)
                    T.gemm(k_c, dh_buf, du_corr_frag)
                    T.copy(
                        du_corr_frag,
                        du_corr[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :],
                        disable_tma=True,
                    )
                    T.copy(du_corr_frag, k_dh_shared)

                    # dk_corr = v_new @ dh_buf^T
                    T.clear(dP_frag)
                    T.gemm(v_new_c, dh_buf, dP_frag, transpose_B=True)
                    T.copy(
                        dP_frag,
                        dk_corr[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :],
                        disable_tma=True,
                    )

                    # dh = dh_local + (I - W^T @ K) @ dh_future
                    #    = dh_local + dh_future - W^T @ (K @ dh_future)
                    T.clear(wk_dh_frag)
                    T.gemm(w_c, k_dh_shared, wk_dh_frag, transpose_A=True)
                    for i, j in T.Parallel(dim_k, dim_v):
                        dh_fp32[i, j] = dh_fp32[i, j] + dh_loc[i, j] - wk_dh_frag[i, j]

        return dh_recurrence_bwd_kernel

    return _func


class DeltaNetBwdKernel(Kernel, DeltaNetBwdInterface):
    """DeltaNet backward kernel.

    Full backward: do -> (dq, dk, dv, dbeta).
    No gate parameter, so no dg output.

    Split pipeline:
      1. fused_prepare_compute_w_u: recompute w, u
      2. bwd_parallel: per-chunk gradients (grid: num_chunks x B x H)
      3. dh_recurrence_bwd: sequential dh propagation + corrections (grid: B x H)
      4. compute_w_u_bwd: dw, du -> dk_wu, dv, dbeta
      5. merge: dk = dk_partial + dk_correction + dk_wu
    """

    @staticmethod
    def _deltanet_bwd_run(
        batch: int,
        head: int,
        seq_len: int,
        chunk_size: int,
        dim_k: int,
        dim_v: int,
        dtype: str,
        num_stages: int,
        threads: int,
        parallel_threads: int,
        recurrence_threads: int,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        S: torch.Tensor,
        Aw: torch.Tensor,
        Au: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from tileops.kernels.linear_attention.deltanet.compute_w_u_bwd import compute_w_u_bwd_tl

        bwd_parallel_fn = _bwd_parallel_tl(
            batch,
            head,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype,
        )(parallel_threads)
        dh_recurrence_bwd_fn = _dh_recurrence_bwd_tl(
            batch,
            head,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype,
        )(num_stages, recurrence_threads)
        wu_bwd_fn = compute_w_u_bwd_tl(
            batch,
            head,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype,
        )(threads)

        dq, dk_partial, dw, du_partial, v_new, dh_local = bwd_parallel_fn(do, q, k, w, u, S)
        dk_corr, du_corr = dh_recurrence_bwd_fn(k, w, v_new, dh_local)

        # Fused: dw_corr + du merge + wu_bwd + A_inv backward + dk merge
        dk, dv, dbeta = wu_bwd_fn(
            dw,
            du_partial,
            du_corr,
            S,
            Aw,
            Au,
            k,
            v,
            beta,
            dk_partial,
            dk_corr,
        )
        return dq, dk, dv, dbeta

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

    @classmethod
    def applies(cls, call: DeltaNetChunkCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: DeltaNetChunkCall) -> Optional[str]:
        """Why no program serves this call, or ``None``.

        Reads lower bounds of the three programs at their smallest configuration (the
        recurrence at one stage): a refused call fits no placement TileLang could choose. A
        call between a lower bound and the size TileLang compiles is built, and fits or not
        as TileLang places it.
        """
        heads = head_count_refusal(call.heads)
        if heads is not None:
            return heads
        if not call.smem_budget:
            return None
        c, k, v, elem = call.chunk_size, call.dim_k, call.dim_v, call.dtype.itemsize
        need = max(
            cls._parallel_live_bytes(c, k, v, elem),
            cls._recurrence_live_bytes(c, k, v, elem),
            cls._wu_live_bytes(c, k, v),
        )
        if need <= call.smem_budget:
            return None
        return (
            f"needs at least {need} bytes of shared memory per block at chunk {c}, head dims "
            f"{k} / {v} in {call.dtype}; the device gives {call.smem_budget}"
        )

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
        threads = 256 if chunk_size >= 64 else 128
        while threads > 64 and dim_v < min_gemm_n(threads):
            threads //= 2
        return {
            "num_stages": max(
                cls._recurrence_stage_options(budget, chunk_size, dim_k, dim_v, elem)
            ),
            "threads": threads,
            "parallel_threads": threads,
            "recurrence_threads": threads,
        }

    @classmethod
    def _recurrence_stage_options(
        cls, budget: int, chunk_size: int, dim_k: int, dim_v: int, elem: int
    ) -> list[int]:
        """Recurrence stage counts to choose from: two only where their upper bound fits
        *budget*; one stays, as ``refusal`` vets it."""
        two = cls._recurrence_shared_bytes(chunk_size, dim_k, dim_v, elem, 2)
        return [1, 2] if two <= budget else [1]

    @staticmethod
    def _recurrence_shared_bytes(c: int, k: int, v: int, elem: int, num_stages: int) -> int:
        """Upper bound on the dh recurrence's shared memory: every buffer it allocates, k, w,
        v_new and the fp32 dh_local tile per stage, then dh_buf and k_dh. TileLang may place
        buffers whose lifetimes do not overlap in one space, so the compiled size can be
        smaller."""
        per_stage = (2 * c * k + c * v) * elem + k * v * 4
        return num_stages * per_stage + (k * v + c * v) * elem

    @staticmethod
    def _parallel_live_bytes(c: int, k: int, v: int, elem: int) -> int:
        """Lower bound on the per-chunk backward's shared memory: q, k, w, h and d_v_new are
        live with do and v_new at the d_attn product, or with do and d_attn at the dq one;
        w, d_v_new, dq and dk are live with dP once it is written."""
        return max(3 * c * k + 2 * c * v + k * v + max(c * v, c * c), 4 * c * k + c * v) * elem

    @staticmethod
    def _recurrence_live_bytes(c: int, k: int, v: int, elem: int) -> int:
        """Lower bound on the one-stage dh recurrence's shared memory: at either GEMM of a
        chunk, w, v_new, dh_buf and the fp32 dh_local are live with k (the first) or k_dh
        (the second)."""
        return (c * k + c * v + k * v + max(c * k, c * v)) * elem + k * v * 4

    @staticmethod
    def _wu_live_bytes(c: int, k: int, v: int) -> int:
        """Lower bound on the w/u backward's shared memory, every buffer fp32: the largest
        set live at once at one of its GEMMs."""
        return 4 * max(
            c * c + 3 * c * v + c,  # dAu: Au, du, du_corr, v_beta, beta
            c * v + k * v + c * k + 2 * c,  # dw correction: du_corr, S, dw, beta, dbeta_v
            c * c + 3 * c * k + 2 * c,  # dAw: Aw, dw, k, k_beta, beta, dbeta_v
            2 * c * c + 2 * c * k + 2 * c,  # A_inv backward: Aw, dA_inv, k, k_beta, beta, dbeta
        )

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Autotune each sub-kernel independently and merge best configs."""
        from tilelang.autotuner import autotune as tl_autotune

        from tileops.kernels.linear_attention.deltanet.compute_w_u_bwd import compute_w_u_bwd_tl

        B, H, S, BC = self.batch, self.head, self.seq_len, self.chunk_size
        DK, DV, dt = self.dim_k, self.dim_v, self.dtype_str

        thread_options = [t for t in [128, 256] if self.dim_v >= min_gemm_n(t)] or [64]
        parallel_configs = [{"threads": t} for t in thread_options]
        print(f"Autotuning bwd_parallel ({len(parallel_configs)} configs)...")
        parallel_jit = _bwd_parallel_tl(B, H, S, BC, DK, DV, dt)
        _parallel_at = dict(configs=parallel_configs, warmup=warmup, rep=rep)
        _parallel_dns = list(
            self._autotune_initial_kwargs(parallel_jit, parallel_configs[0]).keys()
        )
        if _parallel_dns:
            _parallel_at["do_not_specialize"] = _parallel_dns
        tuned_parallel = self._call_autotuned_kernel(
            tl_autotune(**_parallel_at)(parallel_jit),
            parallel_jit,
            parallel_configs[0],
        )
        parallel_best = tuned_parallel.config
        print(f"  Best: {parallel_best}")

        recurrence_configs = [
            {"num_stages": ns, "threads": t}
            for ns in self._recurrence_stage_options(
                get_shared_memory_optin(self.device_index),
                self.chunk_size,
                self.dim_k,
                self.dim_v,
                getattr(torch, self.dtype_str).itemsize,
            )
            for t in thread_options
        ]
        print(f"Autotuning dh_recurrence_bwd ({len(recurrence_configs)} configs)...")
        recurrence_jit = _dh_recurrence_bwd_tl(B, H, S, BC, DK, DV, dt)
        _recurrence_at = dict(configs=recurrence_configs, warmup=warmup, rep=rep)
        _recurrence_dns = list(
            self._autotune_initial_kwargs(recurrence_jit, recurrence_configs[0]).keys()
        )
        if _recurrence_dns:
            _recurrence_at["do_not_specialize"] = _recurrence_dns
        tuned_recurrence = self._call_autotuned_kernel(
            tl_autotune(**_recurrence_at)(recurrence_jit),
            recurrence_jit,
            recurrence_configs[0],
        )
        recurrence_best = tuned_recurrence.config
        print(f"  Best: {recurrence_best}")

        wu_bwd_configs = [{"threads": t} for t in [128, 256]]
        print(f"Autotuning compute_w_u_bwd ({len(wu_bwd_configs)} configs)...")
        wu_bwd_jit = compute_w_u_bwd_tl(B, H, S, BC, DK, DV, dt)
        _wu_bwd_at = dict(configs=wu_bwd_configs, warmup=warmup, rep=rep)
        _wu_bwd_dns = list(self._autotune_initial_kwargs(wu_bwd_jit, wu_bwd_configs[0]).keys())
        if _wu_bwd_dns:
            _wu_bwd_at["do_not_specialize"] = _wu_bwd_dns
        tuned_wu_bwd = self._call_autotuned_kernel(
            tl_autotune(**_wu_bwd_at)(wu_bwd_jit),
            wu_bwd_jit,
            wu_bwd_configs[0],
        )
        wu_bwd_best = tuned_wu_bwd.config
        print(f"  Best: {wu_bwd_best}")

        self.config = {
            "num_stages": recurrence_best["num_stages"],
            "threads": wu_bwd_best["threads"],
            "parallel_threads": parallel_best["threads"],
            "recurrence_threads": recurrence_best["threads"],
        }
        print(f"DeltaNetBwdKernel autotuned config: {self.config}")

    def forward(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        S: torch.Tensor,
        Aw: torch.Tensor,
        Au: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._deltanet_bwd_run(
            self.batch,
            self.head,
            self.seq_len,
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            self.dtype_str,
            self.config.get("num_stages", 2),
            self.config.get("threads", 256),
            self.config.get("parallel_threads", 256),
            self.config.get("recurrence_threads", 256),
            do,
            q,
            k,
            v,
            beta,
            S,
            Aw,
            Au,
            w,
            u,
        )
