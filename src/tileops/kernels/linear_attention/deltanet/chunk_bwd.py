"""
DeltaNet backward: given dL/do, compute dL/d(q, k, v, beta).

Unlike Gated DeltaNet (GDN), there is no gate parameter g, so:
  - No dg output
  - No exp(g) scaling anywhere
  - Returns 4 gradients: (dq, dk, dv, dbeta)

Backward (split for SM utilisation):
  1. bwd_parallel: per-chunk gradients and the walk's operator -(W^T K)
     (grid: num_chunks x B x H)
  2. dh_walk: sequential dh propagation, one matmul a chunk (grid: DV / block_v x B x H)
  3. compute_w_u_bwd: the dh corrections, the w/u backward and the A_inv backward
     -> dk, dv, dbeta (grid: num_chunks x B x H)
"""

import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    DeltaNetChunkBwdInterface,
    DeltaNetChunkCall,
    head_count_refusal,
)
from tileops.kernels.linear_attention.v_tile import min_gemm_n
from tileops.utils import get_shared_memory_optin, get_sm_count

__all__ = [
    "DeltaNetChunkBwdKernel",
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

    Outputs: dq, dk_partial, du_partial, v_new, dh_local and the walk's per-chunk operator
    -(w^T k).
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
            du_partial: T.Tensor([batch, head, seq_len, dim_v], dtype),
            v_new_out: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], accum_dtype),
            neg_wk: T.Tensor([batch, head, num_chunks, dim_k, dim_k], dtype),
        ):
            with T.Kernel(num_chunks, batch, head, threads=threads) as (tid, bid, hid):
                q_c = T.alloc_shared([block_C, dim_k], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                u_c = T.alloc_shared([block_C, dim_v], dtype)
                do_c = T.alloc_shared([block_C, dim_v], dtype)
                h_c = T.alloc_shared([dim_k, dim_v], dtype)
                v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)
                d_v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                d_attn = T.alloc_shared([block_C, block_C], dtype)
                ws_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_v_new_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                d_attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_q_c_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                d_k_c_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dh_frag = T.alloc_fragment([dim_k, dim_v], accum_dtype)

                T.copy(q[bid, hid, tid * block_C : (tid + 1) * block_C, :], q_c, disable_tma=True)
                T.copy(k[bid, hid, tid * block_C : (tid + 1) * block_C, :], k_c, disable_tma=True)
                T.clear(attn_frag)
                T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.if_then_else(i >= j, attn_frag[i, j], T.float32(0.0))

                T.copy(w[bid, hid, tid * block_C : (tid + 1) * block_C, :], w_c, disable_tma=True)
                T.copy(u[bid, hid, tid * block_C : (tid + 1) * block_C, :], u_c, disable_tma=True)
                T.copy(S[bid, hid, tid, :, :], h_c, disable_tma=True)
                # Recompute the values consumed by the backward recurrence.
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

                T.copy(do[bid, hid, tid * block_C : (tid + 1) * block_C, :], do_c, disable_tma=True)
                # Step 2: d_v_new_c = attn^T @ do_c (partial du)
                T.clear(d_v_new_frag)
                T.gemm(attn, do_c, d_v_new_frag, transpose_A=True)
                T.copy(d_v_new_frag, d_v_new_c)

                # d_attn = do_c @ v_new_c^T (causal masked)
                T.clear(d_attn_frag)
                T.gemm(do_c, v_new_c, d_attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    d_attn[i, j] = T.if_then_else(i >= j, d_attn_frag[i, j], T.float32(0.0))

                T.clear(d_q_c_frag)
                # dq from inter-chunk: do * q @ h -> dq += do @ h^T
                T.gemm(do_c, h_c, d_q_c_frag, transpose_B=True)
                q_late = T.alloc_shared([block_C, dim_k], dtype)
                T.copy(
                    q[bid, hid, tid * block_C : (tid + 1) * block_C, :], q_late, disable_tma=True
                )
                k_late = T.alloc_shared([block_C, dim_k], dtype)
                T.copy(
                    k[bid, hid, tid * block_C : (tid + 1) * block_C, :], k_late, disable_tma=True
                )
                # Step 4: dq/dk from d_attn (no Gamma weighting)
                T.gemm(d_attn, k_late, d_q_c_frag)
                T.copy(
                    d_q_c_frag,
                    dq[bid, hid, tid * block_C : (tid + 1) * block_C, :],
                    disable_tma=True,
                )

                T.clear(d_k_c_frag)
                T.gemm(d_attn, q_late, d_k_c_frag, transpose_A=True)
                T.copy(
                    d_k_c_frag,
                    dk_partial[bid, hid, tid * block_C : (tid + 1) * block_C, :],
                    disable_tma=True,
                )

                T.clear(dh_frag)
                # dh from q: q^T @ do
                T.gemm(q_late, do_c, dh_frag, transpose_A=True)

                w_late = T.alloc_shared([block_C, dim_k], dtype)
                T.copy(
                    w[bid, hid, tid * block_C : (tid + 1) * block_C, :], w_late, disable_tma=True
                )
                dh_sub_frag = T.alloc_fragment([dim_k, dim_v], accum_dtype)
                T.clear(dh_sub_frag)
                T.gemm(w_late, d_v_new_c, dh_sub_frag, transpose_A=True)
                for i, j in T.Parallel(dim_k, dim_v):
                    dh_frag[i, j] -= dh_sub_frag[i, j]
                T.copy(
                    d_v_new_c,
                    du_partial[bid, hid, tid * block_C : (tid + 1) * block_C, :],
                    disable_tma=True,
                )
                # Store dh_local for the walk
                T.copy(dh_frag, dh_local[bid, hid, tid, :, :], disable_tma=True)

                # The walk's chunk operator: dh <- dh + dh_local - (w^T k) dh.
                wk_frag = T.alloc_fragment([dim_k, dim_k], accum_dtype)
                T.clear(wk_frag)
                T.gemm(w_late, k_late, wk_frag, transpose_A=True)
                for i, j in T.Parallel(dim_k, dim_k):
                    neg_wk[bid, hid, tid, i, j] = -wk_frag[i, j]

        return bwd_parallel_kernel

    return _func


# Split kernel: dh_walk (sequential backward over chunks)


@functools.lru_cache(maxsize=32)
def _dh_walk_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """Sequential backward walk of the state gradient dh over chunks.

    Grid: (dim_v / block_v, batch, head). Each CTA carries one ``[dim_k, block_v]`` column tile
    of dh from the last chunk to the first. For ungated DeltaNet

        dh_c = dh_{c+1} + dh_local[c] - (W_c^T K_c) dh_{c+1},

    because h_{c+1} = h_c + K_c^T v_new and v_new = U_c - W_c h_c. The column tiles are
    independent, and ``neg_wk`` holds -(W_c^T K_c), so the chain is one matmul a chunk.

    Output: dh_out[c], the gradient of the state chunk c hands on (dh_{c+1}), transposed and
    in the input dtype, which the w/u program reads.
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
        def dh_walk_kernel(
            neg_wk: T.Tensor([batch, head, num_chunks, dim_k, dim_k], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], accum_dtype),
            # Output, transposed: dh_out[..., c, :, :] = dh_{c+1}^T
            dh_out: T.Tensor([batch, head, num_chunks, dim_v, dim_k], dtype),
        ):
            with T.Kernel(dim_v // block_v, batch, head, threads=threads) as (vid, bid, hid):
                op_s = T.alloc_shared([dim_k, dim_k], dtype)
                local_s = T.alloc_shared([dim_k, block_v], accum_dtype)
                # Held transposed, so the matmul reads it K-major. Why: held MN-major, the
                # 16-column walk returned wrong dh under TileLang 0.1.12.
                dh_s = T.alloc_shared([block_v, dim_k], dtype)
                dh_frag = T.alloc_fragment([dim_k, block_v], accum_dtype)
                v0 = vid * block_v

                T.clear(dh_frag)
                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    c = num_chunks - 1 - t
                    T.copy(neg_wk[bid, hid, c, :, :], op_s, disable_tma=True)
                    T.copy(dh_local[bid, hid, c, :, v0 : v0 + block_v], local_s, disable_tma=True)
                    for i, j in T.Parallel(dim_k, block_v):
                        dh_s[j, i] = dh_frag[i, j]
                    T.copy(dh_s, dh_out[bid, hid, c, v0 : v0 + block_v, :], disable_tma=True)
                    for i, j in T.Parallel(dim_k, block_v):
                        dh_frag[i, j] = dh_frag[i, j] + local_s[i, j]
                    T.gemm(op_s, dh_s, dh_frag, transpose_B=True)

        return dh_walk_kernel

    return _func


def walk_block_v_options(dim_v: int) -> list[int]:
    """Value-tile widths of a DeltaNet walk (the backward's dh, the forward's state) that divide
    *dim_v*, narrowest first, else all of *dim_v*."""
    widths = [b for b in (8, 16, 32, 64, 128) if b <= dim_v and dim_v % b == 0]
    return widths or [dim_v]


def _wu_min_n(chunk_size: int, dim_k: int, dim_v: int) -> int:
    """The narrowest N extent among the w/u program's matmuls."""
    return min(chunk_size, dim_k, dim_v)


def walk_threads(block_v: int) -> int:
    """One warp group (WGMMA), or two warps (MMA) under WGMMA's narrowest N extent."""
    return 128 if block_v >= min_gemm_n(128) else 64


class DeltaNetChunkBwdKernel(Kernel, DeltaNetChunkBwdInterface):
    """DeltaNet backward kernel.

    Full backward: do -> (dq, dk, dv, dbeta).
    No gate parameter, so no dg output.

    Split pipeline:
      1. bwd_parallel: per-chunk gradients and the walk's operator -(W^T K)
         (grid: num_chunks x B x H)
      2. dh_walk: sequential dh propagation, one matmul a chunk
         (grid: DV / block_v x B x H)
      3. compute_w_u_bwd: the dh corrections, the w/u backward and the A_inv backward
         -> dk, dv, dbeta (grid: num_chunks x B x H)
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
        block_v: int,
        threads: int,
        parallel_threads: int,
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

        args = (batch, head, seq_len, chunk_size, dim_k, dim_v, dtype)
        bwd_parallel_fn = _bwd_parallel_tl(*args)(parallel_threads)
        dh_walk_fn = _dh_walk_tl(*args)(block_v, num_stages, walk_threads(block_v))
        wu_bwd_fn = compute_w_u_bwd_tl(*args)(threads)

        dq, dk_partial, du_partial, v_new, dh_local, neg_wk = bwd_parallel_fn(do, q, k, w, u, S)
        dh = dh_walk_fn(neg_wk, dh_local)
        dk, dv, dbeta = wu_bwd_fn(dh, v_new, du_partial, S, Aw, Au, k, v, beta, dk_partial)
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
    def refusal(cls, call: DeltaNetChunkCall) -> Optional[str]:
        """Why no program serves this call, or ``None``.

        Reads lower bounds of the three programs at their smallest configuration (the walk
        at one stage over its narrowest value tile): a refused call fits no placement
        TileLang could choose. A call between a lower bound and the size TileLang compiles
        is built, and fits or not as TileLang places it.
        """
        heads = head_count_refusal(call.heads)
        if heads is not None:
            return heads
        if not call.smem_budget:
            return None
        c, k, v, elem = call.chunk_size, call.dim_k, call.dim_v, call.dtype.itemsize
        need = max(
            cls._parallel_live_bytes(c, k, v, elem),
            cls._walk_shared_bytes(k, walk_block_v_options(v)[0], elem, 1),
            cls._wu_live_bytes(c, k, v, elem),
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
    ):
        super().__init__()
        self.batch = batch
        self.head = head
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return self._default_config_for(
            get_shared_memory_optin(self.device_index),
            get_sm_count(self.device_index),
            self.batch * self.head,
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            getattr(torch, self.dtype_str).itemsize,
        )

    @classmethod
    def _default_config_for(
        cls,
        budget: int,
        sm_count: int,
        heads: int,
        chunk_size: int,
        dim_k: int,
        dim_v: int,
        elem: int,
    ) -> dict:
        """The config this kernel builds at *budget* bytes of shared memory per block.

        The walk takes the narrowest value tile whose CTAs fit one wave: a narrower tile
        shortens each CTA's chain of matmuls, a second wave doubles it.
        """
        parallel_threads = 256 if chunk_size >= 64 else 128
        while parallel_threads > 64 and dim_v < min_gemm_n(parallel_threads):
            parallel_threads //= 2
        widths = walk_block_v_options(dim_v)
        block_v = next((b for b in widths if heads * (dim_v // b) <= sm_count), widths[-1])
        # The w/u program keeps several [chunk, head dim] f32 fragments live; one warp group
        # holds a 64 x 64 one in 32 registers a thread, and past that it spills.
        wu_threads = 256 if chunk_size * max(dim_k, dim_v) > 64 * 64 else 128
        while wu_threads > 64 and _wu_min_n(chunk_size, dim_k, dim_v) < min_gemm_n(wu_threads):
            wu_threads //= 2
        return {
            "num_stages": max(cls._walk_stage_options(budget, dim_k, block_v, elem)),
            "block_v": block_v,
            "threads": wu_threads,
            "parallel_threads": parallel_threads,
        }

    @classmethod
    def _walk_stage_options(cls, budget: int, dim_k: int, block_v: int, elem: int) -> list[int]:
        """Walk stage counts to choose from: two only where their upper bound fits
        *budget*; one stays, as ``refusal`` vets it."""
        two = cls._walk_shared_bytes(dim_k, block_v, elem, 2)
        return [1, 2] if two <= budget else [1]

    @staticmethod
    def _walk_shared_bytes(k: int, block_v: int, elem: int, num_stages: int) -> int:
        """Allocated shared buffers: the chunk operator and the FP32 dh_local tile per
        pipeline stage, then the dh tile the matmul reads."""
        per_stage = k * k * elem + k * block_v * 4
        return num_stages * per_stage + k * block_v * elem

    @staticmethod
    def _parallel_live_bytes(c: int, k: int, v: int, elem: int) -> int:
        """Live operands after the early q/k/w loads end and before their later reloads."""
        return (
            max(
                c * k + c * v + k * v + c * c,
                3 * c * v + k * v,
                2 * c * v + k * v + c * c,
                2 * c * k + 2 * c * v + c * c,
            )
            * elem
        )

    @staticmethod
    def _wu_live_bytes(c: int, k: int, v: int, elem: int) -> int:
        """Live operands of the w/u program's phases, with the FP32 beta row."""
        return 4 * c + elem * max(
            c * k + 2 * k * v + 2 * c * v,  # du, dk_corr and dw products
            3 * c * k + 3 * c * v,  # dA_inv products
            3 * c * k + 2 * c * v + 2 * c * c,  # Au and Aw products
            2 * c * k + 4 * c * c,  # inverse derivative
        )

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Autotune each sub-kernel independently and merge best configs."""
        from tilelang.autotuner import autotune as tl_autotune

        from tileops.kernels.linear_attention.deltanet.compute_w_u_bwd import compute_w_u_bwd_tl

        args = (
            self.batch,
            self.head,
            self.seq_len,
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            self.dtype_str,
        )
        default = self.default_config
        elem = getattr(torch, self.dtype_str).itemsize
        budget = get_shared_memory_optin(self.device_index)

        def tune(jit, configs, untuned, label):
            configs = self.tuning_candidates(configs, untuned)
            print(f"Autotuning {label} ({len(configs)} configs)...")
            at = dict(configs=configs, warmup=warmup, rep=rep)
            dns = list(self._autotune_initial_kwargs(jit, configs[0]).keys())
            if dns:
                at["do_not_specialize"] = dns
            best = self._call_autotuned_kernel(tl_autotune(**at)(jit), jit, configs[0]).config
            print(f"  Best: {best}")
            return best

        thread_options = [t for t in [128, 256] if self.dim_v >= min_gemm_n(t)] or [64]
        parallel_best = tune(
            _bwd_parallel_tl(*args),
            [{"threads": t} for t in thread_options],
            {"threads": default["parallel_threads"]},
            "bwd_parallel",
        )
        walk_best = tune(
            _dh_walk_tl(*args),
            [
                {"block_v": b, "num_stages": ns, "threads": walk_threads(b)}
                for b in walk_block_v_options(self.dim_v)
                for ns in self._walk_stage_options(budget, self.dim_k, b, elem)
            ],
            {
                "block_v": default["block_v"],
                "num_stages": default["num_stages"],
                "threads": walk_threads(default["block_v"]),
            },
            "dh_walk",
        )
        wu_n = _wu_min_n(self.chunk_size, self.dim_k, self.dim_v)
        wu_best = tune(
            compute_w_u_bwd_tl(*args),
            [{"threads": t} for t in [128, 256] if wu_n >= min_gemm_n(t)] or [{"threads": 64}],
            {"threads": default["threads"]},
            "compute_w_u_bwd",
        )
        self.config = {
            "num_stages": walk_best["num_stages"],
            "block_v": walk_best["block_v"],
            "threads": wu_best["threads"],
            "parallel_threads": parallel_best["threads"],
        }
        print(f"DeltaNetChunkBwdKernel autotuned config: {self.config}")

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
            self.config["num_stages"],
            self.config["block_v"],
            self.config["threads"],
            self.config["parallel_threads"],
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
