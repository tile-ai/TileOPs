"""Partitioned long-context GLA prefill kernels."""

import functools
from typing import Callable, Optional

import tilelang
import torch
from tilelang import language as T

from tileops.kernels.constants import LOG2E

from .dense_prefill_subchunk import _gla_fwd_a_kernel
from .gla_fwd import GLAFwdKernel, _gla_precompute_g_kernel


@functools.lru_cache(maxsize=32)
def _gla_fwd_h_summary_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    partition_chunks: int,
    dtype: str,
    gate_dtype: str,
    num_v_partitions: int = 4,
    num_k_partitions: int = 2,
) -> Callable:
    """Summarise independent chunk partitions from a zero initial state."""
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    if num_chunks % partition_chunks != 0:
        raise ValueError(
            f"num_chunks ({num_chunks}) must be divisible by partition_chunks ({partition_chunks})"
        )
    num_partitions = num_chunks // partition_chunks
    dim_v_part = dim_v // num_v_partitions
    dim_k_part = dim_k // num_k_partitions
    num_kv = num_k_partitions * num_v_partitions

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _summary_func(num_stages=2, threads=128):
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_shape = [batch, seq_len, heads, dim_k]
        summary_shape = [batch, num_partitions, heads, dim_k, dim_v]
        decay_shape = [batch, num_partitions, heads, dim_k]

        @T.prim_func
        def _main(
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_shape, gate_dtype),
            summaries: T.Tensor(summary_shape, accum_dtype),
            log_decays: T.Tensor(decay_shape, accum_dtype),
        ):
            with T.Kernel(num_partitions * num_kv, batch, heads, threads=threads) as (
                i_pk,
                i_b,
                i_h,
            ):
                i_p = i_pk // num_kv
                i_kv = i_pk % num_kv
                i_kp = i_kv // num_v_partitions
                i_vp = i_kv % num_v_partitions
                k_offset = i_kp * dim_k_part
                v_offset = i_vp * dim_v_part

                h_f = T.alloc_fragment([dim_k_part, dim_v_part], accum_dtype)
                k_s = T.alloc_shared([chunk_size, dim_k_part], dtype)
                v_s = T.alloc_shared([chunk_size, dim_v_part], dtype)
                g_s = T.alloc_shared([chunk_size, dim_k_part], gate_dtype)
                k_adj = T.alloc_fragment([chunk_size, dim_k_part], dtype)
                log_decay = T.alloc_fragment([dim_k_part], accum_dtype)

                T.clear(h_f)
                T.clear(log_decay)
                for i_local in T.Pipelined(partition_chunks, num_stages=num_stages):
                    i_c = i_p * partition_chunks + i_local
                    chunk_start = i_c * chunk_size
                    T.copy(
                        k[
                            i_b,
                            chunk_start : chunk_start + chunk_size,
                            i_h,
                            k_offset : k_offset + dim_k_part,
                        ],
                        k_s,
                        disable_tma=True,
                    )
                    T.copy(
                        v[
                            i_b,
                            chunk_start : chunk_start + chunk_size,
                            i_h,
                            v_offset : v_offset + dim_v_part,
                        ],
                        v_s,
                        disable_tma=True,
                    )
                    T.copy(
                        g_cumsum[
                            i_b,
                            chunk_start : chunk_start + chunk_size,
                            i_h,
                            k_offset : k_offset + dim_k_part,
                        ],
                        g_s,
                        disable_tma=True,
                    )

                    for i_k in T.Parallel(dim_k_part):
                        log_decay[i_k] = log_decay[i_k] + T.cast(
                            g_s[chunk_size - 1, i_k], accum_dtype
                        )
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_f[i_k, i_v] = h_f[i_k, i_v] * T.exp2(
                            T.cast(g_s[chunk_size - 1, i_k], accum_dtype) * LOG2E
                        )
                    for i_t, i_k in T.Parallel(chunk_size, dim_k_part):
                        k_adj[i_t, i_k] = T.cast(
                            T.cast(k_s[i_t, i_k], accum_dtype)
                            * T.exp2(
                                (
                                    T.cast(g_s[chunk_size - 1, i_k], accum_dtype)
                                    - T.cast(g_s[i_t, i_k], accum_dtype)
                                )
                                * LOG2E
                            ),
                            dtype,
                        )
                    T.gemm(
                        k_adj,
                        v_s,
                        h_f,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    summaries[i_b, i_p, i_h, k_offset + i_k, v_offset + i_v] = h_f[i_k, i_v]
                if i_vp == 0:
                    for i_k in T.Parallel(dim_k_part):
                        log_decays[i_b, i_p, i_h, k_offset + i_k] = log_decay[i_k]

        return _main

    return _summary_func


@functools.lru_cache(maxsize=32)
def _gla_fwd_h0_scan_kernel(
    batch: int,
    heads: int,
    num_partitions: int,
    dim_k: int,
    dim_v: int,
    block_v: int = 32,
) -> Callable:
    """Scan affine partition summaries into the true partition start states."""
    accum_dtype = "float32"
    num_v_tiles = dim_v // block_v

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    def _scan_func(threads=128):
        summary_shape = [batch, num_partitions, heads, dim_k, dim_v]
        decay_shape = [batch, num_partitions, heads, dim_k]

        @T.prim_func
        def _main(
            summaries: T.Tensor(summary_shape, accum_dtype),
            log_decays: T.Tensor(decay_shape, accum_dtype),
            initial_state: T.Tensor([batch, heads, dim_k, dim_v], accum_dtype),
            initial_states: T.Tensor(summary_shape, accum_dtype),
        ):
            with T.Kernel(num_v_tiles, batch, heads, threads=threads) as (i_vt, i_b, i_h):
                v_offset = i_vt * block_v
                h_s = T.alloc_shared([dim_k, block_v], accum_dtype)
                summary_s = T.alloc_shared([dim_k, block_v], accum_dtype)
                T.copy(initial_state[i_b, i_h, :, v_offset : v_offset + block_v], h_s)

                for i_p in T.Pipelined(num_partitions, num_stages=2):
                    for i_k, i_v in T.Parallel(dim_k, block_v):
                        initial_states[i_b, i_p, i_h, i_k, v_offset + i_v] = h_s[i_k, i_v]
                    T.copy(
                        summaries[i_b, i_p, i_h, :, v_offset : v_offset + block_v],
                        summary_s,
                        disable_tma=True,
                    )
                    for i_k, i_v in T.Parallel(dim_k, block_v):
                        h_s[i_k, i_v] = (
                            h_s[i_k, i_v] * T.exp2(log_decays[i_b, i_p, i_h, i_k] * LOG2E)
                            + summary_s[i_k, i_v]
                        )

        return _main

    return _scan_func


@functools.lru_cache(maxsize=32)
def _gla_fwd_partitioned_replay_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    partition_chunks: int,
    scale: float,
    dtype: str,
    gate_dtype: str,
) -> Callable:
    """Replay independent partitions while producing output in the same CTA.

    This is the GLA specialization of the GDN prefill replay skeleton.  GLA
    has no delta-rule correction, so the transform warpgroup only prepares
    gate-adjusted Q/K while the state and output warpgroups run concurrently.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    if num_chunks % partition_chunks != 0:
        raise ValueError(
            f"num_chunks ({num_chunks}) must be divisible by partition_chunks ({partition_chunks})"
        )
    if chunk_size != 64 or dim_k not in (64, 128) or dim_v not in (64, 128):
        raise ValueError("fused GLA prefill replay requires chunk_size=64 and DK,DV in {64,128}")
    num_partitions = num_chunks // partition_chunks
    work_dim_k = max(128, dim_k)
    work_dim_v = max(128, dim_v)
    work_dtype = dtype

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "-include", "tl_templates/cuda/gemm.h"],
    )
    def _replay_func(threads=512):
        qk_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_shape = [batch, seq_len, heads, dim_k]
        a_shape = [batch, seq_len, heads, chunk_size]
        initial_shape = [batch, num_partitions, heads, dim_k, dim_v]
        o_shape = [batch, seq_len, heads, dim_v]
        final_shape = [batch, heads, dim_k, dim_v]

        @T.prim_func
        def _main(
            q: T.Tensor(qk_shape, dtype),
            k: T.Tensor(qk_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_shape, gate_dtype),
            initial_states: T.Tensor(initial_shape, accum_dtype),
            A: T.Tensor(a_shape, dtype),
            o: T.Tensor(o_shape, dtype),
            final_state: T.Tensor(final_shape, accum_dtype),
        ):
            with T.Kernel(num_partitions, batch, heads, threads=threads) as (i_p, i_b, i_h):
                q_s = T.alloc_shared([chunk_size, work_dim_k], work_dtype)
                k_s = T.alloc_shared([chunk_size, work_dim_k], work_dtype)
                v_s = T.alloc_shared([chunk_size, work_dim_v], work_dtype)
                g_s = T.alloc_shared([chunk_size, work_dim_k], gate_dtype)
                h_s = T.alloc_shared([work_dim_k, work_dim_v], work_dtype)
                p_s = T.alloc_shared([chunk_size, chunk_size], work_dtype)

                h_f = T.alloc_fragment([work_dim_k, work_dim_v], accum_dtype)
                o_f = T.alloc_fragment([chunk_size, work_dim_v], accum_dtype)

                tx = T.get_thread_binding()
                if tx < 128:
                    T.set_max_nreg(160, 1)
                    T.clear(h_f)
                    T.copy(initial_states[i_b, i_p, i_h, :, :], h_f[:dim_k, :dim_v])
                elif tx < 256:
                    T.set_max_nreg(64, 1)
                elif tx < 384:
                    T.set_max_nreg(160, 1)
                else:
                    T.set_max_nreg(24, 0)
                T.sync_threads()

                for i_local in T.serial(partition_chunks):
                    i_c = i_p * partition_chunks + i_local
                    chunk_start = i_c * chunk_size

                    if tx >= 384:
                        T.clear(q_s)
                        T.clear(k_s)
                        T.clear(v_s)
                        T.clear(g_s)
                        T.copy(
                            q[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                            q_s[:, :dim_k],
                            disable_tma=True,
                        )
                        T.copy(
                            k[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                            k_s[:, :dim_k],
                            disable_tma=True,
                        )
                        T.copy(
                            v[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                            v_s[:, :dim_v],
                            disable_tma=True,
                        )
                        T.copy(
                            g_cumsum[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                            g_s[:, :dim_k],
                            disable_tma=True,
                        )
                        T.copy(
                            A[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                            p_s,
                            disable_tma=True,
                        )
                    T.sync_threads()

                    if tx < 128:
                        T.copy(h_f, h_s)
                        for i_k, i_v in T.Parallel(dim_k, dim_v):
                            h_f[i_k, i_v] *= T.exp2(
                                T.cast(g_s[chunk_size - 1, i_k], accum_dtype) * LOG2E
                            )
                    elif tx < 256:
                        for i_t, i_k in T.Parallel(chunk_size, dim_k):
                            g_last = T.cast(g_s[chunk_size - 1, i_k], accum_dtype)
                            g_value = T.cast(g_s[i_t, i_k], accum_dtype)
                            q_value = T.cast(q_s[i_t, i_k], accum_dtype)
                            k_value = T.cast(k_s[i_t, i_k], accum_dtype)
                            k_s[i_t, i_k] = T.cast(
                                k_value * T.exp2((g_last - g_value) * LOG2E), work_dtype
                            )
                            q_s[i_t, i_k] = T.cast(q_value * T.exp2(g_value * LOG2E), work_dtype)
                    T.sync_threads()

                    if tx < 128:
                        T.gemm(k_s, v_s, h_f, transpose_A=True, clear_accum=False)
                    elif tx < 384:
                        T.clear(o_f)
                        T.gemm(q_s, h_s, o_f)
                        for i_t, i_v in T.Parallel(chunk_size, dim_v):
                            o_f[i_t, i_v] *= scale
                        T.gemm(p_s, v_s, o_f, clear_accum=False)
                        for i_t, i_v in T.Parallel(chunk_size, dim_v):
                            o[i_b, chunk_start + i_t, i_h, i_v] = T.cast(o_f[i_t, i_v], dtype)
                    T.sync_threads()

                if tx < 128 and i_p == num_partitions - 1:
                    for i_k, i_v in T.Parallel(dim_k, dim_v):
                        final_state[i_b, i_h, i_k, i_v] = h_f[i_k, i_v]

        return _main

    return _replay_func


class GLADensePrefillPartitionedKernel(GLAFwdKernel):
    """GLA prefill with parallel partition summaries and fused output replay."""

    supported_archs = [90]

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        scale: float,
        dtype: torch.dtype,
        device_index: Optional[int] = None,
    ) -> None:
        del device_index
        if seq_len % (64 * 16):
            raise ValueError("partitioned GLA requires T divisible by 1024")
        if dim_k not in (64, 128) or dim_v not in (64, 128):
            raise ValueError("partitioned GLA requires K,V in {64,128}")
        super().__init__(
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            dim_k=dim_k,
            dim_v=dim_v,
            chunk_size=64,
            scale=scale,
            output_final_state=True,
            dtype=dtype,
        )

    @property
    def default_config(self) -> dict:
        return {
            "partition_chunks": 16,
            "num_v_partitions": 1,
            "num_k_partitions": 2,
            "g_num_stages": 2,
            "g_threads": 128,
            "h_num_stages": 2,
            "h_threads": 128,
            "scan_threads": 128,
        }

    def _build_kernels(self, config: dict) -> None:
        partition_chunks = config["partition_chunks"]
        gate_dtype = "float16"
        self._g_fn = _gla_precompute_g_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            64,
            self.dtype_name,
            gate_dtype,
        )(config["g_num_stages"], config["g_threads"])
        self._summary_fn = _gla_fwd_h_summary_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            64,
            partition_chunks,
            self.dtype_name,
            gate_dtype,
            num_v_partitions=config["num_v_partitions"],
            num_k_partitions=config["num_k_partitions"],
        )(config["h_num_stages"], config["h_threads"])
        self._scan_fn = _gla_fwd_h0_scan_kernel(
            self.batch,
            self.heads,
            self.seq_len // (64 * partition_chunks),
            self.dim_k,
            self.dim_v,
        )(config["scan_threads"])
        self._a_fn = _gla_fwd_a_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            64,
            self.scale,
            self.dtype_name,
            gate_dtype,
        )(64)
        self._replay_fn = _gla_fwd_partitioned_replay_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            64,
            partition_chunks,
            self.scale,
            self.dtype_name,
            gate_dtype,
        )(512)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cu_seqlens is not None or cu_seqlens_cpu is not None:
            raise ValueError("the in-tree GLA dense-prefill kernel does not support packed varlen")
        state = (
            initial_state
            if initial_state is not None
            else torch.zeros(
                self.batch,
                self.heads,
                self.dim_k,
                self.dim_v,
                dtype=torch.float32,
                device=q.device,
            )
        )
        gate = self._g_fn(g)
        summaries, log_decays = self._summary_fn(k, v, gate)
        partition_states = self._scan_fn(summaries, log_decays, state)
        a = self._a_fn(q, k, gate)
        return self._replay_fn(q, k, v, gate, partition_states, a)
