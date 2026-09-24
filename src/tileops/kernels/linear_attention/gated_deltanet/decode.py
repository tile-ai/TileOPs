"""Hopper single-token Gated DeltaNet inference decode."""

import functools
from typing import Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Kernel

__all__ = ["GatedDeltaNetDenseDecodeFwdKernel"]


@functools.lru_cache(maxsize=32)
def _gated_deltanet_dense_decode_sm90_tl(
    batch: int,
    heads: int,
    dim: int,
    scale: float,
    dtype: str,
    v_tile: int,
    lane_group: int,
    maxrregcount: int,
):
    """Build the one-warp-per-state-column-tile decode program."""
    if dim != 128:
        raise ValueError("Hopper Gated DeltaNet decode currently requires K == V == 128")
    if dim % v_tile != 0:
        raise ValueError(f"dim={dim} must be divisible by v_tile={v_tile}")
    if lane_group * v_tile != 32:
        raise ValueError("lane_group * v_tile must equal one warp")
    if dim % lane_group != 0:
        raise ValueError(f"dim={dim} must be divisible by lane_group={lane_group}")

    total_blocks = batch * heads * (dim // v_tile)
    k_chunk = dim // lane_group
    compile_flags = ["-O3", "-DENABLE_BF16", "--use_fast_math"]
    if maxrregcount > 0:
        compile_flags.append(f"--maxrregcount={maxrregcount}")

    @tilelang.jit(out_idx=[-2, -1], compile_flags=compile_flags)
    def _decode(threads=32):
        @T.prim_func
        def gated_deltanet_dense_decode_sm90(
            q: T.Tensor([batch, 1, heads, dim], dtype),
            k: T.Tensor([batch, 1, heads, dim], dtype),
            v: T.Tensor([batch, 1, heads, dim], dtype),
            g: T.Tensor([batch, 1, heads], dtype),
            beta: T.Tensor([batch, 1, heads], dtype),
            state: T.Tensor([batch, heads, dim, dim], "float32"),
            o: T.Tensor([batch, 1, heads, dim], dtype),
            final_state: T.Tensor([batch, heads, dim, dim], "float32"),
        ):
            with T.Kernel(total_blocks, threads=threads) as (block,):
                tx = T.get_thread_binding()
                value_tile = block % (dim // v_tile)
                batch_head = block // (dim // v_tile)
                batch_idx = batch_head // heads
                head_idx = batch_head - batch_idx * heads
                value_lane = tx // lane_group
                k_rank = tx - value_lane * lane_group
                value_idx = value_tile * v_tile + value_lane
                k_begin = k_rank * k_chunk

                k_shared = T.alloc_shared([dim], "float32")
                q_shared = T.alloc_shared([dim], "float32")
                state_local = T.alloc_local([k_chunk], "float32")
                old_partial = T.alloc_var("float32", init=0.0)
                out_partial = T.alloc_var("float32", init=0.0)
                value_new = T.alloc_var("float32", init=0.0)

                for i in T.serial(T.ceildiv(dim, 32)):
                    kk = tx + i * 32
                    if kk < dim:
                        k_shared[kk] = T.cast(k[batch_idx, 0, head_idx, kk], "float32")
                        q_shared[kk] = T.cast(q[batch_idx, 0, head_idx, kk], "float32") * scale
                T.sync_threads()

                decay = T.exp2(T.cast(g[batch_idx, 0, head_idx], "float32") * LOG2E)
                beta_value = T.cast(beta[batch_idx, 0, head_idx], "float32")

                for ii in T.serial(k_chunk):
                    kk = k_begin + ii
                    decayed_state = decay * state[batch_idx, head_idx, kk, value_idx]
                    state_local[ii] = decayed_state
                    old_partial += decayed_state * k_shared[kk]

                old_value = old_partial + T.shfl_down(old_partial, 1, width=lane_group)
                if k_rank == 0:
                    value_new = beta_value * (
                        T.cast(v[batch_idx, 0, head_idx, value_idx], "float32") - old_value
                    )
                value_new = T.shfl_sync(value_new, value_lane * lane_group, width=lane_group)

                for ii in T.serial(k_chunk):
                    kk = k_begin + ii
                    updated_state = state_local[ii] + k_shared[kk] * value_new
                    final_state[batch_idx, head_idx, kk, value_idx] = updated_state
                    out_partial += updated_state * q_shared[kk]

                out_value = out_partial + T.shfl_down(out_partial, 1, width=lane_group)
                if k_rank == 0:
                    o[batch_idx, 0, head_idx, value_idx] = T.cast(out_value, dtype)

        return gated_deltanet_dense_decode_sm90

    return _decode


class GatedDeltaNetDenseDecodeFwdKernel(Kernel):
    """SM90 FP16/BF16 decode with FP32 recurrent state.

    One warp owns a 16-column state tile. Two lanes reduce the K dimension for
    each output column, keeping the decayed state slice in registers so the
    state update and output projection reuse the same load.
    """

    supported_archs = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        dim: int,
        scale: float,
        dtype: torch.dtype,
        *,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.dim = dim
        self.scale = scale
        self.dtype = dtype
        self.init_config()
        self._kernel_fn = _gated_deltanet_dense_decode_sm90_tl(
            batch,
            heads,
            dim,
            scale,
            self.dtype_str,
            self.config["v_tile"],
            self.config["lane_group"],
            self.config["maxrregcount"],
        )(self.config["threads"])

    @property
    def default_config(self) -> dict:
        return {
            "threads": 32,
            "v_tile": 16,
            "lane_group": 2,
            "maxrregcount": 146,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del cu_seqlens, cu_seqlens_cpu, A_log, dt_bias
        if initial_state is None:
            raise ValueError("Gated DeltaNet decode requires initial_state")
        self._require_cuda(q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state)
        return self._kernel_fn(q, k, v, g, beta, initial_state)
