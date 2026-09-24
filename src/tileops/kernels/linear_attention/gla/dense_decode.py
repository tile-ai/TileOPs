"""Hopper dense GLA decode with caller-owned FP32 recurrent state."""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Kernel

__all__ = ["GLADenseDecodeKernel"]


@functools.lru_cache(maxsize=32)
def _gla_dense_decode_tl(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    scale: float,
):
    threads = 64 if dim_v == 64 else 128

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def decode():
        @T.prim_func
        def main(
            q: T.Tensor([batch, 1, heads, dim_k], dtype),
            k: T.Tensor([batch, 1, heads, dim_k], dtype),
            v: T.Tensor([batch, 1, heads, dim_v], dtype),
            g: T.Tensor([batch, 1, heads, dim_k], dtype),
            initial_state: T.Tensor([batch, heads, dim_k, dim_v], "float32"),
            o: T.Tensor([batch, 1, heads, dim_v], dtype),
            final_state: T.Tensor([batch, heads, dim_k, dim_v], "float32"),
        ):
            with T.Kernel(batch, heads, threads=threads) as (bid, hid):
                tx = T.get_thread_binding()
                q_shared = T.alloc_shared([dim_k], dtype)
                k_shared = T.alloc_shared([dim_k], dtype)
                g_shared = T.alloc_shared([dim_k], dtype)
                v_shared = T.alloc_shared([dim_v], dtype)
                acc = T.alloc_var("float32", init=0.0)

                for i in T.Serial(T.ceildiv(dim_k, threads)):
                    idx = tx + i * threads
                    if idx < dim_k:
                        q_shared[idx] = q[bid, 0, hid, idx]
                        k_shared[idx] = k[bid, 0, hid, idx]
                        g_shared[idx] = g[bid, 0, hid, idx]
                if tx < dim_v:
                    v_shared[tx] = v[bid, 0, hid, tx]
                T.sync_threads()

                if tx < dim_v:
                    for kk in T.Serial(dim_k):
                        state = T.exp2(T.cast(g_shared[kk], "float32") * LOG2E) * initial_state[
                            bid, hid, kk, tx
                        ] + T.cast(k_shared[kk], "float32") * T.cast(v_shared[tx], "float32")
                        final_state[bid, hid, kk, tx] = state
                        acc += T.cast(q_shared[kk], "float32") * state
                    o[bid, 0, hid, tx] = T.cast(scale * acc, dtype)

        return main

    return decode()


class GLADenseDecodeKernel(Kernel):
    """Fuse one GLA recurrence step and output projection in one state pass."""

    supported_archs = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        scale: float,
        dtype: torch.dtype,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        if dim_k not in (64, 128) or dim_v not in (64, 128):
            raise ValueError("GLA dense decode requires K and V dimensions of 64 or 128")
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("GLA dense decode requires float16 or bfloat16 activations")
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.scale = scale
        self.dtype = dtype
        self._kernel = _gla_dense_decode_tl(
            batch,
            heads,
            dim_k,
            dim_v,
            self.dtype_to_str(dtype),
            scale,
        )

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
            raise ValueError("GLA dense decode does not support packed varlen inputs")
        state = (
            torch.zeros(
                self.batch,
                self.heads,
                self.dim_k,
                self.dim_v,
                dtype=torch.float32,
                device=q.device,
            )
            if initial_state is None
            else initial_state
        )
        return self._kernel(q, k, v, g, state)
