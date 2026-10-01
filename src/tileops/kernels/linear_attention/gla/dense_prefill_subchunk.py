"""GLA prefill with tiled, gate-aware intra-chunk products."""

from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import BF16_SPLIT_EXP2_SPAN, LOG2E
from tileops.kernels.kernel_base import Entry
from tileops.kernels.linear_attention.gla.call_spec import (
    GLAInferenceCallSpec,
    GLAInferenceFwdInterface,
    build_entry,
    serves_dense,
)
from tileops.kernels.linear_attention.gla.gla_fwd import (
    GLAChunkedFwdKernel,
    gla_fwd_h_kernel,
    gla_precompute_g_kernel,
)

__all__ = ["GLADensePrefillSubchunkKernel"]


def gla_fwd_a_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    chunk_size: int,
    scale: float,
    dtype: str,
    gate_dtype: str = "float32",
):
    """Contract 16-token blocks on tensor cores, and keep the diagonal exact where it must be.

    A block below the diagonal anchors both exponents on the query block's first row, where
    the causal pairing makes both nonpositive, so neither factor can overflow. The diagonal
    block's key exponent is positive and the two factors together span the block's whole gate
    decay; it takes the same product where that span is representable and the exact per-pair
    form where it is not.
    """
    num_chunks = seq_len // chunk_size
    block_c = 16
    num_subchunks = chunk_size // block_c

    @tilelang.jit(out_idx=[-1])
    def _a_func(threads: int = 128):
        @T.prim_func
        def _main(
            q: T.Tensor([batch, seq_len, heads, dim_k], dtype),
            k: T.Tensor([batch, seq_len, heads, dim_k], dtype),
            g_cumsum: T.Tensor([batch, seq_len, heads, dim_k], gate_dtype),
            a: T.Tensor([batch, seq_len, heads, chunk_size], dtype),
        ):
            with T.Kernel(
                batch * heads * num_chunks * num_subchunks * num_subchunks,
                threads=threads,
            ) as bx:
                pair = bx % (num_subchunks * num_subchunks)
                bi = pair // num_subchunks
                bj = pair % num_subchunks
                chunk_idx = bx // (num_subchunks * num_subchunks)
                i_b = chunk_idx // (heads * num_chunks)
                i_h = (chunk_idx // num_chunks) % heads
                i_c = chunk_idx % num_chunks
                start = i_c * chunk_size

                q_s = T.alloc_shared([block_c, dim_k], dtype)
                k_s = T.alloc_shared([block_c, dim_k], dtype)
                g_q = T.alloc_shared([block_c, dim_k], "float32")
                g_k = T.alloc_shared([block_c, dim_k], "float32")
                a_s = T.alloc_shared([block_c, block_c], dtype)
                # The gate-scaled operands are staged in bfloat16 whatever the activations
                # are: it is the wider exponent of the two, and the diagonal block needs every
                # bit of it. Where even bfloat16 is too narrow the block takes the exact form.
                q_gated = T.alloc_shared([block_c, dim_k], "bfloat16")
                k_gated = T.alloc_shared([block_c, dim_k], "bfloat16")
                spans = T.alloc_fragment([dim_k], "float32")
                span = T.alloc_fragment([1], "float32")

                # Only the diagonal block reads a span, and it is the one that measures it.
                span[0] = 0.0
                if bj <= bi:
                    T.copy(
                        q[i_b, start + bi * block_c : start + (bi + 1) * block_c, i_h, :],
                        q_s,
                        disable_tma=True,
                    )
                    T.copy(
                        k[i_b, start + bj * block_c : start + (bj + 1) * block_c, i_h, :],
                        k_s,
                        disable_tma=True,
                    )
                    T.copy(
                        g_cumsum[i_b, start + bi * block_c : start + (bi + 1) * block_c, i_h, :],
                        g_q,
                        disable_tma=True,
                    )
                    T.copy(
                        g_cumsum[i_b, start + bj * block_c : start + (bj + 1) * block_c, i_h, :],
                        g_k,
                        disable_tma=True,
                    )

                    if bj == bi:
                        for d in T.Parallel(dim_k):
                            spans[d] = (g_q[0, d] - g_k[block_c - 1, d]) * LOG2E
                        T.reduce_max(spans, span, dim=0)

                if bj < bi or (bj == bi and span[0] < BF16_SPLIT_EXP2_SPAN):
                    # Below the diagonal the query block's first row makes both exponents
                    # nonpositive, so neither factor can overflow.
                    for i, d in T.Parallel(block_c, dim_k):
                        q_gated[i, d] = T.cast(
                            T.cast(q_s[i, d], "float32")
                            * T.exp2((g_q[i, d] - g_q[0, d]) * LOG2E)
                            * scale,
                            "bfloat16",
                        )
                    for j, d in T.Parallel(block_c, dim_k):
                        k_gated[j, d] = T.cast(
                            T.cast(k_s[j, d], "float32") * T.exp2((g_q[0, d] - g_k[j, d]) * LOG2E),
                            "bfloat16",
                        )
                    product = T.alloc_fragment([block_c, block_c], "float32")
                    T.fill(product, 0.0)
                    T.gemm(q_gated, k_gated, product, transpose_B=True)
                    for i, j in T.Parallel(block_c, block_c):
                        a_s[i, j] = T.cast(
                            T.if_then_else(bj < bi or j <= i, product[i, j], 0.0), dtype
                        )
                elif bj == bi:
                    products = T.alloc_fragment([block_c, dim_k], "float32")
                    sums = T.alloc_fragment([block_c], "float32")
                    for j in T.Serial(block_c):
                        for i, d in T.Parallel(block_c, dim_k):
                            products[i, d] = (
                                T.cast(q_s[i, d], "float32")
                                * T.cast(k_s[j, d], "float32")
                                * T.exp2((g_q[i, d] - g_k[j, d]) * LOG2E)
                            )
                        T.reduce_sum(products, sums, dim=1)
                        for i in T.Parallel(block_c):
                            a_s[i, j] = T.cast(T.if_then_else(j <= i, sums[i] * scale, 0.0), dtype)
                else:
                    for i, j in T.Parallel(block_c, block_c):
                        a_s[i, j] = 0.0

                T.copy(
                    a_s,
                    a[
                        i_b,
                        start + bi * block_c : start + (bi + 1) * block_c,
                        i_h,
                        bj * block_c : (bj + 1) * block_c,
                    ],
                )

        return _main

    return _a_func


def _gla_fwd_o_from_a_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
):
    """Form the recurrent and intra-chunk outputs with two tensor-core GEMMs."""
    num_chunks = seq_len // chunk_size

    @tilelang.jit(out_idx=[-1])
    def _o_func(threads: int = 128):
        @T.prim_func
        def _main(
            q: T.Tensor([batch, seq_len, heads, dim_k], dtype),
            v: T.Tensor([batch, seq_len, heads, dim_v], dtype),
            g_cumsum: T.Tensor([batch, seq_len, heads, dim_k], "float32"),
            h: T.Tensor([batch, num_chunks + 1, heads, dim_k, dim_v], "float32"),
            a: T.Tensor([batch, seq_len, heads, chunk_size], dtype),
            o: T.Tensor([batch, seq_len, heads, dim_v], dtype),
        ):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = (bx // num_chunks) % heads
                i_c = bx % num_chunks
                start = i_c * chunk_size

                q_s = T.alloc_shared([chunk_size, dim_k], dtype)
                v_s = T.alloc_shared([chunk_size, dim_v], dtype)
                g_s = T.alloc_shared([chunk_size, dim_k], "float32")
                a_s = T.alloc_shared([chunk_size, chunk_size], dtype)
                q_gated = T.alloc_shared([chunk_size, dim_k], dtype)
                h_s = T.alloc_shared([dim_k, dim_v], dtype)

                T.copy(q[i_b, start : start + chunk_size, i_h, :], q_s, disable_tma=True)
                T.copy(v[i_b, start : start + chunk_size, i_h, :], v_s, disable_tma=True)
                T.copy(
                    g_cumsum[i_b, start : start + chunk_size, i_h, :],
                    g_s,
                    disable_tma=True,
                )
                T.copy(a[i_b, start : start + chunk_size, i_h, :], a_s, disable_tma=True)
                for d, j in T.Parallel(dim_k, dim_v):
                    h_s[d, j] = T.cast(h[i_b, i_c, i_h, d, j], dtype)
                for i, d in T.Parallel(chunk_size, dim_k):
                    q_gated[i, d] = T.cast(
                        T.cast(q_s[i, d], "float32") * T.exp2(g_s[i, d] * LOG2E),
                        dtype,
                    )

                acc = T.alloc_fragment([chunk_size, dim_v], "float32")
                T.fill(acc, 0.0)
                T.gemm(q_gated, h_s, acc)
                for i, j in T.Parallel(chunk_size, dim_v):
                    acc[i, j] *= scale
                T.gemm(a_s, v_s, acc)
                for i, j in T.Parallel(chunk_size, dim_v):
                    o[i_b, start + i, i_h, j] = T.cast(acc[i, j], dtype)

        return _main

    return _o_func


class GLADensePrefillSubchunkKernel(GLAChunkedFwdKernel, GLAInferenceFwdInterface):
    """Retain the proven state pass while replacing the costly output pass."""

    supported_archs = [90]
    general = True

    @classmethod
    def applies(cls, call: GLAInferenceCallSpec) -> bool:
        return serves_dense(call) and call.seq_len >= 64 and call.seq_len % 64 == 0

    @classmethod
    def entry_for(cls, call: GLAInferenceCallSpec) -> Entry:
        return build_entry(
            cls,
            call,
            batch=call.batch,
            seq_len=call.seq_len,
            heads=call.heads,
            dim_k=call.dim_k,
            dim_v=call.dim_v,
        )

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

    def _build_kernels(self, config: dict) -> None:
        ns = config.get("num_stages", 3)
        thr_seq = config.get("threads_seq", config.get("threads", 64))
        thr_par = config.get("threads_par", config.get("threads", 64))
        num_vp = config.get("num_v_partitions", 4)
        num_kp = config.get("num_k_partitions", 2)
        self._g_fn = gla_precompute_g_kernel(
            self.batch, self.seq_len, self.heads, self.dim_k, 64, self.dtype_name
        )(ns, thr_par)
        self._h_fn = gla_fwd_h_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            64,
            self.dtype_name,
            num_v_partitions=num_vp,
            num_k_partitions=num_kp,
        )(ns, thr_seq)
        self._a_fn = gla_fwd_a_kernel(
            self.batch, self.seq_len, self.heads, self.dim_k, 64, self.scale, self.dtype_name
        )(thr_par)
        self._o_fn = _gla_fwd_o_from_a_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim_k,
            self.dim_v,
            64,
            self.scale,
            self.dtype_name,
        )(thr_par)

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
        gate = self._g_fn(g)
        h = self._h_fn(k, v, gate, state)
        a = self._a_fn(q, k, gate)
        o = self._o_fn(q, v, gate, h, a)
        return o, h[:, -1]
