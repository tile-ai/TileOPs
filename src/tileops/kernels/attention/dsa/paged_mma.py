"""SM89 paged DSA over the serving FP8 latent-cache byte layout."""

import functools
from typing import ClassVar

import tilelang
import tilelang.language as T

from tileops.kernels.attention.dsa.paged import DSAPagedKernelBase
from tileops.kernels.attention.online_softmax import (
    make_online_softmax_with_mask_guard,
    make_rescale,
)
from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Kernel

H_TILE, K_TILE, LATENT, ROPE = 16, 32, 512, 64
DIM, ROW_BYTES, THREADS, GROUP = 576, 656, 128, 128
SCALE_START, ROPE_START = 512, 528


@T.macro
def load_rows(Cache, Rows, Key, Value):
    for i, j in T.Parallel(K_TILE, LATENT):
        row = T.cast(T.max(Rows[i], 0), "int64")
        col = SCALE_START + (j // GROUP) * 4
        bits = (
            T.cast(Cache[row, col], "uint32")
            | (T.cast(Cache[row, col + 1], "uint32") << 8)
            | (T.cast(Cache[row, col + 2], "uint32") << 16)
            | (T.cast(Cache[row, col + 3], "uint32") << 24)
        )
        scale = T.reinterpret(bits, "float32")
        code = T.reinterpret(Cache[row, j], "float8_e4m3fn")
        value = T.if_then_else(Rows[i] >= 0, T.cast(code, "float32") * scale, 0)
        Key[i, j] = value
        Value[i, j] = value
    for i, j in T.Parallel(K_TILE, ROPE):
        row = T.cast(T.max(Rows[i], 0), "int64")
        col = ROPE_START + 2 * j
        bits = T.cast(Cache[row, col], "uint16") | (T.cast(Cache[row, col + 1], "uint16") << 8)
        Key[i, LATENT + j] = T.if_then_else(Rows[i] >= 0, T.reinterpret(bits, "bfloat16"), 0)


@T.macro
def locate_rows(Indices, Table, Lengths, Rows, token, chunk, seq_len, page_size):
    for i in T.Parallel(K_TILE):
        pos = Indices[token, chunk * K_TILE + i]
        live = (pos >= 0) & (pos < Lengths[token // seq_len])
        safe = T.if_then_else(live, pos, 0)
        Rows[i] = T.if_then_else(
            live, Table[token // seq_len, safe // page_size] * page_size + safe % page_size, -1
        )


@T.macro
def mask_scores(Scores, Rows, scale):
    for i, j in T.Parallel(H_TILE, K_TILE):
        Scores[i, j] = T.if_then_else(Rows[j] >= 0, Scores[i, j] * scale, -T.infinity("float32"))


@T.macro
def finish_output(Acc, Sum, Maximum, Output, Lse, token, head):
    for i, j in T.Parallel(H_TILE, LATENT):
        Output[token, head * H_TILE + i, j] = T.if_then_else(Sum[i] > 0, Acc[i, j] / Sum[i], 0)
    for i in T.Parallel(H_TILE):
        Lse[token, head * H_TILE + i] = T.if_then_else(
            Sum[i] > 0, (T.log2(Sum[i]) + Maximum[i]) / LOG2E, -T.infinity("float32")
        )


@functools.lru_cache(maxsize=32)
@tilelang.jit(out_idx=[5, 6])
def _dsa_paged_mma(*, batch, seq_len, heads, topk, page_size, scale):
    pages = T.dynamic("pages")
    width = T.dynamic("width")
    softmax = make_online_softmax_with_mask_guard(1.0, "float32", H_TILE, K_TILE)
    rescale = make_rescale(H_TILE, LATENT)

    @T.prim_func
    def kernel(
        Q: T.Tensor((batch * seq_len, heads, DIM), "bfloat16"),
        Cache: T.Tensor((pages * page_size, ROW_BYTES), "uint8"),
        Table: T.Tensor((batch, width), "int32"),
        Lengths: T.Tensor((batch,), "int32"),
        Indices: T.Tensor((batch * seq_len, topk), "int32"),
        Output: T.Tensor((batch * seq_len, heads, LATENT), "bfloat16"),
        Lse: T.Tensor((batch * seq_len, heads), "float32"),
    ):
        with T.Kernel(batch * seq_len, heads // H_TILE, threads=THREADS) as (token, head):
            q = T.alloc_shared((H_TILE, DIM), "bfloat16")
            k = T.alloc_shared((K_TILE, DIM), "bfloat16")
            v = T.alloc_shared((K_TILE, LATENT), "bfloat16")
            p = T.alloc_shared((H_TILE, K_TILE), "bfloat16")
            rows = T.alloc_shared((K_TILE,), "int32")
            scores = T.alloc_fragment((H_TILE, K_TILE), "float32")
            acc = T.alloc_fragment((H_TILE, LATENT), "float32")
            maximum = T.alloc_fragment((H_TILE,), "float32")
            previous = T.alloc_fragment((H_TILE,), "float32")
            alpha = T.alloc_fragment((H_TILE,), "float32")
            partial = T.alloc_fragment((H_TILE,), "float32")
            total = T.alloc_fragment((H_TILE,), "float32")
            T.copy(Q[token, head * H_TILE : (head + 1) * H_TILE, :], q)
            T.clear(acc)
            T.clear(total)
            T.fill(maximum, -T.infinity("float32"))
            for chunk in T.serial(topk // K_TILE):
                locate_rows(Indices, Table, Lengths, rows, token, chunk, seq_len, page_size)
                load_rows(Cache, rows, k, v)
                T.clear(scores)
                T.gemm(q, k, scores, transpose_B=True)
                mask_scores(scores, rows, scale * LOG2E)
                softmax(scores, maximum, previous, alpha, partial, total)
                T.copy(scores, p)
                rescale(acc, alpha)
                T.gemm(p, v, acc)
            finish_output(acc, total, maximum, Output, Lse, token, head)

    return kernel


class DSAPagedMMAKernel(DSAPagedKernelBase):
    """Gather and dequantize 32 selected cache rows per MMA iteration."""

    supported_archs: ClassVar[list[int]] = [89]
    cluster: ClassVar[int] = 1

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        index = call.device.index if call.device is not None else None
        Kernel.__init__(self, device_index=index)
        self.call = call
        self.kernel = _dsa_paged_mma(
            batch=call.batch,
            seq_len=call.seq_len,
            heads=call.heads,
            topk=call.topk,
            page_size=call.page_size,
            scale=DIM**-0.5 if call.sm_scale is None else call.sm_scale,
        )

    def forward(self, q, kv_cache, block_table, cache_seqlens, indices):
        self._require_devices(
            q=q,
            kv_cache=kv_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            indices=indices,
        )
        if kv_cache.numel() >= 2**32:
            raise ValueError("kv_cache must contain fewer than 2^32 bytes")
        call = self.call
        tokens = call.batch * call.seq_len
        out, lse = self.kernel(
            q.view(tokens, call.heads, DIM),
            kv_cache.view(-1, ROW_BYTES),
            block_table,
            cache_seqlens,
            indices.view(tokens, call.topk),
        )
        shape = (call.batch, call.seq_len, call.heads)
        return out.view(*shape, LATENT), lse.view(shape)
