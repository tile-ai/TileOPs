"""SM89 paged DSA over the serving FP8 latent-cache byte layout."""

import functools
from typing import ClassVar

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.dsa.paged import DSAPagedKernelBase, _dsa_paged_combine_kernel
from tileops.kernels.attention.online_softmax import (
    make_online_softmax_with_mask_guard,
    make_rescale,
)
from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import CompiledProgram, Kernel


@functools.lru_cache(maxsize=32)
@tilelang.jit(out_idx=[5, 6])
def _dsa_paged_mma(*, batch, seq_len, heads, topk, page_size, sm_scale, split=1):
    block_h = 16  # query heads per CTA: one m16 MMA row block
    block_k = 32  # selected cache rows per iteration
    # A cache row is 512 FP8 latent codes, one float32 scale per 128 codes from byte 512,
    # then 64 bfloat16 rope values from byte 528: 656 bytes.
    latent, rope, group = 512, 64, 128
    scale_start, rope_start, row_bytes = 512, 528, 656
    dim = latent + rope
    scale = (dim**-0.5 if sm_scale is None else sm_scale) * LOG2E
    pages = T.dynamic("pages")
    width = T.dynamic("width")
    softmax = make_online_softmax_with_mask_guard(1.0, "float32", block_h, block_k)
    rescale = make_rescale(block_h, latent)
    output_dtype = "bfloat16" if split == 1 else "float32"

    @T.macro
    def load_rows(Cache, Rows, Key):
        scales = T.alloc_shared((block_k, latent // group), "float32")
        for i, g in T.Parallel(block_k, latent // group):
            row = T.cast(T.max(Rows[i], 0), "int64")
            col = scale_start + g * 4
            bits = (
                T.cast(Cache[row, col], "uint32")
                | (T.cast(Cache[row, col + 1], "uint32") << 8)
                | (T.cast(Cache[row, col + 2], "uint32") << 16)
                | (T.cast(Cache[row, col + 3], "uint32") << 24)
            )
            scales[i, g] = T.reinterpret(bits, "float32")
        for i, j in T.Parallel(block_k, latent):
            row = T.cast(T.max(Rows[i], 0), "int64")
            code = T.reinterpret(Cache[row, j], "float8_e4m3fn")
            Key[i, j] = T.if_then_else(
                Rows[i] >= 0, T.cast(code, "float32") * scales[i, j // group], 0
            )
        for i, j in T.Parallel(block_k, rope):
            row = T.cast(T.max(Rows[i], 0), "int64")
            col = rope_start + 2 * j
            bits = T.cast(Cache[row, col], "uint16") | (T.cast(Cache[row, col + 1], "uint16") << 8)
            Key[i, latent + j] = T.if_then_else(Rows[i] >= 0, T.reinterpret(bits, "bfloat16"), 0)

    @T.macro
    def locate_rows(Indices, Table, Lengths, Rows, token, chunk):
        for i in T.Parallel(block_k):
            pos = Indices[token, chunk * block_k + i]
            live = (pos >= 0) & (pos < Lengths[token // seq_len])
            safe = T.if_then_else(live, pos, 0)
            Rows[i] = T.if_then_else(
                live, Table[token // seq_len, safe // page_size] * page_size + safe % page_size, -1
            )

    @T.prim_func
    def kernel(
        Q: T.Tensor((batch * seq_len, heads, dim), "bfloat16"),
        Cache: T.Tensor((pages * page_size, row_bytes), "uint8"),
        Table: T.Tensor((batch, width), "int32"),
        Lengths: T.Tensor((batch,), "int32"),
        Indices: T.Tensor((batch * seq_len, topk), "int32"),
        Output: T.Tensor((split, batch * seq_len, heads, latent), output_dtype),
        Lse: T.Tensor((split, batch * seq_len, heads), "float32"),
    ):
        with T.Kernel(batch * seq_len, heads // block_h, split, threads=128) as (token, head, part):
            q = T.alloc_shared((block_h, dim), "bfloat16")
            k = T.alloc_shared((block_k, dim), "bfloat16")
            p = T.alloc_shared((block_h, block_k), "bfloat16")
            rows = T.alloc_shared((block_k,), "int32")
            scores = T.alloc_fragment((block_h, block_k), "float32")
            acc = T.alloc_fragment((block_h, latent), "float32")
            maximum = T.alloc_fragment((block_h,), "float32")
            previous = T.alloc_fragment((block_h,), "float32")
            alpha = T.alloc_fragment((block_h,), "float32")
            partial = T.alloc_fragment((block_h,), "float32")
            total = T.alloc_fragment((block_h,), "float32")
            T.copy(Q[token, head * block_h : (head + 1) * block_h, :], q)
            T.clear(acc)
            T.clear(total)
            T.fill(maximum, -T.infinity("float32"))
            for step in T.serial(topk // (block_k * split)):
                chunk = part * (topk // (block_k * split)) + step
                locate_rows(Indices, Table, Lengths, rows, token, chunk)
                load_rows(Cache, rows, k)
                T.clear(scores)
                T.gemm(q, k, scores, transpose_B=True)
                for i, j in T.Parallel(block_h, block_k):
                    scores[i, j] = T.if_then_else(
                        rows[j] >= 0, scores[i, j] * scale, -T.infinity("float32")
                    )
                softmax(scores, maximum, previous, alpha, partial, total)
                T.copy(scores, p)
                rescale(acc, alpha)
                T.gemm(p, k[:, :latent], acc)
            for i, j in T.Parallel(block_h, latent):
                Output[part, token, head * block_h + i, j] = T.if_then_else(
                    total[i] > 0, acc[i, j] / total[i], 0
                )
            for i in T.Parallel(block_h):
                Lse[part, token, head * block_h + i] = T.if_then_else(
                    total[i] > 0,
                    (T.log2(total[i]) + maximum[i]) / LOG2E,
                    -T.infinity("float32"),
                )

    return kernel


class DSAPagedMMAKernel(DSAPagedKernelBase):
    """Gather and dequantize 32 selected cache rows per MMA iteration."""

    supported_archs: ClassVar[list[int]] = [89]
    cluster: ClassVar[int] = 1

    @classmethod
    def entry_for(cls, call):
        split = cls.split_for(
            call.batch * call.seq_len * (call.heads // 16), call.topk, call.sm_count
        )
        return (call, split), lambda: cls(call, split)

    def __init__(self, call, split=1):
        index = call.device.index if call.device is not None else None
        Kernel.__init__(self, device_index=index)
        self.call = call
        self.split = split
        self.program(self._compile)

    def _compile(self, cfg: dict):
        """The attention program and, over several key splits, the merge; no configuration."""
        call, split = self.call, self.split
        self.kernel = _dsa_paged_mma(
            batch=call.batch,
            seq_len=call.seq_len,
            heads=call.heads,
            topk=call.topk,
            page_size=call.page_size,
            sm_scale=call.sm_scale,
            split=split,
        )
        combine = (
            CompiledProgram(
                _dsa_paged_combine_kernel(
                    call.batch * call.seq_len, call.heads, split, target="sm_89"
                )
            )
            if split > 1
            else None
        )
        return CompiledProgram(self.kernel), combine

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
        main, combine = self.program(self._compile)
        out, lse = main(
            q.view(tokens, call.heads, q.shape[-1]),
            kv_cache.view(-1, kv_cache.shape[-1]),
            block_table,
            cache_seqlens,
            indices.view(tokens, call.topk),
        )
        shape = (call.batch, call.seq_len, call.heads)
        if self.split > 1:
            output = torch.empty(
                (tokens, call.heads, out.shape[-1]), dtype=q.dtype, device=q.device
            )
            final = torch.empty((tokens, call.heads), dtype=torch.float32, device=q.device)
            combine(out, lse, output, final)
            out, lse = output, final
        return out.view(*shape, out.shape[-1]), lse.view(shape)
