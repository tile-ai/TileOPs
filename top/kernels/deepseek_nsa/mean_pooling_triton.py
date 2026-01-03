# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from .utils import input_guard, prepare_chunk_indices 


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BT']
)
@triton.jit(do_not_specialize=['T'])
def mean_pooling_fwd_kernel(
    x,
    o,
    offsets,
    indices,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NT: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_tg = i_t
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    p_x = tl.make_block_ptr(x + (bos * H + i_h) * D, (T, D), (H*D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    p_o = tl.make_block_ptr(o + (i_tg * H + i_h) * D, (D,), (1,), (i_d * BD,), (BD,), (0,))
    # [BT, BD]
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    # [BD]
    b_o = tl.sum(b_x, axis=0) / min(BT, T - i_t * BT)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def mean_pooling_fwd(
    x: torch.Tensor,
    chunk_size: int,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None
) -> torch.Tensor:
    B, T, H, D = x.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)

    o = x.new_empty(B, NT, H, D)
    def grid(meta): return (triton.cdiv(D, meta['BD']), NT, B * H)
    mean_pooling_fwd_kernel[grid](
        x,
        o,
        offsets,
        indices,
        T=T,
        H=H,
        D=D,
        BT=BT,
        NT=NT,
    )
    return o


class MeanPoolingFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        chunk_size: int,
        offsets: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        # 2-d indices denoting the offsets of chunks in each sequence
        # for example, if the passed `offsets` is [0, 100, 356] and `chunk_size` is 64,
        # then there are 2 and 4 chunks in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = prepare_chunk_indices(offsets, chunk_size) if offsets is not None else None
        o = mean_pooling_fwd(x, chunk_size, offsets, indices)
        ctx.batch_size = x.shape[0]
        ctx.seq_len = x.shape[1]
        ctx.chunk_size = chunk_size
        ctx.offsets = offsets
        ctx.indices = indices
        return o


def mean_pooling(
    x: torch.Tensor,
    chunk_size: int,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    if head_first:
        x = x.transpose(1, 2)
    if cu_seqlens is not None:
        if x.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {x.shape[0]} when using `cu_seqlens`."
                             f"Please flatten variable-length inputs before processing.")
    o = MeanPoolingFunction.apply(x, chunk_size, cu_seqlens)
    if head_first:
        o = o.transpose(1, 2)
    return o


def test_mean_pooling():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 2
    seq_len = 1024
    num_heads = 4
    head_dim = 64
    chunk_size = 32
    
    x = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True)
    output = mean_pooling(x, chunk_size=chunk_size, head_first=False)
    
    x_hf = x.permute(0, 2, 1, 3).contiguous().requires_grad_(True)  # (B, H, T, D)
    output_hf = mean_pooling(x_hf, chunk_size=chunk_size, head_first=True)
    out1 = output_hf.permute(0, 2, 1, 3).contiguous()
    out2 = output.contiguous()

    print("max abs diff:", (out1 - out2).abs().max().item())
    print("mean abs diff:", (out1 - out2).abs().mean().item())
    assert torch.allclose(output_hf.permute(0, 2, 1, 3).contiguous().clone(), output.contiguous().clone(), atol=1e-4)


if __name__ == "__main__":
    test_mean_pooling()