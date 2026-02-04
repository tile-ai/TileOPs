"""Test MultiHeadAttentionDecodePagedWithKVCacheOp (paged MHA decode with dynamic KV cache)."""

import math

import pytest
import torch
import torch.nn.functional as F

from top.ops import MultiHeadAttentionDecodePagedWithKVCacheOp


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(12345)


def _torch_ref_mha_decode_paged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    real_seqlen_kv: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Reference: reassemble paged K/V to logical layout per batch, then SDPA. Q [B,S_q,H,D]; K,V [S_kv,H,D]."""
    batch, seqlen_q, heads, dim = q.shape
    seqlen_kv = k.shape[0]
    out_list = []
    for i_b in range(batch):
        q_b = q[i_b:i_b + 1, :, :, :]
        k_logical = torch.zeros(seqlen_kv, heads, dim, dtype=q.dtype, device=q.device)
        v_logical = torch.zeros(seqlen_kv, heads, dim, dtype=q.dtype, device=q.device)
        num_pages = math.ceil(real_seqlen_kv[i_b].item() / page_size)
        for i_paged in range(num_pages):
            start_pos = block_table[i_b, i_paged].item() * page_size
            end_pos = min(start_pos + page_size, seqlen_kv)
            page_len = end_pos - start_pos
            k_logical[i_paged * page_size:i_paged * page_size +
                      page_len, :, :] = k[start_pos:end_pos, :, :]
            v_logical[i_paged * page_size:i_paged * page_size +
                      page_len, :, :] = v[start_pos:end_pos, :, :]
        k_logical = k_logical[:real_seqlen_kv[i_b].item(), :, :]
        v_logical = v_logical[:real_seqlen_kv[i_b].item(), :, :]
        k_b = k_logical.unsqueeze(0)
        v_b = v_logical.unsqueeze(0)
        q_bhsd = q_b.transpose(1, 2)
        k_bhsd = k_b.transpose(1, 2)
        v_bhsd = v_b.transpose(1, 2)
        out_b = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
        out_b = out_b.transpose(1, 2).contiguous()
        out_list.append(out_b)
    return torch.cat(out_list, dim=0)


@pytest.mark.parametrize(
    ("batch", "heads", "seqlen_q", "seqlen_kv", "dim", "page_size", "is_causal", "dtype", "tune"),
    [
        (1, 16, 1, 512, 128, 128, False, torch.float16, False),
        (1, 8, 1, 1024, 64, 256, False, torch.float16, False),
        (2, 8, 1, 1024, 64, 256, False, torch.float16, False),
        (1, 8, 1, 512, 64, 256, False, torch.float16, False),
    ],
)
def test_mha_decode_paged_op(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    is_causal: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    num_pages = seqlen_kv // page_size
    real_seqlen_kv = torch.randint(
        page_size, seqlen_kv + 1, (batch,), dtype=torch.int32, device="cuda")
    real_seqlen_kv = (real_seqlen_kv // page_size) * page_size
    real_seqlen_kv[0] = min(real_seqlen_kv[0].item(), seqlen_kv)
    real_seqlen_kv = torch.ones((batch,), dtype=torch.int32, device="cuda") * seqlen_kv
    q = torch.randn(batch, seqlen_q, heads, dim, dtype=dtype, device="cuda")
    k = torch.randn(seqlen_kv, heads, dim, dtype=dtype, device="cuda")
    v = torch.randn(seqlen_kv, heads, dim, dtype=dtype, device="cuda")
    # Identity block_table: logical page i -> physical page i (contiguous layout), so ref and kernel match.
    block_table = torch.arange(
        num_pages, dtype=torch.int32, device="cuda").unsqueeze(0).expand(batch, -1)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    block_table = block_table.contiguous()
    real_seqlen_kv = real_seqlen_kv.contiguous()

    op = MultiHeadAttentionDecodePagedWithKVCacheOp(
        batch=batch,
        heads=heads,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        dim=dim,
        page_size=page_size,
        is_causal=is_causal,
        dtype=dtype,
        tune=tune,
    )
    output = op(q, k, v, real_seqlen_kv, block_table)
    output_ref = _torch_ref_mha_decode_paged(q, k, v, real_seqlen_kv, block_table, page_size)

    max_diff = (output - output_ref).abs().max().item()
    assert max_diff < 0.001, f"max diff {max_diff} too large (real_seqlen_kv={real_seqlen_kv.tolist()})"
    cos_sim = F.cosine_similarity(
        output.reshape(batch, -1), output_ref.reshape(batch, -1), dim=-1, eps=1e-8)
    assert cos_sim.min() > 0.99, f"cosine similarity {cos_sim.min().item()} too low"
    torch.cuda.empty_cache()
