"""Test GroupQueryAttentionDecodePagedWithKVCacheOp (paged GQA decode with dynamic KV cache)."""

import math
import sys

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from top.ops import GroupQueryAttentionDecodePagedWithKVCacheOp


def _torch_ref_gqa_decode_paged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    real_seqlen_kv: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
    heads: int,
    groups: int,
) -> torch.Tensor:
    """Reference: reassemble paged K/V to logical layout per batch, then GQA (expand to heads) + SDPA."""
    batch, _, dim = q.shape
    seqlen_kv, _, _ = k.shape
    kv_group_num = heads // groups
    out_list = []
    for i_b in range(batch):
        q_b = q[i_b:i_b + 1, :, :]
        k_logical = torch.zeros(seqlen_kv, groups, dim, dtype=q.dtype, device=q.device)
        v_logical = torch.zeros(seqlen_kv, groups, dim, dtype=q.dtype, device=q.device)
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
        group_id = torch.arange(heads, dtype=torch.long, device=q.device) // kv_group_num
        k_bhsd = k_logical[:, group_id, :].unsqueeze(0).transpose(1, 2)
        v_bhsd = v_logical[:, group_id, :].unsqueeze(0).transpose(1, 2)
        q_bhsd = q_b.unsqueeze(2)
        with sdpa_kernel(SDPBackend.MATH):
            out_b = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
        out_b = out_b.squeeze(2)
        out_list.append(out_b)
    return torch.cat(out_list, dim=0)


@pytest.mark.parametrize(
    ("batch", "heads", "groups", "seqlen_kv", "dim", "page_size", "dtype", "tune"),
    [
        (1, 16, 8, 512, 128, 128, torch.float16, False),
        (2, 8, 4, 1024, 64, 256, torch.float16, False),
        (1, 32, 8, 256, 128, 64, torch.float16, False),
        (1, 8, 4, 1024, 64, 256, torch.float16, False),
        (2, 16, 8, 512, 128, 128, torch.float16, False),
        (1, 16, 4, 2048, 128, 512, torch.float16, False),
        (1, 32, 16, 512, 64, 128, torch.float16, False),
    ],
)
def test_gqa_decode_paged_op(
    batch: int,
    heads: int,
    groups: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    num_pages = seqlen_kv // page_size
    real_seqlen_kv = torch.randint(
        page_size, seqlen_kv + 1, (batch,), dtype=torch.int32, device="cuda")
    real_seqlen_kv = (real_seqlen_kv // page_size) * page_size
    real_seqlen_kv[0] = min(real_seqlen_kv[0].item(), seqlen_kv)

    q = torch.randn(batch, heads, dim, dtype=dtype, device="cuda")
    k = torch.randn(seqlen_kv, groups, dim, dtype=dtype, device="cuda")
    v = torch.randn(seqlen_kv, groups, dim, dtype=dtype, device="cuda")
    block_table = torch.arange(
        num_pages, dtype=torch.int32, device="cuda").unsqueeze(0).expand(batch, -1)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    block_table = block_table.contiguous()
    real_seqlen_kv = real_seqlen_kv.contiguous()

    op = GroupQueryAttentionDecodePagedWithKVCacheOp(
        batch=batch,
        heads=heads,
        groups=groups,
        seqlen_kv=seqlen_kv,
        dim=dim,
        page_size=page_size,
        dtype=dtype,
        tune=tune,
    )
    output = op(q, k, v, real_seqlen_kv, block_table)
    output_ref = _torch_ref_gqa_decode_paged(q, k, v, real_seqlen_kv, block_table, page_size, heads,
                                             groups)

    max_diff = (output - output_ref).abs().max().item()
    assert max_diff < 0.001, (
        f"max diff {max_diff} too large (real_seqlen_kv={real_seqlen_kv.tolist()})")
    cos_sim = F.cosine_similarity(
        output.reshape(batch, -1), output_ref.reshape(batch, -1), dim=-1, eps=1e-8)
    assert cos_sim.min() > 0.99, f"cosine similarity {cos_sim.min().item()} too low"


if __name__ == "__main__":
    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
