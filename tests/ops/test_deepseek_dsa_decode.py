import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import DeepSeekSparseAttentionDecodeWithKVCacheFwdOp
from workloads.attention.deepseek import DsaDecodeWorkload


class DsaDecodeTest(DsaDecodeWorkload, TestBase):
    pass


class DsaDecodeFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, "
            "q_start_index_s, sm_scale, dtype, tune",
            [
                pytest.param(
                    1,
                    128,
                    1024,
                    2048,
                    512,
                    64,
                    2048,
                    1,
                    1,
                    1024,
                    None,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    1,
                    128,
                    1024,
                    2048,
                    512,
                    64,
                    2048,
                    1,
                    1,
                    1024,
                    None,
                    torch.float16,
                    True,
                    marks=pytest.mark.full,
                    id="full-fp16-tuned",
                ),
            ],
        ),
    ]


@DsaDecodeFixture
def test_sparse_mla_decode(
    batch: int,
    heads: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    dim_tail: int,
    topk: int,
    stride_kv: int,
    heads_kv: int,
    q_start_index_s: int,
    sm_scale: float,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = DsaDecodeTest(
        batch,
        heads,
        seq_len_q,
        seq_len_kv,
        dim,
        dim_tail,
        topk,
        stride_kv,
        heads_kv,
        q_start_index_s,
        sm_scale=sm_scale,
        dtype=dtype,
    )
    op = DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(
        dim_tail, stride_kv, q_start_index_s, sm_scale=sm_scale, tune=tune
    )
    test.check(op, *test.gen_inputs(), atol=3e-4, rtol=1e-5)


def _padded_topk_indices(
    batch: int,
    seq_len: int,
    heads_kv: int,
    topk: int,
    seq_len_kv: int,
    pad: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """A top-k list that fills half its slots and pads the rest with *pad*."""
    indices = torch.full((batch, seq_len, heads_kv, topk), pad, dtype=torch.int32, device="cuda")
    for b in range(batch):
        for t in range(seq_len):
            for h in range(heads_kv):
                selected = torch.randperm(seq_len_kv, generator=generator)[: topk // 2]
                indices[b, t, h, : topk // 2] = selected.to(torch.int32).cuda()
    return indices


@pytest.mark.smoke
def test_sparse_mla_decode_ignores_padded_topk_slots() -> None:
    """A slot no row fills must not reach kv, whatever value pads it.

    The cache here is shorter than the causal window, so the causal limit alone
    accepts a padding index; the kernel has to reject it on the kv extent.
    """
    batch, heads, seq_len, seq_len_kv = 1, 128, 64, 1024
    dim, dim_tail, topk, stride_kv, heads_kv, q_start = 512, 64, 128, 1, 1, 1024
    generator = torch.Generator().manual_seed(0)

    test = DsaDecodeTest(
        batch, heads, seq_len, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, q_start
    )
    op = DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(dim_tail, stride_kv, q_start)
    q, kv, _ = test.gen_inputs()

    # seq_len_kv is the padding the workloads write and the reference reads.
    in_range_pad = _padded_topk_indices(
        batch, seq_len, heads_kv, topk, seq_len_kv, seq_len_kv, generator
    )
    # The reference sums in float32 and rounds once; one fp16 ulp here is 1e-3.
    test.check(op, q, kv, in_range_pad, atol=2e-3, rtol=2e-3)

    expected = op(q, kv, in_range_pad)
    assert torch.isfinite(expected).all(), "an in-range padding slot produced a non-finite output"
    for pad in (-1, 2**30):
        padded = in_range_pad.clone()
        padded[padded == seq_len_kv] = pad
        assert torch.equal(op(q, kv, padded), expected), f"padding with {pad} changed the output"

    # Masked rows carry no state from one launch into the next.
    assert torch.equal(op(q, kv, in_range_pad), expected)
