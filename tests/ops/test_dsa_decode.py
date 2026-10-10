import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops import DSADecodeWithKVCacheFwdOp
from workloads.attention.dsa import DSADecodeWorkload
from workloads.device import run_device


class DSADecodeTest(DSADecodeWorkload, TestBase):
    pass


class DSADecodeFixture(FixtureBase):
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


@DSADecodeFixture
def test_dsa_decode_decode(
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
    test = DSADecodeTest(
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
    op = DSADecodeWithKVCacheFwdOp(
        dim_tail, stride_kv, q_start_index_s, sm_scale=sm_scale, tune=tune
    )
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("dim_tail", "dtype"),
    [
        pytest.param(64, torch.bfloat16, id="bf16-tail"),
        pytest.param(0, torch.bfloat16, id="bf16-no-tail"),
        pytest.param(0, torch.float16, id="fp16-no-tail"),
    ],
)
def test_dsa_decode_decode_tail_and_dtype(dim_tail, dtype) -> None:
    """BF16 preserves the output dtype; a zero tail omits the extra QK contraction."""
    test = DSADecodeTest(1, 64, 7, 256, 512, dim_tail, 128, 1, 1, 256, dtype=dtype)
    op = DSADecodeWithKVCacheFwdOp(dim_tail, 1, 256)
    test.check(op, *test.gen_inputs())


@pytest.mark.sm89
@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_dsa_decode_refuses_what_99_kb_cannot_hold() -> None:
    """SM89 has no WGMMA and gives a block 99 KB of opt-in shared memory, too little for a
    head dim 2048 query tile, so the op refuses it before anything is built."""
    test = DSADecodeTest(1, 16, 1, 2048, 2048, 64, 128, 1, 1, 2048)
    op = DSADecodeWithKVCacheFwdOp(64, 1, 2048)
    with pytest.raises(ValueError, match="bytes of shared memory per block"):
        op(*test.gen_inputs())
    for interface in DSADecodeWithKVCacheFwdOp.interfaces:
        assert not op.built_kernels(interface)


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
    indices = torch.full(
        (batch, seq_len, heads_kv, topk), pad, dtype=torch.int32, device=run_device()
    )
    for b in range(batch):
        for t in range(seq_len):
            for h in range(heads_kv):
                selected = torch.randperm(seq_len_kv, generator=generator)[: topk // 2]
                indices[b, t, h, : topk // 2] = selected.to(torch.int32).to(run_device())
    return indices


@pytest.mark.smoke
def test_dsa_decode_decode_ignores_padded_topk_slots() -> None:
    """A slot no row fills must not reach kv, whatever value pads it.

    The cache here is shorter than the causal window, so the causal limit alone
    accepts a padding index; the kernel has to reject it on the kv extent.
    """
    batch, heads, seq_len, seq_len_kv = 1, 128, 64, 1024
    dim, dim_tail, topk, stride_kv, heads_kv, q_start = 512, 64, 128, 1, 1, 1024
    generator = torch.Generator().manual_seed(0)

    test = DSADecodeTest(
        batch, heads, seq_len, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, q_start
    )
    op = DSADecodeWithKVCacheFwdOp(dim_tail, stride_kv, q_start)
    q, kv, _ = test.gen_inputs()

    # seq_len_kv is the padding the workloads write and the reference reads.
    in_range_pad = _padded_topk_indices(
        batch, seq_len, heads_kv, topk, seq_len_kv, seq_len_kv, generator
    )
    # The reference sums in float32 and rounds once; one fp16 ulp here is 1e-3.
    test.check(op, q, kv, in_range_pad)

    expected = op(q, kv, in_range_pad)
    assert torch.isfinite(expected).all(), "an in-range padding slot produced a non-finite output"
    for pad in (-1, 2**30):
        padded = in_range_pad.clone()
        padded[padded == seq_len_kv] = pad
        assert torch.equal(op(q, kv, padded), expected), f"padding with {pad} changed the output"

    # Masked rows carry no state from one launch into the next: a launch that stages
    # NaN rows in every slot must not change the output.
    op(q, torch.full_like(kv, float("nan")), torch.zeros_like(in_range_pad))
    assert torch.equal(op(q, kv, in_range_pad), expected)


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.parametrize(
    ("dim", "dim_tail", "heads", "heads_kv", "stride_kv", "dtype"),
    [
        pytest.param(512, 0, 128, 2, 1, torch.float16, id="512-kv-groups"),
        pytest.param(512, 64, 64, 1, 2, torch.bfloat16, id="512-kv-stride"),
        pytest.param(256, 64, 64, 1, 1, torch.bfloat16, id="256"),
    ],
)
def test_dsa_decode_ignores_keys_past_the_causal_limit(
    dim, dim_tail, heads, heads_kv, stride_kv, dtype
) -> None:
    """A selected key past the causal limit, which stride_kv scales, carries no weight.

    Each row selects distinct keys from the whole cache, so most of them lie past the limit.
    """
    seq_len, seq_len_kv, topk, q_start = 33, 1024, 256, 256
    test = DSADecodeTest(
        1, heads, seq_len, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, q_start,
        dtype=dtype,
    )  # fmt: skip
    op = DSADecodeWithKVCacheFwdOp(dim_tail, stride_kv, q_start)
    q, kv, _ = test.gen_inputs()
    generator = torch.Generator().manual_seed(0)
    rows = [
        torch.randperm(seq_len_kv, generator=generator)[:topk] for _ in range(seq_len * heads_kv)
    ]
    indices = torch.stack(rows).view(1, seq_len, heads_kv, topk).to(torch.int32).to(q.device)
    test.check(op, q, kv, indices)
