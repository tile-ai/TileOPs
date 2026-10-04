"""Tests for the Native Sparse Attention (NSA) ops: the sparse forward, compression and top-k selection."""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import NSACompressedVarlenFwdOp, NSATopKVarlenFwdOp, NSAVarlenFwdOp
from workloads.attention.nsa import NsaCmpFwdWorkload, NsaFwdWorkload, NsaTopkWorkload


class NsaFwdTest(NsaFwdWorkload, TestBase):
    pass


class NsaFwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, c_seq_len, dim, is_causal, scale, block_size, "
            "groups, selected_blocks, dtype, tune, seq_lens",
            [
                pytest.param(
                    1,
                    16,
                    1024,
                    64,
                    True,
                    0.1,
                    32,
                    16,
                    1,
                    torch.float16,
                    False,
                    None,
                    marks=pytest.mark.smoke,
                ),
                # Non-causal, with lengths no block size divides.
                pytest.param(
                    2,
                    16,
                    256,
                    64,
                    False,
                    0.1,
                    32,
                    16,
                    4,
                    torch.float16,
                    False,
                    [100, 156],
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    4,
                    16,
                    8192,
                    64,
                    True,
                    0.1,
                    32,
                    16,
                    1,
                    torch.float16,
                    False,
                    None,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2,
                    16,
                    8192,
                    64,
                    True,
                    0.1,
                    32,
                    16,
                    4,
                    torch.float16,
                    False,
                    None,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@NsaFwdFixture
def test_nsa_varlen_op(
    batch: int,
    heads: int,
    c_seq_len: int,
    dim: int,
    is_causal: bool,
    scale: float,
    block_size: int,
    groups: int,
    selected_blocks: int,
    dtype: torch.dtype,
    tune: bool,
    seq_lens: "list[int] | None",
) -> None:
    assert groups % 16 == 0, "Group size must be a multiple of 16 in NSA"

    test = NsaFwdTest(
        batch,
        heads,
        c_seq_len,
        dim,
        is_causal,
        scale,
        block_size,
        groups,
        selected_blocks,
        dtype,
        seq_lens,
    )
    op = NSAVarlenFwdOp(
        is_causal=is_causal,
        scale=scale,
        block_size=block_size,
        tune=tune,
    )
    test.check(op, *test.gen_inputs(), atol=5e-4, rtol=1e-5)


class NsaCmpFwdTest(NsaCmpFwdWorkload, TestBase):
    pass


class NsaCmpFwdFixture(FixtureBase):
    PARAMS = [
        (
            "seq_num, c_seq_len, heads, dim_k, dim_v, group, scale, bs, dtype, tune",
            [
                pytest.param(
                    9,
                    8192,
                    32,
                    128,
                    128,
                    16,
                    128**-0.5,
                    32,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                ),
            ],
        ),
    ]


@NsaCmpFwdFixture
def test_nsa_cmp_fwd_varlen_op(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    group: int,
    scale: float,
    bs: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    assert group % 16 == 0, "Group size must be a multiple of 16 in NSA"

    test = NsaCmpFwdTest(seq_num, c_seq_len, heads, dim_k, dim_v, group, scale, bs, dtype)
    inputs = test.gen_inputs()

    op = NSACompressedVarlenFwdOp(scale=scale, bs=bs, tune=tune)
    test.check(op, *inputs, atol=4e-3, rtol=1e-5)


class NsaTopkTest(NsaTopkWorkload, TestBase):
    def check_topk(self, op, *inputs) -> torch.Tensor:
        """Check selection scores, allowing ties, with exact index invariants."""
        got, expected = op(*inputs), self.ref_program(*inputs)
        assert got.shape == expected.shape and got.dtype == expected.dtype
        assert torch.equal(got == -1, expected == -1), "unfilled top-k slots differ"
        current = inputs[-1][:, 1, None, None] // self.bs
        assert ((got >= -1) & (got <= current)).all(), "non-causal or invalid block id"
        ordered = got.sort(-1).values
        assert not ((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] >= 0)).any(), (
            "duplicate selected block"
        )
        torch.testing.assert_close(
            self.selection_scores(got, *inputs),
            self.selection_scores(expected, *inputs),
            rtol=1e-5,
            atol=1e-6,
        )
        return got


class NsaTopkFixture(FixtureBase):
    PARAMS = [
        (
            "seq_num, c_seq_len, heads, dim, group, scale, selected_block_num, bs, dtype, tune",
            [
                pytest.param(
                    5, 1024, 32, 128, 16, 1.0, 16, 32, dtype, False, marks=pytest.mark.smoke
                )
                for dtype in (torch.float16, torch.bfloat16)
            ]
            + [
                pytest.param(
                    3, 512, 32, 128, 16, 1.0, 16, 32, torch.float16, False, marks=pytest.mark.full
                ),
                # Two sort pairs per lane must not put barriers under a partial-lane mask.
                pytest.param(
                    1, 65, 16, 64, 16, 0.125, 8, 64, torch.float16, False, marks=pytest.mark.full
                ),
            ],
        ),
    ]


@NsaTopkFixture
def test_nsa_topk_varlen_op(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim: int,
    group: int,
    scale: float,
    selected_block_num: int,
    bs: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    assert group % 16 == 0, "Group size must be a multiple of 16 in NSA"

    test = NsaTopkTest(seq_num, c_seq_len, heads, dim, group, scale, selected_block_num, bs, dtype)
    inputs = test.gen_inputs()
    op = NSATopKVarlenFwdOp(
        scale=scale,
        selected_block_num=selected_block_num,
        bs=bs,
        tune=tune,
    )
    test.check_topk(op, *inputs)


@pytest.mark.smoke
def test_nsa_topk_reference_keeps_fp32_dot_products() -> None:
    """Promoting a rounded half matmul is too late to rank close candidates."""
    torch.manual_seed(1235)
    workload = NsaTopkWorkload(1, 512, 32, 64, 16, 1.0, 16, 32, torch.float16)
    q, k, *metadata = workload.gen_inputs()
    torch.testing.assert_close(
        workload.ref_program(q, k, *metadata),
        workload.ref_program(q.float(), k.float(), *metadata),
        rtol=0,
        atol=0,
    )


@pytest.mark.smoke
def test_nsa_topk_ranks_unquantized_scores() -> None:
    """Priority blocks survive; the remaining slot takes the highest raw score."""
    workload = NsaTopkTest(1, 256, 16, 16, 16, 1.0, 4, 32, torch.float16)
    q, k, *metadata = workload.gen_inputs()
    q = torch.zeros_like(q)
    q[-1].fill_(1)
    k = torch.empty_like(k)
    k.copy_(torch.arange(k.shape[0] - 1, -1, -1, device=k.device)[:, None, None] * 2**-24)
    op = NSATopKVarlenFwdOp(scale=1.0, selected_block_num=4, bs=32)
    result = workload.check_topk(op, q, k, *metadata)
    # Sub-1e-5 gaps must still select block 1, alongside priority blocks 0, 6, 7.
    torch.testing.assert_close(
        result[-1, 0].sort().values,
        torch.tensor([0, 1, 6, 7], dtype=torch.int32, device=q.device),
        rtol=0,
        atol=0,
    )
