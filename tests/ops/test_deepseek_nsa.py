"""Tests for the Native Sparse Attention (NSA) ops: the sparse forward, compression and top-k selection."""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import NSACompressedVarlenFwdOp, NSATopKVarlenFwdOp, NSAVarlenFwdOp
from workloads.deepseek_attention import NsaCmpFwdWorkload, NsaFwdWorkload, NsaTopkWorkload


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
    def check_topk(self, op, *inputs, threshold: float = 1e-3) -> None:
        """Custom check for topk indices (not floating point closeness)."""
        outputs_ref = self.ref_program(*inputs)
        outputs = op(*inputs)

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        for out, ref in zip(outputs, outputs_ref, strict=True):
            print("[Top-K Indices Comparison - TileLang vs PyTorch]")

            indices_match = torch.all(out == ref)
            if indices_match:
                print("Top-K Indices Matched!")
            else:
                mismatch_count = (out != ref).sum().item()
                total_count = out.numel()
                mismatch_ratio = mismatch_count / total_count

                assert mismatch_ratio <= threshold, (
                    f"Top-K mismatch ratio {mismatch_ratio:.3%} exceeds threshold {threshold:.3%}"
                )
                print(
                    f"Top-K Indices Mismatched slightly within threshold: "
                    f"{mismatch_ratio * 100:.3f}%"
                )


class NsaTopkFixture(FixtureBase):
    PARAMS = [
        (
            "seq_num, c_seq_len, heads, dim, group, scale, selected_block_num, bs, dtype, tune",
            [
                pytest.param(
                    5,
                    1024,
                    32,
                    128,
                    16,
                    1,
                    16,
                    32,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    3,
                    512,
                    32,
                    128,
                    16,
                    1,
                    16,
                    32,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
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
