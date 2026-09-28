"""Tests for multi-head attention paged decode."""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.attention import MHADecodePagedWsKernel
from tileops.ops import MultiHeadAttentionDecodePagedWithKVCacheFwdOp
from workloads.device import run_device
from workloads.mha import MhaDecodePagedWorkload


class MhaDecodePagedTest(MhaDecodePagedWorkload, TestBase):
    #: bfloat16 carries 8 explicit mantissa bits against float16's 10, so one
    #: rounding of a unit-scale output is four times coarser.
    ATOL = {torch.float16: 0.001, torch.bfloat16: 0.005}

    def _maxdiff_cosine_compare(self, output: torch.Tensor, output_ref: torch.Tensor) -> None:
        """Compare using max-diff and cosine similarity."""
        atol = self.ATOL[self.dtype]
        if isinstance(output, (tuple, list)):
            output = output[0]
        max_diff = (output - output_ref).abs().max().item()
        assert max_diff < atol, f"max diff {max_diff} too large (atol={atol})"
        cos_sim = F.cosine_similarity(
            output.reshape(self.batch, -1), output_ref.reshape(self.batch, -1), dim=-1, eps=1e-8
        )
        assert cos_sim.min() > 0.99, f"cosine similarity {cos_sim.min().item()} too low"


class MhaDecodePagedFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune",
            [
                pytest.param(
                    1,
                    16,
                    1,
                    512,
                    128,
                    128,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                ),
                # bfloat16 dispatch: the same signature admits it and the paged
                # decode kernels are selected on dtype.
                pytest.param(
                    1,
                    16,
                    1,
                    1024,
                    128,
                    128,
                    False,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    1,
                    8,
                    1,
                    1024,
                    64,
                    256,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2,
                    8,
                    1,
                    1024,
                    64,
                    256,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1,
                    8,
                    1,
                    512,
                    64,
                    256,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@MhaDecodePagedFixture
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
    test = MhaDecodePagedTest(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype)
    op = MultiHeadAttentionDecodePagedWithKVCacheFwdOp(
        page_size=page_size, is_causal=is_causal, tune=tune
    )
    test.check(op, *test.gen_inputs(), compare=test._maxdiff_cosine_compare)


@pytest.mark.parametrize(
    "seqlen_q, is_causal, real_lengths",
    [
        pytest.param(1, False, [1], marks=pytest.mark.smoke),
        # Several queries; a cache shorter than the query block leaves the first ones no key.
        pytest.param(2, True, [65], marks=pytest.mark.smoke),
        pytest.param(4, True, [700, 3], marks=pytest.mark.smoke),
        pytest.param(4, False, [700, 3], marks=pytest.mark.smoke),
        pytest.param(1, False, [37], marks=pytest.mark.full),
        pytest.param(1, False, [700, 1024, 1], marks=pytest.mark.full),
    ],
)
def test_mha_decode_paged_cache_shorter_than_bound(
    seqlen_q: int, is_causal: bool, real_lengths: list
) -> None:
    """Splits, warps and query rows that see no key keep the output finite and exact.

    A cache far shorter than the static bound, or shorter than the causal queries,
    leaves them with no live score. Pool rows no request reads hold NaN.
    """
    batch, heads, seqlen_kv, dim, page_size = len(real_lengths), 8, 1024, 64, 256
    test = MhaDecodePagedTest(
        batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, torch.float16
    )
    q, k, v, _full, block_table = test.gen_inputs()
    real_seqlen_kv = torch.tensor(real_lengths, dtype=torch.int32, device=q.device)
    pos = torch.arange(block_table.shape[1] * page_size, device=q.device)
    rows = block_table[:, pos // page_size] * page_size + pos % page_size
    read = torch.zeros(seqlen_kv, dtype=torch.bool, device=q.device)
    read[rows[pos < real_seqlen_kv[:, None]]] = True
    k[~read] = float("nan")
    v[~read] = float("nan")

    op = MultiHeadAttentionDecodePagedWithKVCacheFwdOp(page_size=page_size, is_causal=is_causal)
    output = op(q, k, v, real_seqlen_kv, block_table)

    assert torch.isfinite(output).all(), "output is not finite for a partly filled cache"
    test._maxdiff_cosine_compare(output, test.ref_program(q, k, v, real_seqlen_kv, block_table))


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.in_tree_kernels
def test_mha_decode_paged_dispatch_bounds_multi_query_work() -> None:
    """Several query rows run on the warp-specialized kernel only below the work bound.

    One query row always does; past ``MHADecodePagedWsKernel._MAX_MULTI_QUERY_MACS`` the
    tensor-core kernel serves several.
    """
    op = MultiHeadAttentionDecodePagedWithKVCacheFwdOp(page_size=256, is_causal=True)
    heads, dim = 32, 128
    large = 2 * MHADecodePagedWsKernel._MAX_MULTI_QUERY_MACS // (4 * heads * dim)

    def chosen(seqlen_q: int, seqlen_kv: int) -> str:
        q = torch.empty(1, seqlen_q, heads, dim, dtype=torch.float16, device=run_device())
        k = torch.empty(seqlen_kv, heads, dim, dtype=torch.float16, device=run_device())
        block_table = torch.zeros(1, seqlen_kv // 256, dtype=torch.int32, device=run_device())
        return op.select_kernel(op._attention_call(q, k, block_table)).__name__

    assert chosen(4, 1024) == "MHADecodePagedWsKernel"
    assert chosen(1, large) == "MHADecodePagedWsKernel"
    assert chosen(4, large) == "GQADecodePagedKernel"
