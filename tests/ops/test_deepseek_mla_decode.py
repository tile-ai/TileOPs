import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MultiHeadLatentAttentionDecodeWithKVCacheFwdOp
from workloads.deepseek_attention import MlaDecodeWorkload


class MlaDecodeTest(MlaDecodeWorkload, TestBase):
    pass


class MlaDecodeFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype, tune",
            [
                pytest.param(
                    32, 128, 1, 8192, 512, 64, torch.float16, False, marks=pytest.mark.smoke
                ),
                # The KV gather walks dim in 128-column steps; D = 128 takes one.
                pytest.param(
                    2,
                    128,
                    1,
                    256,
                    128,
                    64,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                    id="dim-128",
                ),
                # 96 heads leave the second 64-row head block half full.
                pytest.param(
                    2,
                    96,
                    1,
                    256,
                    512,
                    64,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                    id="tail-heads",
                ),
            ],
        ),
    ]


@MlaDecodeFixture
def test_mla_decode(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_kv: int,
    dim: int,
    dim_pe: int,
    dtype: torch.dtype,
    tune: bool,
):
    test = MlaDecodeTest(batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype)
    op = MultiHeadLatentAttentionDecodeWithKVCacheFwdOp(tune=tune)
    test.check(op, *test.gen_inputs(), atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "seq_len_kv",
    [
        # Each split ends inside a tile and leaves the step's second tile empty.
        pytest.param(300, id="ragged-tail"),
        # The first split's first tile holds one key; the second split holds none.
        pytest.param(1, id="empty-split"),
        # No keys at all: the output is the empty sum.
        pytest.param(0, id="no-keys"),
    ],
)
def test_mla_decode_masks_keys_past_the_cache_end(seq_len_kv: int) -> None:
    """A cache the tiles do not fill must not reach past its last key."""
    test = MlaDecodeTest(2, 128, 1, seq_len_kv, 512, 64, torch.float16)
    op = MultiHeadLatentAttentionDecodeWithKVCacheFwdOp()
    test.check(op, *test.gen_inputs(), atol=2e-3, rtol=2e-3)
