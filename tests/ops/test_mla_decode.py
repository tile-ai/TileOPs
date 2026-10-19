import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops import MLADecodeWithKVCacheFwdOp
from workloads.attention.mla import MLADecodeWorkload


class MLADecodeTest(MLADecodeWorkload, TestBase):
    pass


class MLADecodeFixture(FixtureBase):
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
                # One 64-row head block a request: no other block shares its tiles, so the
                # SM90 seesaw takes them without an L2 prefetch.
                pytest.param(
                    2,
                    64,
                    1,
                    1000,
                    512,
                    64,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="one-head-block",
                ),
                pytest.param(
                    1,
                    128,
                    1,
                    4096,
                    512,
                    64,
                    torch.float16,
                    True,
                    marks=[pytest.mark.full, pytest.mark.sm89],
                    id="sm89-autotune",
                ),
            ],
        ),
    ]


@MLADecodeFixture
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
    test = MLADecodeTest(batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype)
    op = MLADecodeWithKVCacheFwdOp()
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "seq_len_kv",
    [
        # Five tiles in runs of three and two: the first run's last tile has no partner,
        # and the last tile ends inside it.
        pytest.param(300, id="ragged-tail"),
        # One key: a single run whose only tile holds it.
        pytest.param(1, id="empty-split"),
        # Nine tiles in four runs of three: the last run starts past the cache end.
        pytest.param(520, id="empty-run"),
        # No keys at all: the output is the empty sum.
        pytest.param(0, id="no-keys"),
    ],
)
def test_mla_decode_masks_keys_past_the_cache_end(seq_len_kv: int) -> None:
    """A cache the tiles do not fill must not reach past its last key."""
    test = MLADecodeTest(2, 128, 1, seq_len_kv, 512, 64, torch.float16)
    op = MLADecodeWithKVCacheFwdOp()
    test.check(op, *test.gen_inputs())
