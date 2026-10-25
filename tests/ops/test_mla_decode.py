import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.attention import MLADecodeWithKVCacheFwdOp
from tileops.backend import BUILTIN
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
        # The cache ends inside a tile.
        pytest.param(300, id="ragged-tail"),
        # One key: the first tile holds it, and every later key position is masked.
        pytest.param(1, id="empty-split"),
        # No keys at all: the output is the empty sum.
        pytest.param(0, id="no-keys"),
    ],
)
def test_mla_decode_masks_keys_past_the_cache_end(seq_len_kv: int) -> None:
    """A cache the tiles do not fill must not reach past its last key."""
    test = MLADecodeTest(2, 128, 1, seq_len_kv, 512, 64, torch.float16)
    op = MLADecodeWithKVCacheFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.parametrize(
    "batch, heads, seq_len_kv, dtype, splits",
    [
        # Two requests of two head blocks leave the SMs mostly idle: runs of two tiles.
        pytest.param(2, 128, 1000, torch.float16, 8, id="split"),
        # Nine tiles in four runs of three: the last run starts past the cache end.
        pytest.param(2, 128, 520, torch.float16, 4, id="empty-run"),
        # One head block a request, which no other block shares tiles with: no L2 prefetch.
        pytest.param(2, 64, 1000, torch.bfloat16, 8, id="one-head-block"),
        # 68 blocks fill a wave: one run a request.
        pytest.param(34, 128, 130, torch.bfloat16, 1, id="whole"),
    ],
)
def test_mla_decode_takes_the_seesaw_on_sm90(batch, heads, seq_len_kv, dtype, splits) -> None:
    """A 512 + 64 latent takes the seesaw, which cuts runs only while blocks are short of
    the SMs."""
    test = MLADecodeTest(batch, heads, 1, seq_len_kv, 512, 64, dtype)
    op = MLADecodeWithKVCacheFwdOp(target=BUILTIN)
    test.check(op, *test.gen_inputs())
    (kernel,) = op._entries_built("mla_decode_kernel").values()
    assert type(kernel).__name__ == "MLADecodeSeesawKernel"
    assert kernel.num_splits(batch, seq_len_kv) == splits
