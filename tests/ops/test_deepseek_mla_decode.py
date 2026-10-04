import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.attention import MlaDecodeCall, MLADecodeMmaKernel
from tileops.ops import MultiHeadLatentAttentionDecodeWithKVCacheFwdOp
from workloads.attention.mla import MlaDecodeWorkload


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


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("budget", "dim", "pe_dim", "seqlen_kv", "expected"),
    [
        pytest.param(101376, 512, 64, 4096, 32, id="sm89-d512"),
        pytest.param(101376, 768, 64, 4096, 16, id="sm89-d768"),
        pytest.param(101376, 1024, 64, 4096, "needs 106496 bytes", id="sm89-d1024-refused"),
        pytest.param(101376, 1024, 16, 4096, "needs 101888 bytes", id="sm89-pe16-refused"),
        pytest.param(101376, 1024, 16, 64, 16, id="sm89-pe16-one-tile"),
        pytest.param(101376, 1024, 16, 128, 16, id="sm89-pe16-two-tiles"),
        pytest.param(101376, 1024, 64, 0, 16, id="sm89-d1024-empty"),
        pytest.param(101376, 192, 64, 4096, "dim must be", id="d192-refused"),
        pytest.param(166912, 1024, 64, 4096, 32, id="sm80-d1024"),
    ],
)
def test_mla_decode_mma_config_follows_the_shared_memory_budget(
    budget: int,
    dim: int,
    pe_dim: int,
    seqlen_kv: int,
    expected: int | str,
) -> None:
    """The head block and the refusal follow the default's shared memory at a budget."""
    call = MlaDecodeCall(
        arch=89,
        sm_count=1,
        smem_budget=budget,
        batch=2,
        heads=128,
        heads_kv=1,
        seqlen_kv=seqlen_kv,
        dim=dim,
        pe_dim=pe_dim,
        dtype=torch.float16,
    )
    if isinstance(expected, str):
        assert expected in MLADecodeMmaKernel.refusal(call)
        return
    assert MLADecodeMmaKernel.refusal(call) is None
    config = MLADecodeMmaKernel._default_config_for(
        budget, dim, pe_dim, torch.float16.itemsize, seqlen_kv
    )
    assert config["block_H"] == expected
