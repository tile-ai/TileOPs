import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops import FP8LightningIndexerFwdOp
from workloads.attention.fp8_lightning_indexer import FP8LightningIndexerWorkload
from workloads.device import run_device


class FP8LightningIndexerTest(FP8LightningIndexerWorkload, TestBase):
    pass


class FP8LightningIndexerFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits, tune",
            [
                pytest.param(1, 4096, 32, 64, 8192, 1, True, False, marks=pytest.mark.smoke),
                pytest.param(1, 4096, 32, 64, 8192, 1, True, True, marks=pytest.mark.full),
                # On 99 KB of shared memory these take two queries a block and one; at dim 512
                # only one fits, so tuning has to offer it.
                pytest.param(1, 1024, 128, 256, 2048, 1, True, False, marks=pytest.mark.full),
                pytest.param(1, 1024, 128, 256, 2048, 2, True, False, marks=pytest.mark.full),
                pytest.param(1, 1024, 128, 512, 2048, 1, True, True, marks=pytest.mark.nightly),
            ],
        ),
    ]


@FP8LightningIndexerFixture
def test_indexer(
    batch: int,
    seq_len: int,
    heads: int,
    index_dim: int,
    seq_len_kv: int,
    kv_group: int,
    clean_logits: bool,
    tune: bool,
) -> None:
    test = FP8LightningIndexerTest(
        batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits
    )
    op = FP8LightningIndexerFwdOp(clean_logits=clean_logits)
    if tune:
        op.autotune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_indexer_rejects_bf16_inputs_with_external_scale() -> None:
    op = FP8LightningIndexerFwdOp()
    batch, seq_len, heads, index_dim, seq_len_kv, kv_group = 1, 8, 4, 16, 16, 1
    index_q = torch.randn(
        batch, seq_len, heads, index_dim, device=run_device(), dtype=torch.bfloat16
    )
    index_k = torch.randn(
        batch, seq_len_kv, kv_group, index_dim, device=run_device(), dtype=torch.bfloat16
    )
    weights = torch.randn(seq_len, heads, device=run_device(), dtype=torch.float32)
    cu_seqlen_ks = torch.zeros(seq_len, device=run_device(), dtype=torch.int32)
    cu_seqlen_ke = torch.full((seq_len,), seq_len_kv, device=run_device(), dtype=torch.int32)
    index_k_scale = torch.ones(
        batch, seq_len_kv, kv_group, device=run_device(), dtype=torch.float32
    )

    with pytest.raises(ValueError, match="float8_e4m3fn"):
        op(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, index_k_scale)
