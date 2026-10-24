import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops import FP8LightningIndexerFwdOp
from workloads.attention.fp8_lightning_indexer import FP8LightningIndexerWorkload
from workloads.device import run_device


class FP8LightningIndexerTest(FP8LightningIndexerWorkload, TestBase):
    pass


@pytest.mark.smoke
@pytest.mark.sm90
def test_indexer_preserves_small_normal_weights() -> None:
    workload = FP8LightningIndexerTest(1, 8, 64, 128, 128, 1)
    workload.check(FP8LightningIndexerFwdOp(), *workload.gen_weight_boundary_inputs())


class FP8LightningIndexerFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits, tune",
            [
                pytest.param(1, 4096, 32, 64, 8192, 1, True, False, marks=pytest.mark.smoke),
                # The SM90 warp-specialized kernel's other instance: four queries of 64 heads
                # of dim 128 a block, where 32 heads of dim 64 take eight.
                pytest.param(1, 2048, 64, 128, 4096, 1, True, False, marks=pytest.mark.smoke),
                # Two batches, a partial last query block and key tile; the 65 blocks of eight
                # queries fall short of any SM90 part's multiprocessors, so each block's keys
                # split over several CTAs.
                pytest.param(2, 514, 32, 64, 2052, 1, True, False, marks=pytest.mark.full),
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
        op.request_tune()
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


@pytest.mark.smoke
def test_indexer_without_clean_logits_matches_inside_windows() -> None:
    """Without ``clean_logits`` the kernel skips the -inf fill and the per-key window mask; the
    logits inside every window are still the clean call's, which match the reference."""
    test = FP8LightningIndexerTest(1, 2048, 64, 128, 4096, 1, True)
    inputs = test.gen_inputs()
    clean_op = FP8LightningIndexerFwdOp(clean_logits=True)
    test.check(clean_op, *inputs)
    clean = clean_op(*inputs)
    raw = FP8LightningIndexerFwdOp(clean_logits=False)(*inputs)
    live = torch.isfinite(clean)
    assert live.any()
    assert torch.equal(raw[live], clean[live])


@pytest.mark.smoke
def test_indexer_empty_and_inverted_windows() -> None:
    """Rows whose window is empty, inverted or starts at the end of the keys hold -inf
    everywhere, and a block mixing them with live rows still scores the live ones."""
    test = FP8LightningIndexerTest(1, 2048, 64, 128, 4096, 1, True)
    index_q, index_k, weights, ks, ke = test.gen_inputs()
    ks, ke = ks.clone(), ke.clone()
    ke[1::4] = ks[1::4]  # empty
    ks[2::4], ke[2::4] = ke[2::4] + 7, ks[2::4]  # inverted
    ks[3::8] = 4096  # starts at the end of the keys
    test.check(FP8LightningIndexerFwdOp(), index_q, index_k, weights, ks, ke)
