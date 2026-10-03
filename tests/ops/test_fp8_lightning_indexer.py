import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.attention import (
    FP8LightningIndexerCall,
    FP8LightningIndexerKernel,
    fp8_lightning_indexer,
)
from tileops.ops import FP8LightningIndexerFwdOp
from workloads.attention.fp8_lightning_indexer import FP8LightningIndexerWorkload
from workloads.device import run_device


class FP8LightningIndexerTest(FP8LightningIndexerWorkload, TestBase):
    @staticmethod
    def _compute_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
        a, b = a.data.double(), b.data.double()
        norm_sum = (a * a + b * b).sum()
        return 2 * (a * b).sum() / norm_sum

    @staticmethod
    def _validate_tensor_match(
        output: torch.Tensor, output_ref: torch.Tensor, tolerance: float = 1e-3
    ) -> None:
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output_ref, tuple):
            output_ref = output_ref[0]

        a_finite = torch.isfinite(output)
        b_finite = torch.isfinite(output_ref)
        assert torch.all(a_finite == b_finite), "Error: isfinite mask mismatch"
        assert torch.isclose(
            output.masked_fill(a_finite, 0),
            output_ref.masked_fill(b_finite, 0),
            rtol=0,
            atol=0,
            equal_nan=True,
        ).all(), "Error: nonfinite value mismatch"
        output = output.masked_fill(~a_finite, 0)
        output_ref = output_ref.masked_fill(~b_finite, 0)
        correlation = FP8LightningIndexerTest._compute_correlation(output, output_ref)
        difference = 1.0 - correlation
        assert 0 <= difference <= tolerance, (
            f"outputs is not close to outputs_ref, difference: {difference}"
        )


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
    op = FP8LightningIndexerFwdOp(clean_logits=clean_logits, tune=tune)
    test.check(op, *test.gen_inputs(), compare=FP8LightningIndexerTest._validate_tensor_match)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("budget", "heads", "index_dim", "kv_group", "block_q"),
    [
        pytest.param(101376, 64, 128, 1, 4, id="sm89-h64-d128"),
        pytest.param(101376, 128, 256, 1, 2, id="sm89-h128-d256"),
        pytest.param(101376, 128, 256, 2, 1, id="sm89-h128-d256-g2"),
        pytest.param(101376, 128, 512, 2, None, id="sm89-d512-g2-refused"),
        pytest.param(232448, 128, 256, 1, 4, id="sm90-h128-d256"),
    ],
)
def test_indexer_block_q_follows_the_shared_memory_budget(
    monkeypatch: pytest.MonkeyPatch,
    budget: int,
    heads: int,
    index_dim: int,
    kv_group: int,
    block_q: int | None,
) -> None:
    """The queries a block takes and the refusal follow the default's shared memory."""
    call = FP8LightningIndexerCall(
        arch=89,
        sm_count=1,
        smem_budget=budget,
        batch=1,
        seq_len=1024,
        heads=heads,
        index_dim=index_dim,
        seq_len_kv=2048,
        kv_group=kv_group,
    )
    if block_q is None:
        assert "needs" in FP8LightningIndexerKernel.refusal(call)
        return
    assert FP8LightningIndexerKernel.refusal(call) is None
    monkeypatch.setattr(FP8LightningIndexerKernel, "_check_arch", lambda self: None)
    monkeypatch.setattr(fp8_lightning_indexer, "get_shared_memory_optin", lambda index=None: budget)
    kernel = FP8LightningIndexerKernel(1, 1024, heads, index_dim, 2048, kv_group)
    assert kernel.config["block_Q"] == block_q


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
