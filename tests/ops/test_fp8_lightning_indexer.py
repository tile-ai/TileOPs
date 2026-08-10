from typing import Optional

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import FP8LightningIndexerOp
from workloads.fp8_lightning_indexer import FP8LightningIndexerWorkload


class FP8LightningIndexerTest(FP8LightningIndexerWorkload, TestBase):

    @staticmethod
    def _compute_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
        a, b = a.data.double(), b.data.double()
        norm_sum = (a * a + b * b).sum()
        return 2 * (a * b).sum() / norm_sum

    @staticmethod
    def _validate_tensor_match(output: torch.Tensor,
                               output_ref: torch.Tensor,
                               tolerance: float = 1e-3) -> None:
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
        assert 0 <= difference <= tolerance, \
            f"outputs is not close to outputs_ref, difference: {difference}"


class FP8LightningIndexerFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits, config, tune", [
             pytest.param(1, 4096, 32, 64, 8192, 1, True, None, False, marks=pytest.mark.smoke),
             pytest.param(1, 4096, 32, 64, 8192, 1, True, None, True, marks=pytest.mark.full),
        ]),
    ]


@FP8LightningIndexerFixture
def test_indexer(batch: int, seq_len: int, heads: int, index_dim: int, seq_len_kv: int,
                 kv_group: int, clean_logits: bool, config: Optional[dict], tune: bool) -> None:
    test = FP8LightningIndexerTest(batch, seq_len, heads, index_dim, seq_len_kv, kv_group,
                                  clean_logits, config)
    op = FP8LightningIndexerOp(clean_logits=clean_logits, config=config, tune=tune)
    test.check(op, *test.gen_inputs(), compare=FP8LightningIndexerTest._validate_tensor_match)


@pytest.mark.smoke
def test_indexer_rejects_bf16_inputs_with_external_scale() -> None:
    op = FP8LightningIndexerOp()
    batch, seq_len, heads, index_dim, seq_len_kv, kv_group = 1, 8, 4, 16, 16, 1
    index_q = torch.randn(
        batch, seq_len, heads, index_dim, device="cuda", dtype=torch.bfloat16)
    index_k = torch.randn(
        batch, seq_len_kv, kv_group, index_dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(seq_len, heads, device="cuda", dtype=torch.float32)
    cu_seqlen_ks = torch.zeros(seq_len, device="cuda", dtype=torch.int32)
    cu_seqlen_ke = torch.full((seq_len,), seq_len_kv, device="cuda", dtype=torch.int32)
    index_k_scale = torch.ones(batch, seq_len_kv, kv_group, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="float8_e4m3fn"):
        op(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, index_k_scale)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
