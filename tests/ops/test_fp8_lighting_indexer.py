from typing import Optional

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import FP8LightingIndexerOp
from workloads.fp8_lighting_indexer import (
    FP8LightingIndexerTest as _FP8LightingIndexerTestWorkload,
)


class FP8LightingIndexerTest(_FP8LightingIndexerTestWorkload, TestBase):

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
        correlation = FP8LightingIndexerTest._compute_correlation(output, output_ref)
        difference = 1.0 - correlation
        assert 0 <= difference <= tolerance, \
            f"outputs is not close to outputs_ref, difference: {difference}"

    def ref_program(self, q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                    cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor) -> tuple[torch.Tensor]:
        k = kv
        q = q.float()
        k = k.float()
        batch, seq_len, heads, index_dim = q.shape
        seq_len_kv = self.seq_len_kv
        kv_group = self.kv_group
        heads_per_group = heads // kv_group

        k = k.view(batch, seq_len_kv, kv_group, index_dim)
        q = q.view(batch, seq_len, kv_group, heads_per_group, index_dim)

        mask_lo = torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
        mask_hi = torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
        mask = mask_lo & mask_hi

        score = torch.einsum("bsghd,bngd->bghsn", q, k)
        weights = weights.view(seq_len, kv_group, heads_per_group)
        weights = weights.permute(1, 2, 0).unsqueeze(0).unsqueeze(-1)
        score = score.relu() * weights
        logits = score.sum(dim=2)
        logits = logits.permute(0, 2, 3, 1)
        mask_expanded = mask.unsqueeze(0).unsqueeze(-1)
        logits = logits.masked_fill(~mask_expanded, float("-inf"))
        return (logits,)


class FP8LightingIndexerFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits, config, tune", [
             pytest.param(1, 4096, 32, 64, 8192, 1, True, None, False, marks=pytest.mark.smoke),
        ]),
    ]


@FP8LightingIndexerFixture
def test_indexer(batch: int, seq_len: int, heads: int, index_dim: int, seq_len_kv: int,
                 kv_group: int, clean_logits: bool, config: Optional[dict], tune: bool) -> None:
    test = FP8LightingIndexerTest(batch, seq_len, heads, index_dim, seq_len_kv, kv_group,
                                  clean_logits, config)
    op = FP8LightingIndexerOp(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        index_dim=index_dim,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        clean_logits=clean_logits,
        config=config,
        tune=tune)
    test.check(op, *test.gen_inputs(), compare=FP8LightingIndexerTest._validate_tensor_match)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
