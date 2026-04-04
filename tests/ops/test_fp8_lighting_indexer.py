from typing import Optional

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import Fp8LightingIndexerOp
from workloads.ops.fp8_lighting_indexer import (
    Fp8LightingIndexerTest as _Fp8LightingIndexerTestWorkload,
)


class Fp8LightingIndexerTest(_Fp8LightingIndexerTestWorkload, TestBase):

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
        correlation = Fp8LightingIndexerTest._compute_correlation(output, output_ref)
        difference = 1.0 - correlation
        assert 0 <= difference <= tolerance, \
            f"outputs is not close to outputs_ref, difference: {difference}"


class Fp8LightingIndexerFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits, config, tune", [
             pytest.param(1, 4096, 32, 64, 8192, 1, True, None, False, marks=pytest.mark.smoke),
        ]),
    ]


@Fp8LightingIndexerFixture
def test_indexer(batch: int, seq_len: int, heads: int, index_dim: int, seq_len_kv: int,
                 kv_group: int, clean_logits: bool, config: Optional[dict], tune: bool) -> None:
    test = Fp8LightingIndexerTest(batch, seq_len, heads, index_dim, seq_len_kv, kv_group,
                                  clean_logits, config)
    op = Fp8LightingIndexerOp(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        index_dim=index_dim,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        clean_logits=clean_logits,
        config=config,
        tune=tune)
    test.check(op, *test.gen_inputs(), compare=Fp8LightingIndexerTest._validate_tensor_match)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
