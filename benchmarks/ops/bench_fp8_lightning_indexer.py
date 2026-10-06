"""FP8 indexer against DeepGEMM, with complete chunked-reference validation."""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import DEEPGEMM_TAG, deepgemm_op
from tileops.ops import FP8LightningIndexerFwdOp


@pytest.mark.parametrize("case", bench.cases(FP8LightningIndexerFwdOp), ids=lambda case: case.id)
def test_fp8_lightning_indexer_bench(case) -> None:
    op = FP8LightningIndexerFwdOp(**case.arguments)
    logits = deepgemm_op("fp8_mqa_logits")

    def deepgemm_fn(q, k, weights, start, end, k_scale):
        if k_scale is None:
            q = q.to(torch.float8_e4m3fn)
            scale = k.float().abs().amax(-1, keepdim=True).clamp(min=0.0001) / 448.0
            k = (k.float() / scale).to(torch.float8_e4m3fn)
            k_scale = scale.squeeze(-1)
        batches = []
        groups = k.shape[2]
        heads = q.shape[2] // groups
        for batch in range(q.shape[0]):
            scores = [
                logits(
                    q[batch, :, group * heads : (group + 1) * heads].contiguous(),
                    (k[batch, :, group].contiguous(), k_scale[batch, :, group].contiguous()),
                    weights[:, group * heads : (group + 1) * heads].contiguous(),
                    start,
                    end,
                    clean_logits=True,
                )
                for group in range(groups)
            ]
            batches.append(scores[0].unsqueeze(-1) if groups == 1 else torch.stack(scores, -1))
        return batches[0].unsqueeze(0) if len(batches) == 1 else torch.stack(batches)

    bench.Runner(op, case).compare({DEEPGEMM_TAG: deepgemm_fn, "torch-ref": case.reference})
