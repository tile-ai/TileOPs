"""FP8 indexer against DeepGEMM, with complete chunked-reference validation."""

import pytest
import torch

from benchmarks.baselines import DEEPGEMM_TAG, deepgemm_op
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import FP8LightningIndexerFwdOp
from workloads.attention.fp8_lightning_indexer import FP8LightningIndexerCall
from workloads.numerics import Custom, assert_normalized_error, zeroed_input


@pytest.mark.parametrize("call", manifest_calls(FP8LightningIndexerFwdOp))
def test_fp8_lightning_indexer_bench(call) -> None:
    workload = FP8LightningIndexerCall(call)
    inputs = workload.gen_inputs()

    op = FP8LightningIndexerFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)

    logits = deepgemm_op("fp8_mqa_logits")

    def deepgemm_fn(q, k, weights, start, end, k_scale):
        if k_scale is None:
            q = q.to(torch.float8_e4m3fn)
            scale = k.float().abs().amax(-1, keepdim=True).clamp(min=1e-4) / 448.0
            k = (k.float() / scale).to(torch.float8_e4m3fn)
            k_scale = scale.squeeze(-1)
        # DeepGEMM handles one batch and KV group per invocation.
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

    checked = Custom(
        assert_normalized_error,
        "symmetric normalized squared error <= 1e-3; nonfinite values match",
        controls=(zeroed_input(0, "query-zeroed"),),
    )
    bm.compare(
        {
            "tileops": op,
            DEEPGEMM_TAG: deepgemm_fn,
            "torch-ref": workload.ref_program,
        },
        *inputs,
        count_copies=True,
        evidence={"tileops": checked, DEEPGEMM_TAG: checked},
    )
