"""Benchmark the TileOPs masked scatter against ``torch.Tensor.masked_scatter``, eager and compiled,
and vLLM's multimodal embedding merge."""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    compiled_reference,
    private_inputs,
    vllm_op,
)
from tileops.indexing import MaskedScatterFwdOp


@pytest.mark.parametrize("case", bench.cases(MaskedScatterFwdOp), ids=lambda case: case.id)
def test_masked_scatter_bench(case) -> None:
    op = MaskedScatterFwdOp(**case.arguments)
    merge = vllm_op("_merge_multimodal_embeddings", "model_executor.models.utils")
    input, mask, source = case.inputs
    # vLLM writes the text embeddings in place through a token mask, from exactly as many
    # source rows as the mask has true tokens.
    tokens = mask.flatten()
    rows = source[: int(tokens.sum())]
    bench.Runner(op, case).compare(
        {
            "torch": case.torch,
            TORCH_COMPILE_TAG: compiled_reference(case.torch),
            VLLM_TAG: private_inputs(
                merge,
                (input, rows, tokens),
                0,
                noncomparable_reason="writes input in place and takes exactly the true rows of source",
            ),
        }
    )
