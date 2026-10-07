"""Benchmark for the seeded categorical draw op.

Workload shapes come from the ops manifest; roofline FLOP and byte counts come
from the op's ``eval_roofline()``.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    compiled_reference,
    flashinfer_op,
    private_inputs,
    vllm_op,
)
from tileops.sampling import SamplingFromProbsFwdOp


@pytest.mark.parametrize("case", bench.cases(SamplingFromProbsFwdOp), ids=lambda case: case.id)
def test_sampling_from_probs_bench(case) -> None:
    probs, seed, _offset = case.inputs

    op = SamplingFromProbsFwdOp(**case.arguments)

    # flashinfer and torch.multinomial take a torch.Generator rather than a Philox pair, and
    # vllm's random_sample takes one per request; each is seeded once here so no timed call
    # carries set-up the others do not.
    flashinfer_draw = flashinfer_op("sampling.sampling_from_probs")
    random_sample = vllm_op("random_sample", "v1.sample.ops.topk_topp_sampler")
    generator = torch.Generator(device=probs.device).manual_seed(int(seed.item()))
    generators = {0: torch.Generator(device=probs.device).manual_seed(int(seed.item()))}

    def flashinfer_from_probs(probs, _seed, _offset):
        return flashinfer_draw(probs, generator=generator).to(torch.int32)

    def vllm_from_probs(probs, _seed, _offset):
        return random_sample(probs, generators).to(torch.int32)

    def multinomial_from_probs(probs, _seed, _offset):
        return torch.multinomial(probs, 1, generator=generator)[:, 0].to(torch.int32)

    bench.Runner(op, case).compare(
        {
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
            FLASHINFER_TAG: flashinfer_from_probs,
            # vllm's random_sample divides the probabilities it is handed by a draw of
            # exponentials and takes the argmax, in place.
            VLLM_TAG: private_inputs(vllm_from_probs, case.inputs, 0),
            "torch-multinomial": multinomial_from_probs,
            "torch-ref": case.reference,
        }
    )
