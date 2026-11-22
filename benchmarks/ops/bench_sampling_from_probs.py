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
    vllm_op,
)
from tileops.sampling import SamplingFromProbsFwdOp


@pytest.mark.parametrize("case", bench.cases(SamplingFromProbsFwdOp), ids=lambda case: case.id)
def test_sampling_from_probs_bench(case) -> None:
    probs, seed, _offset = case.inputs

    op = SamplingFromProbsFwdOp(**case.arguments)

    # flashinfer and torch.multinomial take a torch.Generator rather than a Philox pair; it
    # is seeded once here so no timed call carries set-up the others do not.
    flashinfer_draw = flashinfer_op("sampling.sampling_from_probs")
    gumbel_sample = vllm_op("gumbel_sample", "v1.worker.gpu.sample.gumbel")
    generator = torch.Generator(device=probs.device).manual_seed(int(seed.item()))
    # vllm keys a row's Gumbel noise by its request's seed and the row's position; every row
    # maps to request 0, which carries the call's seed, and the offset is unused. The index
    # tensors of each batch size are built on its first call.
    temperature = torch.ones(1, dtype=torch.float32, device=probs.device)
    row_indices: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def flashinfer_from_probs(probs, _seed, _offset):
        return flashinfer_draw(probs, generator=generator).to(torch.int32)

    def vllm_from_probs(probs, seed, _offset):
        batch = probs.shape[0]
        if batch not in row_indices:
            row_indices[batch] = (
                torch.zeros(batch, dtype=torch.int32, device=probs.device),
                torch.arange(batch, dtype=torch.int64, device=probs.device),
            )
        requests, positions = row_indices[batch]
        # gumbel_sample takes logits; the log stays in the call because the distribution
        # check calls it with probabilities of its own.
        return gumbel_sample(
            probs.log(),
            requests,
            temperature,
            seed,
            positions,
            apply_temperature=False,
            is_drafting=False,
        ).to(torch.int32)

    def multinomial_from_probs(probs, _seed, _offset):
        return torch.multinomial(probs, 1, generator=generator)[:, 0].to(torch.int32)

    bench.Runner(op, case).compare(
        {
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
            FLASHINFER_TAG: flashinfer_from_probs,
            VLLM_TAG: vllm_from_probs,
            "torch-multinomial": multinomial_from_probs,
            "torch-ref": case.reference,
        }
    )
