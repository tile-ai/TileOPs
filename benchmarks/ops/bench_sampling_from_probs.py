"""Benchmark for the seeded categorical draw op.

Workload shapes come from the ops manifest; roofline FLOP and byte counts come
from the op's ``eval_roofline()`` via :class:`ManifestBenchmark`.
"""

import pytest
import torch

from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    compiled_reference,
    flashinfer_op,
    vllm_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.sampling import SamplingFromProbsFwdOp
from workloads.sampling import SamplingFromProbsWorkload


@pytest.mark.parametrize("call", manifest_calls(SamplingFromProbsFwdOp))
def test_sampling_from_probs_bench(call) -> None:
    workload = SamplingFromProbsWorkload(call)
    probs, seed, offset = workload.gen_inputs()

    op = SamplingFromProbsFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)

    # flashinfer and torch.multinomial take a torch.Generator rather than a Philox pair, and
    # vllm's random_sample takes one per request; each is seeded once here so no timed call
    # carries set-up the others do not.
    flashinfer_draw = flashinfer_op("sampling.sampling_from_probs")
    random_sample = vllm_op("random_sample", "v1.sample.ops.topk_topp_sampler")
    generator = torch.Generator(device=probs.device).manual_seed(int(seed.item()))
    generators = {0: torch.Generator(device=probs.device).manual_seed(int(seed.item()))}
    # vllm's random_sample divides the probabilities it is handed by a draw of exponentials
    # and takes the argmax, in place. Dividing is not idempotent, so its row is restored
    # inside its call; the restoring copy is harness work that keeps repeated iterations
    # reading what the workload built, not part of the draw, and ``count_copies`` stays off
    # so it is not timed, as ``bench_top_p_mask.py`` leaves it.
    vllm_probs = torch.empty_like(probs)

    def flashinfer_from_probs(probs, seed, offset):
        return flashinfer_draw(probs, generator=generator).to(torch.int32)

    def vllm_from_probs(probs, seed, offset):
        held = vllm_probs if probs.shape == vllm_probs.shape else torch.empty_like(probs)
        return random_sample(held.copy_(probs), generators).to(torch.int32)

    def multinomial_from_probs(probs, seed, offset):
        return torch.multinomial(probs, 1, generator=generator)[:, 0].to(torch.int32)

    candidates = {
        "tileops": op,
        TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        FLASHINFER_TAG: flashinfer_from_probs,
        VLLM_TAG: vllm_from_probs,
        "torch-multinomial": multinomial_from_probs,
    }

    bm.compare(
        {**candidates, "torch-ref": workload.ref_program},
        probs,
        seed,
        offset,
    )
