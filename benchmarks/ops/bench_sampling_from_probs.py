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
from workloads.numerics import Custom
from workloads.sampling import SamplingFromProbsWorkload


@pytest.mark.parametrize("call", manifest_calls(SamplingFromProbsFwdOp))
def test_sampling_from_probs_bench(call) -> None:
    workload = SamplingFromProbsWorkload(call)
    probs, seed, offset = workload.gen_inputs()

    op = SamplingFromProbsFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)

    # No two candidates draw from the same random stream, so none of them can be compared
    # with the reference token for token. What is checked instead is the distribution each
    # one draws from, on a row of its own: every fourth token has weight zero, and 65536
    # draws from one row have to leave each token's count within six sigma of its
    # expectation and every zero-weight token undrawn. A candidate that read the row it was
    # handed and then ignored it fails this; the shape and positive-weight checks the timed
    # rows carry would not catch that.
    draws = 65536
    num_tokens = 64
    sigmas = 6
    device = probs.device
    torch.manual_seed(20258)
    weights = torch.rand(num_tokens, device=device)
    weights[::4] = 0
    share = (weights / weights.sum()).double()
    trial = (weights / weights.sum()).expand(draws, num_tokens).contiguous()
    bound = sigmas * (draws * share * (1 - share)).sqrt() + 5 * (share > 0)

    def follows_the_row(draw, tag: str) -> None:
        tokens = draw(trial, seed, offset)
        assert tokens.shape == (draws,), tag
        count = torch.bincount(tokens.long(), minlength=num_tokens).double()
        assert ((count - draws * share).abs() <= bound).all(), (tag, count, draws * share)

    def drawn_from(tokens: torch.Tensor, tag: str) -> None:
        assert tokens.shape == (call.ix["B"],), tag
        assert not tokens.is_floating_point(), tag
        assert (probs.gather(1, tokens.long().view(-1, 1)) > 0).all(), tag

    # flashinfer and torch.multinomial take a torch.Generator rather than a Philox pair, and
    # vllm's random_sample takes one per request; each is seeded once here so no timed call
    # carries set-up the others do not.
    flashinfer_draw = flashinfer_op("sampling.sampling_from_probs")
    random_sample = vllm_op("random_sample", "v1.sample.ops.topk_topp_sampler")
    generator = torch.Generator(device=device).manual_seed(int(seed.item()))
    generators = {0: torch.Generator(device=device).manual_seed(int(seed.item()))}
    # vllm's random_sample divides the probabilities it is handed by a draw of exponentials
    # and takes the argmax, in place. Dividing is not idempotent, so its row is restored
    # inside its call; the restoring copy is harness work that keeps repeated iterations
    # reading what the workload built, not part of the draw, and ``count_copies`` stays off
    # so it is not timed, as ``bench_top_p_mask.py`` leaves it.
    vllm_probs = torch.empty_like(probs)
    vllm_trial = torch.empty_like(trial)

    def flashinfer_from_probs(probs, seed, offset):
        return flashinfer_draw(probs, generator=generator)

    def vllm_from_probs(probs, seed, offset):
        held = vllm_trial if probs.shape == trial.shape else vllm_probs
        return random_sample(held.copy_(probs), generators)

    def multinomial_from_probs(probs, seed, offset):
        return torch.multinomial(probs, 1, generator=generator)[:, 0]

    candidates = {
        "tileops": op,
        TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        FLASHINFER_TAG: flashinfer_from_probs,
        VLLM_TAG: vllm_from_probs,
        "torch-multinomial": multinomial_from_probs,
    }

    def validator(draw, tag):
        def check(tokens, _expected):
            drawn_from(tokens, tag)
            if tag == "tileops":
                assert tokens.dtype == torch.int32
            follows_the_row(draw, tag)

        return check

    bm.compare(
        {**candidates, "torch-ref": workload.ref_program},
        probs,
        seed,
        offset,
        evidence={
            tag: Custom(validator(draw, tag), "support, shape and six-sigma distribution check")
            for tag, draw in candidates.items()
        },
    )
