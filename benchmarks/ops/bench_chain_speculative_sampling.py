"""Benchmark for the chain speculative sampling op.

Workload shapes come from the ops manifest; roofline FLOP and byte counts come from the op's
``eval_roofline()`` via :class:`ManifestBenchmark`.
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
from tileops.sampling import ChainSpeculativeSamplingFwdOp
from workloads.sampling import ChainSpeculativeSamplingWorkload

# Standard deviations of the difference between two batches of accepted lengths that one is
# allowed to sit from the other, plus a constant covering the lengths a batch this size
# expects a handful of. The chain stops where the draws put it, so two implementations of one
# rule agree on the distribution of that length, never on the batch they drew.
_LENGTH_SIGMAS = 5.0
_LENGTH_SLACK = 5.0


def _accepted_lengths(result, num_draft: int) -> torch.Tensor:
    """The accepted prefix length of each row of an ``[B, N + 1]`` output padded after the draw.

    Each implementation returns that tensor first and its own extra counts after it; the
    padding is the only mark of where the chain stopped that all of them carry.
    """
    tokens = result[0] if isinstance(result, tuple) else result
    return (tokens >= 0).sum(-1).clamp(max=num_draft + 1) - 1


def _assert_same_acceptance(result, reference: torch.Tensor, num_draft: int, batch: int) -> None:
    """Accepted lengths distributed as the reference's, length by length, in int32 ``[B, N+1]``."""
    tokens = result[0] if isinstance(result, tuple) else result
    assert tokens.shape == (batch, num_draft + 1), tokens.shape
    assert tokens.dtype == torch.int32, tokens.dtype
    bins = num_draft + 1
    got = torch.bincount(_accepted_lengths(result, num_draft).long(), minlength=bins).double()
    want = torch.bincount(reference.long(), minlength=bins).double()
    share = want / batch
    # Two independent batches of the same length distribution, so twice one batch's variance.
    bound = _LENGTH_SIGMAS * (2 * batch * share * (1 - share)).sqrt() + _LENGTH_SLACK
    assert ((got - want).abs() <= bound).all(), (got, want, bound)


@pytest.mark.parametrize("call", manifest_calls(ChainSpeculativeSamplingFwdOp))
def test_chain_speculative_sampling_bench(call) -> None:
    workload = ChainSpeculativeSamplingWorkload(call)
    draft_probs, draft_token_ids, target_probs, seed, offset = workload.gen_inputs()
    inputs = (draft_probs, draft_token_ids, target_probs, seed, offset)
    batch, num_draft, vocab = draft_probs.shape

    op = ChainSpeculativeSamplingFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)

    lengths = workload.ref_program(*inputs)[1]
    _assert_same_acceptance(op(*inputs), lengths, num_draft, batch)

    # The reference holds no graph break, so the torch-compile row times the compiled
    # reference rather than an eager one under a compiled tag.
    torch._dynamo.utils.counters.clear()
    compiled = compiled_reference(workload.ref_program)
    _assert_same_acceptance(compiled(*inputs), lengths, num_draft, batch)
    assert not torch._dynamo.utils.counters.get("graph_break", {})

    functors = {"tileops": op, "torch-ref": workload.ref_program, TORCH_COMPILE_TAG: compiled}

    # flashinfer's kernel takes the same inputs and the same Philox pair and applies the same
    # rule, drawing from its own stream.
    flashinfer_chain = flashinfer_op("sampling.chain_speculative_sampling")

    def flashinfer_verify(*args):
        return flashinfer_chain(
            draft_probs, draft_token_ids, target_probs, seed=seed, offset=offset
        )

    _assert_same_acceptance(flashinfer_verify(), lengths, num_draft, batch)
    functors[FLASHINFER_TAG] = flashinfer_verify

    # vllm's rejection sampler works on the flattened draft positions and takes target
    # logits, so it softmaxes every draft position itself; the bonus token is the caller's,
    # which its own sampler draws, so the timed call draws it too. That sampler divides its
    # probabilities in place, so the bonus row it is handed is a copy the timed call makes;
    # with count_copies left false the copy is not charged to the tag. rejection_sample
    # itself writes none of its inputs, and its are built once here, outside the timed call.
    rejection_sample = vllm_op("rejection_sample", "v1.sample.rejection_sampler")
    random_sample = vllm_op("random_sample", "v1.sample.ops.topk_topp_sampler")
    sampling_metadata = vllm_op("SamplingMetadata", "v1.sample.metadata")
    device = draft_probs.device
    metadata = sampling_metadata(
        temperature=torch.ones(batch, device=device),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch, device=device),
        presence_penalties=torch.zeros(batch, device=device),
        repetition_penalties=torch.ones(batch, device=device),
        output_token_ids=[[] for _ in range(batch)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=None,
    )
    vllm_draft_probs = draft_probs.reshape(batch * num_draft, vocab)
    vllm_target_logits = target_probs[:, :num_draft].reshape(batch * num_draft, vocab).log()
    vllm_bonus_probs = target_probs[:, num_draft]
    vllm_draft_token_ids = draft_token_ids.reshape(-1)
    counts = [num_draft] * batch
    cumulative = torch.arange(1, batch + 1, dtype=torch.int32, device=device) * num_draft

    def vllm_verify(*args):
        bonus = random_sample(vllm_bonus_probs.clone(), {}).to(torch.int32)[:, None]
        return rejection_sample(
            vllm_draft_token_ids,
            counts,
            num_draft,
            cumulative,
            vllm_draft_probs,
            vllm_target_logits,
            bonus,
            metadata,
        )

    _assert_same_acceptance(vllm_verify(), lengths, num_draft, batch)
    functors[VLLM_TAG] = vllm_verify

    bm.compare(functors, *inputs)
