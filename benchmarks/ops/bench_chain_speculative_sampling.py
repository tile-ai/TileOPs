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


def _accepted_lengths(result, num_draft: int) -> torch.Tensor:
    """The accepted prefix length of each row of an ``[B, N + 1]`` output padded after the draw.

    Each implementation returns that tensor first and its own extra counts after it; the
    padding is the only mark of where the chain stopped that all of them carry.
    """
    tokens = result[0] if isinstance(result, tuple) else result
    return (tokens >= 0).sum(-1).clamp(max=num_draft + 1) - 1


@pytest.mark.parametrize("call", manifest_calls(ChainSpeculativeSamplingFwdOp))
def test_chain_speculative_sampling_bench(call) -> None:
    workload = ChainSpeculativeSamplingWorkload(call)
    draft_probs, draft_token_ids, target_probs, seed, offset = workload.gen_inputs()
    inputs = (draft_probs, draft_token_ids, target_probs, seed, offset)
    batch, num_draft, vocab = draft_probs.shape
    op = ChainSpeculativeSamplingFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)
    compiled = compiled_reference(workload.ref_program)
    functors = {"tileops": op, "torch-ref": workload.ref_program, TORCH_COMPILE_TAG: compiled}
    flashinfer_chain = flashinfer_op("sampling.chain_speculative_sampling")

    def flashinfer_verify(*args):
        result = flashinfer_chain(
            draft_probs, draft_token_ids, target_probs, seed=seed, offset=offset
        )
        return result[0], _accepted_lengths(result, num_draft).to(torch.int32)

    functors[FLASHINFER_TAG] = flashinfer_verify
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
        result = rejection_sample(
            vllm_draft_token_ids,
            counts,
            num_draft,
            cumulative,
            vllm_draft_probs,
            vllm_target_logits,
            bonus,
            metadata,
        )
        return result, _accepted_lengths(result, num_draft).to(torch.int32)

    functors[VLLM_TAG] = vllm_verify
    bm.compare(functors, *inputs)
