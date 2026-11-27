"""Benchmark for the chain speculative sampling op."""

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
from tileops.sampling import ChainSpeculativeSamplingFwdOp


def _accepted_lengths(result, num_draft: int) -> torch.Tensor:
    """The accepted prefix length of each row of an ``[B, N + 1]`` output padded after the draw.

    Each implementation returns that tensor first and its own extra counts after it; the
    padding is the only mark of where the chain stopped that all of them carry.
    """
    tokens = result[0] if isinstance(result, tuple) else result
    return (tokens >= 0).sum(-1).clamp(max=num_draft + 1) - 1


@pytest.mark.parametrize("case", bench.cases(ChainSpeculativeSamplingFwdOp))
def test_chain_speculative_sampling_bench(case) -> None:
    op = ChainSpeculativeSamplingFwdOp(**case.arguments)
    compiled = compiled_reference(case.reference)
    implementations = {"torch-ref": case.reference, TORCH_COMPILE_TAG: compiled}
    flashinfer_chain = flashinfer_op("sampling.chain_speculative_sampling")

    def flashinfer_verify(draft, draft_ids, target, seed, offset):
        result = flashinfer_chain(draft, draft_ids, target, seed=seed, offset=offset)
        return result[0], _accepted_lengths(result, draft.shape[1]).to(torch.int32)

    implementations[FLASHINFER_TAG] = flashinfer_verify
    rejection_sample = vllm_op(
        "rejection_sample", "v1.worker.gpu.spec_decode.rejection_sampler_utils"
    )
    draft_logits_cache = vllm_op("DraftLogitsCache", "v1.worker.gpu.spec_decode.draft_logits")

    # vllm verifies logits: request b owns logits b * (N + 1) .. b * (N + 1) + N, the last
    # one the bonus, and reads draft step i's token at flat position b * (N + 1) + i + 1.
    # Everything is built from the case's inputs here, outside the timed call.
    draft, draft_ids, target, seed, _offset = case.inputs
    batch, num_draft, vocab = draft.shape
    device = draft.device
    cache = draft_logits_cache(
        batch, num_draft, vocab, dtype=torch.float32, fill=0.0, device=device
    )
    torch.log(draft, out=cache.values)
    target_logits = target.log().reshape(batch * (num_draft + 1), vocab)
    draft_sampled = torch.zeros(batch, num_draft + 1, dtype=torch.int64, device=device)
    draft_sampled[:, 1:] = draft_ids
    requests = torch.arange(batch, dtype=torch.int32, device=device)
    steps = torch.arange(num_draft + 1, dtype=torch.int32, device=device)
    verify_args = (
        target_logits,
        cache,
        draft_sampled.reshape(-1),
        torch.arange(batch + 1, dtype=torch.int32, device=device) * (num_draft + 1),
        torch.arange(batch * (num_draft + 1), dtype=torch.int64, device=device),
        requests,
        requests.repeat_interleave(num_draft + 1),
        steps.repeat(batch),
        torch.ones(batch, dtype=torch.float32, device=device),
        torch.full((batch,), int(seed.item()), dtype=torch.int64, device=device),
        num_draft,
    )

    def vllm_verify():
        sampled, num_sampled = rejection_sample(*verify_args)
        accepted = num_sampled - 1
        # vllm leaves the positions after the drawn token unwritten; the op pads them with -1.
        tokens = sampled.masked_fill(steps > accepted[:, None], -1).to(torch.int32)
        return tokens, accepted

    implementations[VLLM_TAG] = bench.Implementation(
        run=vllm_verify,
        args=(),
        noncomparable_reason=(
            "adapter runs on logits built from the case's inputs and draws from vLLM's "
            "per-request seed and position streams instead of the call's seed/offset"
        ),
    )
    bench.Runner(op, case).compare(implementations)
