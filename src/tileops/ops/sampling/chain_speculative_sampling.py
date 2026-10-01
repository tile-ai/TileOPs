"""The chain speculative sampling op."""

from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.sampling import (
    ChainSpeculativeSamplingFwdInterface,
    ChainSpeculativeSamplingFwdKernel,
    SamplingCall,
)
from tileops.ops.op_base import Op

__all__ = ["ChainSpeculativeSamplingFwdOp"]


class ChainSpeculativeSamplingFwdOp(Op):
    """Chain speculative sampling: verifies each request's ``N`` draft tokens in order.

    Draft token ``i`` is accepted with probability ``min(1, target / draft)`` at its id, as
    FlashInfer's ``chain_speculative_sampling`` tests ``u * draft < target``. At the first
    rejection a token is drawn from ``normalize(max(0, target - draft))`` of that position
    and the rest of the row is ``-1``; when every draft is accepted a bonus token is drawn
    from target row ``N``. ``num_accepted`` is the accepted prefix length, the bonus
    excluded. Every probability row finite, non-negative and normalized, and each draft
    token of positive draft probability, is the caller's obligation.

    The draws are a function of the Philox state ``(seed, offset)``: the same pair and
    inputs give the same result. Which stream a pair selects belongs to the implementation,
    so two implementations need not draw the same tokens from one pair.

    The in-tree kernel takes the stream ``workloads/sampling.py`` states: uniform ``j`` of
    a row is the first output word of Philox4x32-10 keyed by ``(seed, offset)`` and counted
    by ``(j, row)``, draws ``0`` to ``N - 1`` testing the drafts and draw ``N`` placing the
    token. The counter names no launch fact and the acceptance test is the reference's own
    float32 comparison, so ``num_accepted`` and the accepted prefix are the reference's
    exactly, whatever the batch and whatever grid the call takes.
    """

    compile_boundary: ClassVar[bool] = True

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "chain_speculative_sampling": ChainSpeculativeSamplingFwdKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "chain_speculative_sampling": ChainSpeculativeSamplingFwdInterface
    }

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtypes are taken from each call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(
        self,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_probs: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Verify the draft chains and draw the token after each accepted prefix.

        Args:
            draft_probs: ``[B, N, V]`` float32 draft model probabilities.
            draft_token_ids: ``[B, N]`` int32 draft tokens, each in ``[0, V)``.
            target_probs: ``[B, N + 1, V]`` float32 target model probabilities.
            seed: ``[1]`` int64 Philox seed.
            offset: ``[1]`` int64 Philox offset.

        Returns:
            ``output_token_ids``, ``[B, N + 1]`` int32: the accepted drafts, the drawn token,
            then ``-1``; and ``num_accepted``, ``[B]`` int32.
        """
        return self._call_boundary(draft_probs, draft_token_ids, target_probs, seed, offset)

    def _eager_forward(
        self,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_probs: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve the kernel and launch, inside the operator."""
        draft_probs = draft_probs.contiguous()
        draft_token_ids = draft_token_ids.contiguous()
        target_probs = target_probs.contiguous()
        seed = seed.contiguous()
        offset = offset.contiguous()
        batch, num_draft, vocab = draft_probs.shape
        call = SamplingCall(
            device=draft_probs.device,
            batch=batch,
            vocab=vocab,
            dtype=draft_probs.dtype,
            num_draft=num_draft,
        )
        inputs = (draft_probs, draft_token_ids, target_probs, seed, offset)
        kernel = self.kernel_for("chain_speculative_sampling", call)
        return kernel(*inputs)
