"""Hashed n-gram ids with end-of-sequence bounded history."""

from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.sequence_modeling.ngram_hash import (
    NGramHashCall,
    NGramHashFwdInterface,
    NGramHashKernel,
)
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["NGramHashFwdOp"]


class NGramHashFwdOp(Op):
    """Hash n-grams and update their context in place.

    Each head hashes an order from two through the multiplier count, with equal
    heads per order. Products and sums wrap in int64, and remainders are nonnegative.
    A previous end-of-sequence (EOS) token blocks older history; the current EOS
    participates normally. Context stores raw tokens, oldest first.

    Distinct context_indices select pool rows. num_accepted_tokens selects the
    history window starting at accepted minus one, then preserves its last N-2
    tokens and every input token for speculative rollback. Without it, only the
    last N-1 tokens are committed. Empty sequences leave their rows untouched.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"ngram_hash": NGramHashKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "ngram_hash": NGramHashFwdInterface
    }

    def __init__(self, eos_token_id: int, *, validate_inputs: bool = False, target: Target = None):
        """Build the op.

        Args:
            eos_token_id: Token used at missing history and EOS boundaries.
            validate_inputs: Check packed offsets and metadata bounds on the CPU.
            target: Backend target, or None to select from the input device.
        """
        self.eos_token_id = eos_token_id
        self.validate_inputs = validate_inputs
        super().__init__(target=target)

    def forward(
        self,
        input_ids: torch.Tensor,
        context: torch.Tensor,
        multipliers: torch.Tensor,
        primes: torch.Tensor,
        offsets: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        context_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return int64 hashed ids, updating the addressed context rows.

        Args:
            input_ids: Dense [B, L] or packed [T] int64 tokens.
            context: Int64 context pool [S, W], written in place.
            multipliers: Int64 [N] hash multipliers.
            primes: Positive int64 [heads] divisors.
            offsets: Int64 [heads] embedding offsets.
            cu_seqlens: Optional int32 [B+1] packed prefix offsets.
            context_indices: Optional distinct int32 [B] pool row indices.
            num_accepted_tokens: Optional int32 [B] accepted counts, at least one.

        Returns:
            Ids shaped [B, L, heads] or [T, heads].
        """
        kernel = self.kernel_for(
            "ngram_hash",
            lambda: NGramHashCall(
                device=input_ids.device,
                width=context.shape[1],
                row_stride=context.stride(0),
                col_stride=context.stride(1),
                order=multipliers.numel(),
                heads=primes.numel(),
                eos=self.eos_token_id,
                packed=cu_seqlens is not None,
                slotted=context_indices is not None,
                rollback=num_accepted_tokens is not None,
            ),
        )
        return kernel(
            input_ids.contiguous(),
            context,
            multipliers.contiguous(),
            primes.contiguous(),
            offsets.contiguous(),
            None if cu_seqlens is None else cu_seqlens.contiguous(),
            None if context_indices is None else context_indices.contiguous(),
            None if num_accepted_tokens is None else num_accepted_tokens.contiguous(),
        )
