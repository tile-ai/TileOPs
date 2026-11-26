from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.attention.gated_pool_compress import (
    GatedPoolCompressCall,
    GatedPoolCompressFwdInterface,
    GatedPoolCompressKernel,
)
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["GatedPoolCompressFwdOp"]


class GatedPoolCompressFwdOp(Op):
    """Pool every ``ratio`` positions of each request into one entry, advancing its ring in place.

    Per channel, a window's entry is the softmax over its positions of ``score + ape``
    (``ape`` indexed by the position's slot in the window) in float32, the weights and their
    products with ``kv`` rounded to ``kv``'s dtype and summed in float32, rounded once to
    bfloat16. A window of more than 16 positions with bfloat16 ``kv`` sums unrounded float32
    products instead, within the entry's bfloat16 tolerance. The kernel flushes float32
    subnormals to zero, a ring's ``score + ape`` rows among them. With ``overlap`` a window also
    pools the first ``D`` channels of the previous window's positions, its own giving the
    last ``D``; the first window has no previous one.

    ``C`` is 2 with ``overlap``, else 1. Request ``i`` holds the packed tokens
    ``cu_seqlens[i]:cu_seqlens[i + 1]`` at positions from ``start_pos[i]``. The token at
    position ``p`` with ``(p + 1) % ratio == 0`` closes a window: ``valid`` is true at it and
    ``entries`` holds the window's entry; elsewhere ``entries`` is unspecified. Positions
    before ``start_pos[i]`` come from the ring of slot ``state_slots[i]``, row
    ``p % (C * ratio)``, which holds ``kv`` in ``kv_state`` and ``score + ape`` in
    ``score_state``; the call writes the rows of the request's last ``C * ratio`` positions.
    ``start_pos[i]`` 0 starts a request afresh; otherwise it is the end of the request's last
    call or one position before it, rolling back the last token it passed. The entries and
    the ring rows of a request's tokens do not depend on how the tokens split across calls: a
    prefill and then one call a token give exactly what one call of them all gives, and calls
    that each pass a token and a draft the next call rolls back give exactly what one call a
    token gives. Distinct ``state_slots`` are the caller's obligation. The
    caller guarantees well-formed ``cu_seqlens`` and ``state_slots`` in ``[0, NS)``, or
    passes ``validate_inputs=True``, which checks both on the CPU on every call.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gated_pool_compress": GatedPoolCompressKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "gated_pool_compress": GatedPoolCompressFwdInterface
    }

    def __init__(
        self,
        ratio: int,
        overlap: bool = False,
        *,
        validate_inputs: bool = False,
        target: Target = None,
    ) -> None:
        """Build the op.

        Args:
            ratio: Manifest ``params.ratio``, ``int``: the positions a window pools.
            overlap: Manifest ``params.overlap``, ``bool``: whether a window also pools the
                previous window's first ``D`` channels, ``kv`` and ``score`` then holding
                ``2 * D`` channels.
            validate_inputs: Manifest ``params.validate_inputs``, ``bool``: check
                ``cu_seqlens`` and the ``state_slots`` range on the CPU before each call.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.ratio = ratio
        self.overlap = overlap
        self.validate_inputs = validate_inputs
        super().__init__(target=target)

    def forward(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        start_pos: torch.Tensor,
        state_slots: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Pool the windows the tokens close and write their ring rows.

        Args:
            kv: Values ``[T_q, C * D]``, ``float32`` or ``bfloat16``; ``C`` is 2 with
                ``overlap``, else 1. Any strides: the column halves of one
                ``[T_q, 2 * C * D]`` projection are read where they lie.
            score: Gate logits, as ``kv``.
            ape: Per-slot gate bias ``[ratio, C * D]``, ``kv``'s dtype.
            kv_state: Ring of ``kv`` rows ``[NS, C * ratio, C * D]``, ``float32``; written.
            score_state: Ring of ``score + ape`` rows, as ``kv_state``; written.
            cu_seqlens: Packed offsets of the requests ``[N + 1]``, ``int32``.
            start_pos: Position of each request's first token ``[N]``, ``int32``.
            state_slots: Ring slot of each request ``[N]``, ``int32``.

        Returns:
            ``entries`` ``[T_q, D]`` ``bfloat16``, defined at the closing tokens, and
            ``valid`` ``[T_q]`` ``bool``, true at them.
        """
        dim = kv.shape[1] // (2 if self.overlap else 1)
        kernel = self.kernel_for(
            "gated_pool_compress",
            lambda: GatedPoolCompressCall(
                ratio=self.ratio,
                overlap=self.overlap,
                dim=dim,
                dtype=kv.dtype,
                device=kv.device,
            ),
        )
        return kernel(
            kv if kv.stride(-1) == 1 else kv.contiguous(),
            score if score.stride(-1) == 1 else score.contiguous(),
            ape.contiguous(),
            kv_state,
            score_state,
            cu_seqlens.contiguous(),
            start_pos.contiguous(),
            state_slots.contiguous(),
        )
