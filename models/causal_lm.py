"""The request bookkeeping and greedy generation every model here shares."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

__all__ = ["CausalLM", "indexed_device"]


class CausalLM:
    """A decoder on one device that advances one request at a time.

    A subclass sets ``config`` (with ``eos_token_id``), ``max_seq_len``, ``ops`` (with
    ``argmax``), ``device`` and ``layers``, and its ``forward`` calls ``_continue`` first and
    sets ``_next_pos`` to the position after its last token.
    """

    config: Any
    max_seq_len: int
    ops: Any
    device: torch.device
    layers: list[Any]
    _next_pos: int | None = None

    def reset(self) -> None:
        """End the request; the next call must start at position 0."""
        self._next_pos = None

    def _continue(self, input_ids: torch.Tensor, start_pos: int) -> int:
        """Check that ``input_ids`` (``[1, S]``) is one request's tokens from ``start_pos``: 0
        resets and starts a new request, any later position must be the number of tokens
        already passed. Returns ``S``."""
        batch, seqlen = input_ids.shape
        if batch != 1:
            raise ValueError(f"forward runs one request, got a batch of {batch}")
        if seqlen == 0:
            raise ValueError("forward takes at least one token")
        if start_pos + seqlen > self.max_seq_len:
            raise ValueError(
                f"position {start_pos + seqlen} exceeds max_seq_len {self.max_seq_len}"
            )
        if start_pos == 0:
            self.reset()
        elif self._next_pos is None:
            raise ValueError("a call at start_pos > 0 needs a prefill at start_pos 0 first")
        elif start_pos != self._next_pos:
            raise ValueError(f"the request continues at position {self._next_pos}, got {start_pos}")
        return seqlen

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        *,
        eos_token_id: int | Sequence[int] | None = None,
    ) -> list[int]:
        """Greedily extend one prompt by up to ``max_new_tokens`` tokens; stops after an EOS
        token (``config.eos_token_id`` unless given) or once prompt and new tokens fill
        ``max_seq_len``."""
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        tokens = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.device)
        logits = self.forward(tokens, 0)
        pos = tokens.shape[1]
        budget = min(max_new_tokens, self.max_seq_len - pos)
        out: list[int] = []
        for _ in range(budget):
            with torch.accelerator.device_index(self.device.index):
                nxt = self.ops.argmax(logits)
            token = int(nxt.item())
            out.append(token)
            if token in eos_ids or len(out) == budget:
                break
            logits = self.forward(nxt.view(1, 1), pos)
            pos += 1
        return out


def indexed_device(device: torch.device | str) -> torch.device:
    """``device`` with an index: a device without one takes the current device's."""
    device = torch.device(device)
    if device.index is None:
        return torch.device(device.type, torch.accelerator.current_device_index())
    return device
