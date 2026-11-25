"""The request bookkeeping, decode-step CUDA graphs and greedy generation every model here
shares."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import torch

__all__ = ["CausalLM", "indexed_device"]


class CausalLM:
    """A decoder on one device that advances one request at a time.

    A subclass sets ``config`` (with ``eos_token_id``), ``max_seq_len``, ``ops`` (with
    ``argmax``), ``device``, ``layers`` and ``cuda_graph``, and implements
    ``_step(start_pos, seqlen)``, the index metadata of a call, and ``_run(ids, step)``, the
    float32 logits ``[1, vocab_size]`` of the last of ``ids`` (``[1, S]`` on ``device``).

    With ``cuda_graph`` on a CUDA device, decode steps, one token at ``start_pos > 0``, replay
    CUDA graphs of ``_run`` on a token buffer and a step that the step's ``moved_to`` rewrites
    in place for each position; the weights and the decoding state must keep the addresses the
    graphs captured. A decode step's host-side branches, which ``_phase`` names, select one
    graph per phase. The first decode step of a phase instead runs eagerly and then captures
    the phase's graph, on the model's own stream; all phases share one memory pool. Replay
    rests on each model's tests that it equals the eager step bitwise, not on
    ``tests/cuda_graph_contract.py``: most ops a decode step calls are outside the CUDA Graph
    support scope of ``docs/stability.md``.
    """

    config: Any
    max_seq_len: int
    ops: Any
    device: torch.device
    layers: list[Any]
    cuda_graph: bool
    _next_pos: int | None = None
    _graphs: dict[Hashable, tuple[torch.cuda.CUDAGraph, torch.Tensor]] | None = None
    _graph_pool: tuple[int, int]
    _graph_stream: torch.cuda.Stream
    _graph_ids: torch.Tensor
    _graph_step: Any

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

    def _phase(self, step: Any) -> Hashable:
        """The host-side branches the decode step ``step`` takes; none by default."""
        return ()

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Run ``input_ids`` (``[1, S]``) from ``start_pos``; return the float32 logits of its
        last token, ``[1, vocab_size]``."""
        seqlen = self._continue(input_ids, start_pos)
        ids = input_ids.to(self.device)
        with torch.accelerator.device_index(self.device.index):
            if self.cuda_graph and self.device.type == "cuda" and start_pos and seqlen == 1:
                logits = self._replay(ids, start_pos)
            else:
                logits = self._run(ids, self._step(start_pos, seqlen))
        self._next_pos = start_pos + seqlen
        return logits

    __call__ = forward

    def _replay(self, ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """The logits of the decode step of ``ids`` (``[1, 1]``) at ``start_pos``, replayed from
        the graph of its phase. The first step of a phase runs eagerly on a step of its own, so
        that the step the graph captures derives anything it caches inside the capture, then
        captures it."""
        if self._graphs is None:
            self._graphs = {}
            self._graph_pool = torch.cuda.graph_pool_handle()
            self._graph_stream = torch.cuda.Stream(self.device)
            self._graph_ids = torch.empty_like(ids)
            self._graph_step = self._step(start_pos, 1)
        self._graph_ids.copy_(ids)
        step = self._graph_step.moved_to(start_pos)
        phase = self._phase(step)
        if phase in self._graphs:
            graph, logits = self._graphs[phase]
            graph.replay()
            return logits.clone()
        logits = self._run(ids, self._step(start_pos, 1))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._graph_pool, stream=self._graph_stream):
            out = self._run(self._graph_ids, step)
        self._graphs[phase] = graph, out
        return logits

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
