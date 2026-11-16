"""Torch reference of ``NGramHashFwdOp``."""

import torch

from workloads.device import run_device
from workloads.numerics import Exact
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = ["NGramHashWorkload", "NGramHashWorkloadCall", "ngram_hash", "ngram_hash_result"]


def _shift_right_ignore_eos(tokens: torch.Tensor, shift: int, eos: int) -> torch.Tensor:
    """transformers ``Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos`` on one row."""
    if shift == 0:
        return tokens
    positions = torch.arange(tokens.shape[0], device=tokens.device)
    eos_positions = torch.where(tokens == eos, positions, -1)
    previous_eos = torch.cat(
        (eos_positions.new_full((1,), -1), eos_positions.cummax(0).values[:-1])
    )
    source = positions - shift
    shifted = tokens[source.clamp_min(0)]
    valid = (positions - (previous_eos + 1) >= shift) & (source >= 0)
    return torch.where(valid, shifted, torch.full_like(tokens, eos))


def _hash(history: torch.Tensor, multipliers, primes, offsets, eos: int) -> torch.Tensor:
    """The ids of every position of *history*, ``[len, heads]``."""
    n, heads = multipliers.shape[0], primes.shape[0]
    per_order = heads // (n - 1)
    shifted = [_shift_right_ignore_eos(history, i, eos) * multipliers[i] for i in range(n)]
    blocks = []
    for order in range(2, n + 1):
        mixed = shifted[0]
        for i in range(1, order):
            mixed = torch.bitwise_xor(mixed, shifted[i])
        heads_of_order = slice((order - 2) * per_order, (order - 1) * per_order)
        ids = torch.remainder(mixed.unsqueeze(-1), primes[heads_of_order].to(mixed.device))
        blocks.append(ids + offsets[heads_of_order])
    return torch.cat(blocks, dim=-1)


def ngram_hash(
    input_ids: torch.Tensor,
    context: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    eos_token_id: int,
    cu_seqlens: torch.Tensor | None = None,
    context_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """``NGramHashFwdOp``'s reference: transformers ``Qwen4ExpTextNGramEmbedding``'s hashing
    per sequence, its history read from and written back to *context* in place.

    Sequence ``b`` uses row ``context_indices[b]`` of *context* (row ``b`` without it). Its
    history is the ``n - 1`` columns from ``o = num_accepted_tokens[b] - 1`` (``o = 0``
    without it). Afterwards the row holds, from column 0, the last ``n - 1`` tokens of history
    and call tokens; with *num_accepted_tokens*, the history from column ``o + 1`` followed by
    every call token. A sequence with no tokens leaves its row untouched.
    """
    n = multipliers.shape[0]
    width = context.shape[1]
    if n < 2 or primes.shape != offsets.shape or primes.shape[0] % (n - 1) or width < n - 1:
        raise ValueError("multipliers, primes, offsets and context do not fit together")
    if cu_seqlens is None:
        sequences = list(input_ids)
    else:
        bounds = cu_seqlens.tolist()
        sequences = [input_ids[a:e] for a, e in zip(bounds, bounds[1:], strict=False)]
    slots = range(len(sequences)) if context_indices is None else context_indices.tolist()
    if context.shape[0] != len(sequences) and context_indices is None:
        raise ValueError("without context_indices the context holds a row per sequence")
    accepted = None if num_accepted_tokens is None else num_accepted_tokens.tolist()
    results = []
    for b, (tokens, slot) in enumerate(zip(sequences, slots, strict=True)):
        if tokens.shape[0] == 0:
            results.append(tokens.new_empty((0, primes.shape[0])))
            continue
        start = 0 if accepted is None else accepted[b] - 1
        before = context[slot, start : start + n - 1]
        history = torch.cat((before, tokens))
        results.append(_hash(history, multipliers, primes, offsets, eos_token_id)[n - 1 :])
        kept = history[1:] if accepted is not None else history[-(n - 1) :]
        if kept.shape[0] > width:
            raise ValueError("the context row is too short for the history it keeps")
        context[slot, : kept.shape[0]] = kept
    if cu_seqlens is not None:
        return torch.cat(results) if results else input_ids.new_empty((0, primes.shape[0]))
    return (
        torch.stack(results)
        if results
        else input_ids.new_empty((0, *input_ids.shape[1:], primes.shape[0]))
    )


def ngram_hash_result(subject, *inputs):
    """Expose both ids and the complete mutated pool for numerical verification."""
    ids = subject(*inputs)
    return ids, inputs[1]


class NGramHashWorkload(WorkloadBase):
    """Dense or packed tokens with EOS boundaries, overflow and optional state slots."""

    def __init__(
        self,
        lengths,
        *,
        n=3,
        heads=16,
        packed=False,
        slots=None,
        pool=None,
        width=None,
        accepted=None,
        eos_token_id=248044,
        extremes=False,
    ):
        self.lengths = list(lengths)
        self.n, self.heads, self.packed = n, heads, packed
        self.slots, self.accepted = slots, accepted
        self.pool = len(lengths) if pool is None else pool
        self.width = n - 1 if width is None else width
        self.eos_token_id = eos_token_id
        self.extremes = extremes

    def gen_inputs(self):
        device = run_device()
        tokens = torch.randint(
            0, 248044, (sum(self.lengths),), dtype=torch.int64, generator=self.rng("tokens")
        ).to(device)
        pos = 0
        for length in self.lengths:
            if length:
                tokens[pos + length // 2] = self.eos_token_id
            if length > 3:
                tokens[pos] = self.eos_token_id
                tokens[pos + length // 2 + 1] = self.eos_token_id
            pos += length
        context = torch.randint(
            0, 248044, (self.pool, self.width), dtype=torch.int64, generator=self.rng("context")
        ).to(device)
        if self.pool:
            context[0, 0] = self.eos_token_id
        # Large, odd multipliers force signed overflow in normal vocabulary inputs.
        mult = torch.tensor(
            [6364136223846793005 - 2 * i for i in range(self.n)], dtype=torch.int64, device=device
        )
        primes = torch.tensor(
            [20000003 + 2 * i for i in range(self.heads)], dtype=torch.int64, device=device
        )
        offsets = torch.arange(self.heads, dtype=torch.int64, device=device) * 20000003
        if self.extremes:
            values = torch.tensor([-(2**63), 2**63 - 1, -1, 0, 1], device=device)
            tokens.copy_(values.repeat((tokens.numel() + 4) // 5)[: tokens.numel()])
            primes.copy_(
                torch.tensor([1, 2**63 - 1, 20000003], device=device).repeat((self.heads + 2) // 3)[
                    : self.heads
                ]
            )
            offsets.fill_(2**63 - 1)
        cu = (
            torch.tensor(
                [0, *torch.tensor(self.lengths).cumsum(0).tolist()],
                dtype=torch.int32,
                device=device,
            )
            if self.packed
            else None
        )
        if not self.packed:
            tokens = tokens.reshape(len(self.lengths), self.lengths[0] if self.lengths else 0)
        slots = (
            None
            if self.slots is None
            else torch.tensor(self.slots, dtype=torch.int32, device=device)
        )
        accepted = (
            None
            if self.accepted is None
            else torch.tensor(self.accepted, dtype=torch.int32, device=device)
        )
        return tokens, context, mult, primes, offsets, cu, slots, accepted

    def ref_program(
        self,
        input_ids,
        context,
        multipliers,
        primes,
        offsets,
        cu_seqlens=None,
        context_indices=None,
        num_accepted_tokens=None,
    ):
        context = context.clone()
        ids = ngram_hash(
            input_ids,
            context,
            multipliers,
            primes,
            offsets,
            self.eos_token_id,
            cu_seqlens,
            context_indices,
            num_accepted_tokens,
        )
        return ids, context

    def torch_program(
        self,
        input_ids,
        context,
        multipliers,
        primes,
        offsets,
        cu_seqlens=None,
        context_indices=None,
        num_accepted_tokens=None,
    ):
        """The torch computation on private context, reset by the benchmark."""
        ids = ngram_hash(
            input_ids,
            context,
            multipliers,
            primes,
            offsets,
            self.eos_token_id,
            cu_seqlens,
            context_indices,
            num_accepted_tokens,
        )
        return ids, context

    def verification(self, *inputs):
        return Exact()


class NGramHashWorkloadCall(CallWorkload, NGramHashWorkload):
    """The manifest call with nontrivial token, multiplier and offset values."""

    def __init__(self, call):
        CallWorkload.__init__(self, call)
        lengths = (
            [
                b - a
                for a, b in zip(
                    call.metadata_values("cu_seqlens")[:-1],
                    call.metadata_values("cu_seqlens")[1:],
                    strict=True,
                )
            ]
            if call.present("cu_seqlens")
            else [call.indices["L"]] * call.indices["B"]
        )
        NGramHashWorkload.__init__(
            self,
            lengths,
            n=call.indices["N"],
            heads=call.indices["Hd"],
            packed=call.present("cu_seqlens"),
            slots=call.metadata_values("context_indices")
            if call.present("context_indices")
            else None,
            pool=call.indices.get("S", len(lengths)),
            width=call.indices["W"],
            accepted=call.metadata_values("num_accepted_tokens")
            if call.present("num_accepted_tokens")
            else None,
            eos_token_id=call.params["eos_token_id"],
        )

    gen_inputs = NGramHashWorkload.gen_inputs
