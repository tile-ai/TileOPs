"""Torch reference of ``NGramHashFwdOp``."""

import torch

from workloads.device import run_device
from workloads.numerics import Exact
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = [
    "NGramHashWorkload",
    "NGramHashWorkloadCall",
    "ngram_hash",
    "ngram_hash_result",
]


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
    (``_shift_right_ignore_eos``, ``forward``) per sequence, its history read from and written
    back to *context* in place.

    Sequence ``b`` uses row ``context_indices[b]`` of *context* (row ``b`` without it). Its
    history is the ``n - 1`` columns from ``o = num_accepted_tokens[b] - 1`` (``o = 0``
    without it). Afterwards the row holds, from column 0, the last ``n - 1`` tokens of history
    and call tokens; with *num_accepted_tokens*, the history from column ``o + 1`` followed by
    every call token. A sequence with no tokens leaves its row untouched.

    Every sequence is computed at once: each sequence's history is laid before its tokens in
    one stream, and a shift stops at the last EOS before it, a running maximum of EOS places
    floored at the sequence start. There is no host loop or sync, so ``torch.compile``
    captures the whole function.
    """
    n, heads = multipliers.shape[0], primes.shape[0]
    width = context.shape[1]
    if n < 2 or primes.shape != offsets.shape or heads % (n - 1) or width < n - 1:
        raise ValueError("multipliers, primes, offsets and context do not fit together")
    hist, per_order = n - 1, heads // (n - 1)
    device = input_ids.device
    multipliers, primes, offsets = (t.to(device) for t in (multipliers, primes, offsets))
    if cu_seqlens is None:
        batch, length = input_ids.shape
        tokens = input_ids.reshape(-1)
        lengths = torch.full((batch,), length, dtype=torch.int64, device=device)
        starts = torch.arange(batch, dtype=torch.int64, device=device) * length
    else:
        batch = cu_seqlens.shape[0] - 1
        tokens = input_ids
        bounds = cu_seqlens.to(device, torch.int64)
        lengths, starts = bounds[1:] - bounds[:-1], bounds[:-1]
    total = tokens.shape[0]
    if batch == 0 or total == 0:
        return input_ids.new_empty((*input_ids.shape, heads))
    slots = (
        torch.arange(batch, device=device)
        if context_indices is None
        else context_indices.to(device, torch.int64)
    )
    first = (
        torch.zeros(batch, dtype=torch.int64, device=device)
        if num_accepted_tokens is None
        else num_accepted_tokens.to(device, torch.int64) - 1
    )
    # The stream: per sequence, its history then its tokens; j is the place in the sequence.
    stream_len = lengths + hist
    begin = torch.cumsum(stream_len, 0) - stream_len
    size = total + batch * hist
    seq = torch.repeat_interleave(torch.arange(batch, device=device), stream_len, output_size=size)
    p = torch.arange(size, device=device)
    j = p - begin[seq]
    history = context[slots[seq], (first[seq] + j).clamp(0, width - 1)]
    token = tokens[(starts[seq] + j - hist).clamp(0, total - 1)]
    stream = torch.where(j < hist, history, token)
    # The last EOS before each place, floored at the sequence start.
    floor = begin[seq] - 1
    eos_at = torch.cummax(torch.where(stream == eos_token_id, p, floor), 0).values
    previous_eos = torch.maximum(torch.cat((eos_at.new_full((1,), -1), eos_at[:-1])), floor)
    mixed = stream * multipliers[0]
    blocks = []
    for shift in range(1, n):
        valid = (p - (previous_eos + 1) >= shift) & (j >= shift)
        source = stream[(p - shift).clamp_min(0)]
        shifted = torch.where(valid, source, torch.full_like(source, eos_token_id))
        mixed = torch.bitwise_xor(mixed, shifted * multipliers[shift])
        heads_of_order = slice((shift - 1) * per_order, shift * per_order)
        blocks.append(
            torch.remainder(mixed.unsqueeze(-1), primes[heads_of_order]) + offsets[heads_of_order]
        )
    ids = torch.cat(blocks, dim=-1)
    token_seq = torch.repeat_interleave(
        torch.arange(batch, device=device), lengths, output_size=total
    )
    ids = ids[torch.arange(total, device=device) + (token_seq + 1) * hist]
    # Write back the kept history; places that keep nothing land on a spare last element.
    if num_accepted_tokens is None:
        column = j - (stream_len[seq] - hist)
        kept = column >= 0
    else:
        column = j - 1
        kept = j >= 1
    kept &= lengths[seq] > 0
    spare = context.shape[0] * width
    target = torch.where(kept, slots[seq] * width + column.clamp(0, width - 1), spare)
    flat = torch.cat((context.reshape(-1), context.new_zeros(1))).scatter(0, target, stream)
    context.copy_(flat[:spare].view(context.shape))
    return ids.view(*input_ids.shape, heads)


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
