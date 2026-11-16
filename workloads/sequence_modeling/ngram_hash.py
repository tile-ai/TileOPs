"""Torch reference of ``NGramHashFwdOp``."""

import torch

__all__ = ["ngram_hash"]


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
