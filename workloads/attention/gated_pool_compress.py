"""Torch reference of the learned gated-softmax pooling of every ``ratio`` tokens into one entry.

The pooling follows DeepSeek-V4-Flash ``inference/model.py`` ``Compressor.forward`` and
transformers ``Glm5NextTextIndexer.get_pooled_states``. The per-request state is a ring:
row ``p % (c * ratio)`` holds the request's token at position ``p``, its kv row in
``kv_state`` and its score plus ``ape`` in ``score_state``.
"""

import torch


def gated_pool_compress(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    start_pos: torch.Tensor,
    state_slots: torch.Tensor,
    ratio: int,
    overlap: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """``(entries [T, D] bfloat16, valid [T] bool)`` of packed ``kv``/``score`` ``[T, c * D]``.

    Request ``i`` holds tokens ``cu_seqlens[i]:cu_seqlens[i + 1]`` at positions from
    ``start_pos[i]``; the token at position ``p`` with ``(p + 1) % ratio == 0`` closes a
    window, sets ``valid`` and writes its entry. The entry softmaxes ``score + ape`` per
    channel over the window's tokens in float32, rounds the weights and their products with
    kv to kv's dtype, and sums the products in float32. With ``overlap`` (``c = 2``) the
    window also takes the first ``D`` channels of the previous window's tokens, the current
    ones giving the last ``D``. Earlier positions come from the state of slot
    ``state_slots[i]``, and the call writes the state rows of its tokens.
    """
    c = 2 if overlap else 1
    span, width = c * ratio, kv.shape[1]
    d = width // c
    if (
        score.shape != kv.shape
        or ape.shape != (ratio, width)
        or kv_state.shape[1:] != (span, width)
        or score_state.shape != kv_state.shape
        or width % c
    ):
        raise ValueError("kv, score, ape and the states must agree with ratio and overlap")
    bounds, starts = cu_seqlens.tolist(), start_pos.tolist()
    slots = state_slots.tolist()
    if len(bounds) != len(starts) + 1 or len(slots) != len(starts) or bounds[-1] != kv.shape[0]:
        raise ValueError("cu_seqlens, start_pos and state_slots must agree with kv")
    entries = torch.zeros(kv.shape[0], d, dtype=torch.bfloat16, device=kv.device)
    valid = torch.zeros(kv.shape[0], dtype=torch.bool, device=kv.device)
    for a, e, first, slot in zip(bounds[:-1], bounds[1:], starts, slots, strict=True):
        own = torch.arange(first, first + e - a)
        prior = torch.arange(max(0, first - span + 1), first)
        rows = torch.cat([prior, own]) % span
        kv_seq = torch.cat([kv_state[slot, prior % span], kv[a:e].float()])
        score_seq = torch.cat(
            [score_state[slot, prior % span], score[a:e].float() + ape.float()[own % ratio]]
        )
        closing = own[(own + 1) % ratio == 0]
        if len(closing):
            # Slot j of a window is the token at its start + j; position p is row
            # p - first + len(prior) of the sequences.
            current = closing[:, None] - ratio + 1 + torch.arange(ratio)
            at = current - first + prior.numel()
            parts_kv = [kv_seq[at][..., width - d :]]
            parts_score = [score_seq[at][..., width - d :]]
            if overlap:
                previous = current - ratio
                missing = (previous < 0)[..., None].to(kv.device)
                at = previous.clamp(min=0) - first + prior.numel()
                parts_kv.insert(0, kv_seq[at][..., :d].masked_fill(missing, 0.0))
                parts_score.insert(0, score_seq[at][..., :d].masked_fill(missing, float("-inf")))
            weights = torch.cat(parts_score, 1).softmax(1).to(kv.dtype)
            pooled = (weights * torch.cat(parts_kv, 1).to(kv.dtype)).float().sum(1)
            out = a + closing - first
            entries[out] = pooled.to(torch.bfloat16)
            valid[out] = True
        kept = slice(max(0, rows.numel() - span), rows.numel())
        kv_state[slot, rows[kept]] = kv_seq[kept]
        score_state[slot, rows[kept]] = score_seq[kept]
    return entries, valid
