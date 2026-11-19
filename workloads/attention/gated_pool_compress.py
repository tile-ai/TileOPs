"""Torch reference of the learned gated-softmax pooling of every ``ratio`` tokens into one entry.

The pooling follows DeepSeek-V4-Flash ``inference/model.py`` ``Compressor.forward`` and
transformers ``Glm5NextTextIndexer.get_pooled_states``. The per-request state is a ring:
row ``p % (c * ratio)`` holds the request's token at position ``p``, its kv row in
``kv_state`` and its score plus ``ape`` in ``score_state``.
"""

import torch

from workloads.workload_base import CallWorkload


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


def gated_pool_compress_torch(
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
    """:func:`gated_pool_compress` as batched torch ops, every request at once.

    Each closing token gathers its window's rows from the tokens and the rings, and the
    rings take the rows of each request's last ``c * ratio`` positions after every window is
    read. The softmax runs in float32 and the weights and products in kv's dtype, as the
    models run it.
    """
    c = 2 if overlap else 1
    t, width = kv.shape
    d, span = width // c, c * ratio
    dev = kv.device
    cu = cu_seqlens.long()
    tok = torch.arange(t, device=dev)
    req = torch.searchsorted(cu[1:], tok, right=True)
    first = start_pos.long()[req]
    pos = first + tok - cu[req]
    slot = state_slots.long()[req]
    tok_score = score.float() + ape.float()[pos % ratio]
    # The tokens' rows, then every ring's.
    kv_rows = torch.cat([kv.float(), kv_state.view(-1, width)])
    score_rows = torch.cat([tok_score, score_state.view(-1, width)])
    closing = ((pos + 1) % ratio == 0).nonzero().squeeze(1)
    q = pos[closing, None] - span + 1 + torch.arange(span, device=dev)
    row = torch.where(
        q >= first[closing, None],
        closing[:, None] + q - pos[closing, None],
        t + slot[closing, None] * span + q.remainder(span),
    )
    # With overlap the previous window's positions give the first d channels, the window's
    # own the last d; positions before 0 are masked.
    halves = [(0, ratio, 0), (ratio, span, d)] if overlap else [(0, span, 0)]
    kv_w = torch.cat([kv_rows[:, lo : lo + d][row[:, a:b]] for a, b, lo in halves], 1)
    score_w = torch.cat([score_rows[:, lo : lo + d][row[:, a:b]] for a, b, lo in halves], 1)
    missing = (q < 0)[..., None]
    kv_w = kv_w.masked_fill(missing, 0.0)
    score_w = score_w.masked_fill(missing, float("-inf"))
    weights = score_w.softmax(1).to(kv.dtype)
    pooled = (weights * kv_w.to(kv.dtype)).float().sum(1)
    entries = torch.empty(t, d, dtype=torch.bfloat16, device=dev)
    entries[closing] = pooled.to(torch.bfloat16)
    valid = torch.zeros(t, dtype=torch.bool, device=dev)
    valid[closing] = True
    end = first + cu[req + 1] - cu[req]
    kept = (pos >= end - span).nonzero().squeeze(1)
    ring = slot[kept] * span + pos[kept] % span
    kv_state.view(-1, width)[ring] = kv_rows[kept]
    score_state.view(-1, width)[ring] = score_rows[kept]
    return entries, valid


def gated_pool_compress_result(subject, *inputs):
    """Observe the entries and valid flags *subject* returns and the two states it writes."""
    entries, valid = subject(*inputs)
    return entries, valid, inputs[3], inputs[4]


class GatedPoolCompressFwdCall(CallWorkload):
    """A manifest call of GatedPoolCompressFwdOp."""

    def ref_program(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        start_pos: torch.Tensor,
        state_slots: torch.Tensor,
    ) -> "tuple[torch.Tensor, ...]":
        # The states are written on copies: the shared inputs are restored after the
        # reference runs, and the expected states must outlive that.
        kv_state, score_state = kv_state.clone(), score_state.clone()
        params = self.call.params
        entries, valid = gated_pool_compress(
            kv,
            score,
            ape,
            kv_state,
            score_state,
            cu_seqlens,
            start_pos,
            state_slots,
            params["ratio"],
            params["overlap"],
        )
        return entries, valid, kv_state, score_state

    def torch_program(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        start_pos: torch.Tensor,
        state_slots: torch.Tensor,
    ) -> "tuple[torch.Tensor, ...]":
        # Writes the states it is given, as the op does.
        params = self.call.params
        entries, valid = gated_pool_compress_torch(
            kv,
            score,
            ape,
            kv_state,
            score_state,
            cu_seqlens,
            start_pos,
            state_slots,
            params["ratio"],
            params["overlap"],
        )
        return entries, valid, kv_state, score_state

    def verification(self, *inputs):
        from workloads.numerics import Custom, reference_tolerance

        def validate(got, expected):
            entries, valid, kv_state, score_state = got
            want_entries, want_valid, want_kv_state, want_score_state = expected
            assert torch.equal(valid, want_valid), "valid flags differ"
            # Only the closing tokens' entries are defined.
            torch.testing.assert_close(
                entries[want_valid].float(),
                want_entries[want_valid].float(),
                **reference_tolerance(torch.bfloat16),
            )
            # A state row is a copy of kv, or score plus ape added in float32.
            torch.testing.assert_close(kv_state, want_kv_state, rtol=0, atol=0)
            torch.testing.assert_close(score_state, want_score_state, rtol=0, atol=0)

        return Custom(validate, "valid flags, entries at the closing tokens and both states")
