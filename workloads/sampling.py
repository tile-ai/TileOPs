"""Workloads and references of the sampling ops: logit filters and token draws.

The references of the random ops draw their uniforms from Philox4x32-10 keyed by ``seed``
and counted by ``(draw, row, offset)``, written in integer tensor arithmetic so that the
draws are the same on every device, meta included. A kernel draws from its own Philox
stream, so its samples are compared with these by distribution, never one by one.
"""

import torch

from tileops.manifest import load_adts, load_manifest
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import Call, instantiate
from workloads.workload_base import CallWorkload

_INF = float("inf")
_MASK = 0xFFFFFFFF
# Philox4x32 round multipliers and Weyl key increments (Salmon et al., SC'11).
_M0, _M1 = 0xD2511F53, 0xCD9E8D57
_W0, _W1 = 0x9E3779B9, 0xBB67AE85


def sampling_call(op: str, dtype_case: dict | None = None, **row) -> Call:
    """The manifest call of *op* that *row* describes, as a workload row would."""
    plan = entry_plan(op, load_manifest()[op], load_adts(), resolve=False)
    return instantiate(plan, {**row, "label": "test"}, dtype_case or {})


def _mulhilo(a: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The high and low 32-bit words of ``a * m`` for 32-bit ``a`` and ``m``, without int64 overflow."""
    low = (a & 0xFFFF) * m
    high = (a >> 16) * m
    mid = low + ((high & 0xFFFF) << 16)
    return ((high >> 16) + (mid >> 32)) & _MASK, mid & _MASK


def philox_uniform(seed: torch.Tensor, offset: torch.Tensor, rows: int, draws: int) -> torch.Tensor:
    """``[rows, draws]`` float32 uniforms in ``[0, 1)``, a function of ``seed`` and ``offset`` alone.

    Draw ``j`` of row ``r`` is the first output word of Philox4x32-10 with key
    ``(seed_lo, seed_hi)`` and counter ``(j, r, offset_lo, offset_hi)``, its top 24 bits
    scaled by ``2**-24``.
    """
    device = seed.device
    s, o = seed.reshape(()), offset.reshape(()).to(device)
    shape = (rows, draws)
    c0 = torch.arange(draws, device=device).expand(shape)
    c1 = torch.arange(rows, device=device)[:, None].expand(shape)
    c2 = (o & _MASK).expand(shape)
    c3 = ((o >> 32) & _MASK).expand(shape)
    k0, k1 = s & _MASK, (s >> 32) & _MASK
    for step in range(10):
        if step:
            k0, k1 = (k0 + _W0) & _MASK, (k1 + _W1) & _MASK
        hi0, lo0 = _mulhilo(c0, _M0)
        hi1, lo1 = _mulhilo(c2, _M1)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return (c0 >> 8).to(torch.float32) * 2.0**-24


def _draw(weights: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Per row, the index an inverse-CDF draw at ``u * total`` lands on; never a zero weight.

    ``weights`` is ``[B, V]``, non-negative with a positive row total; ``u`` is ``[B]`` in ``[0, 1)``.
    """
    batch, vocab = weights.shape
    cdf = weights.double().cumsum(-1)
    point = u.double() * cdf[:, -1]
    index = torch.searchsorted(cdf, point[:, None], right=True).clamp(max=vocab - 1)
    # A draw on a zero weight, which cumsum rounding allows, moves to the next positive one.
    positive = weights > 0
    columns = torch.arange(vocab, device=weights.device).expand(batch, vocab)
    next_positive = torch.where(positive, columns, vocab).flip(-1).cummin(-1).values.flip(-1)
    chosen = next_positive.gather(1, index)[:, 0]
    last_positive = vocab - 1 - positive.flip(-1).int().argmax(-1)
    return torch.where(chosen == vocab, last_positive, chosen).to(torch.int32)


def top_k_mask(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Keep every logit at least the ``k[b]``-th largest of row ``b``; ``k[b] >= V`` keeps the row."""
    batch, vocab = logits.shape
    k = k.view(batch).to(logits.device, torch.long)
    if vocab == 0:
        return logits.clone()
    values = logits.float()
    kth = values.sort(-1, descending=True).values.gather(1, (k.clamp(max=vocab) - 1)[:, None])
    kept = (values >= kth) | (k >= vocab)[:, None]
    return logits.masked_fill(~kept, -_INF)


def min_p_mask(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    """Mask every logit below ``max_logit + log(min_p[b])``."""
    batch, _vocab = logits.shape
    values = logits.float()
    threshold = values.amax(-1) + min_p.view(batch).to(logits.device).log()
    return logits.masked_fill(values < threshold[:, None], -_INF)


def probability_above(probs: torch.Tensor) -> torch.Tensor:
    """Per token, the total probability of the tokens strictly more probable than it.

    Tokens of equal probability get the same value, whatever order a sort leaves them in.
    """
    ordered, order = probs.sort(-1, descending=True)
    exclusive = ordered.cumsum(-1) - ordered
    # The first position of each run of equal values, where the exclusive sum excludes the run.
    # A row of NaN probabilities, which a row whose largest logit is not finite gives, has no
    # position a search settles on; it is clamped into the row, and its total stays NaN.
    first = torch.searchsorted(-ordered, -ordered, side="left").clamp(max=probs.shape[-1] - 1)
    return torch.empty_like(probs).scatter_(-1, order, exclusive.gather(-1, first))


def top_p_mask(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Mask every token whose more probable tokens already hold at least ``p[b]``.

    Tokens tied at the boundary are kept or masked together.
    """
    batch, _vocab = logits.shape
    probs = logits.float().softmax(-1)
    removed = probability_above(probs) >= p.view(batch, 1).to(logits.device)
    return logits.masked_fill(removed, -_INF)


def sampling_from_probs(
    probs: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
    """One index per row, drawn with probability proportional to ``probs[b]``."""
    batch, _vocab = probs.shape
    return _draw(probs, philox_uniform(seed, offset, batch, 1)[:, 0])


def chain_speculative_sampling(
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify each row's draft chain; draws ``0..N-1`` test acceptance, draw ``N`` the token.

    Draft ``i`` is accepted while ``u_i * draft < target`` at its id. The token after the
    accepted prefix is drawn from ``max(0, target - draft)`` at that position, or from target
    row ``N`` when every draft is accepted; later positions are ``-1``.
    """
    batch, n, vocab = draft_probs.shape
    ids = draft_token_ids.view(batch, n).to(draft_probs.device, torch.long)
    target = target_probs.view(batch, n + 1, vocab)
    u = philox_uniform(seed, offset, batch, n + 1)
    d = draft_probs.gather(2, ids[..., None])[..., 0]
    q = target[:, :n].gather(2, ids[..., None])[..., 0]
    num = (u[:, :n] * d < q).int().cumprod(1).sum(1)
    padded_draft = torch.cat([draft_probs, torch.zeros_like(draft_probs[:, :1])], 1)
    at = num.long()[:, None, None].expand(batch, 1, vocab)
    residual = (target.gather(1, at) - padded_draft.gather(1, at))[:, 0].clamp_min(0)
    token = _draw(residual, u[:, n]).long()
    position = torch.arange(n + 1, device=ids.device)[None]
    drafts = torch.cat([ids, torch.full_like(ids[:, :1], -1)], 1)
    stop = num.long()[:, None]
    tokens = torch.where(position < stop, drafts, torch.where(position == stop, token[:, None], -1))
    return tokens.to(torch.int32), num.to(torch.int32)


class TopKMaskWorkload(CallWorkload):
    """Logits and per-row ``k`` of one ``TopKMaskFwdOp`` call."""

    def ref_program(self, logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return top_k_mask(logits, k)


class MinPMaskWorkload(CallWorkload):
    """Logits of one ``MinPMaskFwdOp`` call, with ``min_p`` uniform in ``[0.05, 0.25)``.

    The first two rows take the contract's endpoints instead: 1, which keeps only the logits
    equal to the row max, and 0, which masks nothing. 1 comes first so that a one-row call
    still exercises a threshold.
    """

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        logits, min_p = CallWorkload.gen_inputs(self)
        min_p = torch.rand_like(min_p) * 0.2 + 0.05
        min_p[:2] = torch.tensor([1.0, 0.0], device=min_p.device)[: min_p.numel()]
        return logits, min_p

    def ref_program(self, logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
        return min_p_mask(logits, min_p)


class TopPMaskWorkload(CallWorkload):
    """Logits of one ``TopPMaskFwdOp`` call, with ``p`` uniform in ``[0.5, 0.95)``.

    The first row takes 1 instead, the contract's upper endpoint, which keeps every token
    the float32 softmax gives a probability to. The lower endpoint 0 masks a row whole, a
    row no comparator agrees on, so ``tests/ops/test_sampling.py`` covers it rather than a
    workload every benchmark reads.
    """

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        logits, p = CallWorkload.gen_inputs(self)
        p = torch.rand_like(p) * 0.45 + 0.5
        p[0] = 1.0
        return logits, p

    def ref_program(self, logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return top_p_mask(logits, p)


class TopKTopPMaskWorkload(CallWorkload):
    """Logits and per-row ``k`` of one ``TopKTopPMaskFwdOp`` call, with ``p`` uniform in ``[0.5, 0.95)``."""

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, k, p = CallWorkload.gen_inputs(self)
        return logits, k, torch.rand_like(p) * 0.45 + 0.5

    def ref_program(self, logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return top_p_mask(top_k_mask(logits, k), p)


class SamplingFromProbsWorkload(CallWorkload):
    """One ``SamplingFromProbsFwdOp`` call: each row the softmax of standard normal logits."""

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, seed, offset = CallWorkload.gen_inputs(self)
        return probs.softmax(-1), seed, offset

    def ref_program(
        self, probs: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        return sampling_from_probs(probs, seed, offset)


class ChainSpeculativeSamplingWorkload(CallWorkload):
    """One ``ChainSpeculativeSamplingFwdOp`` call: draft and target rows the softmax of
    standard normal logits, the draft ids the row's generated ones."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        draft_probs, draft_token_ids, target_probs, seed, offset = CallWorkload.gen_inputs(self)
        return draft_probs.softmax(-1), draft_token_ids, target_probs.softmax(-1), seed, offset

    def ref_program(
        self,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_probs: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return chain_speculative_sampling(draft_probs, draft_token_ids, target_probs, seed, offset)
