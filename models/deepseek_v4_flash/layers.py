"""DeepSeek-V4-Flash blocks composed from TileOPs ops: attention with KV compression, MoE, mHC.

Each layer holds its weights as tensors on one device, in the layouts the ops read, and the
decoding state of one request, allocated once on that device. A request starting at position 0
needs no clearing: no row an earlier request left in the state reaches a call's output.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch

from models.deepseek_v4_flash.config import DeepseekV4FlashConfig
from models.deepseek_v4_flash.ops import ModelOps, RoPETable

__all__ = ["Attention", "Block", "Compressor", "FP8Linear", "Indexer", "MoE", "Step"]


@dataclasses.dataclass(frozen=True)
class Step:
    """The index metadata of one forward call on one device, shared by its layers.

    ``window`` holds each query's sliding-window keys, int32, -1 where none: token positions in a
    prefill, ``[S, min(S, window_size)]``, ring slots in a decode step, ``[1, window_size]``.
    ``offset`` is where the compressed entries start among a layer's keys: after the prompt in
    a prefill, after the ring in a decode step. ``cu_seqlens`` and ``slots`` are the int32
    ``[0, S]`` and ``[0]`` the pooling ops read; ``starts``, int32 ``[start_pos]``, is the
    position the pooling ops, the RoPE offsets, ``closed`` and the cache writes read. Past
    telling a prefill from a decode step, a layer branches on ``start_pos`` only through
    ``closes`` and ``has_entries`` and reads the position from the step's tensors, so a decode
    step moved in place to a position where those two answer the same serves a CUDA graph
    captured earlier.

    The index tensors many layers read are derived once per ``Step`` object, by the first layer
    that reads them; ``moved_to`` returns a new object, whose first use derives them again.
    """

    start_pos: int
    seqlen: int
    offset: int
    window: torch.Tensor
    cu_seqlens: torch.Tensor
    starts: torch.Tensor
    slots: torch.Tensor
    _derived: dict[tuple, Any] = dataclasses.field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @classmethod
    def build(
        cls, cfg: DeepseekV4FlashConfig, start_pos: int, seqlen: int, device: torch.device
    ) -> Step:
        win = cfg.window_size
        if start_pos == 0:
            pos = torch.arange(seqlen, device=device).unsqueeze(1)
            idx = (pos - win + 1).clamp(min=0) + torch.arange(min(seqlen, win), device=device)
            window = torch.where(idx > pos, -1, idx)
        else:
            idx = torch.arange(win, device=device)
            window = torch.where(idx > start_pos, -1, idx).unsqueeze(0)
        return cls(
            start_pos=start_pos,
            seqlen=seqlen,
            offset=seqlen if start_pos == 0 else win,
            window=window.to(torch.int32),
            cu_seqlens=torch.tensor([0, seqlen], dtype=torch.int32, device=device),
            starts=torch.tensor([start_pos], dtype=torch.int32, device=device),
            slots=torch.zeros(1, dtype=torch.int32, device=device),
        )

    def closes(self, ratio: int) -> bool:
        """Whether this decode step closes a window of ``ratio`` tokens."""
        return (self.start_pos + 1) % ratio == 0

    def has_entries(self, ratio: int) -> bool:
        """Whether a window of ``ratio`` tokens has closed by this call's last token."""
        return self.start_pos + self.seqlen >= ratio

    def _once(self, key: tuple, derive: Callable[[], Any]) -> Any:
        """``derive()``, computed the first time this object is asked for ``key``."""
        if key not in self._derived:
            self._derived[key] = derive()
        return self._derived[key]

    def closed(self, ratio: int) -> torch.Tensor:
        """``[1, S]`` int32: how many compressed entries of ``ratio`` each query sees closed."""

        def derive() -> torch.Tensor:
            steps = torch.arange(1, self.seqlen + 1, device=self.window.device)
            return ((self.starts + steps) // ratio).int().unsqueeze(0)

        return self._once(("closed", ratio), derive)

    def first_entries(self) -> torch.Tensor:
        """``[1, S]`` int32 zeros: the first compressed entry each query may select."""
        device = self.window.device
        return self._once(
            ("first",), lambda: torch.zeros(1, self.seqlen, dtype=torch.int32, device=device)
        )

    def key_indices(self, ratio: int, width: int) -> torch.Tensor:
        """``[1, S, 1, W + width]`` int32, ``W`` the window's width: each query's window keys,
        then the first ``width`` compressed entries of ``ratio`` from ``offset``, -1 where the
        query does not see one closed: the keys of a layer that attends to every entry it sees
        closed."""

        def derive() -> torch.Tensor:
            idx = torch.arange(width, dtype=torch.int32, device=self.window.device)
            closed = self.closed(ratio).view(-1, 1)
            entries = torch.where(idx < closed, idx + self.offset, -1)
            return torch.cat([self.window, entries], dim=-1).view(1, self.seqlen, 1, -1)

        return self._once(("key_indices", ratio, width), derive)

    def ring(self) -> torch.Tensor:
        """``[1]`` int64: the window ring slot a decode step's token takes."""
        return self._once(("ring",), lambda: (self.starts % self.window.shape[-1]).long())

    def moved_to(self, start_pos: int) -> Step:
        """This decode step at ``start_pos``: its tensors rewritten in place."""
        slots = torch.arange(self.window.shape[-1], device=self.window.device)
        self.window.copy_(torch.where(slots > start_pos, -1, slots))
        self.starts.fill_(start_pos)
        return dataclasses.replace(self, start_pos=start_pos)


class FP8Linear:
    """A dense linear layer: FP8 e4m3 weight ``[N, K]`` and float32 128x128 block scales."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        self.weight = weight
        self.scale = scale

    def __call__(self, ops: ModelOps, x: torch.Tensor) -> torch.Tensor:
        """Return ``x @ weight^T`` in bfloat16 for 2-D bfloat16 ``x``, which is quantized per
        1x128 to FP8 with power-of-two scales."""
        q, s, _ = ops.act_quant(x)
        return ops.gemm_fp8(q, self.weight, s, self.scale)

    def rows(self, start: int, stop: int) -> FP8Linear:
        """The layer of output rows ``[start, stop)``, viewing this one's weight and scales;
        ``start`` must open a 128-row scale block, as the loader's stacking ensures, so the
        view's scales are its own."""
        return FP8Linear(self.weight[start:stop], self.scale[start // 128 : -(-stop // 128)])


class Compressor:
    """Pools every ``ratio`` tokens of a request's KV into one compressed entry.

    An indexed ratio pools overlapping windows of twice the ratio. ``rotate`` marks the indexer's
    compressor, whose entries are Hadamard-rotated and FP4 fake-quantized; the attention
    compressor's entries are FP8 fake-quantized outside their trailing RoPE dimensions.
    """

    def __init__(
        self,
        cfg: DeepseekV4FlashConfig,
        ratio: int,
        head_dim: int,
        rotate: bool,
        device: torch.device,
    ) -> None:
        windows = 2 if cfg.indexed(ratio) else 1
        self.ratio = ratio
        self.rotate = rotate
        self.rope_dim = cfg.rope_head_dim
        self.width = windows * head_dim
        self.wkv_gate: torch.Tensor
        self.ape: torch.Tensor
        self.norm: torch.Tensor
        shape = (1, windows * ratio, self.width)
        self.kv_state = torch.zeros(shape, dtype=torch.float32, device=device)
        self.score_state = torch.zeros(shape, dtype=torch.float32, device=device)

    def __call__(
        self, ops: ModelOps, rope: RoPETable, x: torch.Tensor, step: Step
    ) -> torch.Tensor | None:
        """Pool ``x`` (``[S, dim]``) and return the entries of the windows it closes,
        ``[1, n, head_dim]``, or None."""
        projected = ops.gemm_f32(x, self.wkv_gate)
        kv, score = projected[:, : self.width], projected[:, self.width :]
        entries, _ = ops.pools[self.ratio](
            kv,
            score,
            self.ape,
            self.kv_state,
            self.score_state,
            step.cu_seqlens,
            step.starts,
            step.slots,
        )
        ratio, start_pos = self.ratio, step.start_pos
        if start_pos == 0:
            n = step.seqlen // ratio
            if n == 0:
                return None
            closed = entries.view(1, step.seqlen, -1)[:, ratio - 1 : n * ratio : ratio]
            first = 0
            cos, sin = rope.entries[ratio]
        else:
            if not step.closes(ratio):
                return None
            closed = entries.view(1, 1, -1)
            first = step.starts + (1 - ratio)
            cos, sin = rope.cos, rope.sin
        closed = ops.norm(closed.contiguous(), self.norm)
        ops.rope(
            closed.unsqueeze(2)[..., -self.rope_dim :],
            cos,
            sin,
            seqlen_offsets=first,
        )
        if self.rotate:
            closed = ops.hadamard(closed)
            ops.fp4_fake_quant(closed)
        else:
            ops.kv_fake_quant(closed[..., : -self.rope_dim])
        return closed


class Indexer:
    """The lightning indexer: selects each query's top compressed entries of an indexed layer.

    ``cache`` holds the entries as columns, ``[1, index_head_dim, max_entries]``, the layout the
    scores' BMM reads, so a decode step reads it in place.
    """

    def __init__(
        self, cfg: DeepseekV4FlashConfig, ratio: int, max_entries: int, device: torch.device
    ) -> None:
        self.n_heads = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.rope_dim = cfg.rope_head_dim
        self.ratio = ratio
        self.max_entries = max_entries
        self.compressor = Compressor(cfg, ratio, self.head_dim, rotate=True, device=device)
        self.weights_proj: torch.Tensor
        self.weight_scale = torch.tensor(
            [self.head_dim**-0.5 * self.n_heads**-0.5], dtype=torch.float32, device=device
        )
        self.cache = torch.zeros(1, self.head_dim, max_entries, dtype=torch.bfloat16, device=device)

    def __call__(
        self,
        ops: ModelOps,
        rope: RoPETable,
        x: torch.Tensor,
        q: torch.Tensor,
        step: Step,
    ) -> torch.Tensor:
        """Return ``[1, S, index_topk]`` int32 indices of compressed entries, shifted by
        ``step.offset`` and padded with -1, for the queries ``q`` (``[S, n_heads *
        head_dim]``) that the attention's ``indexer_wq_b`` projects from its low-rank
        queries."""
        start_pos, seqlen = step.start_pos, step.seqlen
        q = q.view(1, seqlen, self.n_heads, self.head_dim)
        ops.rope(q[..., -self.rope_dim :], rope.cos, rope.sin, seqlen_offsets=step.starts)
        q = ops.hadamard(q)
        ops.fp4_fake_quant(q)
        entries = self.compressor(ops, rope, x, step)
        if entries is not None and start_pos == 0:
            self.cache[..., : entries.shape[1]] = entries.transpose(1, 2)
        elif entries is not None:
            self.cache.index_copy_(2, (step.starts // self.ratio).long(), entries.transpose(1, 2))
        if not step.has_entries(self.ratio):
            return torch.full(
                (1, seqlen, ops.index_topk.top_k), -1, dtype=torch.int32, device=x.device
            )
        weights = ops.mul(ops.gemm(x, self.weights_proj).float(), self.weight_scale)
        weights = weights.to(torch.bfloat16)
        width = seqlen // self.ratio if start_pos == 0 else self.max_entries
        keys = self.cache[..., :width]
        scores = ops.bmm(q.view(1, seqlen * self.n_heads, self.head_dim), keys)
        scores = ops.relu(scores).view(seqlen, self.n_heads, width)
        logits = ops.bmm(weights.view(seqlen, 1, self.n_heads), scores)
        logits = logits.float().view(1, seqlen, width, 1)
        idx = ops.index_topk(logits, step.first_entries(), step.closed(self.ratio)).squeeze(2)
        return torch.where(idx >= 0, idx + step.offset, idx)


class Attention:
    """MQA over a sliding window plus, on compressed layers, pooled KV entries.

    Queries come from a low-rank projection; keys and values share one ``head_dim``-wide row
    whose trailing ``rope_head_dim`` dimensions carry interleaved RoPE. The output is rotated
    back, projected per group by ``wo_a`` and mixed by ``wo_b``. A call of at most 64 tokens reads
    ``wo_a``'s weights more than it multiplies them, and runs ``BMMW8A16FwdOp`` on the checkpoint's
    FP8 codes and block scales, half the bytes; a longer call runs the bfloat16 batched matmul on a
    dequantized copy. The 64 is the W8A16 kernel's tile limit: up to 64 rows it splits K over
    warpgroups that widen the weights together, and past it the kernel runs slower than the bfloat16
    batched matmul. Both give the bfloat16 product of the same weights. ``wq_a_kv`` stacks the rows
    of ``wq_a`` and ``wkv``, which read the same input, and ``wq_b_index`` those of ``wq_b`` and
    ``indexer_wq_b``, the indexer's query projection; each layer of a pair is a row view of its
    stack. A one-token call quantizes the input once and runs one GEMM for each pair, whose output
    columns lie in place; a longer call's would be strided, so it runs the two views.
    """

    def __init__(
        self, cfg: DeepseekV4FlashConfig, layer_id: int, max_seq_len: int, device: torch.device
    ) -> None:
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_dim = cfg.rope_head_dim
        self.window = cfg.window_size
        self.o_groups = cfg.o_groups
        self.o_rank = cfg.o_lora_rank
        self.q_rank = cfg.q_lora_rank
        self.q_width = cfg.n_heads * cfg.head_dim
        self.ratio = cfg.compress_ratios[layer_id]
        self.max_entries = max_seq_len // self.ratio if self.ratio else 0
        self.compressor = (
            Compressor(cfg, self.ratio, self.head_dim, rotate=False, device=device)
            if self.ratio
            else None
        )
        self.indexer = (
            Indexer(cfg, self.ratio, self.max_entries, device) if cfg.indexed(self.ratio) else None
        )
        self.wq_a_kv: FP8Linear
        self.wq_a: FP8Linear
        self.q_norm: torch.Tensor
        self.wq_b_index: FP8Linear | None
        self.wq_b: FP8Linear
        self.indexer_wq_b: FP8Linear | None
        self.wkv: FP8Linear
        self.kv_norm: torch.Tensor
        self.wo_a: torch.Tensor
        self.wo_a_codes: torch.Tensor
        self.wo_a_scales: torch.Tensor
        self.wo_b: FP8Linear
        self.attn_sink: torch.Tensor
        self.kv_cache = torch.zeros(
            1, self.window + self.max_entries, self.head_dim, dtype=torch.bfloat16, device=device
        )

    def __call__(self, ops: ModelOps, rope: RoPETable, x: torch.Tensor, step: Step) -> torch.Tensor:
        """Attend ``x`` (``[S, dim]`` bfloat16) and return ``[S, dim]``."""
        seqlen, start_pos = step.seqlen, step.start_pos
        win, rd = self.window, self.rope_dim
        one = seqlen == 1
        if one:
            q_kv = self.wq_a_kv(ops, x)
            q_low, kv = q_kv[:, : self.q_rank], q_kv[:, self.q_rank :]
        else:
            q_low, kv = self.wq_a(ops, x), self.wkv(ops, x)
        qr = ops.norm(q_low, self.q_norm)
        if one and self.wq_b_index is not None:
            q_both = self.wq_b_index(ops, qr)
            q, index_q = q_both[:, : self.q_width], q_both[:, self.q_width :]
        else:
            q = self.wq_b(ops, qr)
            index_q = self.indexer_wq_b(ops, qr) if self.indexer_wq_b is not None else None
        q = ops.norm(q.view(1, seqlen, self.n_heads, self.head_dim), None)
        ops.rope(q[..., -rd:], rope.cos, rope.sin, seqlen_offsets=step.starts)
        kv = ops.norm(kv, self.kv_norm)
        ops.rope(
            kv.view(1, seqlen, 1, self.head_dim)[..., -rd:],
            rope.cos,
            rope.sin,
            seqlen_offsets=step.starts,
        )
        ops.kv_fake_quant(kv[:, :-rd])
        kv = kv.view(1, seqlen, self.head_dim)

        if self.indexer is not None:
            selected = self.indexer(ops, rope, x, index_q, step)[0]
            idx = torch.cat([step.window, selected], dim=-1).view(1, seqlen, 1, -1)
        elif self.ratio:
            width = seqlen // self.ratio if start_pos == 0 else self.max_entries
            idx = step.key_indices(self.ratio, width)
        else:
            idx = step.window.view(1, seqlen, 1, -1)

        entries = self.compressor(ops, rope, x, step) if self.compressor is not None else None
        if start_pos == 0:
            first = max(0, seqlen - win)
            slot = first % win
            head = min(win - slot, seqlen - first)
            self.kv_cache[:, slot : slot + head] = kv[:, first : first + head]
            self.kv_cache[:, : seqlen - first - head] = kv[:, first + head :]
            keys = kv
            if entries is not None:
                self.kv_cache[:, win : win + entries.shape[1]] = entries
                keys = torch.cat([kv, entries], dim=1)
        else:
            self.kv_cache.index_copy_(1, step.ring(), kv)
            if entries is not None:
                self.kv_cache.index_copy_(1, (win + step.starts // self.ratio).long(), entries)
            keys = self.kv_cache

        o = ops.attention(q, keys.unsqueeze(2).contiguous(), idx, attn_sink=self.attn_sink)
        ops.rope(o[..., -rd:], rope.cos, rope.neg_sin, seqlen_offsets=step.starts)
        grouped = o.view(seqlen, self.o_groups, -1).transpose(0, 1)
        projected = torch.empty(seqlen, self.o_groups, self.o_rank, dtype=o.dtype, device=o.device)
        if seqlen <= 64:
            ops.bmm_w8a16(grouped, self.wo_a_codes, self.wo_a_scales, out=projected.transpose(0, 1))
        else:
            ops.bmm(grouped, self.wo_a, out=projected.transpose(0, 1))
        return self.wo_b(ops, projected.view(seqlen, -1))


class MoE:
    """Routed MXFP4 experts plus one FP8 shared expert, with clamped SwiGLU.

    The first ``n_hash_layers`` route by a token-id table; the rest by ``score_func`` scores
    and a selection bias. The routing weights, times ``route_scale``, multiply the SwiGLU
    output before the down projection, and the shared expert joins the float32 combine.
    """

    def __init__(self, cfg: DeepseekV4FlashConfig, layer_id: int) -> None:
        self.hash = layer_id < cfg.n_hash_layers
        self.gate: torch.Tensor
        self.bias: torch.Tensor | None = None
        self.tid2eid: torch.Tensor | None = None
        self.w13: torch.Tensor
        self.w13_scale: torch.Tensor
        self.w2: torch.Tensor
        self.w2_scale: torch.Tensor
        self.shared_w13: FP8Linear
        self.shared_w2: FP8Linear

    def __call__(self, ops: ModelOps, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the MoE output for ``x`` (``[T, dim]`` bfloat16) and its ``[T]`` token ids."""
        logits = ops.gemm_f32(x, self.gate)
        if self.hash:
            expert_ids = ops.embedding(input_ids, self.tid2eid)
            weights, ids = ops.route(logits, expert_ids=expert_ids)
        else:
            weights, ids = ops.route(logits, self.bias)
        routed, metadata, inverse = ops.permute(x, ids)
        a, a_scale, _ = ops.act_quant_e8m0(routed)
        gate_up = ops.expert_gemm(a, a_scale, self.w13, self.w13_scale, metadata)
        row_weights = torch.empty(routed.shape[0], dtype=torch.float32, device=x.device)
        row_weights[inverse.long()] = weights.view(-1)
        hidden = ops.swiglu(gate_up, scale=row_weights)
        a, a_scale, _ = ops.act_quant_e8m0(hidden)
        expert_out = ops.expert_gemm(a, a_scale, self.w2, self.w2_scale, metadata)
        shared = self.shared_w2(ops, ops.swiglu(self.shared_w13(ops, x)))
        return ops.unpermute(expert_out, torch.ones_like(weights), inverse, residual=shared)


class Block:
    """One decoder layer: mHC around attention and around the MoE, on ``hc_mult`` streams."""

    def __init__(
        self, cfg: DeepseekV4FlashConfig, layer_id: int, max_seq_len: int, device: torch.device
    ) -> None:
        self.layer_id = layer_id
        self.attn = Attention(cfg, layer_id, max_seq_len, device)
        self.ffn = MoE(cfg, layer_id)
        self.attn_norm: torch.Tensor
        self.ffn_norm: torch.Tensor
        self.hc_attn: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        self.hc_ffn: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        self.rope: RoPETable

    def __call__(
        self, ops: ModelOps, h: torch.Tensor, input_ids: torch.Tensor, step: Step
    ) -> torch.Tensor:
        """Advance the streams ``h`` (``[1, S, hc_mult, dim]``) of the tokens ``input_ids``
        (``[1, S]``) through this layer."""
        seqlen, dim = step.seqlen, h.shape[-1]
        post, comb, x = ops.hc_pre(h, *self.hc_attn)
        x = ops.norm(x, self.attn_norm).view(seqlen, dim)
        x = self.attn(ops, self.rope, x, step)
        h = ops.hc_post(x.view(1, seqlen, dim), h, post, comb)
        post, comb, x = ops.hc_pre(h, *self.hc_ffn)
        x = ops.norm(x, self.ffn_norm).view(seqlen, dim)
        x = self.ffn(ops, x, input_ids.view(-1))
        return ops.hc_post(x.view(1, seqlen, dim), h, post, comb)
