"""DeepSeek-V4 blocks composed from TileOPs ops: attention with KV compression, MoE, mHC.

Each layer holds its weights as tensors on one device, in the layouts the ops read, and the
decoding state of one request, allocated once on that device. A request starting at position 0
needs no clearing: no row an earlier request left in the state reaches a call's output.
"""

from __future__ import annotations

import dataclasses

import torch

from models.deepseek_v4.config import DeepseekV4Config
from models.deepseek_v4.ops import ModelOps, RoPETable

__all__ = ["Attention", "Block", "Compressor", "FP8Linear", "Indexer", "MoE", "Step"]


@dataclasses.dataclass(frozen=True)
class Step:
    """The index metadata of one forward call on one device, shared by its layers.

    ``window`` holds each query's sliding-window keys, -1 where none: token positions in a
    prefill, ``[S, min(S, window_size)]``, ring slots in a decode step, ``[1, window_size]``.
    ``offset`` is where the compressed entries start among a layer's keys: after the prompt in
    a prefill, after the ring in a decode step. ``cu_seqlens``, ``starts`` and ``slots`` are the
    int32 ``[0, S]``, ``[start_pos]`` and ``[0]`` the pooling ops read.
    """

    start_pos: int
    seqlen: int
    offset: int
    window: torch.Tensor
    cu_seqlens: torch.Tensor
    starts: torch.Tensor
    slots: torch.Tensor

    @classmethod
    def build(
        cls, cfg: DeepseekV4Config, start_pos: int, seqlen: int, device: torch.device
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
            window=window,
            cu_seqlens=torch.tensor([0, seqlen], dtype=torch.int32, device=device),
            starts=torch.tensor([start_pos], dtype=torch.int32, device=device),
            slots=torch.zeros(1, dtype=torch.int32, device=device),
        )


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


class Compressor:
    """Pools every ``ratio`` tokens of a request's KV into one compressed entry.

    An indexed ratio pools overlapping windows of twice the ratio. ``rotate`` marks the indexer's
    compressor, whose entries are Hadamard-rotated and FP4 fake-quantized; the attention
    compressor's entries are FP8 fake-quantized outside their trailing RoPE dimensions.
    """

    def __init__(
        self,
        cfg: DeepseekV4Config,
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
            cos, sin = rope.cos[: n * ratio : ratio], rope.sin[: n * ratio : ratio]
        else:
            if (start_pos + 1) % ratio != 0:
                return None
            closed = entries.view(1, 1, -1)
            first = start_pos + 1 - ratio
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
    """The lightning indexer: selects each query's top compressed entries of an indexed layer."""

    def __init__(
        self, cfg: DeepseekV4Config, ratio: int, max_entries: int, device: torch.device
    ) -> None:
        self.n_heads = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.rope_dim = cfg.rope_head_dim
        self.ratio = ratio
        self.max_entries = max_entries
        self.compressor = Compressor(cfg, ratio, self.head_dim, rotate=True, device=device)
        self.wq_b: FP8Linear
        self.weights_proj: torch.Tensor
        self.weight_scale = torch.tensor(
            [self.head_dim**-0.5 * self.n_heads**-0.5], dtype=torch.float32, device=device
        )
        self.cache = torch.zeros(1, max_entries, self.head_dim, dtype=torch.bfloat16, device=device)

    def __call__(
        self,
        ops: ModelOps,
        rope: RoPETable,
        x: torch.Tensor,
        qr: torch.Tensor,
        step: Step,
    ) -> torch.Tensor:
        """Return ``[1, S, index_topk]`` int32 indices of compressed entries, shifted by
        ``step.offset`` and padded with -1."""
        start_pos, seqlen = step.start_pos, step.seqlen
        end_pos = start_pos + seqlen
        q = self.wq_b(ops, qr).view(1, seqlen, self.n_heads, self.head_dim)
        ops.rope(q[..., -self.rope_dim :], rope.cos, rope.sin, seqlen_offsets=start_pos)
        q = ops.hadamard(q)
        ops.fp4_fake_quant(q)
        entries = self.compressor(ops, rope, x, step)
        if entries is not None:
            first = start_pos // self.ratio
            self.cache[:, first : first + entries.shape[1]] = entries
        n_keys = end_pos // self.ratio
        device = x.device
        if n_keys == 0:
            return torch.full(
                (1, seqlen, ops.index_topk.top_k), -1, dtype=torch.int32, device=device
            )
        weights = ops.mul(ops.gemm(x, self.weights_proj).float(), self.weight_scale)
        weights = weights.to(torch.bfloat16)
        width = n_keys if start_pos == 0 else self.max_entries
        keys = self.cache[:, :width].transpose(1, 2).contiguous()
        scores = ops.bmm(q.view(1, seqlen * self.n_heads, self.head_dim), keys)
        scores = ops.relu(scores).view(seqlen, self.n_heads, width)
        logits = ops.bmm(weights.view(seqlen, 1, self.n_heads), scores)
        logits = logits.float().view(1, seqlen, width, 1)
        if start_pos == 0:
            visible = torch.arange(1, seqlen + 1, dtype=torch.int32, device=device) // self.ratio
            ends = visible.unsqueeze(0)
        else:
            ends = torch.full((1, 1), n_keys, dtype=torch.int32, device=device)
        starts = torch.zeros_like(ends)
        idx = ops.index_topk(logits, starts, ends).squeeze(2)
        return torch.where(idx >= 0, idx + step.offset, idx)


class Attention:
    """MQA over a sliding window plus, on compressed layers, pooled KV entries.

    Queries come from a low-rank projection; keys and values share one ``head_dim``-wide row
    whose trailing ``rope_head_dim`` dimensions carry interleaved RoPE. The output is rotated
    back, projected per group by ``wo_a`` and mixed by ``wo_b``.
    """

    def __init__(
        self, cfg: DeepseekV4Config, layer_id: int, max_seq_len: int, device: torch.device
    ) -> None:
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_dim = cfg.rope_head_dim
        self.window = cfg.window_size
        self.o_groups = cfg.o_groups
        self.o_rank = cfg.o_lora_rank
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
        self.wq_a: FP8Linear
        self.q_norm: torch.Tensor
        self.wq_b: FP8Linear
        self.wkv: FP8Linear
        self.kv_norm: torch.Tensor
        self.wo_a: torch.Tensor
        self.wo_b: FP8Linear
        self.attn_sink: torch.Tensor
        self.kv_cache = torch.zeros(
            1, self.window + self.max_entries, self.head_dim, dtype=torch.bfloat16, device=device
        )

    def _entry_indices(self, step: Step, device: torch.device) -> torch.Tensor:
        """Each query's closed compressed entries on a layer without an indexer, ``[S, n]``."""
        ratio, seqlen, start_pos = self.ratio, step.seqlen, step.start_pos
        if start_pos == 0:
            idx = torch.arange(seqlen // ratio, device=device).expand(seqlen, -1)
            visible = torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
            return torch.where(idx >= visible, -1, idx + step.offset)
        idx = torch.arange(self.max_entries, device=device)
        return torch.where(idx < (start_pos + 1) // ratio, idx + step.offset, -1).unsqueeze(0)

    def __call__(self, ops: ModelOps, rope: RoPETable, x: torch.Tensor, step: Step) -> torch.Tensor:
        """Attend ``x`` (``[S, dim]`` bfloat16) and return ``[S, dim]``."""
        seqlen, start_pos = step.seqlen, step.start_pos
        device = x.device
        win, rd = self.window, self.rope_dim
        qr = ops.norm(self.wq_a(ops, x), self.q_norm)
        q = self.wq_b(ops, qr).view(1, seqlen, self.n_heads, self.head_dim)
        q = ops.norm(q, None)
        ops.rope(q[..., -rd:], rope.cos, rope.sin, seqlen_offsets=start_pos)
        kv = ops.norm(self.wkv(ops, x), self.kv_norm)
        ops.rope(
            kv.view(1, seqlen, 1, self.head_dim)[..., -rd:],
            rope.cos,
            rope.sin,
            seqlen_offsets=start_pos,
        )
        ops.kv_fake_quant(kv[:, :-rd])
        kv = kv.view(1, seqlen, self.head_dim)

        indices = [step.window]
        if self.indexer is not None:
            indices.append(self.indexer(ops, rope, x, qr, step)[0])
        elif self.ratio:
            indices.append(self._entry_indices(step, device))

        entries = self.compressor(ops, rope, x, step) if self.compressor is not None else None
        if start_pos == 0:
            kept = torch.arange(max(0, seqlen - win), seqlen, device=device)
            self.kv_cache[:, kept % win] = kv[:, kept]
            keys = kv
            if entries is not None:
                self.kv_cache[:, win : win + entries.shape[1]] = entries
                keys = torch.cat([kv, entries], dim=1)
        else:
            self.kv_cache[:, start_pos % win] = kv[:, 0]
            if entries is not None:
                self.kv_cache[:, win + start_pos // self.ratio] = entries[:, 0]
            keys = self.kv_cache

        idx = torch.cat(indices, dim=-1).to(torch.int32).view(1, seqlen, 1, -1)
        o = ops.attention(q, keys.unsqueeze(2).contiguous(), idx, attn_sink=self.attn_sink)
        ops.rope(o[..., -rd:], rope.cos, rope.neg_sin, seqlen_offsets=start_pos)
        grouped = o.view(seqlen, self.o_groups, -1).transpose(0, 1).contiguous()
        projected = ops.bmm(grouped, self.wo_a)
        projected = projected.transpose(0, 1).reshape(seqlen, self.o_groups * self.o_rank)
        return self.wo_b(ops, projected)


class MoE:
    """Routed MXFP4 experts plus one FP8 shared expert, with clamped SwiGLU.

    The first ``n_hash_layers`` route by a token-id table; the rest by ``score_func`` scores
    and a selection bias. The routing weights, times ``route_scale``, multiply the SwiGLU
    output before the down projection, and the shared expert joins the float32 combine.
    """

    def __init__(self, cfg: DeepseekV4Config, layer_id: int) -> None:
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
        ones = torch.ones_like(weights)
        return ops.unpermute(expert_out, ones, inverse, residual=shared)


class Block:
    """One decoder layer: mHC around attention and around the MoE, on ``hc_mult`` streams."""

    def __init__(
        self, cfg: DeepseekV4Config, layer_id: int, max_seq_len: int, device: torch.device
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
