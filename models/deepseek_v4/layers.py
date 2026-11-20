"""DeepSeek-V4 blocks composed from TileOPs ops: attention with KV compression, MoE, mHC.

Each layer holds its weights as tensors on one device, in the layouts the ops read, and the
decoding state of one batch of requests that advance together from position 0.
"""

from __future__ import annotations

import torch

from models.deepseek_v4.config import DeepseekV4Config
from models.deepseek_v4.ops import ModelOps, RoPETable

__all__ = ["Block", "FP8Linear"]


class FP8Linear:
    """A dense linear layer: FP8 e4m3 weight ``[N, K]`` and float32 128x128 block scales."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        self.weight = weight
        self.scale = scale

    def __call__(self, ops: ModelOps, x: torch.Tensor) -> torch.Tensor:
        """Return ``x @ weight^T`` in bfloat16 for 2-D bfloat16 ``x``."""
        return ops.fp8_linear(x, self.weight, self.scale)


class Compressor:
    """Pools every ``ratio`` tokens of a request's KV into one compressed entry.

    Ratio 4 pools overlapping windows of 8 tokens. ``rotate`` marks the indexer's
    compressor, whose entries are Hadamard-rotated and FP4 fake-quantized; the attention
    compressor's entries are FP8 fake-quantized outside their trailing RoPE dimensions.
    """

    def __init__(self, cfg: DeepseekV4Config, ratio: int, head_dim: int, rotate: bool) -> None:
        windows = 2 if ratio == 4 else 1
        self.ratio = ratio
        self.rotate = rotate
        self.rope_dim = cfg.rope_head_dim
        self.width = windows * head_dim
        self.state_rows = windows * ratio
        self.wkv_gate: torch.Tensor
        self.ape: torch.Tensor
        self.norm: torch.Tensor
        self.kv_state: torch.Tensor
        self.score_state: torch.Tensor

    def reset(self, batch: int) -> None:
        """Allocate fresh pooling state for ``batch`` requests."""
        self.kv_state = torch.zeros(
            batch, self.state_rows, self.width, dtype=torch.float32, device=self.ape.device
        )
        self.score_state = torch.zeros_like(self.kv_state)

    def __call__(
        self, ops: ModelOps, rope: RoPETable, x: torch.Tensor, batch: int, start_pos: int
    ) -> torch.Tensor | None:
        """Pool ``x`` (``[batch * S, dim]``, positions from ``start_pos``) and return the
        entries of the windows it closes, ``[batch, n, head_dim]``, or None."""
        seqlen = x.shape[0] // batch
        projected = ops.gemm_f32(x, self.wkv_gate)
        kv, score = projected[:, : self.width], projected[:, self.width :]
        device = x.device
        cu_seqlens = torch.arange(0, batch + 1, dtype=torch.int32, device=device) * seqlen
        starts = torch.full((batch,), start_pos, dtype=torch.int32, device=device)
        slots = torch.arange(batch, dtype=torch.int32, device=device)
        entries, _ = ops.pools[self.ratio](
            kv.contiguous(),
            score.contiguous(),
            self.ape,
            self.kv_state,
            self.score_state,
            cu_seqlens,
            starts,
            slots,
        )
        ratio = self.ratio
        if start_pos == 0:
            n = seqlen // ratio
            if n == 0:
                return None
            closed = entries.view(batch, seqlen, -1)[:, ratio - 1 : n * ratio : ratio]
            first = 0
            cos, sin = rope.cos[: n * ratio : ratio], rope.sin[: n * ratio : ratio]
        else:
            if (start_pos + 1) % ratio != 0:
                return None
            closed = entries.view(batch, 1, -1)
            first = start_pos + 1 - ratio
            cos, sin = rope.cos, rope.sin
        closed = ops.norm(closed.contiguous(), self.norm)
        ops.rope(
            closed.unsqueeze(2)[..., -self.rope_dim :],
            cos.contiguous(),
            sin.contiguous(),
            seqlen_offsets=first,
        )
        if self.rotate:
            closed = ops.hadamard(closed)
            ops.fp4_fake_quant(closed)
        else:
            ops.kv_fake_quant(closed[..., : -self.rope_dim])
        return closed


class Indexer:
    """The lightning indexer: selects each query's top compressed entries of a ratio-4 layer."""

    def __init__(self, cfg: DeepseekV4Config, max_seq_len: int) -> None:
        self.n_heads = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.rope_dim = cfg.rope_head_dim
        self.ratio = 4
        self.max_entries = max_seq_len // self.ratio
        self.weight_scale: torch.Tensor
        self.compressor = Compressor(cfg, self.ratio, self.head_dim, rotate=True)
        self.wq_b: FP8Linear
        self.weights_proj: torch.Tensor
        self.cache: torch.Tensor

    def reset(self, batch: int) -> None:
        """Allocate an empty entry cache for ``batch`` requests."""
        device = self.weights_proj.device
        self.compressor.reset(batch)
        self.cache = torch.zeros(
            batch, self.max_entries, self.head_dim, dtype=torch.bfloat16, device=device
        )
        self.weight_scale = torch.tensor(
            [self.head_dim**-0.5 * self.n_heads**-0.5], dtype=torch.float32, device=device
        )

    def __call__(
        self,
        ops: ModelOps,
        rope: RoPETable,
        x: torch.Tensor,
        qr: torch.Tensor,
        batch: int,
        start_pos: int,
        offset: int,
    ) -> torch.Tensor:
        """Return ``[batch, S, index_topk]`` int32 indices of compressed entries, shifted by
        ``offset`` and padded with -1."""
        seqlen = x.shape[0] // batch
        end_pos = start_pos + seqlen
        q = self.wq_b(ops, qr).view(batch, seqlen, self.n_heads, self.head_dim)
        ops.rope(q[..., -self.rope_dim :], rope.cos, rope.sin, seqlen_offsets=start_pos)
        q = ops.hadamard(q)
        ops.fp4_fake_quant(q)
        entries = self.compressor(ops, rope, x, batch, start_pos)
        if entries is not None:
            first = start_pos // self.ratio
            self.cache[:batch, first : first + entries.shape[1]] = entries
        n_keys = end_pos // self.ratio
        device = x.device
        if n_keys == 0:
            return torch.full(
                (batch, seqlen, ops.index_topk.top_k), -1, dtype=torch.int32, device=device
            )
        weights = ops.mul(ops.gemm(x, self.weights_proj).float(), self.weight_scale)
        weights = weights.to(torch.bfloat16)
        width = n_keys if start_pos == 0 else self.max_entries
        keys = self.cache[:batch, :width].transpose(1, 2).contiguous()
        scores = ops.bmm(q.view(batch, seqlen * self.n_heads, self.head_dim), keys)
        scores = ops.relu(scores).view(batch * seqlen, self.n_heads, width)
        logits = ops.bmm(weights.view(batch * seqlen, 1, self.n_heads), scores)
        logits = logits.float().view(batch, seqlen, width, 1)
        if start_pos == 0:
            visible = torch.arange(1, seqlen + 1, dtype=torch.int32, device=device) // self.ratio
            ends = visible.expand(batch, seqlen).contiguous()
        else:
            ends = torch.full((batch, seqlen), n_keys, dtype=torch.int32, device=device)
        starts = torch.zeros_like(ends)
        idx = ops.index_topk(logits.contiguous(), starts, ends).squeeze(2)
        return torch.where(idx >= 0, idx + offset, idx)


class Attention:
    """MQA over a 128-token sliding window plus, on compressed layers, pooled KV entries.

    Queries come from a low-rank projection; keys and values share one 512-wide row whose
    trailing 64 dimensions carry interleaved RoPE. The output is rotated back, projected per
    group by ``wo_a`` and mixed by ``wo_b``.
    """

    def __init__(self, cfg: DeepseekV4Config, layer_id: int, max_seq_len: int) -> None:
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_dim = cfg.rope_head_dim
        self.window = cfg.window_size
        self.o_groups = cfg.o_groups
        self.o_rank = cfg.o_lora_rank
        self.ratio = cfg.compress_ratios[layer_id]
        self.max_entries = max_seq_len // self.ratio if self.ratio else 0
        self.compressor = (
            Compressor(cfg, self.ratio, self.head_dim, rotate=False) if self.ratio else None
        )
        self.indexer = Indexer(cfg, max_seq_len) if self.ratio == 4 else None
        self.wq_a: FP8Linear
        self.q_norm: torch.Tensor
        self.wq_b: FP8Linear
        self.wkv: FP8Linear
        self.kv_norm: torch.Tensor
        self.wo_a: torch.Tensor
        self.wo_b: FP8Linear
        self.attn_sink: torch.Tensor
        self.kv_cache: torch.Tensor

    def reset(self, batch: int) -> None:
        """Allocate an empty window ring and entry cache for ``batch`` requests."""
        self.kv_cache = torch.zeros(
            batch,
            self.window + self.max_entries,
            self.head_dim,
            dtype=torch.bfloat16,
            device=self.attn_sink.device,
        )
        if self.compressor is not None:
            self.compressor.reset(batch)
        if self.indexer is not None:
            self.indexer.reset(batch)

    def _window_indices(self, seqlen: int, start_pos: int, device: torch.device) -> torch.Tensor:
        """Each query's window keys: token positions in prefill, ring slots in decode."""
        win = self.window
        if start_pos == 0:
            pos = torch.arange(seqlen, device=device).unsqueeze(1)
            idx = (pos - win + 1).clamp(min=0) + torch.arange(min(seqlen, win), device=device)
            return torch.where(idx > pos, -1, idx)
        if start_pos >= win - 1:
            idx = (torch.arange(win, device=device) + start_pos + 1) % win
        else:
            idx = torch.arange(win, device=device)
            idx = torch.where(idx > start_pos, -1, idx)
        return idx.unsqueeze(0)

    def _entry_indices(
        self, seqlen: int, start_pos: int, offset: int, device: torch.device
    ) -> torch.Tensor:
        """Each query's closed compressed entries on a layer without an indexer."""
        ratio = self.ratio
        if start_pos == 0:
            idx = torch.arange(seqlen // ratio, device=device).expand(seqlen, -1)
            visible = torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
            return torch.where(idx >= visible, -1, idx + offset)
        idx = torch.arange(self.max_entries, device=device)
        return torch.where(idx < (start_pos + 1) // ratio, idx + offset, -1).unsqueeze(0)

    def __call__(
        self,
        ops: ModelOps,
        rope: RoPETable,
        x: torch.Tensor,
        batch: int,
        start_pos: int,
    ) -> torch.Tensor:
        """Attend ``x`` (``[batch * S, dim]`` bfloat16) and return ``[batch * S, dim]``."""
        seqlen = x.shape[0] // batch
        device = x.device
        win, rd = self.window, self.rope_dim
        qr = ops.norm(self.wq_a(ops, x), self.q_norm)
        q = self.wq_b(ops, qr).view(batch, seqlen, self.n_heads, self.head_dim)
        q = ops.norm(q, None)
        ops.rope(q[..., -rd:], rope.cos, rope.sin, seqlen_offsets=start_pos)
        kv = ops.norm(self.wkv(ops, x), self.kv_norm)
        ops.rope(
            kv.view(batch, seqlen, 1, self.head_dim)[..., -rd:],
            rope.cos,
            rope.sin,
            seqlen_offsets=start_pos,
        )
        ops.kv_fake_quant(kv[:, :-rd])
        kv = kv.view(batch, seqlen, self.head_dim)

        indices = [self._window_indices(seqlen, start_pos, device).expand(batch, seqlen, -1)]
        offset = seqlen if start_pos == 0 else win
        if self.indexer is not None:
            indices.append(self.indexer(ops, rope, x, qr, batch, start_pos, offset))
        elif self.ratio:
            indices.append(
                self._entry_indices(seqlen, start_pos, offset, device).expand(batch, seqlen, -1)
            )

        if start_pos == 0:
            kept = torch.arange(max(0, seqlen - win), seqlen, device=device)
            self.kv_cache[:batch, kept % win] = kv[:, kept]
            keys = kv
            if self.compressor is not None:
                entries = self.compressor(ops, rope, x, batch, start_pos)
                if entries is not None:
                    self.kv_cache[:batch, win : win + entries.shape[1]] = entries
                    keys = torch.cat([kv, entries], dim=1)
        else:
            self.kv_cache[:batch, start_pos % win] = kv[:, 0]
            if self.compressor is not None:
                entries = self.compressor(ops, rope, x, batch, start_pos)
                if entries is not None:
                    self.kv_cache[:batch, win + start_pos // self.ratio] = entries[:, 0]
            keys = self.kv_cache[:batch]

        idx = torch.cat(indices, dim=-1).to(torch.int32).unsqueeze(2).contiguous()
        o = ops.attention(q, keys.unsqueeze(2).contiguous(), idx, attn_sink=self.attn_sink)
        ops.rope(o[..., -rd:], rope.cos, rope.neg_sin, seqlen_offsets=start_pos)
        grouped = o.view(batch * seqlen, self.o_groups, -1).transpose(0, 1).contiguous()
        projected = ops.bmm(grouped, self.wo_a)
        projected = projected.transpose(0, 1).reshape(batch * seqlen, self.o_groups * self.o_rank)
        return self.wo_b(ops, projected)


class MoE:
    """256 routed MXFP4 experts (top 6) plus one FP8 shared expert, with clamped SwiGLU.

    The first ``n_hash_layers`` route by a token-id table; the rest by ``sqrt(softplus)``
    scores, a selection bias and a 1.5 scale. Routing weights multiply the SwiGLU output
    before the down projection, and the shared expert joins the float32 combine.
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

    def __init__(self, cfg: DeepseekV4Config, layer_id: int, max_seq_len: int) -> None:
        self.layer_id = layer_id
        self.attn = Attention(cfg, layer_id, max_seq_len)
        self.ffn = MoE(cfg, layer_id)
        self.attn_norm: torch.Tensor
        self.ffn_norm: torch.Tensor
        self.hc_attn: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        self.hc_ffn: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        self.rope: RoPETable

    def reset(self, batch: int) -> None:
        """Clear this layer's decoding state for ``batch`` requests."""
        self.attn.reset(batch)

    def __call__(
        self, ops: ModelOps, h: torch.Tensor, input_ids: torch.Tensor, start_pos: int
    ) -> torch.Tensor:
        """Advance the streams ``h`` (``[B, S, hc_mult, dim]``) through this layer."""
        batch, seqlen, _, dim = h.shape
        post, comb, x = ops.hc_pre(h, *self.hc_attn)
        x = ops.norm(x, self.attn_norm).view(batch * seqlen, dim)
        x = self.attn(ops, self.rope, x, batch, start_pos)
        h = ops.hc_post(x.view(batch, seqlen, dim), h, post, comb)
        post, comb, x = ops.hc_pre(h, *self.hc_ffn)
        x = ops.norm(x, self.ffn_norm).view(batch * seqlen, dim)
        x = self.ffn(ops, x, input_ids.view(-1))
        return ops.hc_post(x.view(batch, seqlen, dim), h, post, comb)
