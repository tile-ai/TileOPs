"""Qwen3.8-27B decoder layers: Gated DeltaNet or gated full attention, then a SwiGLU MLP.

Every layer works on one request's tokens, ``[T, hidden]``, and keeps that request's decoding
state. Weights are filled by the loader in the layouts the ops read.
"""

from __future__ import annotations

import dataclasses

import torch

from models.qwen3_8.config import LINEAR_ATTENTION, Qwen3_8Config
from models.qwen3_8.ops import ModelOps

__all__ = ["DecoderLayer", "GatedAttention", "GatedDeltaNet", "Step"]


@dataclasses.dataclass(frozen=True)
class Step:
    """The index metadata of one forward call on one device, shared by its layers.

    ``positions`` is ``[3, T]`` int64, contiguous as the q/k norm and RoPE reads it, each
    token's position repeated over the temporal, height and width axes, as text tokens take
    it; ``cu_seqlens`` is int32 ``[0, T]``;
    ``cache_seqlens`` is int32 ``[start_pos + T]``; ``cos_sin`` is the device's RoPE table.
    A layer's one host-side branch on ``start_pos``, whether a call is a decode step, goes the
    same way for every decode step, and the layers read the position from these tensors, so a
    decode step moved in place to a new position serves a CUDA graph captured at an earlier one.
    """

    start_pos: int
    positions: torch.Tensor
    cu_seqlens: torch.Tensor
    cache_seqlens: torch.Tensor
    cos_sin: torch.Tensor

    @classmethod
    def build(cls, start_pos: int, tokens: int, cos_sin: torch.Tensor) -> Step:
        device = cos_sin.device
        pos = torch.arange(start_pos, start_pos + tokens, dtype=torch.int64, device=device)
        return cls(
            start_pos=start_pos,
            positions=pos.repeat(3, 1),
            cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32, device=device),
            cache_seqlens=torch.tensor([start_pos + tokens], dtype=torch.int32, device=device),
            cos_sin=cos_sin,
        )

    def moved_to(self, start_pos: int) -> Step:
        """This one-token step at ``start_pos``: its tensors rewritten in place."""
        self.positions.fill_(start_pos)
        self.cache_seqlens.fill_(start_pos + 1)
        return dataclasses.replace(self, start_pos=start_pos)


class GatedDeltaNet:
    """Gated DeltaNet token mixer with its causal-conv and float32 recurrent states, both
    advanced in place.

    ``in_proj`` stacks the rows of HF's ``in_proj_qkv``, ``in_proj_z``, ``in_proj_b`` and
    ``in_proj_a``. ``A_log`` and ``dt_bias`` are float32. ``norm`` is HF's per-head output
    norm weight repeated over the value heads: the norm runs head by head over a token's
    whole row, reading the gate ``z`` where it lies in the projection.
    """

    def __init__(self, cfg: Qwen3_8Config, device: torch.device) -> None:
        self.key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
        self.value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.heads = cfg.linear_num_key_heads
        self.value_heads = cfg.linear_num_value_heads
        self.head_dim = cfg.linear_key_head_dim
        self.in_proj: torch.Tensor
        self.conv_weight: torch.Tensor
        self.A_log: torch.Tensor
        self.dt_bias: torch.Tensor
        self.norm: torch.Tensor
        self.out_proj: torch.Tensor
        width = cfg.linear_conv_kernel_dim
        self.conv_state = torch.zeros(
            1, self.conv_dim, width - 1, dtype=torch.bfloat16, device=device
        )
        self.state = torch.zeros(
            1, self.value_heads, self.head_dim, self.head_dim, dtype=torch.float32, device=device
        )
        self.has_state = torch.ones(1, dtype=torch.bool, device=device)
        self.slot = torch.zeros(1, dtype=torch.int32, device=device)

    def reset(self) -> None:
        self.conv_state.zero_()
        self.state.zero_()

    def __call__(self, ops: ModelOps, x: torch.Tensor, step: Step) -> torch.Tensor:
        tokens = x.shape[0]
        hv, dim = self.value_heads, self.head_dim
        proj = ops.gemm(x, self.in_proj)
        mixed, z, b, a = proj.split([self.conv_dim, self.value_dim, hv, hv], dim=-1)
        if tokens == 1 and step.start_pos > 0:
            mixed = ops.conv_update(mixed, self.conv_weight, self.conv_state)
        else:
            mixed = ops.conv(
                mixed, self.conv_weight, self.conv_state, step.cu_seqlens, None, self.has_state
            )
        q, k, v = mixed.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        o, _ = ops.gdn(
            q.view(1, tokens, self.heads, dim),
            k.view(1, tokens, self.heads, dim),
            v.view(1, tokens, hv, dim),
            a.view(1, tokens, hv),
            b.view(1, tokens, hv),
            self.state,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            state_indices=self.slot,
        )
        o = ops.gated_norm(o.view(tokens, self.value_dim), self.norm, None, z)
        return ops.gemm(o, self.out_proj)


class GatedAttention:
    """Causal GQA over a paged KV cache, its output gated by ``sigmoid(gate)``.

    ``qkv_gate_proj`` stacks the q rows of HF's ``q_proj`` (the first half of each head's
    rows), then ``k_proj`` and ``v_proj``, then the gate rows (the second half of each head's
    rows of ``q_proj``). A one-token call projects all of them in one product, whose q, k and
    v part is contiguous; a longer one projects q, k and v apart from the gate, as the q and
    k norm rewrites them in place and takes them contiguous.
    """

    def __init__(self, cfg: Qwen3_8Config, max_seq_len: int, device: torch.device) -> None:
        self.heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.qkv_gate_proj: torch.Tensor
        self.q_norm: torch.Tensor
        self.k_norm: torch.Tensor
        self.o_proj: torch.Tensor
        # Rows of a KV-cache page. GQAPagedFwdOp's batch-1 decode kernel serves pages of 64 rows
        # or a multiple of 128, so a fixed page keeps decode on that kernel whatever max_seq_len is.
        page_size = 64
        pages = -(-max_seq_len // page_size)
        shape = (pages, page_size, self.kv_heads, self.head_dim)
        self.k_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.v_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.block_table = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)

    def reset(self) -> None:
        """Nothing to clear: attention reads only the cache rows below ``cache_seqlens``."""

    def __call__(self, ops: ModelOps, x: torch.Tensor, step: Step) -> torch.Tensor:
        tokens = x.shape[0]
        h, hkv, d = self.heads, self.kv_heads, self.head_dim
        qkv_rows = (h + 2 * hkv) * d
        if tokens == 1:
            qkv, gate = ops.gemm(x, self.qkv_gate_proj).split([qkv_rows, h * d], dim=-1)
        else:
            qkv = ops.gemm(x, self.qkv_gate_proj[:qkv_rows])
            gate = ops.gemm(x, self.qkv_gate_proj[qkv_rows:])
        ops.qk_norm_rope(qkv, self.q_norm, self.k_norm, step.cos_sin, step.positions)
        q, k, v = qkv.split([h * d, hkv * d, hkv * d], dim=-1)
        # The block table is the identity, so a token's cache slot is its position.
        ops.kv_write(
            k.view(tokens, hkv, d),
            v.view(tokens, hkv, d),
            self.k_cache,
            self.v_cache,
            step.positions[0],
        )
        o = ops.attention(
            q.view(tokens, h, d),
            self.k_cache,
            self.v_cache,
            self.block_table,
            step.cache_seqlens,
            step.cu_seqlens,
        )
        o = ops.mul(o.view(tokens, h * d), ops.sigmoid(gate))
        return ops.gemm(o, self.o_proj)


class DecoderLayer:
    """Pre-norm residual block: token mixer, then SwiGLU MLP.

    ``gate_up_proj`` stacks HF's MLP ``gate_proj`` rows, then ``up_proj``'s.

    A call takes the previous layer's ``(out, residual)`` and returns its own; the layer's
    hidden state, HF's decoder-layer output, is ``out + residual``. The first layer takes
    ``residual=None`` and ``x`` the embeddings.
    """

    def __init__(
        self, cfg: Qwen3_8Config, layer_id: int, max_seq_len: int, device: torch.device
    ) -> None:
        self.layer_id = layer_id
        self.input_norm: torch.Tensor
        self.post_norm: torch.Tensor
        if cfg.layer_types[layer_id] == LINEAR_ATTENTION:
            self.mixer: GatedDeltaNet | GatedAttention = GatedDeltaNet(cfg, device)
        else:
            self.mixer = GatedAttention(cfg, max_seq_len, device)
        self.gate_up_proj: torch.Tensor
        self.down_proj: torch.Tensor

    def reset(self) -> None:
        self.mixer.reset()

    def __call__(
        self, ops: ModelOps, x: torch.Tensor, residual: torch.Tensor | None, step: Step
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = x
            h = ops.norm(x, self.input_norm)
        else:
            h, residual = ops.add_norm(x, residual, self.input_norm)
        h, residual = ops.add_norm(self.mixer(ops, h, step), residual, self.post_norm)
        mlp = ops.gemm(ops.silu_and_mul(ops.gemm(h, self.gate_up_proj)), self.down_proj)
        return mlp, residual
