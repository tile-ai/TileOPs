"""Qwen3.8-27B causal language model (text only) composed from TileOPs ops."""

from __future__ import annotations

from typing import Any

import torch

from models.causal_lm import CausalLM
from models.qwen3_8.config import Qwen3_8Config
from models.qwen3_8.layers import DecoderLayer, Step
from models.qwen3_8.loader import load_weights
from models.qwen3_8.ops import ModelOps, rope_cos_sin

__all__ = ["Qwen3_8ForCausalLM"]


class Qwen3_8ForCausalLM(CausalLM):
    """Qwen3.8-27B text decoder: embedding, decoder layers, final norm, LM head.

    A call at ``start_pos`` 0 clears the decoding state; a later call continues the request
    with one token (a decode step) or several (a chunk of a chunked prefill).
    """

    config_type = Qwen3_8Config
    load_weights = staticmethod(load_weights)

    def __init__(self, config: Qwen3_8Config, **options: Any) -> None:
        super().__init__(config, **options)
        self.ops = ModelOps(config)
        self.layers = [
            DecoderLayer(config, i, self.max_seq_len, self.device)
            for i in range(config.num_hidden_layers)
        ]
        self.cos_sin = rope_cos_sin(config, self.max_seq_len, self.device)
        self.embed: torch.Tensor
        self.norm: torch.Tensor
        self.lm_head: torch.Tensor

    def reset(self) -> None:
        """Clear every layer's decoding state; the next call must start at position 0."""
        for layer in self.layers:
            layer.reset()
        super().reset()

    def _step(self, start_pos: int, seqlen: int) -> Step:
        return Step.build(start_pos, seqlen, self.cos_sin)

    def _run(self, ids: torch.Tensor, step: Step) -> torch.Tensor:
        ops = self.ops
        x = ops.embedding(ids[0], self.embed)
        residual = None
        for layer in self.layers:
            x, residual = layer(ops, x, residual, step)
        h, _ = ops.add_norm(x[-1:], residual[-1:], self.norm)
        return ops.gemm_f32(h, self.lm_head)
