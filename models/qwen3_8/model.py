"""Qwen3.8-27B causal language model (text only) composed from TileOPs ops."""

from __future__ import annotations

from pathlib import Path

import torch

from models.causal_lm import CausalLM, indexed_device
from models.qwen3_8.config import Qwen3_8Config
from models.qwen3_8.layers import DecoderLayer, Step
from models.qwen3_8.ops import ModelOps, rope_cos_sin

__all__ = ["Qwen3_8ForCausalLM"]


class Qwen3_8ForCausalLM(CausalLM):
    """Qwen3.8-27B text decoder: embedding, decoder layers, final norm, LM head.

    Every weight and every layer's decoding state is on ``device``. A forward call advances
    one request: ``start_pos`` 0 starts a new
    request and clears the decoding state; a later call continues it from ``start_pos``, which
    must be the number of tokens already passed, with one token (a decode step) or several (a
    chunk of a chunked prefill). With ``cuda_graph`` on a CUDA device, the first decode step
    runs eagerly and captures a CUDA graph that every later decode step replays.
    """

    def __init__(
        self,
        config: Qwen3_8Config,
        *,
        device: torch.device | str = "cuda",
        max_seq_len: int = 4096,
        cuda_graph: bool = True,
    ) -> None:
        self.config = config
        self.max_seq_len = max_seq_len
        self.device = indexed_device(device)
        self.cuda_graph = cuda_graph
        self.ops = ModelOps(config)
        self.layers = [
            DecoderLayer(config, i, max_seq_len, self.device)
            for i in range(config.num_hidden_layers)
        ]
        self.cos_sin = rope_cos_sin(config, max_seq_len, self.device)
        self.embed: torch.Tensor
        self.norm: torch.Tensor
        self.lm_head: torch.Tensor

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        device: torch.device | str = "cuda",
        max_seq_len: int = 4096,
        cuda_graph: bool = True,
    ) -> Qwen3_8ForCausalLM:
        """Load a Hugging Face ``qwen3_5`` checkpoint directory (``safetensors`` shards)."""
        from models.qwen3_8.loader import load_weights

        model = cls(
            Qwen3_8Config.from_pretrained(path),
            device=device,
            max_seq_len=max_seq_len,
            cuda_graph=cuda_graph,
        )
        load_weights(model, Path(path))
        return model

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
