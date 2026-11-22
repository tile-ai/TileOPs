"""Qwen3.8-27B causal language model (text only) composed from TileOPs ops."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from models.causal_lm import CausalLM
from models.qwen3_8.config import Qwen3_8Config
from models.qwen3_8.layers import DecoderLayer, Step
from models.qwen3_8.ops import ModelOps, rope_cos_sin

__all__ = ["Qwen3_8ForCausalLM"]


class Qwen3_8ForCausalLM(CausalLM):
    """Qwen3.8-27B text decoder: embedding, decoder layers, final norm, LM head.

    Layers are spread over ``devices`` in contiguous runs; the hidden states move to the next
    device between runs. A forward call advances one request: ``start_pos`` 0 starts a new
    request and clears the decoding state; a later call continues it from ``start_pos``, which
    must be the number of tokens already passed, with one token (a decode step) or several (a
    chunk of a chunked prefill).
    """

    def __init__(
        self,
        config: Qwen3_8Config,
        *,
        devices: Sequence[torch.device | str | int] = ("cuda",),
        max_seq_len: int = 4096,
    ) -> None:
        self.config = config
        self.max_seq_len = max_seq_len
        self.devices = [
            torch.device("cuda", d) if isinstance(d, int) else torch.device(d) for d in devices
        ]
        self.ops = ModelOps(config)
        n_dev, n_layers = len(self.devices), config.num_hidden_layers
        self.layer_devices = [self.devices[i * n_dev // n_layers] for i in range(n_layers)]
        self.layers = [
            DecoderLayer(config, i, max_seq_len, device)
            for i, device in enumerate(self.layer_devices)
        ]
        self.cos_sin = {d: rope_cos_sin(config, max_seq_len, d) for d in set(self.layer_devices)}
        self.embed: torch.Tensor
        self.norm: torch.Tensor
        self.lm_head: torch.Tensor

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        devices: Sequence[torch.device | str | int] = ("cuda",),
        max_seq_len: int = 4096,
    ) -> Qwen3_8ForCausalLM:
        """Load a Hugging Face ``qwen3_5`` checkpoint directory (``safetensors`` shards)."""
        from models.qwen3_8.loader import load_weights

        model = cls(Qwen3_8Config.from_pretrained(path), devices=devices, max_seq_len=max_seq_len)
        load_weights(model, Path(path))
        return model

    def reset(self) -> None:
        """Clear every layer's decoding state; the next call must start at position 0."""
        for layer in self.layers:
            layer.reset()
        super().reset()

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Run ``input_ids`` (``[1, S]``) from ``start_pos``; return the float32 logits of its
        last token, ``[1, vocab_size]``."""
        seqlen = self._continue(input_ids, start_pos)
        ops = self.ops
        x = ops.embedding(input_ids[0].to(self.first_device), self.embed)
        residual = None
        step = None
        previous = None
        for layer, device in zip(self.layers, self.layer_devices, strict=True):
            if device != previous:
                previous = device
                step = Step.build(start_pos, seqlen, self.cos_sin[device])
                x = x.to(device)
                residual = None if residual is None else residual.to(device)
            x, residual = layer(ops, x, residual, step)
        h, _ = ops.add_norm(x[-1:], residual[-1:], self.norm)
        logits = ops.gemm_f32(h, self.lm_head)
        self._next_pos = start_pos + seqlen
        return logits

    __call__ = forward
