"""Qwen3.8-27B causal language model (text only) composed from TileOPs ops."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from models.qwen3_8.config import Qwen3_8Config
from models.qwen3_8.layers import DecoderLayer, Step
from models.qwen3_8.ops import ModelOps, rope_cos_sin

__all__ = ["Qwen3_8ForCausalLM"]


class Qwen3_8ForCausalLM:
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
        self._next_pos: int | None = None

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

    @property
    def first_device(self) -> torch.device:
        """The device holding the embedding and the first layer."""
        return self.layer_devices[0]

    @property
    def last_device(self) -> torch.device:
        """The device holding the last layer, the final norm and the LM head."""
        return self.layer_devices[-1]

    def reset(self) -> None:
        """Clear every layer's decoding state; the next call must start at position 0."""
        for layer in self.layers:
            layer.reset()
        self._next_pos = None

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Run ``input_ids`` (``[1, S]``) from ``start_pos``; return the float32 logits of its
        last token, ``[1, vocab_size]``."""
        batch, seqlen = input_ids.shape
        if batch != 1:
            raise ValueError(f"forward runs one request, got a batch of {batch}")
        if seqlen == 0:
            raise ValueError("forward takes at least one token")
        if start_pos + seqlen > self.max_seq_len:
            raise ValueError(
                f"position {start_pos + seqlen} exceeds max_seq_len {self.max_seq_len}"
            )
        if start_pos == 0:
            self.reset()
        elif self._next_pos is None:
            raise ValueError("a call at start_pos > 0 needs a prefill at start_pos 0 first")
        elif start_pos != self._next_pos:
            raise ValueError(f"the request continues at position {self._next_pos}, got {start_pos}")
        ops = self.ops
        x = ops.embedding(input_ids[0].to(self.first_device), self.embed)
        residual = None
        step = None
        for layer, device in zip(self.layers, self.layer_devices, strict=True):
            if step is None or step.cos_sin.device != device:
                step = Step.build(start_pos, seqlen, self.cos_sin[device])
                x = x.to(device)
                residual = None if residual is None else residual.to(device)
            x, residual = layer(ops, x, residual, step)
        h, _ = ops.add_norm(x[-1:], residual[-1:], self.norm)
        logits = ops.gemm_f32(h, self.lm_head)
        self._next_pos = start_pos + seqlen
        return logits

    __call__ = forward

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        *,
        eos_token_id: int | Sequence[int] | None = None,
    ) -> list[int]:
        """Greedily extend one prompt by up to ``max_new_tokens`` tokens; stops after an EOS
        token (``config.eos_token_id`` unless given) or once the sequence fills
        ``max_seq_len``."""
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        tokens = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.first_device)
        logits = self.forward(tokens, 0)
        pos = tokens.shape[1]
        out: list[int] = []
        for _ in range(max_new_tokens):
            nxt = self.ops.argmax(logits)
            token = int(nxt.item())
            out.append(token)
            if token in eos_ids or pos >= self.max_seq_len:
                break
            logits = self.forward(nxt.view(1, 1).to(self.first_device), pos)
            pos += 1
        return out
