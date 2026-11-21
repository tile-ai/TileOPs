"""DeepSeek-V4 causal language model composed from TileOPs ops."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from models.deepseek_v4.config import DeepseekV4Config
from models.deepseek_v4.layers import Block
from models.deepseek_v4.ops import ModelOps, RoPETable

__all__ = ["DeepseekV4ForCausalLM"]


class DeepseekV4ForCausalLM:
    """DeepSeek-V4 text decoder: embedding, mHC blocks, ``hc_head`` collapse and LM head.

    Layers are spread over ``devices`` in contiguous runs; the streams move to the next
    device between runs. A forward call advances one request: ``start_pos`` 0 is a prefill
    that resets the decoding state; any later ``start_pos`` is a decode step of one token and
    must be the number of tokens already passed. The MTP layer is not built.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
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
        n_dev = len(self.devices)
        self.layer_devices = [
            self.devices[i * n_dev // config.n_layers] for i in range(config.n_layers)
        ]
        self.layers = [Block(config, i, max_seq_len) for i in range(config.n_layers)]
        ropes: dict[tuple[torch.device, bool], RoPETable] = {}
        for layer, device in zip(self.layers, self.layer_devices, strict=True):
            compressed = layer.attn.ratio != 0
            key = (device, compressed)
            if key not in ropes:
                base = config.compress_rope_theta if compressed else config.rope_theta
                ropes[key] = RoPETable(config, base, compressed, max_seq_len, device)
            layer.rope = ropes[key]
        self.embed: torch.Tensor
        self.norm: torch.Tensor
        self.head: torch.Tensor
        self.hc_head: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        self._batch = 0

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        devices: Sequence[torch.device | str | int] = ("cuda",),
        max_seq_len: int = 4096,
    ) -> DeepseekV4ForCausalLM:
        """Load a Hugging Face DeepSeek-V4 checkpoint directory (``safetensors`` shards)."""
        from models.deepseek_v4.loader import load_weights

        model = cls(
            DeepseekV4Config.from_pretrained(path), devices=devices, max_seq_len=max_seq_len
        )
        load_weights(model, Path(path))
        return model

    @property
    def first_device(self) -> torch.device:
        """The device holding the embedding and the first layer."""
        return self.layer_devices[0]

    @property
    def last_device(self) -> torch.device:
        """The device holding the last layer and the LM head."""
        return self.layer_devices[-1]

    def reset(self, batch: int) -> None:
        """Clear every layer's decoding state for ``batch`` requests."""
        for layer in self.layers:
            layer.reset(batch)
        self._batch = batch

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Run ``input_ids`` (``[1, S]``) from ``start_pos``; return the float32 logits of its
        last token, ``[1, vocab_size]``."""
        batch, seqlen = input_ids.shape
        if batch != 1:
            raise ValueError(f"forward runs one request, got a batch of {batch}")
        if start_pos == 0:
            self.reset(batch)
        elif self._batch == 0:
            raise ValueError("a decode step needs a prefill at start_pos 0 first")
        elif seqlen != 1:
            raise ValueError(f"a decode step takes one token per request, got {seqlen}")
        if start_pos + seqlen > self.max_seq_len:
            raise ValueError(
                f"position {start_pos + seqlen} exceeds max_seq_len {self.max_seq_len}"
            )
        ops = self.ops
        ids = input_ids.to(self.first_device)
        x = ops.embedding(ids, self.embed)
        h = x.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        for layer, device in zip(self.layers, self.layer_devices, strict=True):
            if h.device != device:
                h = h.to(device)
                ids = ids.to(device)
            h = layer(ops, h, ids, start_pos)
        last = h[:, -1].contiguous()
        x = ops.hc_head(last, *self.hc_head)
        x = ops.norm(x, self.norm)
        return ops.gemm_f32(x, self.head)

    __call__ = forward

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        *,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """Greedily extend one prompt by up to ``max_new_tokens`` tokens; stops after EOS."""
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        tokens = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.first_device)
        logits = self.forward(tokens, 0)
        pos = tokens.shape[1]
        out: list[int] = []
        for _ in range(max_new_tokens):
            nxt = self.ops.argmax(logits)
            token = int(nxt.item())
            out.append(token)
            if token == eos or pos >= self.max_seq_len:
                break
            logits = self.forward(nxt.view(1, 1).to(self.first_device), pos)
            pos += 1
        return out
