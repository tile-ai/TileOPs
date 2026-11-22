"""DeepSeek-V4 causal language model composed from TileOPs ops."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from models.deepseek_v4.config import DeepseekV4Config
from models.deepseek_v4.layers import Block, Step
from models.deepseek_v4.ops import ModelOps, RoPETable

__all__ = ["DeepseekV4ForCausalLM"]


class DeepseekV4ForCausalLM:
    """DeepSeek-V4 text decoder: embedding, mHC blocks, ``hc_head`` collapse and LM head.

    Layers are spread over ``devices`` in contiguous runs; a device without an index is the current
    one at construction. Each run computes with its device current and builds the index metadata its
    layers share, a ``Step``, once; the streams move to the next device between runs, and a forward
    call leaves the current device as it found it. A forward call advances one request:
    ``start_pos`` 0 is a prefill that resets the decoding state; any later ``start_pos`` is a decode
    step of one token and must be the number of tokens already passed. The MTP layer is not built.
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
        self.devices = [_indexed(d) for d in devices]
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
        self._next_pos: int | None = None

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

    def reset(self) -> None:
        """Clear every layer's decoding state; the next call must be a prefill."""
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
            raise ValueError("a decode step needs a prefill at start_pos 0 first")
        elif start_pos != self._next_pos:
            raise ValueError(
                f"the next decode step is at position {self._next_pos}, got {start_pos}"
            )
        elif seqlen != 1:
            raise ValueError(f"a decode step takes one token, got {seqlen}")
        ops = self.ops
        ids = input_ids.to(self.first_device)
        with torch.cuda.device(self.first_device):
            x = ops.embedding(ids, self.embed)
            h = x.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        previous = None
        for layer, device in zip(self.layers, self.layer_devices, strict=True):
            with torch.cuda.device(device):
                if device != previous:
                    previous = device
                    step = Step.build(self.config, start_pos, seqlen, device)
                    h, ids = h.to(device), ids.to(device)
                h = layer(ops, h, ids, step)
        with torch.cuda.device(self.last_device):
            x = ops.hc_head(h[:, -1].contiguous(), *self.hc_head)
            logits = ops.gemm_f32(ops.norm(x, self.norm), self.head)
        self._next_pos = start_pos + seqlen
        return logits

    __call__ = forward

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        *,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """Greedily extend one prompt by up to ``max_new_tokens`` tokens; stops after EOS or
        once the sequence fills ``max_seq_len``."""
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        tokens = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.first_device)
        logits = self.forward(tokens, 0)
        pos = tokens.shape[1]
        out: list[int] = []
        for _ in range(max_new_tokens):
            with torch.cuda.device(self.last_device):
                nxt = self.ops.argmax(logits)
            token = int(nxt.item())
            out.append(token)
            if token == eos or pos >= self.max_seq_len:
                break
            logits = self.forward(nxt.view(1, 1).to(self.first_device), pos)
            pos += 1
        return out


def _indexed(device: torch.device | str | int) -> torch.device:
    """``device`` as a CUDA device with an index: an int is that index, and ``"cuda"`` the
    current device's."""
    if isinstance(device, int):
        return torch.device("cuda", device)
    device = torch.device(device)
    if device.index is None:
        return torch.device(device.type, torch.cuda.current_device())
    return device
