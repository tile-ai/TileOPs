"""DeepSeek-V4 causal language model composed from TileOPs ops."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from models.causal_lm import CausalLM
from models.deepseek_v4.config import DeepseekV4Config
from models.deepseek_v4.layers import Block, Step
from models.deepseek_v4.ops import ModelOps, RoPETable

__all__ = ["DeepseekV4ForCausalLM"]


class DeepseekV4ForCausalLM(CausalLM):
    """DeepSeek-V4 text decoder: embedding, mHC blocks, ``hc_head`` collapse and LM head.

    Layers are spread over ``devices`` in contiguous runs; a device without an index is the current
    one at construction. Each run computes with its device current and builds the index metadata its
    layers share, a ``Step``, once; the streams move to the next device between runs, and a forward
    call leaves the current device as it found it. A forward call advances one request:
    ``start_pos`` 0 is a prefill that starts a new request; any later ``start_pos`` is a decode
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
        self.layers = [
            Block(config, i, max_seq_len, device) for i, device in enumerate(self.layer_devices)
        ]
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

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Run ``input_ids`` (``[1, S]``) from ``start_pos``; return the float32 logits of its
        last token, ``[1, vocab_size]``."""
        seqlen = self._continue(input_ids, start_pos)
        if start_pos and seqlen != 1:
            raise ValueError(f"a decode step takes one token, got {seqlen}")
        ops = self.ops
        ids = input_ids.to(self.first_device)
        with torch.accelerator.device_index(self.first_device.index):
            x = ops.embedding(ids, self.embed)
            h = x.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        previous = None
        for layer, device in zip(self.layers, self.layer_devices, strict=True):
            with torch.accelerator.device_index(device.index):
                if device != previous:
                    previous = device
                    step = Step.build(self.config, start_pos, seqlen, device)
                    h, ids = h.to(device), ids.to(device)
                h = layer(ops, h, ids, step)
        with torch.accelerator.device_index(self.last_device.index):
            x = ops.hc_head(h[:, -1].contiguous(), *self.hc_head)
            logits = ops.gemm_f32(ops.norm(x, self.norm), self.head)
        self._next_pos = start_pos + seqlen
        return logits

    __call__ = forward


def _indexed(device: torch.device | str | int) -> torch.device:
    """``device`` as a CUDA device with an index: an int is that index, and ``"cuda"`` the
    current device's."""
    if isinstance(device, int):
        return torch.device("cuda", device)
    device = torch.device(device)
    if device.index is None:
        return torch.device(device.type, torch.accelerator.current_device_index())
    return device
