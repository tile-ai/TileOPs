"""DeepSeek-V4-Flash causal language model composed from TileOPs ops."""

from __future__ import annotations

from pathlib import Path

import torch

from models.causal_lm import CausalLM, indexed_device
from models.deepseek_v4_flash.config import DeepseekV4FlashConfig
from models.deepseek_v4_flash.layers import Block, Step
from models.deepseek_v4_flash.ops import ModelOps, RoPETable

__all__ = ["DeepseekV4FlashForCausalLM"]


class DeepseekV4FlashForCausalLM(CausalLM):
    """DeepSeek-V4-Flash text decoder: embedding, mHC blocks, ``hc_head`` collapse and LM head.

    Every weight and every layer's decoding state is on ``device``; a device without an index is
    the current one at construction. A forward call runs with ``device`` current and leaves the
    current device as it found it. A forward call advances one request:
    ``start_pos`` 0 is a prefill that starts a new request; any later ``start_pos`` is a decode
    step of one token and must be the number of tokens already passed. The MTP layer is not built.
    """

    def __init__(
        self,
        config: DeepseekV4FlashConfig,
        *,
        device: torch.device | str = "cuda",
        max_seq_len: int = 4096,
    ) -> None:
        self.config = config
        self.max_seq_len = max_seq_len
        self.device = indexed_device(device)
        self.ops = ModelOps(config)
        self.layers = [Block(config, i, max_seq_len, self.device) for i in range(config.n_layers)]
        ropes: dict[bool, RoPETable] = {}
        for layer in self.layers:
            compressed = layer.attn.ratio != 0
            if compressed not in ropes:
                base = config.compress_rope_theta if compressed else config.rope_theta
                ropes[compressed] = RoPETable(config, base, compressed, max_seq_len, self.device)
            layer.rope = ropes[compressed]
        self.embed: torch.Tensor
        self.norm: torch.Tensor
        self.head: torch.Tensor
        self.hc_head: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        device: torch.device | str = "cuda",
        max_seq_len: int = 4096,
    ) -> DeepseekV4FlashForCausalLM:
        """Load a Hugging Face DeepSeek-V4-Flash checkpoint directory (``safetensors`` shards)."""
        from models.deepseek_v4_flash.loader import load_weights

        model = cls(
            DeepseekV4FlashConfig.from_pretrained(path), device=device, max_seq_len=max_seq_len
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
        ids = input_ids.to(self.device)
        with torch.accelerator.device_index(self.device.index):
            x = ops.embedding(ids, self.embed)
            h = x.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
            step = Step.build(self.config, start_pos, seqlen, self.device)
            for layer in self.layers:
                h = layer(ops, h, ids, step)
            x = ops.hc_head(h[:, -1].contiguous(), *self.hc_head)
            logits = ops.gemm_f32(ops.norm(x, self.norm), self.head)
        self._next_pos = start_pos + seqlen
        return logits

    __call__ = forward
