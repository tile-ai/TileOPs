"""DeepSeek-V4-Flash causal language model composed from TileOPs ops."""

from __future__ import annotations

from typing import Any

import torch

from models.causal_lm import CausalLM
from models.deepseek_v4_flash.config import DeepseekV4FlashConfig
from models.deepseek_v4_flash.layers import Block, Step
from models.deepseek_v4_flash.loader import load_weights
from models.deepseek_v4_flash.ops import ModelOps, RoPETable

__all__ = ["DeepseekV4FlashForCausalLM"]


class DeepseekV4FlashForCausalLM(CausalLM):
    """DeepSeek-V4-Flash text decoder: embedding, mHC blocks, ``hc_head`` collapse and LM head.

    A call at ``start_pos`` 0 is a prefill; a later call is a decode step of one token. A
    decode phase is which compressors close a window and whether the indexers hold an entry.
    The MTP layer is not built.
    """

    config_type = DeepseekV4FlashConfig
    load_weights = staticmethod(load_weights)

    def __init__(self, config: DeepseekV4FlashConfig, **options: Any) -> None:
        super().__init__(config, **options)
        max_seq_len = self.max_seq_len
        self.ops = ModelOps(config)
        self.layers = [Block(config, i, max_seq_len, self.device) for i in range(config.n_layers)]
        self._ratios = tuple(sorted({layer.attn.ratio for layer in self.layers} - {0}))
        self._indexed_ratios = tuple(r for r in self._ratios if config.indexed(r))
        ropes: dict[bool, RoPETable] = {}
        for layer in self.layers:
            compressed = layer.attn.ratio != 0
            if compressed not in ropes:
                base = config.compress_rope_theta if compressed else config.rope_theta
                ratios = self._ratios if compressed else ()
                ropes[compressed] = RoPETable(
                    config, base, compressed, max_seq_len, self.device, ratios
                )
            layer.rope = ropes[compressed]
        self.embed: torch.Tensor
        self.norm: torch.Tensor
        self.head: torch.Tensor
        self.hc_head: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def _continue(self, input_ids: torch.Tensor, start_pos: int) -> int:
        seqlen = super()._continue(input_ids, start_pos)
        if start_pos and seqlen != 1:
            raise ValueError(f"a decode step takes one token, got {seqlen}")
        return seqlen

    def _step(self, start_pos: int, seqlen: int) -> Step:
        return Step.build(self.config, start_pos, seqlen, self.device)

    def _run(self, ids: torch.Tensor, step: Step) -> torch.Tensor:
        h = self.streams(ids)
        for layer in self.layers:
            h = layer(self.ops, h, ids, step)
        return self.logits(h)

    def _phase(self, step: Step) -> tuple[bool, ...]:
        """For each compression ratio, whether ``step`` closes a window, then for each indexed
        ratio, whether the indexer holds an entry."""
        return tuple(step.closes(r) for r in self._ratios) + tuple(
            step.has_entries(r) for r in self._indexed_ratios
        )

    def streams(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The ``hc_mult`` residual streams ``[1, S, hc_mult, dim]`` the layers take for
        ``input_ids`` (``[1, S]`` on ``device``): copies of each token's embedding."""
        x = self.ops.embedding(input_ids, self.embed)
        return x.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """The float32 logits ``[1, vocab_size]`` of the last token of the layers' output
        streams ``h``, collapsed by ``hc_head``."""
        x = self.ops.hc_head(h[:, -1].contiguous(), *self.hc_head)
        return self.ops.gemm_f32(self.ops.norm(x, self.norm), self.head)
