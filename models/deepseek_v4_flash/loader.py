"""Load a Hugging Face DeepSeek-V4-Flash checkpoint into the layouts the model's ops read.

The checkpoint's FP8 weights carry one scale per ``_SCALE_BLOCK`` x ``_SCALE_BLOCK`` block, which
the stacked and the dequantized weights both follow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from models.deepseek_v4_flash.layers import Block, Compressor, FP8Linear
from models.loader import read_checkpoint

if TYPE_CHECKING:
    from models.deepseek_v4_flash.model import DeepseekV4FlashForCausalLM

__all__ = ["load_layer", "load_outside", "load_weights"]

_SCALE_BLOCK = 128


def load_weights(model: DeepseekV4FlashForCausalLM, path: Path) -> None:
    """Read every tensor of the checkpoint at ``path`` onto its layer's device, as
    :func:`read_checkpoint` does. The MTP layer's tensors are not read."""
    tensors = read_checkpoint(
        model,
        path,
        prefix="",
        skipped=("mtp.",),
        outside=(
            "embed.weight",
            "norm.weight",
            "head.weight",
            "hc_head_fn",
            "hc_head_scale",
            "hc_head_base",
        ),
        load_layer=lambda i, t: load_layer(model.layers[i], t, model.config.n_routed_experts),
    )
    load_outside(model, tensors)


def load_outside(model: DeepseekV4FlashForCausalLM, t: dict[str, torch.Tensor]) -> None:
    """Fill the embedding, the final norm, the LM head and ``hc_head`` from their tensors."""
    model.embed = t.pop("embed.weight")
    model.norm = t.pop("norm.weight")
    model.head = t.pop("head.weight")
    model.hc_head = (t.pop("hc_head_fn"), t.pop("hc_head_scale"), t.pop("hc_head_base"))


def _fp8(t: dict[str, torch.Tensor], prefix: str) -> FP8Linear:
    """An FP8 linear layer from ``<prefix>.weight`` and its UE8M0 ``<prefix>.scale``."""
    return FP8Linear(t.pop(f"{prefix}.weight"), t.pop(f"{prefix}.scale").float())


def _fp8_stacked(t: dict[str, torch.Tensor], first: str, second: str) -> FP8Linear:
    """One FP8 linear layer whose output rows are ``first``'s, then ``second``'s; ``first``'s
    rows must fill whole scale blocks for the stacked scales to stay aligned."""
    a, b = _fp8(t, first), _fp8(t, second)
    if a.weight.shape[0] % _SCALE_BLOCK:
        raise ValueError(
            f"{first} has {a.weight.shape[0]} rows, not whole {_SCALE_BLOCK}-row blocks"
        )
    return FP8Linear(torch.cat([a.weight, b.weight]), torch.cat([a.scale, b.scale]))


def _dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """A bfloat16 copy of an FP8 weight under 128x128 block scales."""
    n, k = weight.shape
    b = _SCALE_BLOCK
    blocks = weight.float().view(n // b, b, k // b, b)
    return (blocks * scale.float()[:, None, :, None]).view(n, k).to(torch.bfloat16)


def _load_compressor(c: Compressor, t: dict[str, torch.Tensor], prefix: str) -> None:
    c.wkv_gate = torch.cat([t.pop(f"{prefix}.wkv.weight"), t.pop(f"{prefix}.wgate.weight")])
    c.ape = t.pop(f"{prefix}.ape")
    c.norm = t.pop(f"{prefix}.norm.weight")


def load_layer(block: Block, t: dict[str, torch.Tensor], n_experts: int) -> None:
    """Fill one layer, with ``n_experts`` routed experts, from its tensors, named relative to
    ``layers.<i>.``."""
    block.attn_norm = t.pop("attn_norm.weight")
    block.ffn_norm = t.pop("ffn_norm.weight")
    block.hc_attn = (t.pop("hc_attn_fn"), t.pop("hc_attn_scale"), t.pop("hc_attn_base"))
    block.hc_ffn = (t.pop("hc_ffn_fn"), t.pop("hc_ffn_scale"), t.pop("hc_ffn_base"))

    attn = block.attn
    attn.wq_a_kv = _fp8_stacked(t, "attn.wq_a", "attn.wkv")
    attn.wq_a = attn.wq_a_kv.rows(0, attn.q_rank)
    attn.wkv = attn.wq_a_kv.rows(attn.q_rank, attn.wq_a_kv.weight.shape[0])
    attn.q_norm = t.pop("attn.q_norm.weight")
    if attn.indexer is None:
        attn.wq_b_index = None
        attn.wq_b = _fp8(t, "attn.wq_b")
    else:
        attn.wq_b_index = _fp8_stacked(t, "attn.wq_b", "attn.indexer.wq_b")
        width = attn.n_heads * attn.head_dim
        attn.wq_b = attn.wq_b_index.rows(0, width)
        attn.indexer.wq_b = attn.wq_b_index.rows(width, attn.wq_b_index.weight.shape[0])
    attn.kv_norm = t.pop("attn.kv_norm.weight")
    attn.wo_b = _fp8(t, "attn.wo_b")
    attn.attn_sink = t.pop("attn.attn_sink")
    wo_a = _dequant_fp8(t.pop("attn.wo_a.weight"), t.pop("attn.wo_a.scale"))
    attn.wo_a = wo_a.view(attn.o_groups, attn.o_rank, -1).transpose(1, 2).contiguous()
    if attn.compressor is not None:
        _load_compressor(attn.compressor, t, "attn.compressor")
    if attn.indexer is not None:
        attn.indexer.weights_proj = t.pop("attn.indexer.weights_proj.weight")
        _load_compressor(attn.indexer.compressor, t, "attn.indexer.compressor")

    ffn = block.ffn
    ffn.gate = t.pop("ffn.gate.weight")
    if ffn.hash:
        ffn.tid2eid = t.pop("ffn.gate.tid2eid").to(torch.int32)
    else:
        ffn.bias = t.pop("ffn.gate.bias")
    ffn.shared_w13 = _fp8_stacked(t, "ffn.shared_experts.w1", "ffn.shared_experts.w3")
    ffn.shared_w2 = _fp8(t, "ffn.shared_experts.w2")

    def experts(name: str) -> torch.Tensor:
        return torch.stack([t.pop(f"ffn.experts.{e}.{name}") for e in range(n_experts)])

    # Each expert's w13 rows are its w1 rows, then its w3 rows.
    ffn.w13 = torch.cat([experts("w1.weight"), experts("w3.weight")], dim=1).view(torch.uint8)
    ffn.w13_scale = torch.cat([experts("w1.scale"), experts("w3.scale")], dim=1)
    ffn.w2 = experts("w2.weight").view(torch.uint8)
    ffn.w2_scale = experts("w2.scale")
    if t:
        raise ValueError(f"layer {block.layer_id}: unused tensors {sorted(t)[:8]}")
