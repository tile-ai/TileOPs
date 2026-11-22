"""Load a Hugging Face DeepSeek-V4 checkpoint into the layouts the model's ops read.

The checkpoint's FP8 weights carry one scale per ``_SCALE_BLOCK`` x ``_SCALE_BLOCK`` block, which
the stacked and the dequantized weights both follow.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from models.deepseek_v4.layers import Block, Compressor, FP8Linear

if TYPE_CHECKING:
    from models.deepseek_v4.model import DeepseekV4ForCausalLM

__all__ = ["load_weights"]

_SCALE_BLOCK = 128
_LAYER = re.compile(r"layers\.(\d+)\.")


def load_weights(model: DeepseekV4ForCausalLM, path: Path) -> None:
    """Read every tensor of the checkpoint at ``path`` onto its layer's device.

    Needs the ``safetensors`` package. The MTP layer's tensors are not read; any other tensor
    the model has no place for, and a checkpoint missing a layer, raise ``ValueError`` before a
    tensor is read.
    """
    from safetensors import safe_open

    with open(path / "model.safetensors.index.json") as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]
    by_shard: dict[tuple[str, torch.device], list[str]] = defaultdict(list)
    per_layer: dict[int, set[str]] = defaultdict(set)
    unused = []
    for name, shard in weight_map.items():
        if name.startswith("mtp."):
            continue
        device = _device_of(model, name)
        if device is None:
            unused.append(name)
            continue
        by_shard[(shard, device)].append(name)
        if m := _LAYER.match(name):
            per_layer[int(m.group(1))].add(name)
    if unused:
        raise ValueError(f"unused checkpoint tensors: {sorted(unused)[:8]}")
    missing = set(range(len(model.layers))) - set(per_layer)
    if missing:
        raise ValueError(f"the checkpoint holds no tensors of layers {sorted(missing)}")
    tensors: dict[str, torch.Tensor] = {}
    for (shard, device), names in sorted(by_shard.items(), key=lambda item: item[0][0]):
        with safe_open(str(path / shard), framework="pt", device=str(device)) as f:
            for name in names:
                tensors[name] = f.get_tensor(name)
        for layer_id in sorted({int(m.group(1)) for n in names if (m := _LAYER.match(n))}):
            if all(n in tensors for n in per_layer[layer_id]):
                prefix = f"layers.{layer_id}."
                own = {n[len(prefix) :]: tensors.pop(n) for n in per_layer[layer_id]}
                _load_block(model.layers[layer_id], own, model.config.n_routed_experts)
    model.embed = tensors.pop("embed.weight")
    model.norm = tensors.pop("norm.weight")
    model.head = tensors.pop("head.weight")
    model.hc_head = (
        tensors.pop("hc_head_fn"),
        tensors.pop("hc_head_scale"),
        tensors.pop("hc_head_base"),
    )


def _device_of(model: DeepseekV4ForCausalLM, name: str) -> torch.device | None:
    """The device a checkpoint tensor belongs on, or None for one the model does not read."""
    if m := _LAYER.match(name):
        layer_id = int(m.group(1))
        return model.layer_devices[layer_id] if layer_id < len(model.layers) else None
    if name == "embed.weight":
        return model.first_device
    if name in ("norm.weight", "head.weight", "hc_head_fn", "hc_head_scale", "hc_head_base"):
        return model.last_device
    return None


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


def _load_block(block: Block, t: dict[str, torch.Tensor], n_experts: int) -> None:
    """Fill one layer from its tensors, named relative to ``layers.<i>.``."""
    block.attn_norm = t.pop("attn_norm.weight")
    block.ffn_norm = t.pop("ffn_norm.weight")
    block.hc_attn = (t.pop("hc_attn_fn"), t.pop("hc_attn_scale"), t.pop("hc_attn_base"))
    block.hc_ffn = (t.pop("hc_ffn_fn"), t.pop("hc_ffn_scale"), t.pop("hc_ffn_base"))

    attn = block.attn
    attn.wq_a = _fp8(t, "attn.wq_a")
    attn.q_norm = t.pop("attn.q_norm.weight")
    attn.wq_b = _fp8(t, "attn.wq_b")
    attn.wkv = _fp8(t, "attn.wkv")
    attn.kv_norm = t.pop("attn.kv_norm.weight")
    attn.wo_b = _fp8(t, "attn.wo_b")
    attn.attn_sink = t.pop("attn.attn_sink")
    wo_a = _dequant_fp8(t.pop("attn.wo_a.weight"), t.pop("attn.wo_a.scale"))
    attn.wo_a = wo_a.view(attn.o_groups, attn.o_rank, -1).transpose(1, 2).contiguous()
    if attn.compressor is not None:
        _load_compressor(attn.compressor, t, "attn.compressor")
    if attn.indexer is not None:
        attn.indexer.wq_b = _fp8(t, "attn.indexer.wq_b")
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
    ffn.w13 = torch.stack(
        [
            torch.cat([t.pop(f"ffn.experts.{e}.w1.weight"), t.pop(f"ffn.experts.{e}.w3.weight")])
            for e in range(n_experts)
        ]
    ).view(torch.uint8)
    ffn.w13_scale = torch.stack(
        [
            torch.cat([t.pop(f"ffn.experts.{e}.w1.scale"), t.pop(f"ffn.experts.{e}.w3.scale")])
            for e in range(n_experts)
        ]
    )
    ffn.w2 = torch.stack([t.pop(f"ffn.experts.{e}.w2.weight") for e in range(n_experts)]).view(
        torch.uint8
    )
    ffn.w2_scale = torch.stack([t.pop(f"ffn.experts.{e}.w2.scale") for e in range(n_experts)])
    if t:
        raise ValueError(f"layer {block.layer_id}: unused tensors {sorted(t)[:8]}")
