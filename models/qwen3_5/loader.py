"""Load a Hugging Face ``qwen3_5`` checkpoint's text decoder into the layouts its ops read."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from models.qwen3_5.layers import DecoderLayer, GatedDeltaNet

if TYPE_CHECKING:
    from models.qwen3_5.model import Qwen3_5ForCausalLM

__all__ = ["load_weights"]

_PREFIX = "model.language_model."
# The vision tower and the MTP layer, which the text decoder does not read.
_SKIPPED = ("model.visual.", "mtp.")
_LAYER = re.compile(rf"{re.escape(_PREFIX)}layers\.(\d+)\.")


def load_weights(model: Qwen3_5ForCausalLM, path: Path) -> None:
    """Read the text decoder's tensors of the checkpoint at ``path`` onto their layers' devices.

    Needs the ``safetensors`` package. Tensors under ``model.visual.`` and ``mtp.`` are not
    read; any other tensor the decoder has no place for raises ``ValueError``.
    """
    from safetensors import safe_open

    with open(path / "model.safetensors.index.json") as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]
    by_shard: dict[tuple[str, torch.device], list[str]] = defaultdict(list)
    per_layer: dict[int, set[str]] = defaultdict(set)
    unused = []
    for name, shard in weight_map.items():
        if name.startswith(_SKIPPED):
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
    tensors: dict[str, torch.Tensor] = {}
    for (shard, device), names in sorted(by_shard.items(), key=lambda item: item[0][0]):
        with safe_open(str(path / shard), framework="pt", device=str(device)) as f:
            for name in names:
                tensors[name] = f.get_tensor(name)
        for layer_id in sorted({int(m.group(1)) for n in names if (m := _LAYER.match(n))}):
            if all(n in tensors for n in per_layer[layer_id]):
                prefix = f"{_PREFIX}layers.{layer_id}."
                own = {n[len(prefix) :]: tensors.pop(n) for n in per_layer[layer_id]}
                _load_layer(model.layers[layer_id], own)
    missing = set(range(len(model.layers))) - set(per_layer)
    if missing:
        raise ValueError(f"the checkpoint holds no tensors of layers {sorted(missing)}")
    model.embed = tensors.pop(f"{_PREFIX}embed_tokens.weight")
    model.norm = tensors.pop(f"{_PREFIX}norm.weight")
    model.lm_head = tensors.pop("lm_head.weight")


def _device_of(model: Qwen3_5ForCausalLM, name: str) -> torch.device | None:
    """The device a checkpoint tensor belongs on, or None for one the decoder does not read."""
    if m := _LAYER.match(name):
        layer_id = int(m.group(1))
        return model.layer_devices[layer_id] if layer_id < len(model.layers) else None
    if name == f"{_PREFIX}embed_tokens.weight":
        return model.first_device
    if name in (f"{_PREFIX}norm.weight", "lm_head.weight"):
        return model.last_device
    return None


def _load_layer(layer: DecoderLayer, t: dict[str, torch.Tensor]) -> None:
    """Fill one layer from its tensors, named relative to ``layers.<i>.``."""
    layer.input_norm = t.pop("input_layernorm.weight")
    layer.post_norm = t.pop("post_attention_layernorm.weight")
    layer.mlp.gate_up_proj = torch.cat([t.pop("mlp.gate_proj.weight"), t.pop("mlp.up_proj.weight")])
    layer.mlp.down_proj = t.pop("mlp.down_proj.weight")
    mixer = layer.mixer
    if isinstance(mixer, GatedDeltaNet):
        mixer.in_proj = torch.cat(
            [t.pop(f"linear_attn.in_proj_{p}.weight") for p in ("qkv", "z", "b", "a")]
        )
        mixer.conv_weight = t.pop("linear_attn.conv1d.weight").squeeze(1)
        mixer.A_log = t.pop("linear_attn.A_log").float()
        mixer.dt_bias = t.pop("linear_attn.dt_bias").float()
        mixer.norm = t.pop("linear_attn.norm.weight")
        mixer.out_proj = t.pop("linear_attn.out_proj.weight")
    else:
        # Each head's q_proj rows are its query, then its output gate.
        q, gate = (
            t.pop("self_attn.q_proj.weight").view(mixer.heads, 2, mixer.head_dim, -1).unbind(1)
        )
        mixer.qkv_proj = torch.cat(
            [q.flatten(0, 1), t.pop("self_attn.k_proj.weight"), t.pop("self_attn.v_proj.weight")]
        )
        mixer.gate_proj = gate.flatten(0, 1).contiguous()
        mixer.q_norm = t.pop("self_attn.q_norm.weight")
        mixer.k_norm = t.pop("self_attn.k_norm.weight")
        mixer.o_proj = t.pop("self_attn.o_proj.weight")
    if t:
        raise ValueError(f"layer {layer.layer_id}: unused tensors {sorted(t)[:8]}")
