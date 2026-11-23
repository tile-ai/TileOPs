"""Load a Qwen3.8-27B checkpoint's text decoder into the layouts its ops read."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from models.loader import read_checkpoint
from models.qwen3_8.layers import DecoderLayer, GatedDeltaNet

if TYPE_CHECKING:
    from models.qwen3_8.model import Qwen3_8ForCausalLM

__all__ = ["load_weights"]

_PREFIX = "model.language_model."


def load_weights(model: Qwen3_8ForCausalLM, path: Path) -> None:
    """Read the text decoder's tensors of the checkpoint at ``path`` onto their layers' devices,
    as :func:`read_checkpoint` does. The vision tower's and the MTP layer's tensors are not
    read."""
    tensors = read_checkpoint(
        model,
        path,
        prefix=_PREFIX,
        skipped=("model.visual.", "mtp."),
        outside=(f"{_PREFIX}embed_tokens.weight", f"{_PREFIX}norm.weight", "lm_head.weight"),
        load_layer=lambda i, t: _load_layer(model.layers[i], t),
    )
    model.embed = tensors.pop(f"{_PREFIX}embed_tokens.weight")
    model.norm = tensors.pop(f"{_PREFIX}norm.weight")
    model.lm_head = tensors.pop("lm_head.weight")


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
