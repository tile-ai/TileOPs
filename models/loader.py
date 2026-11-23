"""Read a sharded Hugging Face checkpoint onto a model's device."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch

from models.causal_lm import CausalLM

__all__ = ["read_checkpoint"]


def read_checkpoint(
    model: CausalLM,
    path: Path,
    *,
    prefix: str,
    skipped: tuple[str, ...],
    outside: tuple[str, ...],
    load_layer: Callable[[int, dict[str, torch.Tensor]], None],
) -> dict[str, torch.Tensor]:
    """Read the tensors of the checkpoint at ``path`` onto ``model.device``, and return those
    outside the layers.

    Needs the ``safetensors`` package. A layer's tensors are ``<prefix>layers.<i>.<name>``,
    passed as ``{name: tensor}`` to ``load_layer(i, ...)`` once all of them are read;
    ``outside`` names the tensors outside the layers, and tensors starting with one of
    ``skipped`` are not read.
    Any other tensor and a layer without tensors raise ``ValueError`` before any tensor is read;
    a tensor ``load_layer`` asks for and does not find raises ``ValueError``.
    """
    from safetensors import safe_open

    layer_re = re.compile(rf"{re.escape(prefix)}layers\.(\d+)\.")
    with open(path / "model.safetensors.index.json") as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]
    by_shard: dict[str, list[str]] = defaultdict(list)
    per_layer: dict[int, set[str]] = defaultdict(set)
    unused = []
    for name, shard in weight_map.items():
        if name.startswith(skipped):
            continue
        m = layer_re.match(name)
        known = int(m.group(1)) < len(model.layers) if m else name in outside
        if not known:
            unused.append(name)
            continue
        by_shard[shard].append(name)
        if m:
            per_layer[int(m.group(1))].add(name)
    if unused:
        raise ValueError(f"unused checkpoint tensors: {sorted(unused)[:8]}")
    missing = set(range(len(model.layers))) - set(per_layer)
    if missing:
        raise ValueError(f"the checkpoint holds no tensors of layers {sorted(missing)}")
    tensors: dict[str, torch.Tensor] = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(str(path / shard), framework="pt", device=str(model.device)) as f:
            for name in names:
                tensors[name] = f.get_tensor(name)
        for layer_id in sorted({int(m.group(1)) for n in names if (m := layer_re.match(n))}):
            if all(n in tensors for n in per_layer[layer_id]):
                layer_prefix = f"{prefix}layers.{layer_id}."
                own = {n[len(layer_prefix) :]: tensors.pop(n) for n in per_layer[layer_id]}
                try:
                    load_layer(layer_id, own)
                except KeyError as missing_name:
                    raise ValueError(f"layer {layer_id}: missing tensor {missing_name}") from None
    return tensors
