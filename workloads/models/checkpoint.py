"""Hugging Face checkpoint directories for the models' tests."""

import json
from pathlib import Path

import torch

__all__ = ["write_checkpoint"]


def write_checkpoint(path: Path, config: dict, tensors: dict[str, torch.Tensor]) -> None:
    """Write ``tensors`` and ``config`` to ``path`` as a Hugging Face checkpoint whose tensors
    alternate between two shards, so every layer spans both. Needs ``safetensors``."""
    from safetensors.torch import save_file

    names = sorted(tensors)
    weight_map = {n: f"model-0000{1 + i % 2}-of-00002.safetensors" for i, n in enumerate(names)}
    for shard in set(weight_map.values()):
        save_file({n: tensors[n] for n in names if weight_map[n] == shard}, str(path / shard))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (path / "config.json").write_text(json.dumps(config))
