"""Qwen3.8-27B text decoder hyperparameters, read from a Hugging Face ``config.json``."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

__all__ = ["Qwen3_8Config"]

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"


@dataclasses.dataclass(frozen=True)
class Qwen3_8Config:
    """The Qwen3.8-27B text decoder, read from a Hugging Face ``qwen3_5`` config.

    Field names are the Hugging Face ``qwen3_5_text`` keys. ``layer_types[i]`` is
    ``"linear_attention"`` (Gated DeltaNet) or ``"full_attention"`` (gated GQA). The full
    attention rotates the first ``rotary_dim`` values of each q and k head with interleaved
    MRoPE over ``mrope_section`` frequencies.
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    mrope_section: tuple[int, ...]
    eos_token_id: int

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types names {len(self.layer_types)} layers, the model has "
                f"{self.num_hidden_layers}"
            )
        unknown = set(self.layer_types) - {FULL_ATTENTION, LINEAR_ATTENTION}
        if unknown:
            raise ValueError(f"unsupported layer_types {sorted(unknown)}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("linear_num_value_heads must be a multiple of linear_num_key_heads")
        if self.linear_key_head_dim != self.linear_value_head_dim:
            raise ValueError(
                "the Gated DeltaNet state must be square: key and value head dims differ"
            )
        if 2 * sum(self.mrope_section) != self.rotary_dim:
            raise ValueError(
                f"mrope_section {list(self.mrope_section)} does not cover the {self.rotary_dim} "
                "rotary dims"
            )

    @classmethod
    def from_hf(cls, config: dict) -> Qwen3_8Config:
        """Read a ``qwen3_5`` config (its ``text_config``) or a bare ``qwen3_5_text`` one.

        Raises ``ValueError`` on another model type and on any setting this model does not
        compute: attention biases, an ungated attention output, a gate other than SiLU, a rope
        type other than default interleaved MRoPE, tied embeddings, a dtype other than
        bfloat16, and a recurrent state other than float32.
        """
        model_type = config.get("model_type")
        if model_type == "qwen3_5":
            config = config["text_config"]
            model_type = config.get("model_type")
        if model_type != "qwen3_5_text":
            raise ValueError(f"model_type {model_type!r} is not qwen3_5 or qwen3_5_text")
        rope = config["rope_parameters"]
        refused = {
            "attention_bias": config.get("attention_bias", False) is not False,
            "attn_output_gate": config.get("attn_output_gate", True) is not True,
            "hidden_act": config.get("hidden_act", "silu") != "silu",
            "output_gate_type": config.get("output_gate_type", "swish") != "swish",
            "rope_type": rope.get("rope_type", "default") != "default",
            "mrope_interleaved": rope.get("mrope_interleaved", False) is not True,
            "tie_word_embeddings": config.get("tie_word_embeddings", False) is not False,
            "dtype": config.get("dtype", "bfloat16") != "bfloat16",
            "mamba_ssm_dtype": config.get("mamba_ssm_dtype", "float32") != "float32",
        }
        if any(refused.values()):
            raise ValueError(f"unsupported config: {[k for k, v in refused.items() if v]}")
        head_dim = config["head_dim"]
        return cls(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_hidden_layers=config["num_hidden_layers"],
            layer_types=tuple(config["layer_types"]),
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            head_dim=head_dim,
            linear_num_key_heads=config["linear_num_key_heads"],
            linear_num_value_heads=config["linear_num_value_heads"],
            linear_key_head_dim=config["linear_key_head_dim"],
            linear_value_head_dim=config["linear_value_head_dim"],
            linear_conv_kernel_dim=config["linear_conv_kernel_dim"],
            rms_norm_eps=config["rms_norm_eps"],
            rope_theta=rope["rope_theta"],
            rotary_dim=int(head_dim * rope["partial_rotary_factor"]),
            mrope_section=tuple(rope["mrope_section"]),
            eos_token_id=config["eos_token_id"],
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> Qwen3_8Config:
        """Read ``config.json`` from a checkpoint directory."""
        with open(Path(path) / "config.json") as f:
            return cls.from_hf(json.load(f))
