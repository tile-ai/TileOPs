"""DeepSeek-V4-Flash hyperparameters, read from a Hugging Face ``config.json``."""

from __future__ import annotations

import dataclasses

__all__ = ["DeepseekV4FlashConfig"]


@dataclasses.dataclass(frozen=True)
class DeepseekV4FlashConfig:
    """The hyperparameters a DeepSeek-V4-Flash checkpoint's ``config.json`` states.

    Field names are the official ``inference/model.py`` ``ModelArgs`` names; ``from_hf``
    maps the Hugging Face keys onto them. ``compress_ratios[i]`` is layer ``i``'s KV (the MTP
    layers follow the decoder's) compression ratio: 0 is sliding-window attention only, 4 adds
    overlapping compressed entries chosen by the lightning indexer, any other ratio attends to
    every closed entry.
    """

    vocab_size: int
    dim: int
    moe_inter_dim: int
    n_layers: int
    n_hash_layers: int
    n_mtp_layers: int
    n_heads: int
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    score_func: str
    route_scale: float
    swiglu_limit: float
    q_lora_rank: int
    head_dim: int
    rope_head_dim: int
    norm_eps: float
    o_groups: int
    o_lora_rank: int
    window_size: int
    compress_ratios: tuple[int, ...]
    compress_rope_theta: float
    original_seq_len: int
    rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    eos_token_id: int

    def __post_init__(self) -> None:
        if len(self.compress_ratios) != self.n_layers + self.n_mtp_layers:
            raise ValueError(
                f"compress_ratios names {len(self.compress_ratios)} layers, the model has "
                f"{self.n_layers} plus {self.n_mtp_layers} MTP"
            )
        if self.n_shared_experts != 1:
            raise ValueError("DeepSeek-V4-Flash has exactly one shared expert")

    @staticmethod
    def indexed(ratio: int) -> bool:
        """Whether a layer of compression ``ratio`` pools overlapping windows and attends only
        to the entries its lightning indexer selects."""
        return ratio == 4

    @classmethod
    def from_hf(cls, config: dict) -> DeepseekV4FlashConfig:
        """Map a Hugging Face ``deepseek_v4`` config dict onto the official field names.

        Raises ``ValueError`` on another model type and on any setting this model does not
        compute: attention biases, more than one KV head, a gate other than SiLU, a rope
        scaling other than YaRN, tied embeddings, a dtype other than bfloat16, routed experts
        other than FP4, and weights other than FP8 e4m3 under UE8M0 128x128 block scales.
        """
        if config.get("model_type") != "deepseek_v4":
            raise ValueError(f"model_type {config.get('model_type')!r} is not deepseek_v4")
        rope = config["rope_scaling"]
        quant = config.get("quantization_config", {})
        refused = {
            "attention_bias": config.get("attention_bias", False) is not False,
            "num_key_value_heads": config.get("num_key_value_heads", 1) != 1,
            "hidden_act": config.get("hidden_act", "silu") != "silu",
            "rope_scaling": rope.get("type", "yarn") != "yarn",
            "tie_word_embeddings": config.get("tie_word_embeddings", False) is not False,
            "torch_dtype": config.get("torch_dtype", "bfloat16") != "bfloat16",
            "expert_dtype": config.get("expert_dtype", "fp4") != "fp4",
            "quantization_config": (
                quant.get("quant_method"),
                quant.get("fmt"),
                quant.get("scale_fmt"),
                quant.get("weight_block_size"),
            )
            != ("fp8", "e4m3", "ue8m0", [128, 128]),
        }
        if any(refused.values()):
            raise ValueError(f"unsupported config: {[k for k, v in refused.items() if v]}")
        return cls(
            vocab_size=config["vocab_size"],
            dim=config["hidden_size"],
            moe_inter_dim=config["moe_intermediate_size"],
            n_layers=config["num_hidden_layers"],
            n_hash_layers=config["num_hash_layers"],
            n_mtp_layers=config["num_nextn_predict_layers"],
            n_heads=config["num_attention_heads"],
            n_routed_experts=config["n_routed_experts"],
            n_shared_experts=config["n_shared_experts"],
            n_activated_experts=config["num_experts_per_tok"],
            score_func=config["scoring_func"],
            route_scale=config["routed_scaling_factor"],
            swiglu_limit=config["swiglu_limit"],
            q_lora_rank=config["q_lora_rank"],
            head_dim=config["head_dim"],
            rope_head_dim=config["qk_rope_head_dim"],
            norm_eps=config["rms_norm_eps"],
            o_groups=config["o_groups"],
            o_lora_rank=config["o_lora_rank"],
            window_size=config["sliding_window"],
            compress_ratios=tuple(config["compress_ratios"]),
            compress_rope_theta=config["compress_rope_theta"],
            original_seq_len=rope["original_max_position_embeddings"],
            rope_theta=config["rope_theta"],
            rope_factor=rope["factor"],
            beta_fast=rope["beta_fast"],
            beta_slow=rope["beta_slow"],
            index_n_heads=config["index_n_heads"],
            index_head_dim=config["index_head_dim"],
            index_topk=config["index_topk"],
            hc_mult=config["hc_mult"],
            hc_sinkhorn_iters=config["hc_sinkhorn_iters"],
            hc_eps=config["hc_eps"],
            eos_token_id=config["eos_token_id"],
        )
