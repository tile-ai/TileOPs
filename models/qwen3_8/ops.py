"""The TileOPs ops one Qwen3.8-27B model calls, built once and shared by its layers."""

from __future__ import annotations

import torch

from models.qwen3_8.config import Qwen3_8Config
from tileops.attention import GQAPagedFwdOp, PagedKVCacheWriteFwdOp
from tileops.elementwise import MulFwdOp, SigmoidFwdOp, SiLUAndMulFwdOp
from tileops.gemm import GEMMFwdOp
from tileops.indexing import EmbeddingFwdOp
from tileops.linear_attention import GDNFwdOp
from tileops.mamba import CausalConv1dFwdOp, CausalConv1dUpdateFwdOp
from tileops.norm import FusedAddRMSNormFwdOp, FusedQKNormRoPEFwdOp, RMSNormFwdOp, RMSNormGatedFwdOp
from tileops.reduction import ArgmaxFwdOp

__all__ = ["ModelOps", "rope_cos_sin"]


class ModelOps:
    """One instance of every op the model calls; ops take their shapes from each call.

    Every RMSNorm of the decoder, the q/k norms included, scales by ``1 + w``; the Gated
    DeltaNet output norm scales by ``w``.
    """

    def __init__(self, cfg: Qwen3_8Config) -> None:
        eps = cfg.rms_norm_eps
        self.embedding = EmbeddingFwdOp()
        self.norm = RMSNormFwdOp((cfg.hidden_size,), eps, zero_centered=True)
        self.add_norm = FusedAddRMSNormFwdOp(eps, zero_centered=True)
        self.gemm = GEMMFwdOp()
        self.gemm_f32 = GEMMFwdOp(out_dtype=torch.float32)
        self.conv = CausalConv1dFwdOp("silu")
        self.conv_update = CausalConv1dUpdateFwdOp("silu")
        self.gdn = GDNFwdOp(
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            use_gate_in_kernel=True,
        )
        self.gated_norm = RMSNormGatedFwdOp(eps, norm_before_gate=True, activation="silu")
        self.qk_norm_rope = FusedQKNormRoPEFwdOp(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            eps,
            "neox",
            zero_centered=True,
            mrope_section=cfg.mrope_section,
            mrope_interleaved=True,
        )
        self.kv_write = PagedKVCacheWriteFwdOp()
        self.attention = GQAPagedFwdOp(is_causal=True, sm_scale=cfg.head_dim**-0.5)
        self.sigmoid = SigmoidFwdOp()
        self.mul = MulFwdOp()
        self.silu_and_mul = SiLUAndMulFwdOp()
        self.argmax = ArgmaxFwdOp(dim=-1)


def rope_cos_sin(cfg: Qwen3_8Config, max_seq_len: int, device: torch.device) -> torch.Tensor:
    """``[max_seq_len, rotary_dim]`` float32: per position, the cosines then the sines of its
    ``rotary_dim / 2`` rotary angles."""
    dim = cfg.rotary_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
    return torch.cat([angles.cos(), angles.sin()], dim=-1).to(device)
