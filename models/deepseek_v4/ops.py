"""The TileOPs ops one DeepSeek-V4 model calls, built once and shared by its layers."""

from __future__ import annotations

import math

import torch

from tileops.attention import DSADenseFwdOp, GatedPoolCompressFwdOp, TopKSelectFwdOp
from tileops.elementwise import MulFwdOp, ReLUFwdOp, SiLUAndMulFwdOp
from tileops.gemm import BMMFwdOp, GEMMFP8FwdOp, GEMMFwdOp
from tileops.indexing import EmbeddingFwdOp
from models.deepseek_v4.config import DeepseekV4Config
from tileops.moe import (
    ContiguousLayoutSpec,
    FusedTopKFwdOp,
    MoEGroupedGEMMMXFP4FwdOp,
    MoEPostPermuteFwdOp,
    MoEPrePermuteFwdOp,
    RoutingEpilogueSpec,
)
from tileops.norm import RMSNormFwdOp
from tileops.quantization import FP4QuantPerGroupFwdOp, FP8QuantPerTokenGroupFwdOp
from tileops.reduction import ArgmaxFwdOp
from tileops.rope import RoPECosSinFwdOp
from tileops.sequence_modeling import (
    HyperConnectionHeadFwdOp,
    HyperConnectionPostFwdOp,
    HyperConnectionPreFwdOp,
)
from tileops.transform import HadamardTransformFwdOp

__all__ = ["ModelOps", "RoPETable"]


class ModelOps:
    """One instance of every op the model calls; ops take their shapes from each call."""

    _ACT_BLOCK = 128
    _ACT_AMAX_FLOOR = 1e-4
    _KV_QUANT_BLOCK = 64
    _FP4_BLOCK = 32

    def __init__(self, cfg: DeepseekV4Config) -> None:
        eps = cfg.norm_eps
        floor = self._ACT_AMAX_FLOOR
        self.layout = ContiguousLayoutSpec.tight_physical_psum()
        self.embedding = EmbeddingFwdOp()
        self.act_quant = FP8QuantPerTokenGroupFwdOp(self._ACT_BLOCK, eps=floor, use_ue8m0=True)
        self.act_quant_e8m0 = FP8QuantPerTokenGroupFwdOp(
            self._ACT_BLOCK, eps=floor, use_ue8m0=True, scale_dtype=torch.float8_e8m0fnu
        )
        self.kv_fake_quant = FP8QuantPerTokenGroupFwdOp(
            self._KV_QUANT_BLOCK, eps=floor, use_ue8m0=True, inplace=True
        )
        self.fp4_fake_quant = FP4QuantPerGroupFwdOp(self._FP4_BLOCK, inplace=True)
        self.gemm_fp8 = GEMMFP8FwdOp()
        self.gemm = GEMMFwdOp()
        self.gemm_f32 = GEMMFwdOp(out_dtype=torch.float32)
        self.bmm = BMMFwdOp()
        self.rope = RoPECosSinFwdOp(interleaved=True, inplace=True)
        self.hadamard = HadamardTransformFwdOp()
        self.relu = ReLUFwdOp(inplace=True)
        self.mul = MulFwdOp()
        self.argmax = ArgmaxFwdOp(dim=-1)
        self.norms = {
            d: RMSNormFwdOp((d,), eps)
            for d in {cfg.dim, cfg.q_lora_rank, cfg.head_dim, cfg.index_head_dim}
        }
        self.attention = DSADenseFwdOp(
            dim_tail=0,
            stride_kv=1,
            q_start_index_s=0,
            sm_scale=cfg.head_dim**-0.5,
            is_causal=False,
        )
        self.pools = {
            r: GatedPoolCompressFwdOp(r, overlap=r == 4) for r in set(cfg.compress_ratios) if r
        }
        self.index_topk = TopKSelectFwdOp(cfg.index_topk, fill_value=-1)
        self.hc_pre = HyperConnectionPreFwdOp(
            rms_eps=eps,
            hc_pre_eps=cfg.hc_eps,
            hc_sinkhorn_eps=cfg.hc_eps,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=cfg.hc_sinkhorn_iters,
        )
        self.hc_post = HyperConnectionPostFwdOp()
        self.hc_head = HyperConnectionHeadFwdOp(rms_norm_eps=eps, hc_eps=cfg.hc_eps)
        self.route = FusedTopKFwdOp(
            cfg.n_activated_experts,
            scoring_func=cfg.score_func,
            renormalize=True,
            routed_scaling_factor=cfg.route_scale,
        )
        self.swiglu = SiLUAndMulFwdOp(swiglu_limit=cfg.swiglu_limit)
        self.permute = MoEPrePermuteFwdOp(self.layout, cfg.n_routed_experts)
        self.expert_gemm = MoEGroupedGEMMMXFP4FwdOp(self.layout)
        self.unpermute = MoEPostPermuteFwdOp(self.layout, RoutingEpilogueSpec(1.0))

    def norm(self, x: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        """RMS-normalize the last axis of ``x``, scaled by ``weight`` when given."""
        return self.norms[x.shape[-1]](x, weight)

    def fp8_linear(
        self, x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """``x @ weight^T`` for 2-D ``x``, quantized per 1x128 to FP8 with power-of-two scales."""
        q, s, _ = self.act_quant(x)
        return self.gemm_fp8(q, weight, s, scale)


class RoPETable:
    """Per-position cosines and sines of one rotary setting, on one device.

    ``yarn`` applies DeepSeek's YaRN frequency interpolation; ``neg_sin`` rotates back.
    """

    def __init__(
        self,
        cfg: DeepseekV4Config,
        base: float,
        yarn: bool,
        max_seq_len: int,
        device: torch.device,
    ) -> None:
        dim = cfg.rope_head_dim
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if yarn and cfg.original_seq_len > 0:
            low, high = _yarn_correction_range(
                cfg.beta_fast, cfg.beta_slow, dim, base, cfg.original_seq_len
            )
            if low == high:
                high += 0.001
            ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
            freqs = freqs / cfg.rope_factor * ramp + freqs * (1 - ramp)
        angles = torch.outer(torch.arange(max_seq_len), freqs)
        self.cos = torch.cos(angles).to(device)
        self.sin = torch.sin(angles).to(device)
        self.neg_sin = -self.sin


def _yarn_correction_range(
    beta_fast: int, beta_slow: int, dim: int, base: float, original_seq_len: int
) -> tuple[int, int]:
    """The rotary dimensions YaRN interpolates between, clamped to ``[0, dim - 1]``."""

    def corr(rotations: float) -> float:
        return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

    return max(math.floor(corr(beta_fast)), 0), min(math.ceil(corr(beta_slow)), dim - 1)
