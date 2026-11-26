"""The TileOPs ops one DeepSeek-V4-Flash model calls, built once and shared by its layers."""

from __future__ import annotations

import math

import torch

from models.deepseek_v4_flash.config import DeepseekV4FlashConfig
from tileops.attention import DSADenseFwdOp, GatedPoolCompressFwdOp, TopKSelectFwdOp
from tileops.elementwise import MulFwdOp, ReLUFwdOp, SiLUAndMulFwdOp
from tileops.gemm import BMMFwdOp, GEMMFP8FwdOp, GEMMFwdOp
from tileops.indexing import EmbeddingFwdOp
from tileops.moe import (
    ContiguousLayoutSpec,
    FusedTopKFwdOp,
    MoEGroupedGEMMMXFP4FwdOp,
    MoEPostPermuteFwdOp,
    MoEPrePermuteFwdOp,
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
    """One instance of every op the model calls; ops take their shapes from each call.

    The FP8 quantizations are the official model's ``act_quant``: power-of-two scales over
    groups of 128 activations, 64 for the KV rows, with the group maximum floored at 1e-4. The
    mHC post mix is doubled, as the official ``hc_split_sinkhorn`` doubles it.
    """

    def __init__(self, cfg: DeepseekV4FlashConfig) -> None:
        eps = cfg.norm_eps
        floor = 1e-4
        layout = ContiguousLayoutSpec.tight_physical_psum()
        self.embedding = EmbeddingFwdOp()
        self.act_quant = FP8QuantPerTokenGroupFwdOp(128, eps=floor, use_ue8m0=True)
        self.act_quant_e8m0 = FP8QuantPerTokenGroupFwdOp(
            128, eps=floor, use_ue8m0=True, scale_dtype=torch.float8_e8m0fnu
        )
        self.kv_fake_quant = FP8QuantPerTokenGroupFwdOp(64, eps=floor, use_ue8m0=True, inplace=True)
        self.fp4_fake_quant = FP4QuantPerGroupFwdOp(inplace=True)
        self.gemm_fp8 = GEMMFP8FwdOp()
        self.gemm = GEMMFwdOp()
        self.gemm_f32 = GEMMFwdOp(out_dtype=torch.float32)
        self.bmm = BMMFwdOp()
        self.rope = RoPECosSinFwdOp(interleaved=True, inplace=True)
        self.hadamard = HadamardTransformFwdOp()
        self.relu = ReLUFwdOp()
        self.mul = MulFwdOp()
        self.argmax = ArgmaxFwdOp(dim=-1)
        self.norms = {
            d: RMSNormFwdOp((d,), eps)
            for d in {cfg.dim, cfg.q_lora_rank, cfg.head_dim, cfg.index_head_dim}
        }
        self.attention = DSADenseFwdOp(dim_tail=0, stride_kv=1, q_start_index_s=0, is_causal=False)
        self.pools = {
            r: GatedPoolCompressFwdOp(r, overlap=cfg.indexed(r))
            for r in set(cfg.compress_ratios)
            if r
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
            renormalize=cfg.score_func != "softmax",
            routed_scaling_factor=cfg.route_scale,
        )
        self.swiglu = SiLUAndMulFwdOp(swiglu_limit=cfg.swiglu_limit)
        self.permute = MoEPrePermuteFwdOp(layout, cfg.n_routed_experts)
        self.expert_gemm = MoEGroupedGEMMMXFP4FwdOp(layout)
        self.unpermute = MoEPostPermuteFwdOp(layout)

    def norm(self, x: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        """RMS-normalize the last axis of ``x``, scaled by ``weight`` when given."""
        return self.norms[x.shape[-1]](x, weight)


class RoPETable:
    """Per-position cosines and sines of one rotary setting, on one device.

    ``yarn`` applies DeepSeek's YaRN frequency interpolation, whose ramp over a zero-width
    range of dimensions spans 0.001, as the official ``linear_ramp_factor``'s does;
    ``neg_sin`` rotates back. ``entries[ratio]``, for each of ``ratios``, holds the cosines
    and sines of positions ``0, ratio, 2 * ratio, ...``: row ``i`` rotates entry ``i`` of a
    compressor of that ratio, the entry of positions ``i * ratio`` to ``(i + 1) * ratio - 1``,
    which an overlapping compressor pools together with the ``ratio`` positions before them.
    """

    def __init__(
        self,
        cfg: DeepseekV4FlashConfig,
        base: float,
        yarn: bool,
        max_seq_len: int,
        device: torch.device,
        ratios: tuple[int, ...] = (),
    ) -> None:
        dim = cfg.rope_head_dim
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if yarn:
            # The rotary dimensions whose wavelengths fit beta_fast and beta_slow times in the
            # original length bound the interpolated range.
            low, high = (
                dim * math.log(cfg.original_seq_len / (b * 2 * math.pi)) / (2 * math.log(base))
                for b in (cfg.beta_fast, cfg.beta_slow)
            )
            low, high = max(math.floor(low), 0), min(math.ceil(high), dim - 1)
            if low == high:
                high += 0.001
            ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
            freqs = freqs / cfg.rope_factor * ramp + freqs * (1 - ramp)
        angles = torch.outer(torch.arange(max_seq_len), freqs)
        self.cos = torch.cos(angles).to(device)
        self.sin = torch.sin(angles).to(device)
        self.neg_sin = -self.sin
        self.entries = {r: (self.cos[::r].contiguous(), self.sin[::r].contiguous()) for r in ratios}
