"""Structural oracle for roofline ``bytes`` (docs/design/roofline.md §4.6).

Each case recomputes the minimum traffic from the tensors the workload binds
— every distinct input storage read once, every output written once — and
requires equality with ``eval_roofline()``. The oracle enumerates tensors
from the signature, so a formula that drops a term, double-counts a tensor,
or prices a broadcast operand at the output's shape breaks the equality.
"""

from math import prod

import pytest
import torch

pytestmark = pytest.mark.smoke


def _nbytes(*tensors: tuple[tuple[int, ...], torch.dtype]) -> int:
    return sum(prod(shape) * dtype.itemsize for shape, dtype in tensors)


class TestBytesOracle:
    # __new__ + attribute binding keeps the oracle CUDA-free; each case binds
    # exactly the state the op's eval_roofline reads after a forward().

    def test_conv2d_counts_input_weight_output_and_bias(self):
        from tileops.ops.convolution import Conv2dFwdOp

        n, c_in, h, w = 8, 64, 56, 56
        c_out, c_in_g, kh, kw = 128, 64, 3, 3
        out_h = out_w = 54  # stride 1, no padding
        for has_bias in (True, False):
            op = Conv2dFwdOp.__new__(Conv2dFwdOp)
            op._last_roofline_spec = (
                n, c_in, h, w, c_out, c_in_g, kh, kw, out_h, out_w,
                torch.float16, has_bias,
            )  # fmt: skip
            oracle = _nbytes(
                ((n, c_in, h, w), torch.float16),
                ((c_out, c_in_g, kh, kw), torch.float16),
                ((n, c_out, out_h, out_w), torch.float16),
                *((((c_out,), torch.float16),) if has_bias else ()),
            )
            assert op.eval_roofline()[1] == oracle, f"has_bias={has_bias}"

    def test_gemm_fp8_counts_fp8_inputs_fp32_scales_and_out_dtype(self):
        from tileops.ops.gemm.gemm import GemmFp8FwdOp

        m, n, k = 4096, 4096, 8192
        for has_bias in (True, False):
            op = GemmFp8FwdOp.__new__(GemmFp8FwdOp)
            op.m, op.n, op.k = m, n, k
            op.dtype = torch.float8_e4m3fn
            op.out_dtype = torch.bfloat16
            op.scale_a_shape = (m, 1)
            op.scale_b_shape = (1, n)
            op.has_bias = has_bias
            oracle = _nbytes(
                ((m, k), torch.float8_e4m3fn),
                ((k, n), torch.float8_e4m3fn),
                ((m, n), torch.bfloat16),
                ((m, 1), torch.float32),
                ((1, n), torch.float32),
                *((((n,), torch.bfloat16),) if has_bias else ()),
            )
            assert op.eval_roofline()[1] == oracle, f"has_bias={has_bias}"

    def test_add_broadcast_counts_the_operand_at_its_own_shape(self):
        from tileops.ops.elementwise.arithmetic import AddFwdOp

        a_shape, b_shape, out_shape = (4, 4096, 4096), (1, 1, 4096), (4, 4096, 4096)
        op = AddFwdOp.__new__(AddFwdOp)
        op.input_shape = a_shape
        op.other_shape = b_shape  # out_shape derives via _infer_output_shapes
        op.dtype = torch.bfloat16
        oracle = _nbytes(
            (a_shape, torch.bfloat16),
            (b_shape, torch.bfloat16),
            (out_shape, torch.bfloat16),
        )
        assert op.eval_roofline()[1] == oracle

    def test_var_mean_counts_both_outputs(self):
        from tileops.ops.reduction.reduce import VarMeanFwdOp

        m, n = 8192, 4096
        op = VarMeanFwdOp.__new__(VarMeanFwdOp)
        op._last_roofline_mn = (m, n)
        op.dtype = torch.float32
        oracle = _nbytes(
            ((m, n), torch.float32),
            ((m,), torch.float32),  # var
            ((m,), torch.float32),  # mean
        )
        assert op.eval_roofline()[1] == oracle

    def test_argmax_counts_int64_indices(self):
        from tileops.ops.reduction.argreduce import ArgmaxFwdOp

        m, n = 8192, 4096
        op = ArgmaxFwdOp.__new__(ArgmaxFwdOp)
        op._last_roofline_mn = (m, n)
        op.dtype = torch.float16
        oracle = _nbytes(((m, n), torch.float16), ((m,), torch.int64))
        assert op.eval_roofline()[1] == oracle

    def test_rms_norm_counts_x_weight_and_output(self):
        from tileops.ops.norm.rms_norm import RMSNormFwdOp

        m, n = 16384, 8192
        op = RMSNormFwdOp.__new__(RMSNormFwdOp)
        op._last_m = m
        op.N = n
        op.dtype = torch.float16
        oracle = _nbytes(
            ((m, n), torch.float16),
            ((n,), torch.float16),
            ((m, n), torch.float16),
        )
        assert op.eval_roofline()[1] == oracle

    def test_w4a16_counts_packed_weights_and_group_metadata(self):
        from tileops.ops.gemm.gemm import GemmW4A16FwdOp

        m, n, k, group_size = 4096, 8192, 8192, 128
        op = GemmW4A16FwdOp.__new__(GemmW4A16FwdOp)
        op.m, op.n, op.k = m, n, k
        op.dtype = torch.float16
        op.group_size = group_size
        groups = k // group_size
        oracle = (
            _nbytes(((m, k), torch.float16), ((m, n), torch.float16))
            + n * k // 2  # int4 weights: two per byte
            + n * groups * 4  # per-group scales, float32
            + n * groups * 1  # per-group zero points, int8
        )
        assert op.eval_roofline()[1] == oracle

    def test_fused_moe_counts_the_bias_the_call_passed(self):
        from tileops.ops.moe.fused_moe import FusedMoeFwdOp

        tokens, experts, top_k, hidden, ffn = 4096, 64, 8, 4096, 1408
        for has_bias in (True, False):
            op = FusedMoeFwdOp.__new__(FusedMoeFwdOp)
            op.num_tokens, op.num_experts, op.top_k = tokens, experts, top_k
            op.hidden_size, op.ffn_size = hidden, ffn
            op.dtype = torch.bfloat16
            op.correction_bias_shape = (experts,) if has_bias else None
            oracle = _nbytes(
                ((tokens, hidden), torch.bfloat16),  # hidden states in
                ((experts, 2 * ffn, hidden), torch.bfloat16),  # w_gate_up
                ((experts, hidden, ffn), torch.bfloat16),  # w_down
                ((tokens, experts), torch.float32),  # gating logits
                ((tokens, hidden), torch.bfloat16),  # output
                *((((experts,), torch.float32),) if has_bias else ()),
            )
            assert op.eval_roofline()[1] == oracle, f"has_bias={has_bias}"

    def test_gqa_varlen_prefill_counts_packed_qkv_and_cu_seqlens(self):
        from tileops.ops.attention.gqa import GroupedQueryAttentionPrefillFwdOp

        heads, heads_kv, dim = 32, 8, 128
        q_lens, kv_lens = [512, 1024], [2048, 4096]
        total_q, total_kv = sum(q_lens), sum(kv_lens)
        cu = lambda lens: torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()])  # noqa: E731
        op = GroupedQueryAttentionPrefillFwdOp.__new__(GroupedQueryAttentionPrefillFwdOp)
        op._roofline_kwargs = {
            "q_shape": (total_q, heads, dim),
            "k_shape": (total_kv, heads_kv, dim),
            "batch": 2,
            "max_seqlen_q": max(q_lens),
            "max_seqlen_kv": max(kv_lens),
            "cu_seqlens_q": cu(q_lens),
            "cu_seqlens_kv": cu(kv_lens),
            "is_causal": True,
            "dtype": torch.float16,
        }
        oracle = _nbytes(
            ((total_q, heads, dim), torch.float16),  # q
            ((total_kv, heads_kv, dim), torch.float16),  # k
            ((total_kv, heads_kv, dim), torch.float16),  # v
            ((total_q, heads, dim), torch.float16),  # o
            ((2, 3), torch.int32),  # cu_seqlens_q + cu_seqlens_kv, [batch+1] each
        )
        assert op.eval_roofline()[1] == oracle


# Classification registry: every implemented op appears in exactly one of
# AUDITED (has a bytes-oracle case above), EXEMPT (traffic depends on tensor
# content; audited by the NCU script instead, roofline.md §4.5), or PENDING.
# Adding an op to the manifest forces a choice here.
AUDITED = frozenset(
    {
        "AddFwdOp",
        "ArgmaxFwdOp",
        "Conv2dFwdOp",
        "FusedMoeFwdOp",
        "GemmFp8FwdOp",
        "GemmW4A16FwdOp",
        "GroupedQueryAttentionPrefillFwdOp",
        "RMSNormFwdOp",
        "VarMeanFwdOp",
    }
)

# op name -> why the shape-level oracle cannot count its traffic
EXEMPT: dict[str, str] = {}

# FIXME(staged-rollout): most implemented ops lack a bytes-oracle case.
#
# Broken invariant: every implemented op is AUDITED or EXEMPT.
# Why: the oracle landed with the SOL metric; cases are added family by
#   family, highest formula complexity first.
# Cleanup: PENDING is empty; delete it and this marker.
PENDING = frozenset(
    {
        "AbsFwdOp",
        "AdaLayerNormFwdOp",
        "AdaLayerNormZeroFwdOp",
        "AdaptiveAvgPool2dFwdOp",
        "AdaptiveMaxPool2dFwdOp",
        "AdaptiveMaxPool2dIndicesFwdOp",
        "AlibiFwdOp",
        "AllFwdOp",
        "AmaxFwdOp",
        "AminFwdOp",
        "AnyFwdOp",
        "ArgminFwdOp",
        "AvgPool1dFwdOp",
        "AvgPool2dFwdOp",
        "AvgPool3dFwdOp",
        "BatchNormBwdOp",
        "BatchNormFwdOp",
        "BitwiseAndFwdOp",
        "BitwiseNotFwdOp",
        "BitwiseOrFwdOp",
        "BitwiseXorFwdOp",
        "BmmFp8KNFwdOp",
        "BmmFp8NKFwdOp",
        "BmmFwdOp",
        "CBProducerFwdOp",
        "CeilFwdOp",
        "ClampFwdOp",
        "ClampScalarFwdOp",
        "Conv1dFwdOp",
        "Conv3dFwdOp",
        "CosFwdOp",
        "CountNonzeroFwdOp",
        "CumprodFwdOp",
        "CumsumFwdOp",
        "DaCumsumFwdOp",
        "DeepSeekSparseAttentionDecodeWithKVCacheFwdOp",
        "DeltaNetBwdOp",
        "DeltaNetDecodeFwdOp",
        "DeltaNetFwdOp",
        "DivFwdOp",
        "DropoutFwdOp",
        "EluFwdOp",
        "EngramDecodeFwdOp",
        "EngramGateConvBwdOp",
        "EngramGateConvFwdOp",
        "EqFwdOp",
        "ErfFwdOp",
        "ExpFwdOp",
        "Expm1FwdOp",
        "FFTC2CFwdOp",
        "FP8LightningIndexerFwdOp",
        "FP8QuantFwdOp",
        "FloorDivideFwdOp",
        "FloorFwdOp",
        "FusedAddLayerNormFwdOp",
        "FusedAddRMSNormFwdOp",
        "FusedMoEExpertsNopadPersistent3WGFwdOp",
        "GLABwdOp",
        "GLADecodeFwdOp",
        "GLAFwdOp",
        "GatedDeltaNetBHTDFwdOp",
        "GatedDeltaNetBTHDFwdOp",
        "GatedDeltaNetBwdOp",
        "GatedDeltaNetDecodeFwdOp",
        "GatedDeltaNetPrefillBHTDFwdOp",
        "GatedDeltaNetPrefillBTHDFwdOp",
        "GeFwdOp",
        "GeluAndMulFwdOp",
        "GeluFwdOp",
        "GeluTanhAndMulFwdOp",
        "GemmFwdOp",
        "GroupNormFwdOp",
        "GroupedGemmFwdOp",
        "GroupedQueryAttentionBwdOp",
        "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp",
        "GroupedQueryAttentionDecodeWithKVCacheFwdOp",
        "GroupedQueryAttentionFwdOp",
        "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp",
        "GroupedQueryAttentionSlidingWindowFwdOp",
        "GroupedQueryAttentionSlidingWindowVarlenFwdOp",
        "GtFwdOp",
        "HardsigmoidFwdOp",
        "HardswishFwdOp",
        "HardtanhFwdOp",
        "InfNormFwdOp",
        "InstanceNormFwdOp",
        "IsfiniteFwdOp",
        "IsinfFwdOp",
        "IsnanFwdOp",
        "L1NormFwdOp",
        "L2NormFwdOp",
        "LayerNormFwdOp",
        "LeFwdOp",
        "LeakyReluFwdOp",
        "LerpFwdOp",
        "LerpTensorFwdOp",
        "Log1pFwdOp",
        "LogFwdOp",
        "LogSoftmaxFwdOp",
        "LogSumExpFwdOp",
        "LogicalAndFwdOp",
        "LogicalNotFwdOp",
        "LogicalOrFwdOp",
        "LtFwdOp",
        "MHCPostFwdOp",
        "MHCPreFwdOp",
        "Mamba2FwdOp",
        "MaskedFillFwdOp",
        "MaskedFillScalarFwdOp",
        "MaxPool1dFwdOp",
        "MaxPool1dIndicesFwdOp",
        "MaxPool2dFwdOp",
        "MaxPool2dIndicesFwdOp",
        "MaxPool3dFwdOp",
        "MaxPool3dIndicesFwdOp",
        "MaximumFwdOp",
        "MeanFwdOp",
        "MinimumFwdOp",
        "MishFwdOp",
        "MoeGateUpFwdOp",
        "MoeGroupedGemmNopadFwdOp",
        "MoePermuteAlignFwdOp",
        "MoePermuteNopadFwdOp",
        "MoeUnpermuteFwdOp",
        "MulFwdOp",
        "MultiHeadAttentionBwdOp",
        "MultiHeadAttentionDecodePagedWithKVCacheFwdOp",
        "MultiHeadAttentionDecodeWithKVCacheFwdOp",
        "MultiHeadAttentionFwdOp",
        "MultiHeadLatentAttentionDecodeWithKVCacheFwdOp",
        "NanToNumFwdOp",
        "NeFwdOp",
        "NegFwdOp",
        "PowFwdOp",
        "PreluFwdOp",
        "ProdFwdOp",
        "ReciprocalFwdOp",
        "ReluFwdOp",
        "RemainderFwdOp",
        "RopeLlama31FwdOp",
        "RopeLongRopeFwdOp",
        "RopeNeoxFwdOp",
        "RopeNeoxPositionIdsFwdOp",
        "RopeNonNeoxFwdOp",
        "RopeYarnFwdOp",
        "RoundFwdOp",
        "RsqrtFwdOp",
        "SSDChunkScanFwdOp",
        "SSDChunkStateFwdOp",
        "SSDDecodeFwdOp",
        "SSDStatePassingFwdOp",
        "SeluFwdOp",
        "SigmoidFwdOp",
        "SignFwdOp",
        "SiluAndMulFwdOp",
        "SiluFwdOp",
        "SinFwdOp",
        "SinusoidalFwdOp",
        "SoftmaxFwdOp",
        "SoftplusFwdOp",
        "SqrtFwdOp",
        "StdFwdOp",
        "SubFwdOp",
        "SumFwdOp",
        "TanhFwdOp",
        "TopkSelectorFwdOp",
        "TruncFwdOp",
        "VarFwdOp",
        "WhereFwdOp",
    }
)


def test_every_implemented_op_is_classified():
    """A new op cannot ship a bytes formula nothing accounts for."""
    from tileops.manifest import load_manifest

    implemented = {name for name, e in load_manifest().items() if e.get("status") == "implemented"}
    classified = AUDITED | set(EXEMPT) | PENDING
    assert implemented - classified == set(), (
        f"unclassified implemented ops: {sorted(implemented - classified)}; "
        "add an oracle case (AUDITED), an EXEMPT reason, or a PENDING entry"
    )
    assert classified - implemented == set(), (
        f"stale registry entries: {sorted(classified - implemented)}"
    )
    assert not (AUDITED & PENDING) and not (AUDITED & set(EXEMPT))
