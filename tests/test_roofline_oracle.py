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
            op.input_shape = (n, c_in, h, w)
            op.weight_shape = (c_out, c_in_g, kh, kw)
            op.bias_shape = (c_out,) if has_bias else None
            op.bias = None
            op.dtype = torch.float16
            op.stride, op.padding, op.dilation, op.groups = 1, 0, 1, 1
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
        op.alpha = 1  # add/sub price the scale multiply from it
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
        op.x_shape = (m, n)
        op.dim = -1
        op.keepdim = False
        op.correction = 1
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
        op.x_shape = (m, n)
        op.dim = -1
        op.keepdim = False
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

    def test_moe_pre_permute_counts_inputs_and_three_outputs(self):
        from tileops.ops.moe import ContiguousLayoutSpec, MoePrePermuteFwdOp

        tokens, top_k, experts, hidden = 512, 2, 4, 128
        op = MoePrePermuteFwdOp.__new__(MoePrePermuteFwdOp)
        op.layout = ContiguousLayoutSpec.tight_physical_psum()
        op.num_local_experts = experts
        op.input_shapes = [(tokens, hidden), (tokens, top_k)]
        op.dtype = torch.bfloat16
        oracle = _nbytes(
            ((tokens, hidden), torch.bfloat16),
            ((tokens, top_k), torch.int32),
            ((tokens * top_k, hidden), torch.bfloat16),
            ((experts,), torch.int32),
            ((tokens * top_k,), torch.int32),
        )
        assert op.eval_roofline()[1] == oracle

    def test_moe_post_permute_counts_inputs_and_output(self):
        from tileops.ops.moe import MoePostPermuteFwdOp

        tokens, top_k, hidden = 512, 2, 128
        rows = tokens * top_k
        op = MoePostPermuteFwdOp.__new__(MoePostPermuteFwdOp)
        op.input_shapes = [(rows, hidden), (tokens, top_k), (rows,)]
        op.dtype = torch.bfloat16
        oracle = _nbytes(
            ((rows, hidden), torch.bfloat16),
            ((tokens, top_k), torch.float32),
            ((rows,), torch.int32),
            ((tokens, hidden), torch.bfloat16),
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

    def test_fused_moe_counts_active_experts_and_bias(self):
        from tileops.ops.moe.fused_moe import FusedMoeFwdOp

        tokens, experts, top_k, hidden, ffn = 2, 8, 2, 64, 32
        for has_bias in (True, False):
            op = FusedMoeFwdOp.__new__(FusedMoeFwdOp)
            op.num_tokens, op.num_experts, op.top_k = tokens, experts, top_k
            op.hidden_size, op.ffn_size = hidden, ffn
            op.dtype = torch.bfloat16
            op.correction_bias_shape = (experts,) if has_bias else None
            # Only experts 0, 3 and 7 receive rows.
            op._roofline_topk_ids = torch.tensor([[0, 3], [3, 7]], dtype=torch.int32)
            oracle = _nbytes(
                ((tokens, hidden), torch.bfloat16),  # hidden states in
                ((3, 2 * ffn, hidden), torch.bfloat16),  # active w_gate_up
                ((3, hidden, ffn), torch.bfloat16),  # active w_down
                ((tokens, experts), torch.float32),  # gating logits
                ((tokens, hidden), torch.bfloat16),  # output
                *((((experts,), torch.float32),) if has_bias else ()),
            )
            assert op.eval_roofline()[1] == oracle, f"has_bias={has_bias}"

        del op._roofline_topk_ids
        with pytest.raises(RuntimeError, match="requires a prior forward"):
            op.eval_roofline()

    def test_routed_expert_mlp_counts_active_experts_and_the_routing(self):
        from tileops.moe import IndexedExpertMLPFwdOp

        tokens, experts, top_k, hidden, ffn = 2, 8, 2, 64, 32
        op = IndexedExpertMLPFwdOp.__new__(IndexedExpertMLPFwdOp)
        op.num_tokens, op.num_experts, op.top_k = tokens, experts, top_k
        op.hidden_size, op.ffn_size = hidden, ffn
        op.dtype = torch.bfloat16
        # Only experts 0, 3 and 7 receive rows.
        op._roofline_topk_ids = torch.tensor([[0, 3], [3, 7]], dtype=torch.int32)
        oracle = _nbytes(
            ((tokens, hidden), torch.bfloat16),  # hidden states in
            ((3, 2 * ffn, hidden), torch.bfloat16),  # active w_gate_up
            ((3, hidden, ffn), torch.bfloat16),  # active w_down
            ((tokens, top_k), torch.int32),  # topk_ids
            ((tokens, top_k), torch.float32),  # topk_weights
            ((tokens, hidden), torch.bfloat16),  # output
        )
        assert op.eval_roofline()[1] == oracle

        del op._roofline_topk_ids
        with pytest.raises(RuntimeError, match="requires a prior forward"):
            op.eval_roofline()

    def test_fused_moe_experts_counts_active_experts_and_the_routing(self):
        from tileops.moe import FusedMoEExpertsFwdOp

        tokens, experts, top_k, hidden, ffn = 2, 8, 2, 64, 32
        op = FusedMoEExpertsFwdOp.__new__(FusedMoEExpertsFwdOp)
        op.num_tokens, op.num_experts, op.top_k = tokens, experts, top_k
        op.hidden_size, op.ffn_size = hidden, ffn
        op.dtype = torch.bfloat16
        # Only experts 0, 3 and 7 receive rows.
        op._roofline_topk_ids = torch.tensor([[0, 3], [3, 7]], dtype=torch.int32)
        oracle = _nbytes(
            ((tokens, hidden), torch.bfloat16),  # hidden states in
            ((3, 2 * ffn, hidden), torch.bfloat16),  # active w_gate_up
            ((3, hidden, ffn), torch.bfloat16),  # active w_down
            ((tokens, top_k), torch.int32),  # topk_ids
            ((tokens, top_k), torch.float32),  # topk_weights
            ((tokens, hidden), torch.bfloat16),  # output
        )
        assert op.eval_roofline()[1] == oracle

    def test_shared_expert_adds_its_shard_to_the_routed_cost(self):
        from tileops.moe import FusedMoeSharedExpertFwdOp

        tokens, experts, top_k, hidden, ffn = 2, 8, 2, 64, 32
        shard_ffn = 16
        op = FusedMoeSharedExpertFwdOp.__new__(FusedMoeSharedExpertFwdOp)
        op.num_tokens, op.num_experts, op.top_k = tokens, experts, top_k
        op.hidden_size, op.ffn_size = hidden, ffn
        op.dtype = torch.bfloat16
        op.correction_bias_shape = None
        op._roofline_topk_ids = torch.tensor([[0, 3], [3, 7]], dtype=torch.int32)
        routed = _nbytes(
            ((tokens, hidden), torch.bfloat16),  # hidden states in
            ((3, 2 * ffn, hidden), torch.bfloat16),  # active w_gate_up
            ((3, hidden, ffn), torch.bfloat16),  # active w_down
            ((tokens, experts), torch.float32),  # gating logits
            ((tokens, hidden), torch.bfloat16),  # output
        )
        op._shared_mlp_shard_ffn = None
        assert op.eval_roofline()[1] == routed

        op._shared_mlp_shard_ffn = shard_ffn
        shared = _nbytes(
            ((3 * shard_ffn, hidden), torch.bfloat16),  # this rank's shared weights
            ((tokens, hidden), torch.bfloat16),  # its own read of the hidden states
            ((tokens, hidden), torch.bfloat16),  # shared_output, returned separately
        )
        assert op.eval_roofline()[1] == routed + shared

    def test_gqa_dense_counts_qkv_output_and_the_optional_inputs(self):
        from tileops.ops.attention.gqa import GroupedQueryAttentionDenseFwdOp

        batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 256, 1792, 8, 2, 128
        q_shape = (batch, seq_len_q, heads, dim)
        kv_shape = (batch, seq_len_kv, heads_kv, dim)
        scales = (((batch, heads_kv), torch.float32),) * 3
        tables = (((seq_len_kv, dim // 4), torch.float16),) * 2
        # (q/k/v dtype, output dtype, the optional tensors the call passed)
        calls = {
            "16-bit": (torch.float16, torch.float16, ()),
            "fused RoPE": (torch.float16, torch.float16, tables),
            "FP8": (torch.float8_e4m3fn, torch.float16, scales),
        }
        for label, (dtype, out_dtype, optional) in calls.items():
            op = GroupedQueryAttentionDenseFwdOp.__new__(GroupedQueryAttentionDenseFwdOp)
            op._roofline_kwargs = {
                "q_shape": q_shape,
                "k_shape": kv_shape,
                "is_causal": True,
                "dtype": dtype,
                "out_dtype": out_dtype,
                "optional_shapes": optional,
            }
            oracle = _nbytes(
                (q_shape, dtype),
                (kv_shape, dtype),
                (kv_shape, dtype),
                (q_shape, out_dtype),
                *optional,
            )
            assert op.eval_roofline()[1] == oracle, label

    def test_batch_norm_counts_the_running_stat_write_only_when_training(self):
        from tileops.ops.norm.batch_norm import BatchNormFwdOp

        x_shape, channels = (32, 256, 28, 28), 256
        stats = (((channels,), torch.float32),) * 4  # mean, var, weight, bias
        for training in (False, True):
            op = BatchNormFwdOp.__new__(BatchNormFwdOp)
            op.x_shape = x_shape
            op.dtype = torch.float16
            op.training = training
            oracle = _nbytes(
                (x_shape, torch.float16),
                *stats,
                (x_shape, torch.float16),
                # running_mean and running_var are mutated: written back too.
                *((((channels,), torch.float32),) * 2 if training else ()),
            )
            assert op.eval_roofline()[1] == oracle, f"training={training}"

    def test_mamba2_counts_the_public_tensors_and_not_the_stage_intermediates(self):
        from tileops.ops.mamba.mamba2_fwd import Mamba2FwdOp

        batch, seqlen, n_heads, d_head, d_state, n_groups = 2, 2048, 8, 64, 128, 1
        chunk_size = 256
        x_shape = (batch, seqlen, n_heads, d_head)
        bc_shape = (batch, seqlen, n_groups, d_state)
        state_shape = (batch, n_heads, d_head, d_state)
        for has_optional in (False, True):
            op = Mamba2FwdOp.__new__(Mamba2FwdOp)
            op.batch, op.seqlen = batch, seqlen
            op.num_chunks, op.chunk_size = seqlen // chunk_size, chunk_size
            op.n_heads, op.d_head, op.d_state, op.n_groups = n_heads, d_head, d_state, n_groups
            op.dtype = torch.float16
            op.dt_softplus = True
            op.dt_bias_shape = (n_heads,) if has_optional else None
            op.initial_states_shape = state_shape if has_optional else None
            oracle = _nbytes(
                (x_shape, torch.float16),
                ((batch, seqlen, n_heads), torch.float32),  # dt
                ((n_heads,), torch.float32),  # A
                (bc_shape, torch.float16),  # B
                (bc_shape, torch.float16),  # C
                *((((n_heads,), torch.float32),) if has_optional else ()),
                *(((state_shape, torch.float32),) if has_optional else ()),
                (x_shape, torch.float32),  # y
                (state_shape, torch.float32),  # final_states
            )
            assert op.eval_roofline()[1] == oracle, f"optional={has_optional}"

    def test_mha_backward_counts_o_and_lse(self):
        from tileops.ops.attention.mha import MultiHeadAttentionBwdOp

        batch, seq_len, heads, dim = 2, 2048, 16, 128
        shape = (batch, seq_len, heads, dim)
        op = MultiHeadAttentionBwdOp.__new__(MultiHeadAttentionBwdOp)
        op.batch, op.seq_len, op.heads, op.dim = batch, seq_len, heads, dim
        op.is_causal = True
        op.dtype = torch.float16
        oracle = _nbytes(
            *((shape, torch.float16),) * 5,  # q, k, v, o, do
            ((batch, heads, seq_len), torch.float32),  # lse
            *((shape, torch.float16),) * 3,  # dq, dk, dv
        )
        assert op.eval_roofline()[1] == oracle

    def test_gqa_backward_prices_the_kv_tensors_at_the_kv_head_count(self):
        from tileops.ops.attention.gqa import GroupedQueryAttentionBwdOp

        batch, seq_len, heads, heads_kv, dim = 2, 2048, 16, 4, 128
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        op = GroupedQueryAttentionBwdOp.__new__(GroupedQueryAttentionBwdOp)
        op.batch, op.seq_len, op.heads, op.heads_kv, op.dim = batch, seq_len, heads, heads_kv, dim
        op.is_causal = True
        op.dtype = torch.float16
        oracle = _nbytes(
            *((q_shape, torch.float16),) * 3,  # q, o, do
            *((kv_shape, torch.float16),) * 2,  # k, v
            ((batch, heads, seq_len), torch.float32),  # lse
            (q_shape, torch.float16),  # dq
            *((kv_shape, torch.float16),) * 2,  # dk, dv
        )
        assert op.eval_roofline()[1] == oracle

    def test_fp8_indexer_prices_each_input_at_its_own_dtype_and_the_scale_by_presence(self):
        from tileops.ops.fp8_lightning_indexer import FP8LightningIndexerFwdOp

        batch, seq_len, heads, index_dim = 1, 4096, 32, 128
        seq_len_kv, kv_group = 4096, 1
        scale_shape = (batch, seq_len_kv, kv_group)
        fp8, bf16 = torch.float8_e4m3fn, torch.bfloat16
        # Handed bf16, the op quantizes and produces the scale itself: both are
        # intermediates. Only the pre-quantized call reads a caller's scale, and
        # only it requires index_q and index_k to share a dtype.
        calls = {
            "bf16 inputs": (bf16, bf16, None),
            "pre-quantized": (fp8, fp8, scale_shape),
            "fp8 inputs, quantized k": (fp8, fp8, None),
            "mixed dtypes": (fp8, bf16, None),
        }
        for label, (q_dtype, k_dtype, supplied_scale) in calls.items():
            op = FP8LightningIndexerFwdOp.__new__(FP8LightningIndexerFwdOp)
            op.batch, op.seq_len, op.heads, op.index_dim = batch, seq_len, heads, index_dim
            op.seq_len_kv, op.kv_group = seq_len_kv, kv_group
            op.dtype, op.index_k_dtype = q_dtype, k_dtype
            op.index_k_scale_shape = supplied_scale
            oracle = _nbytes(
                ((batch, seq_len, heads, index_dim), q_dtype),  # index_q
                ((batch, seq_len_kv, kv_group, index_dim), k_dtype),  # index_k
                ((seq_len, heads), torch.float32),  # weights
                ((seq_len,), torch.int32),  # cu_seqlen_ks
                ((seq_len,), torch.int32),  # cu_seqlen_ke
                *(((supplied_scale, torch.float32),) if supplied_scale else ()),
                ((batch, seq_len, seq_len_kv, kv_group), torch.float32),  # logits
            )
            assert op.eval_roofline()[1] == oracle, label

    def test_deltanet_autograd_counts_only_the_output_it_returns(self):
        from tileops.ops.linear_attention.deltanet import DeltaNetAutogradOp

        batch, heads, seq_len, dim_k, dim_v = 2, 8, 2048, 128, 128
        op = DeltaNetAutogradOp.__new__(DeltaNetAutogradOp)
        op.batch, op.heads, op.seq_len = batch, heads, seq_len
        op.dim_k, op.dim_v = dim_k, dim_v
        op.chunk_size = 64
        op.dtype = torch.float16
        # The chunk buffers and the per-chunk state stay in the autograd context,
        # so o is the only output; DeltaNetFwdOp returns them and is priced for it.
        oracle = _nbytes(
            ((batch, heads, seq_len, dim_k), torch.float16),  # q
            ((batch, heads, seq_len, dim_k), torch.float16),  # k
            ((batch, heads, seq_len, dim_v), torch.float16),  # v
            ((batch, heads, seq_len), torch.float16),  # beta
            ((batch, heads, seq_len, dim_v), torch.float16),  # o
        )
        assert op.eval_roofline()[1] == oracle

    def test_engram_gate_conv_backward_counts_the_six_gradient_rows(self):
        from tileops.ops.sequence_modeling.engram import EngramGateConvBwdOp

        m, seq_len, d = 4, 2048, 512
        rows = (m, seq_len, d)
        op = EngramGateConvBwdOp.__new__(EngramGateConvBwdOp)
        op.M, op.seq_len, op.d = m, seq_len, d
        op.dtype = torch.float16
        oracle = _nbytes(
            *((rows, torch.float16),) * 5,  # dY, H, k, v, vhat
            *((((d,), torch.float16),) * 2),  # rms_w_h, rms_w_v
            ((4, d), torch.float16),  # conv_w
            *((((m, seq_len), torch.float32),) * 4),  # alpha, rrms_h, rrms_k, rrms_v
            *((rows, torch.float16),) * 3,  # dH, dk, dv
            *((((d,), torch.float32),) * 2),  # drms_w_h, drms_w_v
            ((4, d), torch.float32),  # dconv_w
        )
        assert op.eval_roofline()[1] == oracle

    def test_masked_fill_counts_the_value_tensor_only_where_it_is_declared(self):
        from tileops.ops.elementwise.masked_fill import MaskedFillFwdOp, MaskedFillScalarFwdOp

        # The mask broadcasts against the input, so the two operands and the
        # output all have different sizes.
        input_shape, mask_shape, out_shape = (8, 1, 4096), (8, 4096, 4096), (8, 4096, 4096)
        for cls, has_value in ((MaskedFillFwdOp, True), (MaskedFillScalarFwdOp, False)):
            op = cls.__new__(cls)
            op.input_shape = input_shape
            op.mask_shape = mask_shape
            op.dtype = torch.bfloat16
            if has_value:
                op.value_shape = ()
            oracle = _nbytes(
                (input_shape, torch.bfloat16),
                (mask_shape, torch.bool),
                (out_shape, torch.bfloat16),
                *((((), torch.bfloat16),) if has_value else ()),
            )
            assert op.eval_roofline()[1] == oracle, cls.__name__

    def test_where_prices_each_operand_at_its_own_shape(self):
        from tileops.ops.elementwise.where import WhereFwdOp

        cond_shape, input_shape, other_shape = (8, 4096, 1), (1, 1, 4096), (8, 4096, 4096)
        op = WhereFwdOp.__new__(WhereFwdOp)
        op.condition_shape, op.input_shape, op.other_shape = cond_shape, input_shape, other_shape
        op.dtype = torch.bfloat16
        oracle = _nbytes(
            (cond_shape, torch.bool),
            (input_shape, torch.bfloat16),
            (other_shape, torch.bfloat16),
            ((8, 4096, 4096), torch.bfloat16),  # output
        )
        assert op.eval_roofline()[1] == oracle

    def test_clamp_counts_only_the_bounds_the_call_passed(self):
        from tileops.ops.elementwise.clamp import ClampFwdOp

        input_shape, bound_shape, out_shape = (8, 4096, 4096), (1, 1, 4096), (8, 4096, 4096)
        # (min passed, max passed)
        for has_min, has_max in ((True, True), (True, False), (False, True)):
            op = ClampFwdOp.__new__(ClampFwdOp)
            op.input_shape = input_shape
            op.min_shape = bound_shape if has_min else None
            op.max_shape = bound_shape if has_max else None
            op.dtype = torch.bfloat16
            oracle = _nbytes(
                (input_shape, torch.bfloat16),
                *(((bound_shape, torch.bfloat16),) if has_min else ()),
                *(((bound_shape, torch.bfloat16),) if has_max else ()),
                (out_shape, torch.bfloat16),
            )
            assert op.eval_roofline()[1] == oracle, f"min={has_min} max={has_max}"

    def test_lerp_tensor_prices_the_broadcast_weight_at_its_own_shape(self):
        from tileops.ops.elementwise.arithmetic import LerpTensorFwdOp

        input_shape, end_shape, weight_shape = (8, 4096, 4096), (8, 4096, 4096), (1, 1, 4096)
        op = LerpTensorFwdOp.__new__(LerpTensorFwdOp)
        op.input_shape, op.end_shape, op.weight_shape = input_shape, end_shape, weight_shape
        op.dtype = torch.bfloat16
        oracle = _nbytes(
            (input_shape, torch.bfloat16),
            (end_shape, torch.bfloat16),
            (weight_shape, torch.bfloat16),
            ((8, 4096, 4096), torch.bfloat16),  # output
        )
        assert op.eval_roofline()[1] == oracle


# Classification registry: every implemented op appears in AUDITED (has a
# bytes-oracle case above) or PENDING. There is no exemption: an op whose
# traffic depends on tensor content is recounted the same way, with the case
# constructing the selecting tensor itself, exactly as it constructs shapes.
# Adding an op to the manifest forces a choice here.
AUDITED = frozenset(
    {
        "AddFwdOp",
        "ArgmaxFwdOp",
        "BatchNormFwdOp",
        "ClampFwdOp",
        "Conv2dFwdOp",
        "DeltaNetAutogradOp",
        "EngramGateConvBwdOp",
        "FP8LightningIndexerFwdOp",
        "FusedMoEExpertsFwdOp",
        "FusedMoeFwdOp",
        "FusedMoeSharedExpertFwdOp",
        "GemmFp8FwdOp",
        "GemmW4A16FwdOp",
        "GroupedQueryAttentionBwdOp",
        "GroupedQueryAttentionDenseFwdOp",
        "IndexedExpertMLPFwdOp",
        "LerpTensorFwdOp",
        "Mamba2FwdOp",
        "MaskedFillFwdOp",
        "MaskedFillScalarFwdOp",
        "MoePostPermuteFwdOp",
        "MoePrePermuteFwdOp",
        "MultiHeadAttentionBwdOp",
        "RMSNormFwdOp",
        "VarMeanFwdOp",
        "WhereFwdOp",
    }
)

# FIXME(staged-rollout): most implemented ops lack a bytes-oracle case.
#
# Broken invariant: every implemented op is AUDITED.
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
        "BitwiseAndFwdOp",
        "BitwiseNotFwdOp",
        "BitwiseOrFwdOp",
        "BitwiseXorFwdOp",
        "BmmFp8FwdOp",
        "BmmFwdOp",
        "CBProducerFwdOp",
        "CeilFwdOp",
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
        "EngramGateConvFwdOp",
        "EqFwdOp",
        "ErfFwdOp",
        "ExpFwdOp",
        "Expm1FwdOp",
        "FFTC2CFwdOp",
        "FP8QuantFwdOp",
        "FloorDivideFwdOp",
        "FloorFwdOp",
        "FusedAddLayerNormFwdOp",
        "FusedAddRMSNormFwdOp",
        "FusedTopKOp",
        "GLABwdOp",
        "GLADecodeFwdOp",
        "GLAFwdOp",
        "GeFwdOp",
        "GeluAndMulFwdOp",
        "GeluFwdOp",
        "GeluTanhAndMulFwdOp",
        "GemmFwdOp",
        "GroupNormFwdOp",
        "GroupedGemmFwdOp",
        "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp",
        "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp",
        "GroupedQueryAttentionPrefillVarlenFwdOp",
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
        "MaxPool1dFwdOp",
        "MaxPool1dIndicesFwdOp",
        "MaxPool2dFwdOp",
        "MaxPool2dIndicesFwdOp",
        "MaxPool3dFwdOp",
        "MaxPool3dIndicesFwdOp",
        "MaximumFwdOp",
        "MeanFwdOp",
        "MeanPoolingFwdOp",
        "MinimumFwdOp",
        "MishFwdOp",
        "MoeExpertMLPFwdOp",
        "MoeGroupedGemmFwdOp",
        "MoePermuteAlignFwdOp",
        "MulFwdOp",
        "MultiHeadAttentionDecodePagedWithKVCacheFwdOp",
        "MultiHeadLatentAttentionDecodeWithKVCacheFwdOp",
        "NSACmpFwdVarlenOp",
        "NSAFwdVarlenOp",
        "NSATopkVarlenOp",
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
    }
)


def test_every_implemented_op_is_classified():
    """A new op cannot ship a bytes formula nothing accounts for."""
    from tileops.manifest import load_manifest

    implemented = {name for name, e in load_manifest().items() if e.get("status") == "implemented"}
    classified = AUDITED | PENDING
    assert implemented - classified == set(), (
        f"unclassified implemented ops: {sorted(implemented - classified)}; "
        "add an oracle case (AUDITED) or a PENDING entry"
    )
    assert classified - implemented == set(), (
        f"stale registry entries: {sorted(classified - implemented)}"
    )
    assert not (AUDITED & PENDING)
