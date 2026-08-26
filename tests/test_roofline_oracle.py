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
