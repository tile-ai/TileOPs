"""Real-op smoke tests for the generated ``eval_roofline``."""

import pytest

pytestmark = pytest.mark.smoke


class TestRealOpSmoke:
    def test_prelu_fwd_op_eval_roofline_uses_shape_attrs(self):
        import torch

        from tileops.ops.elementwise.prelu import PreluFwdOp

        # __new__ bypasses kernel construction so the smoke stays CUDA-free.
        op = PreluFwdOp.__new__(PreluFwdOp)
        op.input_shape = (16, 256, 56, 56)
        op.weight_shape = (256,)
        op.dtype = torch.float16

        from math import prod as _prod
        N = _prod(op.input_shape)
        W = op.weight_shape[0]
        elem = op.dtype.itemsize
        flops, total_bytes = op.eval_roofline()
        assert flops == 2 * N
        assert total_bytes == (2 * N + W) * elem

    def test_nan_to_num_fwd_op_eval_roofline_uses_input_shape(self):
        import torch

        from tileops.ops.elementwise.nan_to_num import NanToNumFwdOp

        op = NanToNumFwdOp.__new__(NanToNumFwdOp)
        op.input_shape = (4096 * 4096,)
        op.dtype = torch.float16

        N = op.input_shape[0]
        elem = op.dtype.itemsize
        flops, total_bytes = op.eval_roofline()
        assert flops == 6 * N
        assert total_bytes == 2 * N * elem
