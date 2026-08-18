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

    def test_optional_input_presence_switches_the_formula(self):
        """R18.1: an optional input may appear as a bare presence test."""
        import torch

        from tileops.ops._roofline_codegen import synthesize_eval_roofline

        fn = synthesize_eval_roofline(
            "FakeOp",
            roofline={
                "vars": {"N": "x.shape[0]", "has_bias": "bias is not None"},
                "flops": "(2 if has_bias else 1) * N",
                "bytes": "(N + (8 if has_bias else 0)) * elem_bytes",
            },
            signature={
                "inputs": {
                    "x": {"dtype": "float16"},
                    "bias": {"dtype": "float16", "optional": True},
                },
            },
        )

        class FakeOp:
            dtype = torch.float16

            def __init__(self, bias):
                self.x_shape = (64,)
                self.bias = bias

        assert fn(FakeOp(torch.empty(8, dtype=torch.float16))) == (128, 144)
        assert fn(FakeOp(None)) == (64, 128)

    def test_vars_may_not_read_an_optional_input(self):
        """Only a presence test — its shape is unavailable on an absent call."""
        from tileops.ops._roofline_codegen import synthesize_eval_roofline

        for expr in ("bias.shape[0]", "bias is None or bias.ndim", "bias[0]"):
            with pytest.raises(ValueError, match="optional input"):
                synthesize_eval_roofline(
                    "FakeOp",
                    roofline={
                        "vars": {"n": expr}, "flops": "n", "bytes": "n",
                    },
                    signature={
                        "inputs": {
                            "x": {"dtype": "float16"},
                            "bias": {"dtype": "float16", "optional": True},
                        },
                    },
                )

    def test_optional_input_the_op_never_exposes_still_raises(self):
        """An unexposed binding must not read as "the call omitted it"."""
        import torch

        from tileops.ops._roofline_codegen import synthesize_eval_roofline

        fn = synthesize_eval_roofline(
            "FakeOp",
            roofline={
                "vars": {"N": "x.shape[0]", "has_bias": "bias is not None"},
                "flops": "N",
                "bytes": "(N + (8 if has_bias else 0)) * elem_bytes",
            },
            signature={
                "inputs": {
                    "x": {"dtype": "float16"},
                    "bias": {"dtype": "float16", "optional": True},
                },
            },
        )

        class FakeOp:
            dtype = torch.float16
            x_shape = (64,)

        with pytest.raises(ValueError, match="cannot resolve roofline input"):
            fn(FakeOp())

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
