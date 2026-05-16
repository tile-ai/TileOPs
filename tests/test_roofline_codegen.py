"""End-to-end checks for manifest-driven ``eval_roofline`` codegen.

Guards two contracts:
- ``op.eval_roofline()`` on real ops returns the numbers the manifest declares.
- Every ``status: implemented`` op in the live manifest has a non-stub
  ``eval_roofline`` installed by the ``__init_subclass__`` hook.
"""

import pytest

from tileops.ops.op_base import Op

pytestmark = pytest.mark.smoke


class TestRealOpSmoke:
    """End-to-end checks on concrete ops with both inline-mode tiers:
    ``PreluFwdOp`` uses ``self.<input>_shape`` for both inputs;
    ``NanToNumFwdOp`` uses ``self.<input>_shape`` for its single input.
    """

    def test_prelu_fwd_op_eval_roofline_uses_shape_attrs(self):
        import torch

        from tileops.ops.elementwise.prelu import PreluFwdOp

        # PreluFwdOp.__init__ builds a kernel; bypass via __new__ so the
        # test stays smoke-light and CUDA-free.
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


class TestRealManifestParity:
    """Every status:implemented op in the live manifest has a working
    ``eval_roofline`` installed."""

    def test_no_stubs_for_implemented_ops(self):
        from tileops.manifest import load_manifest
        ops = load_manifest()
        import tileops.ops  # noqa: F401
        stubs = []
        for op_name, entry in ops.items():
            if entry.get("status") != "implemented":
                continue
            cls = _find_op_class(op_name)
            if cls is None:
                continue
            if cls.eval_roofline is Op.eval_roofline:
                stubs.append(op_name)
        assert stubs == [], (
            f"{len(stubs)} implemented ops still have the base "
            f"eval_roofline stub: {stubs[:5]}..."
        )


def _find_op_class(op_name):
    seen = set()
    stack = list(Op.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        if cls.__name__ == op_name:
            return cls
        stack.extend(cls.__subclasses__())
    return None
