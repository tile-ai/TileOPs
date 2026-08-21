"""Compile-boundary contract for the norm family.

One case per op: a cold ``torch.compile(op, fullgraph=True)`` must match eager and the
traced graph must hold nothing but that op's own operator. RMSNorm has its own module.
"""

import pytest
import torch

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tileops.ops.norm import (
    AdaLayerNormFwdOp,
    AdaLayerNormZeroFwdOp,
    BatchNormBwdOp,
    BatchNormFwdOp,
    FusedAddLayerNormFwdOp,
    FusedAddRMSNormFwdOp,
    GroupNormFwdOp,
    InstanceNormFwdOp,
    LayerNormFwdOp,
)

_DTYPE = torch.float16
_N = 256


def _x(*shape, dtype=_DTYPE):
    return torch.randn(*shape, dtype=dtype, device="cuda")


def _cases():
    """One builder per op, returning ``(op, inputs)``.

    The builders run inside the test: this module is imported on the CPU-only runner
    that enforces the compile-contract gate, where a CUDA tensor built at import time
    would fail before any test is selected.
    """

    def layer_norm():
        return LayerNormFwdOp(normalized_shape=(_N,)), (_x(8, _N), _x(_N), _x(_N))

    def ada_layer_norm():
        return AdaLayerNormFwdOp(), (_x(8, _N), _x(8, _N), _x(8, _N))

    def ada_layer_norm_zero():
        return AdaLayerNormZeroFwdOp(), (_x(8, _N), _x(8, _N), _x(8, _N), _x(8, _N))

    def fused_add_layer_norm():
        return FusedAddLayerNormFwdOp(), (_x(8, _N), _x(8, _N), _x(_N), _x(_N))

    def fused_add_rms_norm():
        return FusedAddRMSNormFwdOp(), (_x(8, _N), _x(8, _N), _x(_N))

    def group_norm():
        return GroupNormFwdOp(num_groups=2), (_x(2, 4, 8, 8), _x(4), _x(4))

    def group_norm_no_affine():
        return GroupNormFwdOp(num_groups=2), (_x(2, 4, 8, 8),)

    def instance_norm():
        return InstanceNormFwdOp(), (_x(2, 4, 8, 8), None, None, _x(4), _x(4))

    def batch_norm_infer():
        f32 = dict(dtype=torch.float32)
        return BatchNormFwdOp(), (
            _x(2, 4, 8, 8),
            _x(4, **f32).abs(),
            _x(4, **f32).abs(),
            _x(4, **f32),
            _x(4, **f32),
        )

    def batch_norm_train():
        f32 = dict(dtype=torch.float32)
        return BatchNormFwdOp(training=True), (
            _x(2, 4, 8, 8),
            _x(4, **f32).abs(),
            _x(4, **f32).abs(),
            _x(4, **f32),
            _x(4, **f32),
        )

    def batch_norm_bwd():
        f32 = dict(dtype=torch.float32)
        return BatchNormBwdOp(), (
            _x(2, 4, 8, 8),
            _x(2, 4, 8, 8),
            _x(4, **f32),
            _x(4, **f32),
            _x(4, **f32).abs(),
        )

    return [
        pytest.param(builder, id=name)
        for name, builder in (
            ("layer-norm", layer_norm),
            ("ada-layer-norm", ada_layer_norm),
            ("ada-layer-norm-zero", ada_layer_norm_zero),
            ("fused-add-layer-norm", fused_add_layer_norm),
            ("fused-add-rms-norm", fused_add_rms_norm),
            ("group-norm", group_norm),
            ("group-norm-no-affine", group_norm_no_affine),
            ("instance-norm", instance_norm),
            ("batch-norm-infer", batch_norm_infer),
            ("batch-norm-train", batch_norm_train),
            ("batch-norm-bwd", batch_norm_bwd),
        )
    ]


for _op_cls in (
    LayerNormFwdOp,
    AdaLayerNormFwdOp,
    AdaLayerNormZeroFwdOp,
    FusedAddLayerNormFwdOp,
    FusedAddRMSNormFwdOp,
    GroupNormFwdOp,
    InstanceNormFwdOp,
    BatchNormFwdOp,
    BatchNormBwdOp,
):
    register_compile_contract(_op_cls)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize("build_case", _cases())
def test_a_cold_op_traces_fullgraph_and_matches_eager(build_case) -> None:
    """Cold is the whole contract: a warm op has nothing left for dynamo to trace into."""
    op, inputs = build_case()
    # Each call gets its own copies: an op that writes an input (BatchNorm's running
    # statistics) would otherwise hand the second call a different starting state.
    compiled_inputs = tuple(None if t is None else t.clone() for t in inputs)
    eager_inputs = tuple(None if t is None else t.clone() for t in inputs)

    compiled = torch.compile(op, fullgraph=True)(*compiled_inputs)

    torch.testing.assert_close(compiled, op(*eager_inputs))
    # An input the op writes (BatchNorm's running statistics, R22) must come out of the
    # compiled call holding what eager left in it.
    torch.testing.assert_close(compiled_inputs, eager_inputs)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize("build_case", _cases())
def test_the_traced_graph_holds_only_this_ops_operator(build_case) -> None:
    """The node is the op's, so replacing the kernel cannot change the graph."""
    op, inputs = build_case()

    assert_op_owns_graph_nodes(op, *inputs)
