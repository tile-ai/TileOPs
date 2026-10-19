"""Public FusedQKNormRoPEFwdOp semantics: both layouts, partial rotation, every dtype pair."""

import pytest
import torch

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tests.workload_test_base import TestBase
from tileops.norm import FusedQKNormRoPEFwdOp
from workloads.norm import FusedQKNormRoPEWorkload, fused_qk_norm_rope_result
from workloads.numerics import compare_outputs


class FusedQKNormRoPETest(FusedQKNormRoPEWorkload, TestBase):
    """The workload as a test case: the reference and its policy come from the workload."""


def _packed(*lengths: int) -> list[int]:
    return [p for n in lengths for p in range(n)]


_SMOKE = pytest.mark.smoke
_FULL = pytest.mark.full


def _case(positions, heads, kv_heads, dim, rot, dtype, cache, layout, mark, case_id):
    """A call on a cache of max(positions) + 1 rows."""
    return pytest.param(
        positions,
        heads,
        kv_heads,
        dim,
        rot,
        max(positions) + 1,
        dtype,
        cache,
        layout,
        marks=mark,
        id=case_id,
    )


_DTYPES = tuple(
    _case(_packed(5, 3), 4, 2, 128, 128, t, c, "neox", _SMOKE, f"{t}-{c}")
    for t in ("bfloat16", "float16")
    for c in ("bfloat16", "float16", "float32")
)

# A head is D / (2 vec) lanes of two vec-wide accesses, padded to a power of two; a CTA is
# 64 threads. One case per path through that mapping.
_PATHS = (
    _case(
        _packed(4, 4), 4, 2, 128, 128, "bfloat16", "bfloat16", "interleaved", _SMOKE, "interleaved"
    ),
    # Lanes past R / (2 vec) hold passed-through pairs.
    _case(_packed(9), 6, 2, 128, 64, "bfloat16", "bfloat16", "neox", _SMOKE, "neox-partial"),
    _case(
        _packed(9),
        6,
        2,
        128,
        32,
        "bfloat16",
        "bfloat16",
        "interleaved",
        _SMOKE,
        "interleaved-partial",
    ),
    # Six lanes a head padded to eight; the two padding lanes hold nothing.
    _case(_packed(5), 3, 1, 96, 96, "bfloat16", "bfloat16", "neox", _SMOKE, "six-lanes"),
    # Six q and k heads, the last CTA of each token holding two heads of eight.
    _case(_packed(3, 2), 5, 1, 64, 64, "bfloat16", "bfloat16", "neox", _SMOKE, "head-tail"),
    _case([3], 4, 2, 128, 128, "bfloat16", "bfloat16", "neox", _SMOKE, "one-token"),
    # R / 2 = 20 is no multiple of 8: 4-value accesses.
    _case(_packed(4), 2, 1, 64, 40, "bfloat16", "bfloat16", "neox", _FULL, "vec-4"),
    # D = 12: 2-value accesses, three lanes a head padded to four.
    _case(_packed(4), 2, 1, 12, 12, "bfloat16", "bfloat16", "neox", _FULL, "vec-2"),
    # R = 2: 1-value accesses, 32 lanes a head.
    _case(_packed(3), 2, 1, 64, 2, "bfloat16", "bfloat16", "neox", _FULL, "vec-1"),
    _case(_packed(4), 2, 2, 256, 256, "bfloat16", "bfloat16", "neox", _FULL, "dim-256"),
    # One decode token a sequence, at positions far apart, the last the cache's last row.
    _case(
        [4095, 0, 1777, 3],
        8,
        2,
        128,
        128,
        "bfloat16",
        "bfloat16",
        "neox",
        _FULL,
        "decode-scattered",
    ),
)

_CASES = _DTYPES + _PATHS


@pytest.mark.parametrize(
    "positions, heads, kv_heads, dim, rot, positions_max, dtype, cache_dtype, layout", _CASES
)
def test_fused_qk_norm_rope(
    positions, heads, kv_heads, dim, rot, positions_max, dtype, cache_dtype, layout
) -> None:
    """The q and k heads hold the reference's result; the v heads are untouched."""
    test = FusedQKNormRoPETest(
        positions,
        heads,
        kv_heads,
        dim,
        rot,
        positions_max,
        getattr(torch, dtype),
        getattr(torch, cache_dtype),
        layout,
    )
    op = FusedQKNormRoPEFwdOp(heads, kv_heads, rope_layout=layout)
    test.check(op, *test.gen_inputs(), runs=lambda *args: fused_qk_norm_rope_result(op, *args))


@pytest.mark.smoke
def test_fused_qk_norm_rope_eps() -> None:
    """A row of zeros normalizes to zeros through eps, and eps is the one the op was built with."""
    test = FusedQKNormRoPETest(
        _packed(3), 2, 1, 128, 128, 4, torch.bfloat16, torch.bfloat16, eps=1e-2
    )
    qkv, *rest = test.gen_inputs()
    qkv[:, :128] = 0
    qkv[:, 128:256] *= 0.05
    op = FusedQKNormRoPEFwdOp(2, 1, eps=1e-2)
    test.check(op, qkv, *rest, runs=lambda *args: fused_qk_norm_rope_result(op, *args))


def _off_boundary(t):
    """A copy of *t* whose storage starts one element past a 16-byte boundary."""
    flat = torch.empty(t.numel() + 1, device=t.device, dtype=t.dtype)
    shifted = flat[1:].view_as(t)
    shifted.copy_(t)
    return shifted


@pytest.mark.smoke
def test_fused_qk_norm_rope_unaligned_inputs() -> None:
    """Inputs whose storage starts off a 16-byte boundary give the same result, qkv in place."""
    test = FusedQKNormRoPETest(_packed(5), 4, 2, 128, 128, 8, torch.bfloat16, torch.bfloat16)
    inputs = test.gen_inputs()
    expected = test.ref_program(*inputs)
    qkv, q_weight, k_weight, cache = (_off_boundary(t) for t in inputs[:4])

    assert FusedQKNormRoPEFwdOp(4, 2)(qkv, q_weight, k_weight, cache, inputs[4]) is None

    compare_outputs(qkv, expected, test.verification(*inputs))


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param("qkv-float32", marks=_SMOKE),
        pytest.param("weight-dtype", marks=_FULL),
        pytest.param("positions-int32", marks=_FULL),
        pytest.param("qkv-noncontiguous", marks=_FULL),
    ],
)
def test_fused_qk_norm_rope_refuses_an_invalid_tensor_contract(invalid) -> None:
    """Dtypes outside the signature, and a qkv it cannot write in place, are rejected before
    any kernel is chosen."""
    test = FusedQKNormRoPETest(_packed(2), 2, 1, 64, 64, 4, torch.bfloat16, torch.bfloat16)
    qkv, q_weight, k_weight, cache, positions = test.gen_inputs()
    if invalid == "qkv-float32":
        qkv = qkv.float()
    elif invalid == "weight-dtype":
        q_weight = q_weight.half()
    elif invalid == "positions-int32":
        positions = positions.int()
    else:
        qkv = torch.cat([qkv, qkv], -1)[:, : qkv.shape[1]]
    with pytest.raises(
        ValueError, match="contiguous" if invalid == "qkv-noncontiguous" else "dtype"
    ):
        FusedQKNormRoPEFwdOp(2, 1)(qkv, q_weight, k_weight, cache, positions)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_fused_qk_norm_rope_compiles_cold_to_its_operator() -> None:
    """A cold fullgraph compile owns the graph and writes the same qkv as eager."""
    test = FusedQKNormRoPETest(_packed(3, 2), 4, 2, 128, 64, 8, torch.bfloat16, torch.float32)
    assert_op_owns_graph_nodes(FusedQKNormRoPEFwdOp(4, 2), *test.gen_inputs())
    compiled_inputs = test.gen_inputs()
    torch.compile(FusedQKNormRoPEFwdOp(4, 2), fullgraph=True)(*compiled_inputs)
    eager_inputs = test.gen_inputs()
    FusedQKNormRoPEFwdOp(4, 2)(*eager_inputs)
    torch.testing.assert_close(compiled_inputs[0], eager_inputs[0], atol=0, rtol=0)


register_compile_contract(FusedQKNormRoPEFwdOp)
