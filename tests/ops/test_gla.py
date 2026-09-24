"""Public GLA forward contract and its in-tree prefill/decode paths."""

from functools import partial

import pytest
import torch

from benchmarks.baselines import reference_tolerance
from tests.test_base import TestBase, allclose_compare
from tileops.backend import TensorSpec, registry
from tileops.kernels.linear_attention.gla.dense_decode import GLADenseDecodeKernel
from tileops.kernels.linear_attention.gla.dense_prefill_partitioned import (
    GLADensePrefillPartitionedKernel,
)
from tileops.ops import GLAFwdOp
from tileops.perf.formulas import gla_fwd_roofline
from tileops.utils import is_h200
from workloads.linear_attention import GLAWorkload

pytestmark = pytest.mark.smoke


class GLATest(GLAWorkload, TestBase):
    pass


@pytest.fixture(autouse=True)
def isolated_registry():
    state = registry.snapshot()
    registry.DETECTORS.clear()
    registry.BUILDERS.clear()
    registry.LOAD_FAILURES.clear()
    registry.default_target = None
    registry._loaded = True
    yield
    registry.restore(state)


def test_gla_reaches_external_target_with_optional_inputs() -> None:
    calls = []

    def build_kernel(*specs, **params):
        calls.append((specs, params))

        def kernel(q, k, v, g, initial_state, cu_seqlens, cu_seqlens_cpu):
            del k, g, initial_state, cu_seqlens_cpu
            state_batch = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
            return (
                torch.empty_like(v),
                torch.empty(state_batch, q.shape[2], q.shape[3], v.shape[3], dtype=torch.float32),
            )

        return kernel

    registry.register_kernel_builder("GLAFwdOp", "gla_test", build_kernel)
    q = torch.randn(1, 7, 2, 8, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(1, 7, 2, 6, dtype=torch.float16)
    g = -torch.rand_like(q)
    state = torch.zeros(2, 2, 8, 6, dtype=torch.float32)
    cu = torch.tensor([0, 3, 7], dtype=torch.int64)
    op = GLAFwdOp(scale=0.125, target="gla_test")

    o, final_state = op(q, k, v, g, state, cu, cu.clone())
    assert o.shape == v.shape
    assert final_state.shape == state.shape

    decode_q = torch.randn(1, 1, 2, 8, dtype=torch.float16)
    decode_v = torch.randn(1, 1, 2, 6, dtype=torch.float16)
    decode_state = torch.zeros(1, 2, 8, 6, dtype=torch.float32)
    decode_o, decode_final = op(decode_q, decode_q, decode_v, decode_q, decode_state)
    assert decode_o.shape == decode_v.shape
    assert decode_final.shape == decode_state.shape
    assert calls == [
        (
            tuple(TensorSpec.of(tensor) for tensor in (q, k, v, g, state, cu, cu)),
            {"scale": 0.125},
        ),
        (
            tuple(
                None if tensor is None else TensorSpec.of(tensor)
                for tensor in (decode_q, decode_q, decode_v, decode_q, decode_state, None, None)
            ),
            {"scale": 0.125},
        ),
    ]


@pytest.mark.parametrize("seeded", [False, True])
def test_gla_roofline_counts_packed_states(seeded: bool) -> None:
    def build_kernel(*specs, **params):
        def kernel(q, k, v, g, initial_state, cu_seqlens, cu_seqlens_cpu):
            state_batch = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
            return torch.empty_like(v), torch.empty(
                state_batch, q.shape[2], q.shape[3], v.shape[3], dtype=torch.float32
            )

        return kernel

    registry.register_kernel_builder("GLAFwdOp", "gla_test", build_kernel)
    op = GLAFwdOp(target="gla_test")
    q, k, g = (torch.empty(1, 7, 2, 8, dtype=torch.float16) for _ in range(3))
    v = torch.empty(1, 7, 2, 6, dtype=torch.float16)
    packed_state = torch.empty(2, 2, 8, 6, dtype=torch.float32) if seeded else None
    cu = torch.tensor([0, 3, 7], dtype=torch.int64)

    # Reuse the Op for dense input too: packed sequence metadata must not leak.
    for state, lengths, expected_bytes in (
        (packed_state, cu, 2544 if seeded else 1776),
        (packed_state[:1] if seeded else None, None, 1776 if seeded else 1392),
    ):
        op(q, k, v, g, state, lengths)
        assert op.eval_roofline()[1] == expected_bytes
        assert (
            gla_fwd_roofline(
                q_shape=q.shape,
                v_shape=v.shape,
                dtype=q.dtype,
                initial_state_shape=state.shape if state is not None else None,
                cu_seqlens_shape=lengths.shape if lengths is not None else None,
            )
            == op.eval_roofline()
        )


def test_gla_rejects_float32_activations() -> None:
    q = torch.empty(1, 64, 2, 8, dtype=torch.float32)
    v = torch.empty(1, 64, 2, 6, dtype=torch.float32)
    with pytest.raises(ValueError, match="q must have float16 or bfloat16 dtype"):
        GLAFwdOp().forward(q, q, v, q)


def test_gla_rejects_invalid_state_and_gate() -> None:
    q = torch.empty(1, 64, 2, 8, dtype=torch.float16)
    k = torch.empty_like(q)
    v = torch.empty(1, 64, 2, 6, dtype=torch.float16)
    g = torch.empty_like(q)
    op = GLAFwdOp()
    with pytest.raises(ValueError, match="same.*shape"):
        op.forward(q, k, v, g[:, :-1])
    with pytest.raises(ValueError, match=r"\[N, H, K, V\]"):
        op.forward(q, k, v, g, torch.empty(1, 2, 6, 8))
    with pytest.raises(ValueError, match="requires matching cu_seqlens"):
        op.forward(q, k, v, g, cu_seqlens_cpu=torch.tensor([0, 64]))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the in-tree dense prefill requires SM90",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("seq_len, dim", [(128, 64), (128, 128), (1024, 64)])
def test_gla_dense_prefill_matches_fla(dtype: torch.dtype, seq_len: int, dim: int) -> None:
    torch.manual_seed(2160)
    test = GLATest(2, seq_len, 4, dim, dim, dtype, has_initial_state=True)
    inputs = test.gen_inputs()
    op = GLAFwdOp()
    test.check(op, *inputs, **reference_tolerance(dtype))
    test.check(op, *inputs[:4], **reference_tolerance(dtype))
    inputs[3].mul_(3.0)
    test.check(op, *inputs, **reference_tolerance(dtype))


@pytest.mark.skipif(not is_h200(), reason="partitioned prefill is selected on H200")
@pytest.mark.parametrize(
    "dtype,has_initial_state,gate_scale",
    [(torch.bfloat16, True, 1.0), (torch.float16, False, 3.0)],
)
def test_gla_long_prefill_uses_partitioned_kernel(
    dtype: torch.dtype, has_initial_state: bool, gate_scale: float
) -> None:
    torch.manual_seed(2160)
    test = GLATest(2, 16384, 4, 64, 64, dtype, has_initial_state)
    inputs = test.gen_inputs()
    inputs[3].mul_(gate_scale)
    op = GLAFwdOp()
    tolerance = reference_tolerance(dtype)
    state_tolerance = tolerance.copy()
    if dtype == torch.float16:
        # H200 / FLA 0.5.2, T=16384, K=V=64, gate_scale=3, seed=2160:
        # max absolute error is 1.908e-4 for output and 2.300e-3 for FP32 state.
        # Only final-state atol needs an exception; output and rtol stay standard.
        state_tolerance["atol"] = 2.5e-3
    test.check(
        op,
        *inputs,
        compare=[
            partial(allclose_compare, **tolerance),
            partial(allclose_compare, **state_tolerance),
        ],
    )
    assert any(
        isinstance(kernel, GLADensePrefillPartitionedKernel)
        for kernel in op.built_kernels("gla_dense_prefill").values()
    )


@pytest.mark.skipif(
    not is_h200(),
    reason="the in-tree dense decode requires H200",
)
@pytest.mark.parametrize(
    "dtype,dim,has_initial_state",
    [
        (torch.float16, 128, True),
        (torch.bfloat16, 64, False),
        (torch.bfloat16, 64, True),
    ],
)
def test_gla_dense_decode_matches_fla(
    dtype: torch.dtype, dim: int, has_initial_state: bool
) -> None:
    torch.manual_seed(2174)
    test = GLATest(2, 1, 4, dim, dim, dtype, has_initial_state)
    inputs = test.gen_inputs()
    op = GLAFwdOp()
    test.check(op, *inputs, **reference_tolerance(dtype))
    assert any(
        isinstance(kernel, GLADenseDecodeKernel)
        for kernel in op.built_kernels("gla_dense_decode").values()
    )
