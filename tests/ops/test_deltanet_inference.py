"""The inference DeltaNet contract, before the in-tree kernels are migrated."""

from functools import partial

import pytest
import torch

from tests.test_base import TestBase, allclose_compare
from tileops.backend import TensorSpec, registry
from tileops.ops import DeltaNetInferenceFwdOp
from workloads.linear_attention import DeltaNetInferenceWorkload

pytestmark = pytest.mark.smoke


class DeltaNetInferenceTest(DeltaNetInferenceWorkload, TestBase):
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


def test_deltanet_inference_reaches_target_with_optional_inputs() -> None:
    calls = []

    def build_kernel(*specs, **params):
        calls.append((specs, params))

        def kernel(q, k, v, beta, initial_state, cu_seqlens, cu_seqlens_cpu):
            del k, beta, initial_state, cu_seqlens_cpu
            state_batch = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
            return (
                torch.empty_like(v),
                torch.empty(state_batch, q.shape[2], q.shape[3], v.shape[3], dtype=torch.float32),
            )

        return kernel

    registry.register_kernel_builder("DeltaNetInferenceFwdOp", "deltanet_test", build_kernel)

    q = torch.randn(1, 7, 2, 8, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(1, 7, 2, 6, dtype=torch.float16)
    beta = torch.rand(1, 7, 2, dtype=torch.float16)
    initial_state = torch.zeros(2, 2, 8, 6, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int64)
    cu_seqlens_cpu = cu_seqlens.clone()

    op = DeltaNetInferenceFwdOp(scale=0.125, use_qk_l2norm_in_kernel=True, target="deltanet_test")
    o, final_state = op(q, k, v, beta, initial_state, cu_seqlens, cu_seqlens_cpu)

    assert o.shape == v.shape
    assert final_state.shape == initial_state.shape
    assert op.eval_roofline()[1] == 2348
    decode_q = torch.randn(1, 1, 2, 8, dtype=torch.float16)
    decode_v = torch.randn(1, 1, 2, 6, dtype=torch.float16)
    decode_beta = torch.rand(1, 1, 2, dtype=torch.float16)
    decode_state = torch.zeros(1, 2, 8, 6, dtype=torch.float32)
    decode_o, decode_final_state = op(decode_q, decode_q, decode_v, decode_beta, decode_state)
    assert decode_o.shape == decode_v.shape
    assert decode_final_state.shape == decode_state.shape
    assert calls == [
        (
            tuple(
                TensorSpec.of(tensor)
                for tensor in (q, k, v, beta, initial_state, cu_seqlens, cu_seqlens_cpu)
            ),
            {"scale": 0.125, "use_qk_l2norm_in_kernel": True},
        ),
        (
            tuple(
                None if tensor is None else TensorSpec.of(tensor)
                for tensor in (decode_q, decode_q, decode_v, decode_beta, decode_state, None, None)
            ),
            {"scale": 0.125, "use_qk_l2norm_in_kernel": True},
        ),
    ]


def test_deltanet_inference_rejects_invalid_state() -> None:
    q = torch.empty(1, 1, 2, 8, dtype=torch.float16)
    k = torch.empty_like(q)
    v = torch.empty(1, 1, 2, 6, dtype=torch.float16)
    beta = torch.empty(1, 1, 2, dtype=torch.float16)

    with pytest.raises(ValueError, match=r"\[N, H, K, V\]"):
        DeltaNetInferenceFwdOp().forward(q, k, v, beta, torch.empty(1, 2, 6, 8))
    with pytest.raises(ValueError, match="requires cu_seqlens"):
        DeltaNetInferenceFwdOp().forward(q, k, v, beta, cu_seqlens_cpu=torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        DeltaNetInferenceFwdOp().forward(q.float(), k.float(), v.float(), beta.float())


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the in-tree dense prefill requires SM90",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_deltanet_dense_prefill_matches_fla(dtype: torch.dtype) -> None:
    torch.manual_seed(2163)
    test = DeltaNetInferenceTest(2, 128, 4, 64, dtype)
    inputs = test.gen_inputs()
    op = DeltaNetInferenceFwdOp()
    if dtype == torch.float16:
        # The output meets the standard 1e-3 tolerance. The FP32 final state
        # has a measured 2.10e-3 maximum error for seeded and zero-state calls.
        compare = [
            partial(allclose_compare, atol=1e-3, rtol=1e-3),
            partial(allclose_compare, atol=3e-3, rtol=1e-3),
        ]
        test.check(op, *inputs, compare=compare)
        test.check(op, *inputs[:4], compare=compare)
    else:
        test.check(op, *inputs, atol=1.6e-2, rtol=1.6e-2)
        test.check(op, *inputs[:4], atol=1.6e-2, rtol=1.6e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the in-tree dense prefill requires SM90",
)
def test_deltanet_partitioned_prefill_matches_fla(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TILEOPS_DELTANET_PREFILL_MAX_LOCAL_CHUNKS", "4")
    torch.manual_seed(2163)
    test = DeltaNetInferenceTest(2, 512, 4, 64, torch.bfloat16)
    test.check(DeltaNetInferenceFwdOp(), *test.gen_inputs(), atol=1.6e-2, rtol=1.6e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the in-tree dense prefill requires SM90",
)
def test_deltanet_wide_prefill_matches_fla() -> None:
    torch.manual_seed(2163)
    test = DeltaNetInferenceTest(1, 256, 4, 128, torch.bfloat16)
    test.check(DeltaNetInferenceFwdOp(), *test.gen_inputs(), atol=1.6e-2, rtol=1.6e-2)
