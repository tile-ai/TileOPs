"""Public GLA inference contract and its first in-tree dense-prefill path."""

import pytest
import torch

from tests.test_base import TestBase
from tileops.backend import TensorSpec, registry
from tileops.ops import GLAInferenceFwdOp
from workloads.linear_attention import GLAInferenceWorkload

pytestmark = pytest.mark.smoke


class GLAInferenceTest(GLAInferenceWorkload, TestBase):
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


def test_gla_inference_reaches_external_target_with_optional_inputs() -> None:
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

    registry.register_kernel_builder("GLAInferenceFwdOp", "gla_test", build_kernel)
    q = torch.randn(1, 7, 2, 8, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(1, 7, 2, 6, dtype=torch.float16)
    g = -torch.rand_like(q)
    state = torch.zeros(2, 2, 8, 6, dtype=torch.float32)
    cu = torch.tensor([0, 3, 7], dtype=torch.int64)
    op = GLAInferenceFwdOp(scale=0.125, target="gla_test")

    o, final_state = op(q, k, v, g, state, cu, cu.clone())
    assert o.shape == v.shape
    assert final_state.shape == state.shape

    decode_q = torch.randn(1, 1, 2, 8, dtype=torch.float32)
    decode_v = torch.randn(1, 1, 2, 6, dtype=torch.float32)
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


def test_gla_inference_rejects_invalid_state_and_gate() -> None:
    q = torch.empty(1, 64, 2, 8, dtype=torch.float16)
    k = torch.empty_like(q)
    v = torch.empty(1, 64, 2, 6, dtype=torch.float16)
    g = torch.empty_like(q)
    op = GLAInferenceFwdOp()
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
def test_gla_dense_prefill_matches_fla(dtype: torch.dtype) -> None:
    torch.manual_seed(2160)
    test = GLAInferenceTest(2, 128, 4, 64, 64, dtype, has_initial_state=True)
    inputs = test.gen_inputs()
    op = GLAInferenceFwdOp()
    test.check(op, *inputs, atol=0.03, rtol=0.03)
    test.check(op, *inputs[:4], atol=0.03, rtol=0.03)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the in-tree dense prefill requires SM90",
)
def test_gla_inference_rejects_unmigrated_decode() -> None:
    test = GLAInferenceTest(1, 1, 2, 64, 64, torch.bfloat16)
    with pytest.raises(ValueError, match="T not divisible by 64"):
        GLAInferenceFwdOp()(*test.gen_inputs())
