import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.mamba import CausalConv1dFwdOp, CausalConv1dUpdateFwdOp
from workloads.device import run_device
from workloads.mamba import (
    CausalConv1dFwdCall,
    CausalConv1dUpdateFwdCall,
    causal_conv1d_update_ref,
    conv_state_result,
)
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call


class CausalConv1dFwdTest(CausalConv1dFwdCall, TestBase):
    pass


class CausalConv1dUpdateFwdTest(CausalConv1dUpdateFwdCall, TestBase):
    pass


_SMOKE = pytest.mark.smoke
_FULL = pytest.mark.full
_BOTH = ["bias", "has_initial_state"]

# A warp walks 64 tokens (48 at W = 3) of 256 channels. One case per path through it.
_PREFILL = (
    pytest.param(
        [1024, 1024], 5376, 4, "bfloat16", _BOTH, "silu", marks=_SMOKE, id="one-sequence-a-chunk"
    ),
    pytest.param(
        [100, 900], 8192, 4, "float16", _BOTH, "silu", marks=_SMOKE, id="start-mid-chunk-fp16"
    ),
    pytest.param(
        [64, 64, 128], 512, 4, "bfloat16", _BOTH, "silu", marks=_SMOKE, id="start-on-chunk"
    ),
    pytest.param(
        [62, 70], 256, 4, "bfloat16", _BOTH, "silu", marks=_SMOKE, id="head-past-chunk-end"
    ),
    pytest.param(
        [1, 2, 0, 3, 5, 17, 0, 64, 33, 1, 0],
        264,
        4,
        "bfloat16",
        _BOTH,
        "silu",
        marks=_SMOKE,
        id="short-and-empty",
    ),
    pytest.param(
        [300, 200],
        1024,
        4,
        "float16",
        ["has_initial_state"],
        None,
        marks=_SMOKE,
        id="no-bias-no-act",
    ),
    pytest.param([129, 70], 768, 4, "bfloat16", ["bias"], "silu", marks=_SMOKE, id="no-state"),
    # Regression: an empty sequence without a state got zeros written over its state.
    pytest.param(
        [0, 4, 0, 70], 256, 4, "bfloat16", ["bias"], "silu", marks=_SMOKE, id="empty-keeps-state"
    ),
    pytest.param(
        [0, 0, 7, 70, 0],
        136,
        3,
        "float16",
        ["has_initial_state"],
        "silu",
        marks=_FULL,
        id="width-3",
    ),
    pytest.param([5, 3, 2, 0, 90], 40, 2, "bfloat16", ["bias"], None, marks=_FULL, id="width-2"),
    pytest.param([37, 91, 1, 2], 100, 4, "float16", _BOTH, "silu", marks=_FULL, id="vec-4"),
    pytest.param([9, 70], 6, 4, "float16", _BOTH, "silu", marks=_FULL, id="vec-2-async-4-bytes"),
    pytest.param(
        [13, 60], 7, 4, "bfloat16", ["has_initial_state"], "silu", marks=_FULL, id="vec-1-no-async"
    ),
    pytest.param([1] * 300, 256, 4, "bfloat16", _BOTH, "silu", marks=_FULL, id="tokens-a-sequence"),
    pytest.param(
        [2500, 1000, 500, 96],
        8192,
        4,
        "float16",
        ["has_initial_state"],
        "silu",
        marks=_FULL,
        id="ragged",
    ),
)


@pytest.mark.parametrize("seq_lens, dim, width, dtype, some, activation", _PREFILL)
def test_causal_conv1d(seq_lens, dim, width, dtype, some, activation) -> None:
    call = manifest_call(
        "CausalConv1dFwdOp",
        {"T": dtype},
        D=dim,
        W=width,
        Tt=sum(seq_lens),
        seq_lens=seq_lens,
        activation=activation,
        some=some,
    )
    test = CausalConv1dFwdTest(call)
    op = CausalConv1dFwdOp(**call.arguments({}))
    test.check(op, *test.gen_inputs(), runs=lambda *args: conv_state_result(op, *args))


_DECODE = (
    pytest.param(64, 5376, 4, "bfloat16", ["bias"], "silu", marks=_SMOKE, id="bf16-bias"),
    pytest.param(3, 8192, 4, "float16", [], "silu", marks=_SMOKE, id="fp16"),
    pytest.param(5, 100, 3, "bfloat16", ["bias"], None, marks=_FULL, id="width-3-vec-4"),
    pytest.param(4, 6, 4, "bfloat16", [], "silu", marks=_FULL, id="vec-2"),
    pytest.param(2, 7, 2, "float16", ["bias"], "silu", marks=_FULL, id="width-2-vec-1"),
)


@pytest.mark.parametrize("batch, dim, width, dtype, some, activation", _DECODE)
def test_causal_conv1d_update(batch, dim, width, dtype, some, activation) -> None:
    call = manifest_call(
        "CausalConv1dUpdateFwdOp",
        {"T": dtype},
        B=batch,
        D=dim,
        W=width,
        activation=activation,
        some=some,
    )
    test = CausalConv1dUpdateFwdTest(call)
    op = CausalConv1dUpdateFwdOp(**call.arguments({}))
    test.check(op, *test.gen_inputs(), runs=lambda *args: conv_state_result(op, *args))


@pytest.mark.smoke
def test_causal_conv1d_update_unaligned_state() -> None:
    """A state whose storage starts off a 16-byte boundary is advanced in place all the same."""
    batch, dim, width = 4, 256, 4
    dev, dtype = run_device(), torch.bfloat16
    flat = torch.randn(batch * dim * (width - 1) + 1, device=dev, dtype=dtype)
    state = flat[1:].view(batch, dim, width - 1)
    x = torch.randn(batch, dim, device=dev, dtype=dtype)
    weight = torch.rand(dim, width, device=dev, dtype=dtype) - 0.5
    expected_state = state.clone()
    expected = causal_conv1d_update_ref(x, weight, expected_state)

    got = CausalConv1dUpdateFwdOp()(x, weight, state)

    workload = CausalConv1dUpdateFwdCall(
        manifest_call("CausalConv1dUpdateFwdOp", {"T": "bfloat16"}, B=batch, D=dim, W=width)
    )
    evidence = workload.verification(x, weight, state, None)
    compare_outputs((got, state), (expected, expected_state), evidence)
