"""Tests for unary activation elementwise ops.

Covers L1 smoke correctness, multi-dtype coverage, and L4 edge cases.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops.elementwise import ReluFwdOp
from workloads.device import run_device
from workloads.elementwise import (
    ElementwiseWorkload,
    GeluTailWorkload,
    ReluWorkload,
    UnaryActivationCase,
)
from workloads.numerics import compare_outputs


class ReluTest(ReluWorkload, TestBase):
    pass


class ReluFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype",
            [
                # Smoke: one typical shape per supported dtype
                pytest.param(
                    1_000_000,
                    torch.float16,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="elementwise")],
                ),
                pytest.param(1_000_000, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1_000_000, torch.float32, marks=pytest.mark.smoke),
                # Full: larger follow-up coverage
                pytest.param(4_000_000, torch.float16, marks=pytest.mark.full),
                pytest.param(4_000_000, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@ReluFixture
def test_relu_op(n_total: int, dtype: torch.dtype) -> None:
    test = ReluTest(n_total, dtype)
    op = ReluFwdOp()
    test.check(op, *test.gen_inputs())


# Template-based activation ops


class ActivationFixture(FixtureBase):
    """Parametrize over shapes / dtypes for activation ops."""

    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


class ActivationEdgeFixture(FixtureBase):
    """L4 edge-case fixture: fp32, 4K elements."""

    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(4096, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


class UnaryActivationTest(UnaryActivationCase, TestBase):
    pass


def _randn(n: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(n, device=run_device(), dtype=dtype)


def _make_activation_test(n_total, dtype, gen_fn, op_cls, **op_kwargs):
    """Build test, instantiate op, and run check."""
    test = UnaryActivationTest(n_total, dtype, op_cls.__name__, gen_fn=gen_fn, **op_kwargs)
    op = op_cls(**op_kwargs)
    test.check(op, *test.gen_inputs())


@ActivationFixture
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu(n_total: int, dtype: torch.dtype, approximate: str) -> None:
    from tileops.ops.elementwise import GeluFwdOp

    _make_activation_test(
        n_total,
        dtype,
        _randn,
        GeluFwdOp,
        approximate=approximate,
    )


@ActivationFixture
def test_silu(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import SiluFwdOp

    _make_activation_test(n_total, dtype, _randn, SiluFwdOp)


@ActivationFixture
def test_sigmoid(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import SigmoidFwdOp

    _make_activation_test(n_total, dtype, _randn, SigmoidFwdOp)


@ActivationFixture
def test_tanh(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import TanhFwdOp

    _make_activation_test(n_total, dtype, _randn, TanhFwdOp)


@ActivationFixture
def test_hardswish(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import HardswishFwdOp

    _make_activation_test(n_total, dtype, _randn, HardswishFwdOp)


@ActivationFixture
def test_hardsigmoid(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import HardsigmoidFwdOp

    _make_activation_test(n_total, dtype, _randn, HardsigmoidFwdOp)


@ActivationFixture
def test_mish(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import MishFwdOp

    _make_activation_test(n_total, dtype, _randn, MishFwdOp)


@ActivationFixture
def test_selu(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import SeluFwdOp

    _make_activation_test(n_total, dtype, _randn, SeluFwdOp)


# L4 edge-case tests (fp32, 4K)


@ActivationEdgeFixture
def test_sigmoid_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: sigmoid of large negative -> ~0, large positive -> ~1."""
    from tileops.ops.elementwise import SigmoidFwdOp

    def _extreme(n, dtype):
        x = torch.zeros(n, device=run_device(), dtype=dtype)
        x[: n // 2] = -50.0
        x[n // 2 :] = 50.0
        return x

    _make_activation_test(n_total, dtype, _extreme, SigmoidFwdOp)


@ActivationEdgeFixture
def test_tanh_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: tanh saturates to +/-1 for large inputs."""
    from tileops.ops.elementwise import TanhFwdOp

    def _extreme(n, dtype):
        x = torch.zeros(n, device=run_device(), dtype=dtype)
        x[: n // 2] = -50.0
        x[n // 2 :] = 50.0
        return x

    _make_activation_test(n_total, dtype, _extreme, TanhFwdOp)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gelu_tails_are_exact(dtype: torch.dtype) -> None:
    """Edge: GELU reaches exactly 0 below its tail and exactly x above it.

    It scales the 1 - erf(x / sqrt(2)) residual by x, so an erf tail off by eps
    costs |x| * eps / 2 -- unbounded in |x| unless erf saturates exactly. The
    non-finite cases ride on the same saturation: -inf reaches 0 * -inf.
    """
    from tileops.ops.elementwise import GeluFwdOp

    workload = GeluTailWorkload(dtype)
    TestBase.check(workload, GeluFwdOp(), *workload.gen_inputs())


# Independent activation ops


@ActivationFixture
def test_leaky_relu(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import LeakyReluFwdOp

    _make_activation_test(
        n_total,
        dtype,
        _randn,
        LeakyReluFwdOp,
    )


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(("threads", "npt"), [(128, 4), (256, 4), (512, 8)])
def test_leaky_relu_writes_every_element_under_a_tuned_config(threads: int, npt: int) -> None:
    """Every element is written whatever ``threads * npt`` the selected config asks for.

    The builder is handed the default config and the JIT the actual one, so a block extent
    taken from the builder's arguments leaves part of each block untouched.
    """
    from tileops.backend import BUILTIN
    from tileops.kernels.elementwise import LeakyReluFwdKernel
    from tileops.ops.elementwise import LeakyReluFwdOp

    config = {"threads": threads, "num_per_thread": npt}

    class Tuned(LeakyReluFwdKernel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, config=config, **kwargs)

    test = UnaryActivationTest(4096 * 7 + 13, torch.float16, "LeakyReluFwdOp", gen_fn=_randn)
    op = LeakyReluFwdOp(kernel_map={"leaky_relu": Tuned}, target=BUILTIN)
    test.check(op, *test.gen_inputs())
    (kernel,) = op.built_kernels("elementwise").values()
    assert type(kernel) is Tuned
    assert {k: kernel.config[k] for k in config} == config


@ActivationFixture
def test_elu(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import EluFwdOp

    _make_activation_test(
        n_total,
        dtype,
        _randn,
        EluFwdOp,
    )


@ActivationFixture
def test_hardtanh(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import HardtanhFwdOp

    _make_activation_test(
        n_total,
        dtype,
        _randn,
        HardtanhFwdOp,
    )


@ActivationFixture
def test_softplus(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import SoftplusFwdOp

    _make_activation_test(
        n_total,
        dtype,
        _randn,
        SoftplusFwdOp,
    )


class PreluFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


@PreluFixture
def test_prelu(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import PreluFwdOp

    C = 64
    H = n_total // C
    # Shape (1, C, H): batch=1, channels=C, spatial=H
    shape = (1, C, H)
    x = torch.randn(shape, device=run_device(), dtype=dtype)
    weight = torch.randn(C, device=run_device(), dtype=dtype).abs() * 0.1 + 0.01
    ref = F.prelu(x.float(), weight.float()).to(dtype)

    op = PreluFwdOp()
    out = op(x, weight)
    compare_outputs(
        out, ref, ElementwiseWorkload(type(op).__name__, (x, weight)).verification(*(x, weight))
    )


@pytest.mark.smoke
def test_prelu_batch_dim() -> None:
    """PReLU with a leading batch dimension: shape (2, 4, 8)."""
    from tileops.ops.elementwise import PreluFwdOp

    dtype = torch.float32
    shape = (2, 4, 8)
    x = torch.randn(shape, device=run_device(), dtype=dtype)
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4], device=run_device(), dtype=dtype)
    ref = F.prelu(x, weight)
    op = PreluFwdOp()
    out = op(x, weight)
    compare_outputs(
        out, ref, ElementwiseWorkload(type(op).__name__, (x, weight)).verification(*(x, weight))
    )


@pytest.mark.smoke
def test_prelu_rejects_a_weight_that_does_not_match_the_channel_axis() -> None:
    """PReLU applies one weight per channel, so a weight of another length is wrong.

    The channel axis is dim 1 for a rank >= 2 input: a length-4 weight against a
    ``(2, 8, 4)`` input names 4 channels where the tensor has 8, and the per-channel
    weight would be applied to the wrong elements.
    """
    from tileops.ops.elementwise import PreluFwdOp

    dtype = torch.float32
    op = PreluFwdOp()
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4], device=run_device(), dtype=dtype)
    bad = torch.randn((2, 8, 4), device=run_device(), dtype=dtype)
    with pytest.raises(ValueError, match=r"shape_rules"):
        op(bad, weight)
