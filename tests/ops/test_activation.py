"""Tests for unary activation elementwise ops.

Covers L1 smoke correctness, multi-dtype coverage, and L4 edge cases.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.elementwise import UnaryKernel
from tileops._ops.elementwise import ReLUFwdOp
from workloads.device import run_device
from workloads.elementwise import (
    ElementwiseWorkload,
    Every16BitCase,
    Exhaustive16BitWorkload,
    Float32SpecialValuesWorkload,
    GELUSignWorkload,
    GELUTailWorkload,
    ReLUWorkload,
    SigmoidEdgeWorkload,
    UnaryActivationCase,
)
from workloads.numerics import compare_outputs


class ReLUTest(ReLUWorkload, TestBase):
    pass


class ReLUFixture(FixtureBase):
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


@ReLUFixture
def test_relu_op(n_total: int, dtype: torch.dtype) -> None:
    test = ReLUTest(n_total, dtype)
    op = ReLUFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_name, dtype",
    [
        # NaN payloads and signed zeros.
        ("ReLUFwdOp", torch.float16),
        ("ReLUFwdOp", torch.bfloat16),
        # x near -88, where the divisor passes 2**126.
        ("SiLUFwdOp", torch.bfloat16),
        # A two-ulp float32 divide alone moves two float16 results by one ulp, so float16
        # refines its reciprocal by one Newton step; bfloat16 rounds the fast divide to
        # torch's value everywhere.
        ("SigmoidFwdOp", torch.float16),
        ("SigmoidFwdOp", torch.bfloat16),
    ],
)
def test_matches_torch_bit_for_bit_at_every_16bit_input(op_name: str, dtype: torch.dtype) -> None:
    """The op gives torch's result bit for bit at every 16-bit input."""
    import tileops._ops.elementwise as ops

    workload = Exhaustive16BitWorkload(op_name, dtype)
    TestBase.check(workload, getattr(ops, op_name)(), *workload.gen_inputs())


@pytest.mark.smoke
def test_float32_relu_special_values_in_place() -> None:
    """Float32 special values match torch, except NaN payloads may differ.

    In place, because the out-of-place call returns the same kernel's result uncopied.
    """
    workload = Float32SpecialValuesWorkload("ReLUFwdOp", repeats=256)
    (x,) = workload.gen_inputs()
    expected = workload.ref_program(x.clone())
    y = ReLUFwdOp(inplace=True)(x)
    assert y is x
    compare_outputs(y, expected, workload.verification(x))


def _launch_size(launch: str, dtype: torch.dtype) -> int:
    """Input sizes covering launch-policy boundaries on the test device."""
    props = torch.cuda.get_device_properties(run_device())
    resident = props.multi_processor_count * props.max_threads_per_multi_processor
    run = 256 * 64 // dtype.itemsize
    wave = 20 * resident // run * run
    if launch in ("wave", "ragged") and not UnaryKernel.fits_l2_budget(
        wave, 2 * dtype.itemsize, props.L2_cache_size
    ):
        pytest.skip("twenty elements per resident thread exceed three quarters of this L2")
    return {
        "tiny": 2048,
        "few-blocks": 4096,
        "wave": wave,
        "ragged": wave + 3,
        "past-l2": props.L2_cache_size // (2 * dtype.itemsize) + 4096,
    }[launch]


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "launch,dtype",
    [
        ("tiny", torch.bfloat16),
        ("few-blocks", torch.bfloat16),
        ("wave", torch.bfloat16),
        ("wave", torch.float32),
        ("ragged", torch.float32),
        ("past-l2", torch.float32),
    ],
)
def test_relu_launch_sizes(launch: str, dtype: torch.dtype) -> None:
    """ReLU remains correct across block-size, alignment and cache-size boundaries."""
    test = ReLUTest(_launch_size(launch, dtype), dtype)
    test.check(ReLUFwdOp(), *test.gen_inputs())


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
    from tileops._ops.elementwise import GELUFwdOp

    _make_activation_test(
        n_total,
        dtype,
        _randn,
        GELUFwdOp,
        approximate=approximate,
    )


@ActivationFixture
def test_silu(n_total: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import SiLUFwdOp

    _make_activation_test(n_total, dtype, _randn, SiLUFwdOp)


@pytest.mark.smoke
def test_float16_silu_every_input() -> None:
    """Every float16 input, -65504 included, where the divisor overflows to infinity."""
    from tileops._ops.elementwise import SiLUFwdOp

    workload = Every16BitCase(torch.float16, "SiLUFwdOp")
    TestBase.check(workload, SiLUFwdOp(), *workload.gen_inputs())


@ActivationFixture
def test_sigmoid(n_total: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import SigmoidFwdOp

    _make_activation_test(n_total, dtype, _randn, SigmoidFwdOp)


@ActivationFixture
def test_tanh(n_total: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import TanhFwdOp

    _make_activation_test(n_total, dtype, _randn, TanhFwdOp)


@ActivationFixture
def test_hardswish(n_total: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import HardswishFwdOp

    _make_activation_test(n_total, dtype, _randn, HardswishFwdOp)


@ActivationFixture
def test_hardsigmoid(n_total: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import HardsigmoidFwdOp

    _make_activation_test(n_total, dtype, _randn, HardsigmoidFwdOp)


@ActivationFixture
def test_mish(n_total: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import MishFwdOp

    _make_activation_test(n_total, dtype, _randn, MishFwdOp)


@ActivationFixture
def test_selu(n_total: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import SELUFwdOp

    _make_activation_test(n_total, dtype, _randn, SELUFwdOp)


# L4 edge-case tests (fp32, 4K)


@ActivationEdgeFixture
def test_sigmoid_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: infinities, NaN, signed zeros, subnormals and a divisor past ``2**126``."""
    from tileops._ops.elementwise import SigmoidFwdOp

    workload = SigmoidEdgeWorkload(n_total, dtype)
    TestBase.check(workload, SigmoidFwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
def test_sigmoid_partial_last_block() -> None:
    """A last block that ends one vector in: of a thread's two vectors, the second is past N."""
    from tileops._ops.elementwise import SigmoidFwdOp

    # 2**18 float32 elements keep eight a thread, two 16-byte vectors; one more vector
    # leaves a 1024-element block holding only it.
    _make_activation_test(2**18 + 4, torch.float32, _randn, SigmoidFwdOp)


@pytest.mark.smoke
def test_sigmoid_input_off_a_vector_boundary() -> None:
    """A contiguous input starting one element past a 16-byte boundary matches the reference."""
    from tileops._ops.elementwise import SigmoidFwdOp

    test = UnaryActivationTest(4096, torch.float32, "SigmoidFwdOp", gen_fn=_randn)
    (aligned,) = test.gen_inputs()
    x = torch.empty(aligned.numel() + 1, device=run_device(), dtype=aligned.dtype)[1:]
    x.copy_(aligned)
    test.check(SigmoidFwdOp(), x)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_float32_sigmoid_past_l2() -> None:
    """A float32 input larger than the device's L2, with a partial last block."""
    from tileops._ops.elementwise import SigmoidFwdOp

    l2_bytes = torch.cuda.get_device_properties(run_device()).L2_cache_size
    _make_activation_test(l2_bytes // 4 + 4, torch.float32, _randn, SigmoidFwdOp)


@pytest.mark.smoke
def test_sigmoid_tuned() -> None:
    """Tuned mode picks a launch whose result matches the reference."""
    from tileops._ops.elementwise import SigmoidFwdOp

    test = UnaryActivationTest(2**18 + 4, torch.float32, "SigmoidFwdOp", gen_fn=_randn)
    op = SigmoidFwdOp()
    op.request_tune()
    test.check(op, *test.gen_inputs())


@ActivationEdgeFixture
def test_tanh_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: tanh saturates to +/-1 for large inputs."""
    from tileops._ops.elementwise import TanhFwdOp

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
    from tileops._ops.elementwise import GELUFwdOp

    workload = GELUTailWorkload(dtype)
    TestBase.check(workload, GELUFwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gelu_keeps_negative_inputs_nonpositive(dtype: torch.dtype) -> None:
    """Edge: GELU of a negative input is never positive, at every 16-bit input.

    The erf polynomial overshoots 1 by up to 3.2e-6 short of its clamp. There 1 + erf
    turns negative unless clipped, and x / 2 scales it into a positive result.
    """
    from tileops._ops.elementwise import GELUFwdOp

    workload = GELUSignWorkload(dtype)
    TestBase.check(workload, GELUFwdOp(), *workload.gen_inputs())


# Independent activation ops


@ActivationFixture
def test_leaky_relu(n_total: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import LeakyReLUFwdOp

    _make_activation_test(
        n_total,
        dtype,
        _randn,
        LeakyReLUFwdOp,
    )


@ActivationFixture
def test_elu(n_total: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import ELUFwdOp

    _make_activation_test(
        n_total,
        dtype,
        _randn,
        ELUFwdOp,
    )


@ActivationFixture
def test_hardtanh(n_total: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import HardtanhFwdOp

    _make_activation_test(
        n_total,
        dtype,
        _randn,
        HardtanhFwdOp,
    )


@ActivationFixture
def test_softplus(n_total: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import SoftplusFwdOp

    _make_activation_test(
        n_total,
        dtype,
        _randn,
        SoftplusFwdOp,
    )


class PReLUFixture(FixtureBase):
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


@PReLUFixture
def test_prelu(n_total: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import PReLUFwdOp

    C = 64
    H = n_total // C
    # Shape (1, C, H): batch=1, channels=C, spatial=H
    shape = (1, C, H)
    x = torch.randn(shape, device=run_device(), dtype=dtype)
    weight = torch.randn(C, device=run_device(), dtype=dtype).abs() * 0.1 + 0.01
    ref = F.prelu(x.float(), weight.float()).to(dtype)

    op = PReLUFwdOp()
    out = op(x, weight)
    compare_outputs(
        out, ref, ElementwiseWorkload(type(op).__name__, (x, weight)).verification(*(x, weight))
    )


@pytest.mark.smoke
def test_prelu_batch_dim() -> None:
    """PReLU with a leading batch dimension: shape (2, 4, 8)."""
    from tileops._ops.elementwise import PReLUFwdOp

    dtype = torch.float32
    shape = (2, 4, 8)
    x = torch.randn(shape, device=run_device(), dtype=dtype)
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4], device=run_device(), dtype=dtype)
    ref = F.prelu(x, weight)
    op = PReLUFwdOp()
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
    from tileops._ops.elementwise import PReLUFwdOp

    dtype = torch.float32
    op = PReLUFwdOp()
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4], device=run_device(), dtype=dtype)
    bad = torch.randn((2, 8, 4), device=run_device(), dtype=dtype)
    with pytest.raises(ValueError, match=r"shape_rules"):
        op(bad, weight)


@pytest.mark.parametrize(
    "tune",
    [pytest.param(False, marks=pytest.mark.smoke), pytest.param(True, marks=pytest.mark.full)],
)
def test_gelu_short_row(tune: bool) -> None:
    """A row the grid-filling shrink reaches: two elements a thread, across a block and its tail."""
    from tileops._ops.elementwise import GELUFwdOp

    test = UnaryActivationTest(257, torch.float32, "GELUFwdOp", gen_fn=_randn)
    op = GELUFwdOp()
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "tune",
    [pytest.param(False, marks=pytest.mark.smoke), pytest.param(True, marks=pytest.mark.full)],
)
def test_leaky_relu_under_tuning(tune: bool) -> None:
    from tileops._ops.elementwise import LeakyReLUFwdOp

    test = UnaryActivationTest(4096 * 7 + 13, torch.float16, "LeakyReLUFwdOp", gen_fn=_randn)
    op = LeakyReLUFwdOp()
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())
