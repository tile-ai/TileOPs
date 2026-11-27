"""Tests for the unary activation elementwise ops."""

import pytest
import torch
import torch.nn.functional as F

from tests.cuda_graph_contract import check_replay
from tests.launch_bounds import elementwise_launch_bounds, full_occupancy_bounds
from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.elementwise import UnaryKernel
from tileops._ops.elementwise import (
    ELUFwdOp,
    GELUFwdOp,
    HardsigmoidFwdOp,
    HardswishFwdOp,
    HardtanhFwdOp,
    LeakyReLUFwdOp,
    MishFwdOp,
    ReLUFwdOp,
    SELUFwdOp,
    SigmoidFwdOp,
    SiLUFwdOp,
    SoftplusFwdOp,
    TanhFwdOp,
)
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
                pytest.param(
                    1_000_000,
                    torch.float16,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="elementwise")],
                ),
                pytest.param(1_000_000, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1_000_000, torch.float32, marks=pytest.mark.smoke),
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


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "launch,dtype",
    [
        pytest.param("tiny", torch.bfloat16, marks=pytest.mark.smoke),
        pytest.param("ragged", torch.float32, marks=pytest.mark.smoke),
        pytest.param("few-blocks", torch.bfloat16, marks=pytest.mark.full),
        pytest.param("wave", torch.bfloat16, marks=pytest.mark.full),
        pytest.param("wave", torch.float32, marks=pytest.mark.full),
        pytest.param("past-l2", torch.float32, marks=pytest.mark.full),
    ],
)
def test_relu_launch_sizes(launch: str, dtype: torch.dtype) -> None:
    """ReLU remains correct across block-size, alignment and cache-size boundaries."""
    test = ReLUTest(_launch_size(launch, dtype), dtype)
    test.check(ReLUFwdOp(), *test.gen_inputs())


class ActivationFixture(FixtureBase):
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
    test = UnaryActivationTest(n_total, dtype, op_cls.__name__, gen_fn=gen_fn, **op_kwargs)
    op = op_cls(**op_kwargs)
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "op_cls, op_kwargs",
    [
        pytest.param(GELUFwdOp, {"approximate": "none"}, id="gelu"),
        pytest.param(GELUFwdOp, {"approximate": "tanh"}, id="gelu-tanh"),
        pytest.param(SiLUFwdOp, {}, id="silu"),
        pytest.param(SigmoidFwdOp, {}, id="sigmoid"),
        pytest.param(TanhFwdOp, {}, id="tanh"),
        pytest.param(HardswishFwdOp, {}, id="hardswish"),
        pytest.param(HardsigmoidFwdOp, {}, id="hardsigmoid"),
        pytest.param(MishFwdOp, {}, id="mish"),
        pytest.param(SELUFwdOp, {}, id="selu"),
        pytest.param(LeakyReLUFwdOp, {}, id="leaky-relu"),
        pytest.param(ELUFwdOp, {}, id="elu"),
        pytest.param(HardtanhFwdOp, {}, id="hardtanh"),
        pytest.param(SoftplusFwdOp, {}, id="softplus"),
    ],
)
@ActivationFixture
def test_activation(op_cls, op_kwargs: dict, n_total: int, dtype: torch.dtype) -> None:
    _make_activation_test(n_total, dtype, _randn, op_cls, **op_kwargs)


def _launch_bounds(op, dtype: torch.dtype) -> tuple[int, list[tuple[str, str]]]:
    """*op* run on 1M elements of *dtype*, and its program's threads and launch bounds."""
    op(_randn(1_048_576, dtype))
    return elementwise_launch_bounds(op)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_keeps_an_sm_of_threads_resident(approximate: str) -> None:
    """Both GELU forms ask for an SM's full thread count at 32 registers a thread."""
    threads, bounds = _launch_bounds(GELUFwdOp(approximate=approximate), torch.float16)
    assert bounds == full_occupancy_bounds(threads)


@pytest.mark.smoke
def test_float16_silu_every_input() -> None:
    """Every float16 input, -65504 included, where the divisor overflows to infinity."""
    workload = Every16BitCase(torch.float16, "SiLUFwdOp")
    TestBase.check(workload, SiLUFwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tanh_keeps_an_sm_of_threads_resident_for_16bit_inputs(dtype: torch.dtype) -> None:
    """A 16-bit input asks for as many blocks an SM as the register file holds threads at 32
    registers each."""
    threads, bounds = _launch_bounds(TanhFwdOp(), dtype)
    assert bounds == full_occupancy_bounds(threads)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_float32_tanh_leaves_the_register_budget_to_nvcc() -> None:
    """A float32 input asks for one resident block, nvcc's default budget."""
    threads, bounds = _launch_bounds(TanhFwdOp(), torch.float32)
    assert bounds == [(str(threads), "1")]


@ActivationEdgeFixture
def test_sigmoid_edge(n_total: int, dtype: torch.dtype) -> None:
    """Infinities, NaN, signed zeros, subnormals and a divisor past ``2**126``."""
    workload = SigmoidEdgeWorkload(n_total, dtype)
    TestBase.check(workload, SigmoidFwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
def test_sigmoid_partial_last_block() -> None:
    """A last block that ends one vector in: of a thread's two vectors, the second is past N."""
    # 2**18 float32 elements keep eight a thread, two 16-byte vectors; one more vector
    # leaves a 1024-element block holding only it.
    _make_activation_test(2**18 + 4, torch.float32, _randn, SigmoidFwdOp)


@pytest.mark.smoke
def test_sigmoid_input_off_a_vector_boundary() -> None:
    """A contiguous input starting one element past a 16-byte boundary matches the reference."""
    test = UnaryActivationTest(4096, torch.float32, "SigmoidFwdOp", gen_fn=_randn)
    (aligned,) = test.gen_inputs()
    x = torch.empty(aligned.numel() + 1, device=run_device(), dtype=aligned.dtype)[1:]
    x.copy_(aligned)
    test.check(SigmoidFwdOp(), x)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_float32_sigmoid_past_l2() -> None:
    """A float32 input larger than the device's L2, with a partial last block."""
    l2_bytes = torch.cuda.get_device_properties(run_device()).L2_cache_size
    _make_activation_test(l2_bytes // 4 + 4, torch.float32, _randn, SigmoidFwdOp)


@pytest.mark.smoke
def test_sigmoid_tuned() -> None:
    """Tuned mode picks a launch whose result matches the reference."""
    test = UnaryActivationTest(2**18 + 4, torch.float32, "SigmoidFwdOp", gen_fn=_randn)
    op = SigmoidFwdOp()
    op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.parametrize(
    "op_type, n_total",
    [
        pytest.param(
            ReLUFwdOp,
            64 * (4096 // 4),
            marks=[pytest.mark.smoke, pytest.mark.cuda_graph(op="ReLUFwdOp")],
            id="deepseek-v4-flash-indexer-scores",
        ),
        pytest.param(
            ReLUFwdOp,
            64 * (4128 // 4),
            marks=[pytest.mark.full, pytest.mark.cuda_graph(op="ReLUFwdOp")],
            id="deepseek-v4-flash-indexer-scores-4128",
        ),
        pytest.param(
            SigmoidFwdOp,
            24 * 256,
            marks=[pytest.mark.full, pytest.mark.cuda_graph(op="SigmoidFwdOp")],
            id="qwen3.8-27b-attention-gate",
        ),
    ],
)
def test_activation_graph_replay(op_type, n_total: int) -> None:
    """A bfloat16 activation of one decode token: DeepSeek-V4-Flash's indexer scores rectified,
    Qwen3.8-27B's attention gate."""
    check_replay(UnaryActivationTest(n_total, torch.bfloat16, op_type.__name__), op_type())


@ActivationEdgeFixture
def test_tanh_edge(n_total: int, dtype: torch.dtype) -> None:
    """tanh saturates to +/-1 for large inputs."""

    def _extreme(n, dtype):
        x = torch.zeros(n, device=run_device(), dtype=dtype)
        x[: n // 2] = -50.0
        x[n // 2 :] = 50.0
        return x

    _make_activation_test(n_total, dtype, _extreme, TanhFwdOp)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gelu_tails_are_exact(dtype: torch.dtype) -> None:
    """GELU reaches exactly 0 below its tail and exactly x above it, infinities included."""
    workload = GELUTailWorkload(dtype)
    TestBase.check(workload, GELUFwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gelu_keeps_negative_inputs_nonpositive(dtype: torch.dtype) -> None:
    """GELU of a negative input is never positive, at every 16-bit input."""
    workload = GELUSignWorkload(dtype)
    TestBase.check(workload, GELUFwdOp(), *workload.gen_inputs())


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
    shape = (1, C, n_total // C)
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
    """A weight whose length is not the channel extent (dim 1) is rejected."""
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
    test = UnaryActivationTest(4096 * 7 + 13, torch.float16, "LeakyReLUFwdOp", gen_fn=_randn)
    op = LeakyReLUFwdOp()
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())
