"""Tests for RMSNorm and fused-add RMSNorm."""

import re

import pytest
import torch
import torch.nn.functional as F

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tests.cuda_graph_contract import check_replay
from tests.misaligned import misaligned
from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.norm import rms_norm_staged
from tileops._kernels.norm.rms_norm import (
    RMSNormPairedKernel,
    RMSNormTwoPassKernel,
    _paired_width,
    _two_pass_shape,
)
from tileops._kernels.tiling import ALIGNMENT
from tileops._ops.norm.fused_add_rms_norm import FusedAddRMSNormFwdOp
from tileops._ops.norm.rms_norm import RMSNormFwdOp
from tileops._utils import RESIDENT_CTAS, get_shared_memory_optin, get_sm_count
from tileops.backend._dispatch import device_facts_of
from workloads.device import run_device
from workloads.norm import (
    FusedAddRMSNormWorkload,
    FusedAddRMSNormZeroCenteredOffsetWorkload,
    NormCall,
    RMSNormGroupedCall,
    RMSNormGroupedOffsetCall,
    RMSNormWorkload,
    RMSNormZeroCenteredOffsetWorkload,
    norm_verification,
    normalization_verification,
)
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call

register_compile_contract(RMSNormFwdOp)


class RMSNormTest(RMSNormWorkload, TestBase):
    pass


class RMSNormFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                # Standard aligned shapes; bf16 stands for both 2-byte dtypes past smoke.
                pytest.param(
                    1024,
                    4096,
                    torch.float16,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="norm")],
                ),
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(8192, 8192, torch.bfloat16, marks=pytest.mark.full),
                # Non-aligned N (AC required)
                pytest.param(1024, 3000, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(2048, 5120, torch.bfloat16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m (proves T.copy partial block safety)
                pytest.param(1025, 4096, torch.bfloat16, marks=pytest.mark.full),
                # A short unaligned row: several share a block, with a tail block.
                pytest.param(17, 96, torch.bfloat16, marks=pytest.mark.full),
                # Rows a group of a warp's lanes holds, with a tail; 64 rows of 2048, which a
                # block a row holds on a device of at least 64 SMs; and a row read twice.
                pytest.param(33, 256, torch.float16, marks=pytest.mark.full),
                pytest.param(64, 2048, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(4, 32768, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@RMSNormFixture
def test_rms_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = RMSNormTest(m, n, dtype)
    op = RMSNormFwdOp(normalized_shape=(n,))
    test.check(op, *test.gen_inputs())


def _view(view: str, n: int, dtype: torch.dtype) -> torch.Tensor:
    """A strided ``(1024, n)`` view, or a ``(2, 512, n)`` input whose leading axes fold."""
    if view == "strided":
        return torch.randn(1024, 2 * n, dtype=dtype, device=run_device())[:, :n]
    return torch.randn(2, 512, n, dtype=dtype, device=run_device())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "view, dtype", [("strided", torch.float16), ("3d", torch.bfloat16)], ids=["strided", "3d"]
)
def test_rms_norm_views(view: str, dtype: torch.dtype) -> None:
    n = 4096
    x = _view(view, n, dtype)
    weight = torch.randn(n, dtype=dtype, device=run_device())
    expected = F.rms_norm(x.float(), (n,), weight.float(), eps=1e-6).to(dtype)
    y = RMSNormFwdOp(normalized_shape=(n,))(x, weight)
    compare_outputs(y, expected, norm_verification(dtype))


@pytest.mark.sm90
@pytest.mark.cuda_graph(op="RMSNormFwdOp")
@pytest.mark.parametrize(
    "T, B, n, row",
    [
        pytest.param("float16", 1024, 4096, {}, marks=pytest.mark.smoke, id="fp16"),
        pytest.param(
            "bfloat16",
            3,
            10240,
            {"zero_centered": True, "group_size": 2560},
            marks=pytest.mark.smoke,
            id="grouped",
        ),
        *(
            pytest.param("bfloat16", *case, marks=pytest.mark.full, id=f"deepseek-v4-flash-{name}")
            for name, *case in (
                ("hidden", 1, 4096, {}),
                ("query-lora", 1, 1024, {}),
                ("kv", 1, 512, {}),
                ("indexer-entry", 1, 128, {}),
                ("query-heads", 64, 512, {"some": []}),
            )
        ),
        pytest.param(
            "bfloat16", 1, 5120, {"zero_centered": True}, marks=pytest.mark.full, id="qwen3.8-27b"
        ),
    ],
)
def test_rms_norm_graph_replay(T, B, n, row) -> None:
    """Replay consumes changed input and weight contents at their captured addresses."""
    row = {"eps": 1e-6, "some": ["weight"], **row}
    call = manifest_call("RMSNormFwdOp", {"T": T}, B=[B], normalized_shape=[n], **row)
    check_replay(NormCall(call), RMSNormFwdOp(**call.arguments()))


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_a_cold_op_traces_fullgraph_and_matches_eager() -> None:
    """Cold is the whole contract: a warm op has nothing left for dynamo to trace into."""
    op = RMSNormFwdOp(normalized_shape=(4096,))
    x = torch.randn(64, 4096, dtype=torch.float16, device=run_device())
    weight = torch.randn(4096, dtype=torch.float16, device=run_device())

    assert torch.equal(torch.compile(op, fullgraph=True)(x, weight), op(x, weight))


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_the_traced_graph_holds_only_this_ops_operator() -> None:
    """The node is the op's, so replacing the kernel cannot change the graph."""
    op = RMSNormFwdOp(normalized_shape=(256,))
    x = torch.randn(8, 256, dtype=torch.float16, device=run_device())
    weight = torch.randn(256, dtype=torch.float16, device=run_device())

    assert_op_owns_graph_nodes(op, x, weight)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_a_non_contiguous_input_compiles_to_the_shape_the_fake_promised() -> None:
    """The fake speaks before the body normalizes contiguity, so it promises contiguous."""
    op = RMSNormFwdOp(normalized_shape=(256,))
    x = torch.randn(8, 512, dtype=torch.float16, device=run_device())[:, ::2]
    weight = torch.randn(256, dtype=torch.float16, device=run_device())
    assert not x.is_contiguous()

    output = torch.compile(op, fullgraph=True)(x, weight)

    assert output.is_contiguous()
    assert torch.equal(output, op(x, weight))


@pytest.mark.smoke
def test_an_unaligned_row_comes_back_contiguous() -> None:
    """A row width that is not a multiple of the alignment still returns a contiguous output."""
    x = torch.randn(3, 96, dtype=torch.float16, device=run_device())
    assert RMSNormFwdOp(normalized_shape=(96,))(x).is_contiguous()


@pytest.mark.smoke
def test_no_weight_and_no_eps_match_torch() -> None:
    """An absent weight scales by one; ``eps=None`` is torch's float32 machine epsilon."""
    x = torch.full((2, 4), 1e-3, dtype=torch.float16, device=run_device())
    compare_outputs(
        RMSNormFwdOp(normalized_shape=(4,))(x),
        F.rms_norm(x, [4]),
        normalization_verification("RMSNormFwdOp", x.dtype),
    )


@pytest.mark.parametrize(
    "n,many_rows",
    [
        # Warp rows: four rows a warp at 256, three vectors a lane at 768, and the bound 2048
        # past one row an SM, weighted; the widest held row; a row of 12288 read twice past
        # four rows an SM, weighted, and held in registers below them; a row of 32768 read
        # twice.
        pytest.param(256, False, marks=pytest.mark.smoke),
        pytest.param(768, False, marks=pytest.mark.smoke),
        pytest.param(2048, True, marks=pytest.mark.smoke),
        pytest.param(8192, False, marks=pytest.mark.smoke),
        pytest.param(12288, True, marks=[pytest.mark.smoke, pytest.mark.in_tree_kernels]),
        pytest.param(12288, False, marks=pytest.mark.smoke),
        pytest.param(32768, False, marks=pytest.mark.smoke),
        # Warp rows at two vectors a lane, two rows a lane group, the second row of the last
        # group past the end; the fragment body just past the warp-row bound; and
        # the widest row read twice one vector a thread at a time, weighted.
        pytest.param(512, False, marks=pytest.mark.full),
        pytest.param(2304, False, marks=pytest.mark.full),
        pytest.param(16384, True, marks=[pytest.mark.full, pytest.mark.in_tree_kernels]),
    ],
)
def test_tuned_rms_norm_in_each_row_body(n: int, many_rows: bool) -> None:
    """Each body a row width and row count select is right tuned; the launches of more than
    four rows an SM carry a weight, the others scale by one."""
    rows = 4 * device_facts_of(torch.device(run_device())).sm_count + 1 if many_rows else 5
    x = torch.randn(rows, n, dtype=torch.bfloat16, device=run_device())
    weight = torch.randn(n, dtype=torch.bfloat16, device=run_device()) if many_rows else None
    op = RMSNormFwdOp(normalized_shape=(n,), eps=1e-6)
    op.request_tune()
    ref_weight = None if weight is None else weight.float()
    compare_outputs(
        op(x, weight),
        F.rms_norm(x.float(), (n,), ref_weight, eps=1e-6).to(x.dtype),
        normalization_verification("RMSNormFwdOp", x.dtype),
    )


class RMSNormZeroCenteredOffsetTest(RMSNormZeroCenteredOffsetWorkload, TestBase):
    pass


@pytest.mark.parametrize(
    "m, n",
    [
        # A held row; warp rows with a lane group past the last row; rows of the fragment body
        # sharing a block.
        pytest.param(5, 8192, marks=pytest.mark.smoke),
        pytest.param(33, 256, marks=pytest.mark.full),
        pytest.param(17, 96, marks=pytest.mark.full),
    ],
)
def test_rms_norm_zero_centered_in_each_row_body(m: int, n: int) -> None:
    """Each body scales a zero-centered row by one plus the weight, formed in float32: rounded to
    bfloat16 first, it moves outputs these inputs hold exactly."""
    test = RMSNormZeroCenteredOffsetTest(m, n, torch.bfloat16, zero_centered=True)
    op = RMSNormFwdOp(normalized_shape=(n,), eps=1e-6, zero_centered=True)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_rms_norm_zero_centered_float16() -> None:
    test = RMSNormTest(1024, 4096, torch.float16, zero_centered=True)
    op = RMSNormFwdOp(normalized_shape=(4096,), eps=1e-6, zero_centered=True)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_rms_norm_zero_centered_without_weight() -> None:
    """A zero-centered call without a weight scales by one, as by a zero weight's 1 + 0."""
    test = RMSNormTest(5, 256, torch.bfloat16, zero_centered=True)
    x, _ = test.gen_inputs()
    op = RMSNormFwdOp(normalized_shape=(256,), eps=1e-6, zero_centered=True)
    compare_outputs(op(x), test.ref_program(x, None), test.verification(x))


class RMSNormGroupedTest(RMSNormGroupedCall, TestBase):
    pass


@pytest.mark.parametrize(
    "dtype, batch, shape, group, centered, weighted, tune",
    [
        pytest.param(
            "float16", [3], [3, 33], 33, False, True, False, marks=pytest.mark.smoke, id="fp16-tail"
        ),
        pytest.param(
            "bfloat16", [5], [12], 1, True, True, False, marks=pytest.mark.smoke, id="unit-group"
        ),
        pytest.param(
            "bfloat16",
            [2, 3],
            [4, 2560],
            2560,
            True,
            True,
            False,
            marks=pytest.mark.full,
            id="trailing-axes",
        ),
        pytest.param(
            "bfloat16",
            [1],
            [10240],
            2560,
            True,
            True,
            False,
            marks=pytest.mark.full,
            id="decode-one",
        ),
        pytest.param(
            "bfloat16",
            [64],
            [10240],
            2560,
            True,
            True,
            True,
            marks=pytest.mark.full,
            id="decode-tuned",
        ),
        pytest.param(
            "float16",
            [7],
            [4, 130],
            130,
            True,
            False,
            False,
            marks=pytest.mark.full,
            id="no-weight-tail",
        ),
    ],
)
def test_rms_norm_grouped(dtype, batch, shape, group, centered, weighted, tune) -> None:
    """Each group has independent statistics and its own slice of the full-width weight."""
    call = manifest_call(
        "RMSNormFwdOp",
        {"T": dtype},
        B=batch,
        normalized_shape=shape,
        group_size=group,
        zero_centered=centered,
        eps=1e-6,
        some=["weight"] if weighted else [],
    )
    test = RMSNormGroupedTest(call)
    op = RMSNormFwdOp(**call.arguments())
    inputs = test.gen_inputs()
    if tune:
        op.request_tune()
        op(*inputs)
    test.check(op, *inputs)


@pytest.mark.smoke
def test_rms_norm_grouped_changing_row_count() -> None:
    """One op must not reuse stale row bounds when the input size changes and changes back."""
    op = RMSNormFwdOp(normalized_shape=(10240,), group_size=2560, eps=1e-6, zero_centered=True)
    for rows in (3, 4, 3):
        call = manifest_call(
            "RMSNormFwdOp",
            {"T": "bfloat16"},
            B=[rows],
            normalized_shape=[10240],
            group_size=2560,
            zero_centered=True,
            eps=1e-6,
            some=["weight"],
        )
        test = RMSNormGroupedTest(call)
        test.check(op, *test.gen_inputs())


class RMSNormGroupedOffsetTest(RMSNormGroupedOffsetCall, TestBase):
    pass


@pytest.mark.smoke
def test_rms_norm_grouped_rounds_once() -> None:
    call = manifest_call(
        "RMSNormFwdOp",
        {"T": "bfloat16"},
        B=[3],
        normalized_shape=[10240],
        group_size=2560,
        zero_centered=True,
        eps=1e-6,
        some=["weight"],
    )
    test = RMSNormGroupedOffsetTest(call)
    test.check(RMSNormFwdOp(**call.arguments()), *test.gen_inputs())


def _check_without_weight(test: RMSNormTest, op: RMSNormFwdOp) -> None:
    """*op* scaling by one matches *test*'s reference without a weight."""
    x, _ = test.gen_inputs()
    compare_outputs(op(x), test.ref_program(x, None), test.verification(x))


def _check_served_by(
    rows: int, n: int, dtype: torch.dtype, kernel: str, zero_centered: bool = False
) -> Kernel:
    """*rows* rows of *n* are right with and without a weight, and *kernel* alone serves them;
    returns that kernel."""
    test = RMSNormTest(rows, n, dtype, zero_centered=zero_centered)
    op = RMSNormFwdOp(normalized_shape=(n,), eps=1e-6, zero_centered=zero_centered)
    test.check(op, *test.gen_inputs())
    _check_without_weight(test, op)
    (served,) = op._entries_built("rms_norm").values()
    assert type(served).__name__ == kernel
    return served


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, marks=pytest.mark.sm90, id="bf16-sm90"),
        pytest.param(torch.float16, marks=pytest.mark.sm90, id="fp16-sm90"),
        pytest.param(torch.bfloat16, marks=pytest.mark.sm80, id="bf16-sm80"),
        pytest.param(torch.float16, marks=pytest.mark.sm80, id="fp16-sm80"),
    ],
)
def test_rms_norm_one_wave_of_row_pairs(dtype: torch.dtype) -> None:
    """The staged kernel handles an odd final row, with and without affine weights."""
    sm_count = torch.cuda.get_device_properties(run_device()).multi_processor_count
    wave = rms_norm_staged._CTAS_PER_SM * sm_count
    _check_served_by(rms_norm_staged._ROWS * wave - 1, 3072, dtype, "RMSNormStagedKernel")


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "n",
    [
        pytest.param(2048, id="thread-bound", marks=pytest.mark.smoke),
        pytest.param(1024, id="cap-bound", marks=pytest.mark.full),
    ],
)
def test_rms_norm_many_waves_of_held_rows(n: int) -> None:
    """The paired kernel handles sustained launches with and without affine weights, and a
    block the resident-block cap bounds asks the compiler to fit that cap of blocks an SM while
    a block the SM's threads bound asks for none."""
    props = torch.cuda.get_device_properties(run_device())
    threads = _paired_width(n, torch.bfloat16.itemsize)
    per_sm = props.max_threads_per_multi_processor // threads
    # At least as many blocks an SM as its threads admit, so at least the kernel's waves.
    rows = RMSNormPairedKernel._MIN_WAVES * per_sm * props.multi_processor_count
    kernel = _check_served_by(rows, n, torch.bfloat16, "RMSNormPairedKernel")
    sources = [program.get_kernel_source() for program in kernel._generation.programs.values()]
    cap = RESIDENT_CTAS[props.major * 10 + props.minor]
    bound = f"{threads}, {cap if per_sm >= cap else 1}"
    assert {b for s in sources for b in re.findall(r"__launch_bounds__\(([\d, ]+)\)", s)} == {bound}


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "n,dtype,zero_centered,served",
    [
        pytest.param(
            2048,
            torch.bfloat16,
            True,
            "RMSNormPairedKernel",
            id="block-row-zero-centered",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            2048,
            torch.float16,
            True,
            "RMSNormPairedKernel",
            id="block-row-zero-centered-fp16",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            4096,
            torch.bfloat16,
            False,
            "RMSNormPairedKernel",
            id="block-row",
            marks=pytest.mark.full,
        ),
        pytest.param(
            1024, torch.bfloat16, False, "RMSNormKernel", id="warp-held", marks=pytest.mark.full
        ),
    ],
)
def test_rms_norm_one_row_an_sm(
    n: int, dtype: torch.dtype, zero_centered: bool, served: str
) -> None:
    """A launch of one row an SM is right with and without a weight. The paired kernel takes
    it where a warp row's lane group cannot hold the row within the vectors it packs; one row
    more, or a row a warp holds, stays with ``RMSNormKernel``."""
    sm_count = torch.cuda.get_device_properties(run_device()).multi_processor_count
    for rows, kernel in ((sm_count, served), (sm_count + 1, "RMSNormKernel")):
        _check_served_by(rows, n, dtype, kernel, zero_centered)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_rms_norm_zero_centered_many_waves_of_held_rows_stay_with_the_row_kernel() -> None:
    """A zero-centered launch of the paired kernel's many waves stays with ``RMSNormKernel``,
    with and without a weight: only a launch of at most one row an SM takes a zero-centered
    call to the paired kernel."""
    props = torch.cuda.get_device_properties(run_device())
    threads = _paired_width(2048, torch.bfloat16.itemsize)
    per_sm = min(
        props.max_threads_per_multi_processor // threads,
        RESIDENT_CTAS[props.major * 10 + props.minor],
    )
    rows = RMSNormPairedKernel._MIN_WAVES * per_sm * props.multi_processor_count
    _check_served_by(rows, 2048, torch.bfloat16, "RMSNormKernel", zero_centered=True)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_rms_norm_untuned_warp_rows_take_the_fewest_threads_that_fill_the_sm() -> None:
    """Untuned warp rows run on the fewest threads a block whose resident-block cap still fills
    the SM's threads, and are right with and without a weight past the last full block."""
    props = torch.cuda.get_device_properties(run_device())
    threads = props.max_threads_per_multi_processor // RESIDENT_CTAS[props.major * 10 + props.minor]
    test = RMSNormTest(33, 256, torch.bfloat16)
    op = RMSNormFwdOp(normalized_shape=(256,), eps=1e-6)
    test.check(op, *test.gen_inputs())
    _check_without_weight(test, op)
    (kernel,) = op._entries_built("rms_norm").values()
    assert kernel.config["threads"] == threads


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_rms_norm_rows_of_two_vectors_a_thread_in_long_launches() -> None:
    """Rows of two 16-byte vectors a thread are read twice from ``_MIN_WAVES`` waves on."""
    props = torch.cuda.get_device_properties(run_device())
    _, _, blocks, _ = _two_pass_shape(
        8192, torch.bfloat16.itemsize, props.max_threads_per_multi_processor
    )
    rows = RMSNormTwoPassKernel._MIN_WAVES * blocks * props.multi_processor_count
    _check_served_by(rows, 8192, torch.bfloat16, "RMSNormTwoPassKernel")


@pytest.mark.parametrize(
    "rows,n,dtype,has_weight",
    [
        # On SM90 these four reach the on-chip kernel: one CTA staging by cp.async, a
        # cluster of two, and one CTA read synchronously with and without weight.
        pytest.param(
            3, 131072, torch.float16, True, id="on-chip-async-fp16", marks=pytest.mark.smoke
        ),
        pytest.param(
            3, 262144, torch.bfloat16, True, id="on-chip-cluster-bf16", marks=pytest.mark.smoke
        ),
        pytest.param(3, 65536, torch.float16, True, id="on-chip-fp16", marks=pytest.mark.full),
        pytest.param(
            3, 65536, torch.bfloat16, False, id="on-chip-no-weight", marks=pytest.mark.full
        ),
        # Widths the on-chip kernel cannot split reach the streaming kernel: an odd width,
        # and more rows than resident CTAs, so each CTA walks several rows.
        pytest.param(3, 131073, torch.bfloat16, False, id="tail-no-weight", marks=pytest.mark.full),
        pytest.param(600, 131080, torch.bfloat16, True, id="rows-per-cta", marks=pytest.mark.full),
    ],
)
def test_rms_norm_rows_exceeding_shared_memory(rows, n, dtype, has_weight) -> None:
    x = torch.randn(rows, n, device=run_device(), dtype=dtype)
    weight = torch.randn(n, device=x.device, dtype=dtype) if has_weight else None
    expected = F.rms_norm(x.float(), (n,), None if weight is None else weight.float(), eps=1e-6).to(
        dtype
    )
    op = RMSNormFwdOp(normalized_shape=(n,), eps=1e-6)
    op.request_tune()
    actual = op(x, weight)
    compare_outputs(actual, expected, normalization_verification("RMSNormFwdOp", x.dtype))


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_rms_norm_widest_row_past_the_sm_count() -> None:
    # The widest row shared memory holds, at more rows than SMs: staged beside the reduction's
    # scratch it would pass the budget, so the row kernel holds it in registers.
    n = get_shared_memory_optin() // 2 // ALIGNMENT * ALIGNMENT
    x = torch.randn(get_sm_count() + 1, n, device=run_device(), dtype=torch.float16)
    weight = torch.randn(n, device=x.device, dtype=x.dtype)
    expected = F.rms_norm(x.float(), (n,), weight.float(), eps=1e-6).to(x.dtype)
    actual = RMSNormFwdOp(normalized_shape=(n,), eps=1e-6)(x, weight)
    compare_outputs(actual, expected, normalization_verification("RMSNormFwdOp", x.dtype))


@pytest.mark.parametrize(
    "n",
    [
        pytest.param(4096, id="register-row", marks=pytest.mark.smoke),
        pytest.param(131080, id="streaming", marks=pytest.mark.full),
        # The on-chip kernel on SM90, the streaming kernel before it.
        pytest.param(131072, id="on-chip", marks=pytest.mark.full),
    ],
)
def test_rms_norm_reads_an_input_and_weight_off_the_vector_boundary(n: int) -> None:
    """A contiguous input and weight may start anywhere in their storage."""
    dtype = torch.bfloat16
    x = misaligned(torch.randn(2, n, device=run_device(), dtype=dtype))
    weight = misaligned(torch.randn(n, device=x.device, dtype=dtype))
    expected = F.rms_norm(x.float(), (n,), weight.float(), eps=1e-6).to(dtype)
    actual = RMSNormFwdOp(normalized_shape=(n,), eps=1e-6)(x, weight)
    compare_outputs(actual, expected, normalization_verification("RMSNormFwdOp", dtype))


class FusedAddRMSNormTest(FusedAddRMSNormWorkload, TestBase):
    pass


class FusedAddRMSNormFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                # Standard aligned shapes; bf16 stands for both 2-byte dtypes past smoke.
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-aligned N
                pytest.param(1024, 3000, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(2048, 5120, torch.bfloat16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(1025, 4096, torch.bfloat16, marks=pytest.mark.full),
                # A row long enough for the split path, at the row counts
                # around which it turns on: one row takes it, a few rows do
                # not, and a full grid does not.
                pytest.param(1, 16384, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(2, 16384, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(4, 16384, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@FusedAddRMSNormFixture
def test_fused_add_rms_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = FusedAddRMSNormTest(m, n, dtype)
    test.check(FusedAddRMSNormFwdOp(), *test.gen_inputs())


class FusedAddRMSNormZeroCenteredOffsetTest(FusedAddRMSNormZeroCenteredOffsetWorkload, TestBase):
    pass


@pytest.mark.smoke
def test_fused_add_rms_norm_zero_centered() -> None:
    """A zero-centered call scales the normalized sum by one plus the weight, formed in float32:
    rounded to bfloat16 first, it moves outputs these inputs hold exactly."""
    test = FusedAddRMSNormZeroCenteredOffsetTest(1024, 4096, torch.bfloat16, zero_centered=True)
    test.check(FusedAddRMSNormFwdOp(zero_centered=True), *test.gen_inputs())


@pytest.mark.smoke
def test_fused_add_rms_norm_zero_centered_float16() -> None:
    test = FusedAddRMSNormTest(1024, 4096, torch.float16, zero_centered=True)
    test.check(FusedAddRMSNormFwdOp(zero_centered=True), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_graph(op="FusedAddRMSNormFwdOp")
@pytest.mark.smoke
def test_fused_add_rms_norm_graph_replay() -> None:
    """Qwen3.8-27B's decode call, one zero-centered bfloat16 row."""
    test = FusedAddRMSNormTest(1, 5120, torch.bfloat16, zero_centered=True)
    check_replay(test, FusedAddRMSNormFwdOp(zero_centered=True))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "view, dtype", [("strided", torch.float16), ("3d", torch.bfloat16)], ids=["strided", "3d"]
)
def test_fused_add_rms_norm_views(view: str, dtype: torch.dtype) -> None:
    n = 4096
    x, residual = _view(view, n, dtype), _view(view, n, dtype)
    weight = torch.randn(n, dtype=dtype, device=run_device())
    test = FusedAddRMSNormTest(x.numel() // n, n, dtype)
    y_ref, add_ref = test.ref_program(x.contiguous(), residual.contiguous(), weight)
    y, residual_out = FusedAddRMSNormFwdOp()(x, residual, weight)
    compare_outputs(y, y_ref, norm_verification(dtype))
    compare_outputs(residual_out, add_ref, norm_verification(dtype))


@pytest.mark.smoke
def test_fused_add_rms_norm_reads_inputs_off_the_vector_boundary() -> None:
    """A contiguous input, residual and weight may start anywhere in their storage."""
    test = FusedAddRMSNormTest(4, 1024, torch.float16)
    test.check(FusedAddRMSNormFwdOp(), *(misaligned(t) for t in test.gen_inputs()))


@pytest.mark.smoke
def test_rms_norm_reduces_over_a_multi_axis_normalized_shape() -> None:
    """``eps=None`` takes the default, and ``weight`` matches ``normalized_shape``."""
    op = RMSNormFwdOp(normalized_shape=(2, 3), eps=None)
    assert tuple(op.normalized_shape) == (2, 3)
    x = torch.randn(4, 2, 3, dtype=torch.float16, device=run_device())
    w = torch.randn(2, 3, dtype=torch.float16, device=run_device())
    assert op(x, w).shape == x.shape
