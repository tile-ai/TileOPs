"""Tests for AdaLayerNorm and AdaLayerNormZero."""

import pytest
import torch

from tests.ops.test_layer_norm import (
    assert_raw_sums_fail,
    check_held_launch,
    op_preferring,
    outlier_rows,
    warp_leads,
)
from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.norm import (
    AdaLayerNormHeldKernel,
    AdaLayerNormKernel,
    AdaLayerNormZeroHeldKernel,
    AdaLayerNormZeroKernel,
)
from tileops._ops.norm.ada_layer_norm import AdaLayerNormFwdOp
from tileops._ops.norm.ada_layer_norm_zero import AdaLayerNormZeroFwdOp
from tileops._utils import get_sm_count
from tileops.backend import BUILTIN
from workloads.device import run_device
from workloads.norm import (
    AdaLayerNormFarFromZeroWorkload,
    AdaLayerNormOutlierLeadWorkload,
    AdaLayerNormWorkload,
    AdaLayerNormZeroFarFromZeroWorkload,
    AdaLayerNormZeroOutlierLeadWorkload,
    AdaLayerNormZeroWorkload,
)


class AdaLayerNormTest(AdaLayerNormWorkload, TestBase):
    pass


class AdaLayerNormFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                # Standard aligned shapes -- fp32
                pytest.param(1024, 4096, torch.float32, marks=pytest.mark.smoke),
                # Standard aligned shapes -- fp16
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
                # Standard aligned shapes -- bf16
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.float32, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-power-of-two hidden dims
                pytest.param(1024, 3000, torch.float32, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float16, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(1025, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@AdaLayerNormFixture
def test_ada_layer_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormTest(m, n, dtype)
    op = AdaLayerNormFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ada_layer_norm_kernel_handles_natural_unaligned_shape(
    dtype: torch.dtype,
) -> None:
    m, n = 16, 1152
    test = AdaLayerNormTest(m, n, dtype)
    op = AdaLayerNormFwdOp(eps=test.eps, target=BUILTIN)
    test.check(op, *test.gen_inputs())


class AdaLayerNormFarFromZeroTest(AdaLayerNormFarFromZeroWorkload, TestBase):
    pass


class AdaLayerNormZeroFarFromZeroTest(AdaLayerNormZeroFarFromZeroWorkload, TestBase):
    pass


def _far_case(gated, rows, n, kernel, dtype, row, mark):
    variant = "ada-zero" if gated else "ada"
    return pytest.param(
        gated,
        rows,
        n,
        kernel,
        dtype,
        row,
        marks=mark,
        id=f"{variant}-{kernel}-{n}-{str(dtype)[6:]}-{row}",
    )


# (rows, or rows an SM when a float, n, kernel): "held" holds a row in registers, padded
# (1152) and aligned (4096); "fragment" serves rows that queue past a thread's registers,
# aligned (16384) and padded (16000). The gated variant shares the statistics: one case a
# kernel and width.
_ADA_FAR_CASES = [
    *[
        _far_case(False, 4, n, "held", dtype, row, pytest.mark.smoke)
        for n, dtype, row in [
            (1152, torch.float32, "constant"),
            (1152, torch.float32, "noisy"),
            (1152, torch.float32, "huge"),
            (1152, torch.bfloat16, "huge"),
            (4096, torch.float32, "noisy"),
            (4096, torch.bfloat16, "huge"),
        ]
    ],
    _far_case(True, 4, 1152, "held", torch.float32, "huge", pytest.mark.smoke),
    _far_case(True, 4, 4096, "held", torch.float32, "noisy", pytest.mark.smoke),
    _far_case(False, 4, 4096, "held", torch.float32, "constant", pytest.mark.full),
    *[
        _far_case(False, 2.0, n, "fragment", dtype, row, pytest.mark.full)
        for n, dtype, row in [
            (16384, torch.float32, "constant"),
            (16384, torch.float32, "noisy"),
            (16384, torch.float32, "huge"),
            (16384, torch.bfloat16, "huge"),
            (16000, torch.float32, "noisy"),
            (16000, torch.float32, "huge"),
        ]
    ],
    _far_case(True, 2.0, 16384, "fragment", torch.float32, "constant", pytest.mark.full),
    _far_case(True, 2.0, 16000, "fragment", torch.float32, "huge", pytest.mark.full),
    _far_case(False, 4, 1152, "held", torch.float32, "wide_spread", pytest.mark.full),
    _far_case(True, 2.0, 16000, "fragment", torch.float32, "wide_spread", pytest.mark.full),
]


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("gated, rows, n, kernel, dtype, row", _ADA_FAR_CASES)
def test_ada_layer_norm_normalizes_rows_far_from_zero(gated, rows, n, kernel, dtype, row) -> None:
    m = int(rows * get_sm_count()) if isinstance(rows, float) else rows
    test_cls = AdaLayerNormZeroFarFromZeroTest if gated else AdaLayerNormFarFromZeroTest
    test = test_cls(m, n, dtype, row=row)
    op = (AdaLayerNormZeroFwdOp if gated else AdaLayerNormFwdOp)()
    test.check(op, *test.gen_inputs())
    (built,) = op.iter_kernels()
    held = AdaLayerNormZeroHeldKernel if gated else AdaLayerNormHeldKernel
    assert isinstance(built, held) == (kernel == "held")


class AdaLayerNormOutlierLeadTest(AdaLayerNormOutlierLeadWorkload, TestBase):
    pass


class AdaLayerNormZeroOutlierLeadTest(AdaLayerNormZeroOutlierLeadWorkload, TestBase):
    pass


def _outlier_case(gated, rows, n, kernel, leads, mark):
    variant = "ada-zero" if gated else "ada"
    return pytest.param(
        gated, rows, n, kernel, leads, marks=mark, id=f"{variant}-{kernel}-{n}-{leads}"
    )


# (rows, n, kernel, leads), as `test_layer_norm_normalizes_rows_with_an_outlier_lead` reads
# them; "fragment" serves rows queued past a thread's registers, aligned (16384) and padded
# (16000). The gated variant takes the held kernel alone and queued, where it reads its gate
# late.
_ADA_OUTLIER_CASES = [
    *[_outlier_case(gated, 4, 4096, "split", "row", pytest.mark.smoke) for gated in (False, True)],
    *[
        _outlier_case(False, rows, n, kernel, leads, pytest.mark.full)
        for rows, n, kernel, leads in [
            (1, 262144, "split", "warps"),
            (2.0, 6144, "queued", "row"),
            (2.0, 16384, "fragment", "row"),
            (2.0, 16000, "fragment", "row"),
        ]
    ],
    _outlier_case(True, 2.0, 6144, "queued", "row", pytest.mark.full),
]


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("gated, rows, n, kernel, leads", _ADA_OUTLIER_CASES)
def test_ada_layer_norm_normalizes_rows_with_an_outlier_lead(gated, rows, n, kernel, leads) -> None:
    m, dtype = outlier_rows(rows), torch.float32
    held = AdaLayerNormZeroHeldKernel if gated else AdaLayerNormHeldKernel
    check_held_launch(held, kernel, m, n, dtype)
    positions = (0,) if leads == "row" else warp_leads(held, m, n, dtype)
    test_cls = AdaLayerNormZeroOutlierLeadTest if gated else AdaLayerNormOutlierLeadTest
    test = test_cls(m, n, dtype, leads=positions)
    op = (AdaLayerNormZeroFwdOp if gated else AdaLayerNormFwdOp)()
    inputs = test.gen_inputs()
    assert_raw_sums_fail(inputs[0])
    test.check(op, *inputs)
    (built,) = op.iter_kernels()
    assert isinstance(built, held) == (kernel != "fragment")


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "gated",
    [pytest.param(False, marks=pytest.mark.smoke), pytest.param(True, marks=pytest.mark.full)],
    ids=["ada", "ada-zero"],
)
def test_ada_layer_norm_fragment_prefetches_rows_with_an_outlier_lead(gated) -> None:
    # The held kernel serves 3000-wide fp32 rows; the fragment kernel, made to take them,
    # prefetches the modulation of a padded row that fits shared memory.
    class Queued(AdaLayerNormZeroHeldKernel if gated else AdaLayerNormHeldKernel):
        @classmethod
        def _queued_past_registers(cls, *args) -> bool:
            return True

    op_cls = AdaLayerNormZeroFwdOp if gated else AdaLayerNormFwdOp
    fragment = AdaLayerNormZeroKernel if gated else AdaLayerNormKernel

    class Prefetched(fragment):
        general = False
        preferred_over = frozenset(op_cls.kernel_types)
        _held = Queued

    op = op_preferring(op_cls, "ada_layer_norm_prefetched", Prefetched)
    n = 3000
    test_cls = AdaLayerNormZeroOutlierLeadTest if gated else AdaLayerNormOutlierLeadTest
    test = test_cls(2 * get_sm_count(), n, torch.float32, leads=(0,))
    inputs = test.gen_inputs()
    assert_raw_sums_fail(inputs[0])
    test.check(op, *inputs)
    (built,) = op.iter_kernels()
    assert isinstance(built, Prefetched)
    assert built.use_cp_async


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize(
    "n, dtype",
    [
        pytest.param(511, torch.float16, id="fp16-below"),
        pytest.param(514, torch.float16, id="fp16-lower-inside"),
    ],
)
def test_ada_layer_norm_async_policy_edge_correctness(
    n: int,
    dtype: torch.dtype,
) -> None:
    m = 4
    test = AdaLayerNormTest(m, n, dtype)
    op = AdaLayerNormFwdOp(eps=test.eps, target=BUILTIN)
    test.check(op, *test.gen_inputs())


class AdaLayerNorm3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 512, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@AdaLayerNorm3DFixture
def test_ada_layer_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    scale = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    shift = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())

    op = AdaLayerNormFwdOp()

    test = AdaLayerNormTest(batch * seq, hidden, dtype)
    test.check(op, x, scale, shift)


class AdaLayerNormZeroTest(AdaLayerNormZeroWorkload, TestBase):
    pass


class AdaLayerNormZeroFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                # Standard aligned shapes -- fp32
                pytest.param(1024, 4096, torch.float32, marks=pytest.mark.smoke),
                # Standard aligned shapes -- fp16
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
                # Standard aligned shapes -- bf16
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.float32, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-power-of-two hidden dims
                pytest.param(1024, 3000, torch.float32, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float16, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(1025, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@AdaLayerNormZeroFixture
def test_ada_layer_norm_zero_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormZeroTest(m, n, dtype)
    op = AdaLayerNormZeroFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ada_layer_norm_zero_kernel_handles_natural_unaligned_shape(
    dtype: torch.dtype,
) -> None:
    m, n = 16, 1152
    test = AdaLayerNormZeroTest(m, n, dtype)
    op = AdaLayerNormZeroFwdOp(eps=test.eps, target=BUILTIN)
    test.check(op, *test.gen_inputs())


class AdaLayerNormZero3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 512, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@AdaLayerNormZero3DFixture
def test_ada_layer_norm_zero_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    scale = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    shift = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    gate = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())

    op = AdaLayerNormZeroFwdOp()

    test = AdaLayerNormZeroTest(batch * seq, hidden, dtype)
    test.check(op, x, scale, shift, gate)


_TUNE = [pytest.param(False, marks=pytest.mark.smoke), pytest.param(True, marks=pytest.mark.full)]


@pytest.mark.parametrize("tune", _TUNE)
def test_ada_layer_norm_under_tuning(tune: bool) -> None:
    test = AdaLayerNormTest(17, 514, torch.float16)
    op = AdaLayerNormFwdOp(eps=test.eps)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize("tune", _TUNE)
def test_ada_layer_norm_zero_under_tuning(tune: bool) -> None:
    test = AdaLayerNormZeroTest(17, 514, torch.float16)
    op = AdaLayerNormZeroFwdOp(eps=test.eps)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


def _misaligned(t: torch.Tensor) -> torch.Tensor:
    """*t* copied into a contiguous view that starts one element into its storage."""
    view = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)[1:].view(t.shape)
    view.copy_(t)
    return view


@pytest.mark.smoke
@pytest.mark.parametrize("zero", [False, True], ids=["ada", "ada-zero"])
def test_ada_layer_norm_reads_inputs_off_the_vector_boundary(zero: bool) -> None:
    """A contiguous input and modulation tensors may start anywhere in their storage."""
    test = (AdaLayerNormZeroTest if zero else AdaLayerNormTest)(4, 1024, torch.float16)
    op = AdaLayerNormZeroFwdOp() if zero else AdaLayerNormFwdOp()
    test.check(op, *(_misaligned(t) for t in test.gen_inputs()))
