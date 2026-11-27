"""SoftmaxFwdOp, LogSoftmaxFwdOp and LogSumExpFwdOp against their PyTorch references."""

import math
import re
import warnings

import pytest
import torch
import torch.nn.functional as F
from tilelang.jit.kernel import JITKernel

from tests.workload_test_base import TestBase
from tileops._kernels.kernel_base import CompiledProgram
from tileops._kernels.reduction.call_spec import SoftmaxCall
from tileops._kernels.reduction.softmax_on_chip import SoftmaxClusterSplitKernel
from tileops._kernels.tiling import ALIGNMENT
from tileops._ops.reduction.softmax import LogSoftmaxFwdOp, LogSumExpFwdOp, SoftmaxFwdOp
from tileops._utils import get_shared_memory_optin, get_sm_count
from workloads.device import run_device, run_device_available
from workloads.numerics import compare_outputs
from workloads.reduction import (
    LogSoftmaxCase,
    LogSumExpCase,
    SoftmaxCase,
    softmax_verification,
)

_OPS = {"softmax": SoftmaxFwdOp, "log_softmax": LogSoftmaxFwdOp, "logsumexp": LogSumExpFwdOp}
_CASES = {"softmax": SoftmaxCase, "log_softmax": LogSoftmaxCase, "logsumexp": LogSumExpCase}


class SoftmaxTest(SoftmaxCase, TestBase):
    pass


class LogSoftmaxTest(LogSoftmaxCase, TestBase):
    pass


class LogSumExpTest(LogSumExpCase, TestBase):
    pass


_TESTS = {"softmax": SoftmaxTest, "log_softmax": LogSoftmaxTest, "logsumexp": LogSumExpTest}


def _check(op_kind: str, x: torch.Tensor, dim, keepdim: bool = False) -> None:
    """*op_kind* over *dim* on *x* against its workload's reference."""
    kwargs = {"keepdim": keepdim} if op_kind == "logsumexp" else {}
    y = _OPS[op_kind](dim=dim, **kwargs)(x)
    ref = _CASES[op_kind](tuple(x.shape), x.dtype, dim=dim).ref_program(x)
    if keepdim:
        ref = ref.unsqueeze(dim)
    assert y.shape == ref.shape, f"shape {y.shape} vs ref {ref.shape}"
    compare_outputs(y, ref, softmax_verification(x.dtype, logarithmic=op_kind != "softmax"))


def _op_cases() -> list:
    """Each op at every dtype is smoke, then the row shapes that pick a kernel or a tail."""
    smoke, full = pytest.mark.smoke, pytest.mark.full
    dtypes = (torch.float32, torch.float16, torch.bfloat16)
    cases = []
    for k in _OPS:
        marks = [smoke, pytest.mark.packaging(family="reduction")]
        cases.append(pytest.param(k, (32, 256), -1, torch.float32, False, marks=marks))
        cases += [pytest.param(k, (32, 256), -1, dt, False, marks=smoke) for dt in dtypes[1:]]
    # Rows of whole thread-wide vectors, held in registers; a 2 KB row on a warp.
    cases += [
        pytest.param("softmax", (8, 2048), -1, torch.float16, False, marks=smoke),
        pytest.param("softmax", (8, 1024), -1, torch.float32, False, marks=smoke),
        pytest.param("softmax", (4, 512), -1, torch.float32, False, marks=smoke),
        # A 2 KB row on a warp, tail-M: the last CTA's warps past M store nothing.
        pytest.param("softmax", (5, 512), -1, torch.float32, False, marks=full),
    ]
    # request_tune(): the kernel is built before autotune runs; log_softmax shares softmax's
    # kernels, logsumexp's tuned build is test_logsumexp_tuned.
    cases.append(pytest.param("softmax", (32, 256), -1, torch.float16, True, marks=full))
    for k in _OPS:
        cases += [
            # Rows off the vector width at both element sizes.
            pytest.param(k, (32, 300), -1, torch.float32, False, marks=full),
            pytest.param(k, (32, 300), -1, torch.float16, False, marks=full),
            # Tail-M.
            pytest.param(k, (33, 256), -1, torch.float16, False, marks=full),
            # Leading axes folded into rows.
            pytest.param(k, (2, 16, 256), -1, torch.float16, False, marks=full),
            # A few long rows (split kernel), aligned and not, with a tail-M.
            pytest.param(k, (4, 32768), -1, torch.float16, False, marks=full),
            pytest.param(k, (33, 300), -1, torch.float32, False, marks=full),
            pytest.param(k, (33, 33000), -1, torch.float16, False, marks=full),
            pytest.param(k, (33, 32768), -1, torch.float16, False, marks=full),
            # A leading and a middle axis.
            pytest.param(k, (256, 32), 0, torch.float16, False, marks=full),
            pytest.param(k, (2, 256, 16), 1, torch.float16, False, marks=full),
        ]
    # Long rows on a filled grid: the streaming logsumexp kernel.
    cases.append(pytest.param("logsumexp", (256, 16384), -1, torch.float16, False, marks=full))
    return cases


@pytest.mark.parametrize("op_kind, shape, dim, dtype, tune", _op_cases())
def test_softmax_family_op(
    op_kind: str, shape: tuple, dim: int, dtype: torch.dtype, tune: bool
) -> None:
    test = _TESTS[op_kind](shape, dtype, dim=dim)
    op = _OPS[op_kind](dim=dim)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "shape, tune",
    [
        # A tail-M row off the vector width, which no smoke case of the family test reaches.
        pytest.param((33, 300), False, marks=pytest.mark.smoke),
        pytest.param((32, 256), True, marks=pytest.mark.full),
    ],
)
def test_logsumexp_tuned(shape: tuple, tune: bool) -> None:
    """Separate from the family test: conftest allows one full tune=True case per function."""
    test = LogSumExpTest(shape, torch.float16, dim=-1)
    op = LogSumExpFwdOp(dim=-1)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", list(_OPS))
@pytest.mark.parametrize("view", ["strided", "1d"])
def test_softmax_family_views(op_kind: str, view: str) -> None:
    """A strided view and a 1-D row, both off the vector width."""
    if view == "strided":
        x = torch.randn(32, 600, dtype=torch.float16, device=run_device())[:, :300]
    else:
        x = torch.randn(300, dtype=torch.float16, device=run_device())
    _check(op_kind, x, -1)


@pytest.mark.parametrize(
    "op_cls, ref_fn, shape",
    [
        pytest.param(SoftmaxFwdOp, F.softmax, (8, 1000), marks=pytest.mark.smoke, id="single"),
        pytest.param(SoftmaxFwdOp, F.softmax, (64, 256), marks=pytest.mark.full, id="warp-rows"),
        pytest.param(
            LogSoftmaxFwdOp, F.log_softmax, (512, 40000), marks=pytest.mark.full, id="streaming"
        ),
        # An odd row: no access wider than one element starts every row.
        pytest.param(
            SoftmaxFwdOp, F.softmax, (512, 40001), marks=pytest.mark.full, id="streaming-odd-width"
        ),
        pytest.param(
            LogSoftmaxFwdOp, F.log_softmax, (64, 40000), marks=pytest.mark.full, id="split"
        ),
        pytest.param(
            SoftmaxFwdOp, F.softmax, (4, 128256), marks=pytest.mark.full, id="fused-split"
        ),
        pytest.param(SoftmaxFwdOp, F.softmax, (264, 32768), marks=pytest.mark.full, id="on-chip"),
    ],
)
def test_softmax_dtype_widens_in_kernel(op_cls, ref_fn, shape: tuple) -> None:
    """A float32 ``dtype`` on a bfloat16 input matches torch on every kernel path."""
    x = torch.randn(shape, device=run_device()).to(torch.bfloat16)
    y = op_cls(dim=-1, dtype=torch.float32)(x)
    assert y.dtype == torch.float32
    # Relative only: softmax values sit below any absolute floor that would let a
    # bfloat16-rounded output pass.
    compare_outputs(
        y,
        ref_fn(x, dim=-1, dtype=torch.float32),
        softmax_verification(y.dtype, input_dtype=x.dtype, logarithmic=op_cls is LogSoftmaxFwdOp),
    )


@pytest.mark.parametrize(
    "case_cls, op_cls, dim, shape",
    [
        pytest.param(
            SoftmaxCase, SoftmaxFwdOp, -1, (8, 1000), marks=pytest.mark.smoke, id="softmax"
        ),
        pytest.param(
            SoftmaxCase, SoftmaxFwdOp, -1, (8, 512), marks=pytest.mark.full, id="softmax-warp-rows"
        ),
        pytest.param(
            LogSumExpCase, LogSumExpFwdOp, -1, (32, 1000), marks=pytest.mark.full, id="logsumexp"
        ),
        pytest.param(
            LogSumExpCase,
            LogSumExpFwdOp,
            (0, 2),
            (8, 64, 1024),
            marks=pytest.mark.full,
            id="logsumexp-edge-split",
        ),
    ],
)
def test_contiguous_input_off_a_vector_boundary(case_cls, op_cls, dim, shape: tuple) -> None:
    """A contiguous input starting one element past a 16-byte boundary matches torch."""
    x = torch.randn(math.prod(shape) + 1, device=run_device(), dtype=torch.float16)[1:]
    x = x.view(shape)
    case = case_cls(shape, x.dtype, dim=dim)
    compare_outputs(op_cls(dim=dim)(x), case.ref_program(x), case.verification(x))


@pytest.mark.parametrize(
    "shape, dim, dtype",
    [
        pytest.param((32, 256), -1, torch.float32, marks=pytest.mark.smoke),
        # A non-last axis goes through a transpose before keepdim restores it.
        pytest.param((256, 32), 0, torch.float16, marks=pytest.mark.smoke),
        pytest.param((2, 256, 16), 1, torch.float32, marks=pytest.mark.full),
    ],
)
def test_logsumexp_keepdim(shape: tuple, dim: int, dtype: torch.dtype) -> None:
    _check("logsumexp", torch.randn(*shape, dtype=dtype, device=run_device()), dim, keepdim=True)


@pytest.mark.parametrize(
    "shape, dim, dtype",
    [
        pytest.param((256, 16384), -1, torch.bfloat16, id="streaming", marks=pytest.mark.smoke),
        pytest.param((64, 4096), -1, torch.float16, id="single-tile", marks=pytest.mark.smoke),
        pytest.param((300, 100000), -1, torch.float32, id="tiled", marks=pytest.mark.smoke),
        pytest.param((8, 102400), -1, torch.float32, id="split", marks=pytest.mark.full),
        pytest.param((4, 128, 4096), [0, 2], torch.float16, id="edge-axes", marks=pytest.mark.full),
    ],
)
def test_logsumexp_special_values(shape: tuple, dim, dtype: torch.dtype) -> None:
    """Every kernel path keeps torch's -inf, +inf, and NaN row semantics."""
    x = torch.randn(*shape, dtype=dtype, device=run_device())
    # Row r of the reduced output is x[r] for a trailing dim, x[:, r] for edge axes.
    rows = x if dim == -1 else x.transpose(0, 1)
    rows[0] = float("-inf")
    rows[1] = float("-inf")
    rows[1][..., 7] = 2.0
    rows[2][..., ::2] = float("-inf")
    rows[3][..., 100] = float("nan")
    rows[4][..., 200] = float("inf")
    rows[5][..., 200] = float("inf")
    rows[5][..., 300] = float("nan")
    # A NaN in a run of -inf, far from the row's one finite or +inf value.
    first, final = (0,) * rows[6].dim(), (-1,) * rows[6].dim()
    rows[6] = float("-inf")
    rows[6][first] = float("nan")
    rows[6][final] = -10.0
    rows[7] = float("-inf")
    rows[7][first] = float("nan")
    rows[7][final] = float("inf")

    y = LogSumExpFwdOp(dim=dim)(x).float()
    y_ref = torch.logsumexp(x.float(), dim=dim)
    assert y[0].item() == float("-inf")
    assert torch.isnan(y[3])
    assert y[4].item() == float("inf")
    assert torch.isnan(y[5])
    assert torch.isnan(y[6])
    assert torch.isnan(y[7])
    finite = torch.isfinite(y_ref)
    compare_outputs(y[finite], y_ref[finite], softmax_verification(x.dtype, logarithmic=True))


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls", [SoftmaxFwdOp, LogSoftmaxFwdOp])
def test_softmax_rejects_a_sequence_dim_at_construction(op_cls) -> None:
    """``dim`` is one axis, as in torch."""
    with pytest.raises(ValueError, match="dim = "):
        op_cls(dim=[-1, 0])


@pytest.mark.smoke
def test_logsumexp_accepts_multidim() -> None:
    """LogSumExpFwdOp must accept list dim without error (multi-dim is supported)."""
    x = torch.randn(4, 8, device=run_device(), dtype=torch.float32)
    op = LogSumExpFwdOp(dim=[0, 1])
    y = op(x)
    y_ref = torch.logsumexp(x.float(), dim=[0, 1])
    compare_outputs(y, y_ref, softmax_verification(x.dtype, logarithmic=True))


@pytest.mark.parametrize(
    "op_kind, shape, dtype",
    [
        pytest.param(k, shape, dtype, marks=pytest.mark.smoke)
        for k in ("softmax", "log_softmax")
        for shape, dtype in (
            ((256,), torch.float32),
            ((32, 256), torch.float16),
            ((4, 16, 32), torch.bfloat16),
        )
    ],
)
def test_softmax_dim_none_implicit_axis(op_kind: str, shape: tuple, dtype: torch.dtype) -> None:
    """``dim=None`` takes PyTorch's implicit axis for each rank and warns as it does."""
    x = torch.randn(*shape, dtype=dtype, device=run_device())
    with pytest.warns(UserWarning, match="Implicit dimension choice"):
        y = _OPS[op_kind](dim=None)(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ref = getattr(F, op_kind)(x.float(), dim=None).to(dtype)
    compare_outputs(y, ref, softmax_verification(x.dtype, logarithmic=op_kind != "softmax"))


@pytest.mark.smoke
@pytest.mark.filterwarnings("ignore:Implicit dimension choice:UserWarning")
@pytest.mark.parametrize("op_kind", ["softmax", "log_softmax"])
def test_softmax_dim_none_reused_across_ranks(op_kind: str) -> None:
    """One ``dim=None`` op re-resolves its axis per call and keeps ``op.dim`` None."""
    op = _OPS[op_kind](dim=None)
    for shape in ((4,), (2, 4), (4, 3, 5)):
        x = torch.randn(*shape, dtype=torch.float32, device=run_device())
        ref = getattr(F, op_kind)(x, dim=None)
        compare_outputs(op(x), ref, softmax_verification(x.dtype, logarithmic=op_kind != "softmax"))
    assert op.dim is None, f"op.dim was mutated to {op.dim!r}; expected None"


# LogSoftmax FLOPs equal 5 * M * N.


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
def test_log_softmax_eval_roofline_flops_5mn() -> None:
    """LogSoftmaxFwdOp.eval_roofline() must report flops == 5 * M * N."""
    M, N = 64, 256
    dtype = torch.float16
    op = LogSoftmaxFwdOp(dim=-1)
    x = torch.randn(M, N, dtype=dtype, device=run_device())
    op(x)  # bind dynamic shape
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == 5 * M * N, f"LogSoftmax flops {flops} != 5 * M * N = {5 * M * N}"
    assert mem_bytes == 2 * M * N * elem_bytes, (
        f"LogSoftmax bytes {mem_bytes} != 2 * M * N * elem_bytes = {2 * M * N * elem_bytes}"
    )


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_softmax_few_long_rows_split_across_blocks(dtype: torch.dtype) -> None:
    """A handful of long rows runs as segments plus a fold, not one block per row.

    102392 is not a segment multiple, so the tail segment's masked lanes
    contribute ``exp(-inf) = 0`` to the fold. It is not a whole number of 16-byte
    vectors per CTA of a 16-CTA cluster either, so softmax stays on the split on SM90.
    """
    x = torch.randn(4, 102392, dtype=dtype, device=run_device())

    compare_outputs(
        SoftmaxFwdOp(dim=-1)(x),
        F.softmax(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=False),
    )
    compare_outputs(
        LogSoftmaxFwdOp(dim=-1)(x),
        F.log_softmax(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=True),
    )


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
def test_logsumexp_few_long_rows_split_across_blocks() -> None:
    """logsumexp folds the shared segment statistics without re-reading the input."""
    x = torch.randn(4, 102400, dtype=torch.float32, device=run_device())
    compare_outputs(
        LogSumExpFwdOp(dim=-1)(x),
        torch.logsumexp(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=True),
    )


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
def test_log_softmax_widest_one_tile_row() -> None:
    """The widest fp32 row shared memory holds, at rows enough to fill the grid, where a one
    tile log_softmax would stage it beside the scratch of its reductions."""
    n = get_shared_memory_optin() // 4 // ALIGNMENT * ALIGNMENT
    x = torch.randn(2 * get_sm_count(), n, dtype=torch.float32, device=run_device())
    compare_outputs(
        LogSoftmaxFwdOp(dim=-1)(x),
        F.log_softmax(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=True),
    )


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
def test_split_rows_survive_fully_masked_segments() -> None:
    """A segment of only ``-inf`` contributes zero to the fold, not NaN.

    Masked inputs make whole segments ``-inf``; the segment-local max is then
    ``-inf`` and an unguarded ``exp(-inf - -inf)`` would poison rows the
    single-block path computes fine. An all--inf row stays torch: NaN for
    softmax, ``-inf`` for logsumexp. 102392 columns keep softmax on the split on SM90.
    """
    x = torch.randn(4, 102392, dtype=torch.float32, device=run_device())
    x[:, :4096] = float("-inf")  # first segments fully masked, rest finite
    compare_outputs(
        SoftmaxFwdOp(dim=-1)(x),
        F.softmax(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=False),
    )
    compare_outputs(
        LogSumExpFwdOp(dim=-1)(x),
        torch.logsumexp(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=True),
    )

    compare_outputs(
        LogSoftmaxFwdOp(dim=-1)(x),
        F.log_softmax(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=True),
    )

    x[0, :] = float("-inf")
    compare_outputs(
        SoftmaxFwdOp(dim=-1)(x),
        F.softmax(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=False),
    )
    compare_outputs(
        LogSumExpFwdOp(dim=-1)(x),
        torch.logsumexp(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=True),
    )


@pytest.mark.smoke
def test_large_row_shifts_its_maximum_to_exactly_one() -> None:
    """A row of large values still shifts its maximum to exactly 1 before the sum.

    The shift is applied in base 2. Scaling the value and the shift separately and
    subtracting inside the product lets the pair contract into one ``FFMA``, which keeps
    the value's product exact against an already-rounded shift: equal operands stop
    cancelling, and the error grows with the row's magnitude. Every manifest row draws
    from ``randn``, which holds ``|x|`` under about 5 and cannot show it, so the rows
    here are scaled up and the second one repeats its maximum.

    ``log_softmax`` is the whole guard for the shared ``exp_shifted`` helper. The other
    two ops cannot show the fault at any tolerance: ``softmax`` divides every term by
    their sum, which takes the error out with it, and ``logsumexp`` returns a value of
    the row's own magnitude, whose fp32 spacing against a float64 reference is two
    orders larger than the fault and identical with or without it.
    """
    device = run_device()
    torch.manual_seed(1235)
    x = torch.randn((2, 4096), dtype=torch.float32, device=device) * 1e5
    x[1, ::512] = x[1].max()

    expected = F.log_softmax(x.double(), dim=-1).to(torch.float32)
    compare_outputs(
        LogSoftmaxFwdOp(dim=-1)(x), expected, softmax_verification(x.dtype, logarithmic=True)
    )


@pytest.mark.parametrize(
    "op, reference",
    [
        pytest.param(SoftmaxFwdOp, lambda t: F.softmax(t, dim=-1), id="softmax"),
        pytest.param(LogSoftmaxFwdOp, lambda t: F.log_softmax(t, dim=-1), id="log_softmax"),
    ],
)
@pytest.mark.parametrize(
    "width, prefix, dtype",
    [
        # 16384-element tiles on the streaming kernel: the first tile and most of the
        # second are the -inf prefix.
        pytest.param(40000, 28672, torch.float32, id="streaming", marks=pytest.mark.smoke),
        # Two CTAs a row on the on-chip kernel, whose 16-bit rows take the warp maximum
        # before their terms: the first CTA's share is the prefix.
        pytest.param(65536, 32768, torch.float16, id="on-chip-fp16", marks=pytest.mark.smoke),
        # Eight CTAs a row on the on-chip kernel: the first CTA's share is the prefix.
        pytest.param(40960, 5120, torch.float32, id="on-chip", marks=pytest.mark.full),
    ],
)
def test_leading_tiles_of_only_neg_inf_do_not_poison_a_row(
    op: type, reference, width: int, prefix: int, dtype: torch.dtype
) -> None:
    """A row whose leading part holds only ``-inf`` still reduces its finite tail.

    The streaming path folds one tile at a time and rescales the running sum by the change
    in the row maximum; the on-chip path folds the partials of a row's CTAs. While a
    maximum is still ``-inf`` the rescale subtracts one infinity from another, and the
    ``NaN`` reaches the rest of the row. The second row is ``-inf`` throughout and must
    still come back as torch returns it.
    """
    device = run_device()
    torch.manual_seed(1235)
    # Too few rows dispatch to the split kernel instead.
    x = torch.randn((264, width), dtype=dtype, device=device)
    x[0, :prefix] = float("-inf")
    x[1, :] = float("-inf")

    compare_outputs(
        op(dim=-1)(x),
        reference(x),
        softmax_verification(x.dtype, logarithmic=op is LogSoftmaxFwdOp),
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op, reference, shape, dtype",
    [
        # 32 KB shares: 16 KB in registers, the rest staged, four CTAs an SM.
        pytest.param(SoftmaxFwdOp, F.softmax, (264, 32768), torch.bfloat16, id="share-32k"),
        # 64 KB shares: 32 KB in registers, the rest staged, three CTAs an SM.
        pytest.param(LogSoftmaxFwdOp, F.log_softmax, (264, 65536), torch.float16, id="share-64k"),
        # Eight CTAs a row, each past the share a cluster is aimed at.
        pytest.param(SoftmaxFwdOp, F.softmax, (264, 262144), torch.float32, id="widest"),
    ],
)
def test_rows_read_once_across_a_cluster(op, reference, shape: tuple, dtype) -> None:
    """Rows the on-chip kernel splits across a cluster match torch."""
    torch.manual_seed(1235)
    x = torch.randn(shape, dtype=dtype, device=run_device()) * 4
    compare_outputs(
        op(dim=-1)(x),
        reference(x, dim=-1),
        softmax_verification(x.dtype, logarithmic=op is LogSoftmaxFwdOp),
    )


def _scaled_rows(shape: tuple, dtype) -> torch.Tensor:
    """Standard normal rows times four, from a fixed seed."""
    torch.manual_seed(1235)
    return torch.randn(shape, dtype=dtype, device=run_device()) * 4


def _special_value_rows(shape: tuple, dtype) -> torch.Tensor:
    """``_scaled_rows`` with special values in the first four rows.

    The first eighth of row 0 is ``-inf``, row 1 is only ``-inf``, row 2 holds a ``+inf``
    and row 3 a NaN.
    """
    x = _scaled_rows(shape, dtype)
    x[0, : shape[1] // 8] = float("-inf")
    x[1, :] = float("-inf")
    x[2, shape[1] // 2] = float("inf")
    x[3, shape[1] // 3] = float("nan")
    return x


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "shape, dtype, special",
    [
        # 1002 vectors a CTA: the last vector of a thread's share is past the row in some
        # threads.
        pytest.param((4, 128256), torch.bfloat16, False, id="partial-vectors"),
        # Four-element vectors: 512 a CTA, four a thread.
        pytest.param((4, 32768), torch.float32, False, id="float32"),
        # Every thread holds whole vectors; special values fill rows, see the docstring.
        pytest.param((4, 102400), torch.float16, True, id="special-values"),
    ],
)
def test_few_long_rows_across_a_cluster(shape: tuple, dtype, special: bool) -> None:
    """A handful of long rows, each split across a 16-CTA cluster in registers, matches torch.

    SM90 is the first architecture with thread-block clusters and the 16-CTA cluster size
    the kernel launches.
    In the special-value case (``_special_value_rows``) the first two CTAs of row 0 hold
    only ``-inf``, so whole warps and whole CTAs fold a ``-inf`` maximum and contribute
    nothing instead of the NaN of ``exp(-inf - -inf)``; rows 1 to 3 come back NaN
    throughout, as torch returns them.
    """
    x = _special_value_rows(shape, dtype) if special else _scaled_rows(shape, dtype)
    op = SoftmaxFwdOp(dim=-1)
    compare_outputs(op(x), F.softmax(x, dim=-1), softmax_verification(dtype, logarithmic=False))
    served = {type(kernel).__name__ for kernel in op._entries_built("softmax").values()}
    assert served == {"SoftmaxClusterSplitKernel"}, served


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "shape, dtype, cluster",
    [
        # Two 32 KB shares a row, on a grid of over 100 waves of resident CTAs.
        pytest.param((20480, 32768), torch.bfloat16, 2, id="two-shares", marks=pytest.mark.smoke),
        # Eight 16 KB shares a row.
        pytest.param((10240, 65536), torch.float16, 8, id="eight-shares", marks=pytest.mark.smoke),
        # Six 16 KB shares, a cluster whose width is not a power of two.
        pytest.param((13312, 49152), torch.float16, 6, id="six-shares", marks=pytest.mark.full),
        # 2**31 elements: the loads take an int64 offset.
        pytest.param(
            (32768, 65536), torch.bfloat16, 8, id="two-to-the-31", marks=pytest.mark.nightly
        ),
    ],
)
def test_long_rows_held_whole_across_a_cluster(shape: tuple, dtype, cluster: int) -> None:
    """Many 16-bit rows match torch, held whole in registers across a cluster.

    SM90 is the first architecture with thread-block clusters, which the kernel launches.
    The input takes ``_special_value_rows``: the first eighth of row 0 is ``-inf``, so where
    that is a CTA's share its warps fold a ``-inf`` maximum and contribute nothing; rows 1
    to 3 come back NaN throughout, as torch returns them. Row 4 lies wholly below -900, so
    only its own maximum keeps its terms from flushing to zero. Row 5 lies near the type's
    largest finite value; for bfloat16 its maximum times log2(e) is past the largest fp32.
    Row 6 is one zero among ``-inf``: its sum is a single term, which the bias must raise.
    """
    x = _special_value_rows(shape, dtype)
    x[4] -= 1000
    x[5] += 0.9 * torch.finfo(dtype).max
    x[6] = float("-inf")
    x[6, 7] = 0
    normalize = SoftmaxFwdOp(dim=-1)
    compare_outputs(normalize(x), F.softmax(x, dim=-1), softmax_verification(dtype))
    (kernel,) = normalize.iter_kernels()
    assert isinstance(kernel, SoftmaxClusterSplitKernel), type(kernel).__name__
    assert kernel.cluster == cluster, kernel.cluster


@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_kind, shape, dtype, out_dtype",
    [
        pytest.param("softmax", (20480, 32768), torch.float32, torch.float32, id="float32"),
        pytest.param("softmax", (1024, 32768), torch.bfloat16, torch.bfloat16, id="few-waves"),
        # A row of four and a half shares.
        pytest.param("softmax", (20480, 36864), torch.bfloat16, torch.bfloat16, id="part-share"),
        # Sixteen shares, past the portable cluster.
        pytest.param("softmax", (10240, 131072), torch.float16, torch.float16, id="sixteen-shares"),
        pytest.param("softmax", (20480, 32768), torch.bfloat16, torch.float32, id="widened"),
        pytest.param(
            "log_softmax", (20480, 32768), torch.bfloat16, torch.bfloat16, id="log-softmax"
        ),
        # A row of 16384 elements fits one CTA's registers whole.
        pytest.param("softmax", (40960, 16384), torch.bfloat16, torch.bfloat16, id="two-shares"),
    ],
)
def test_cluster_split_refuses_rows_it_does_not_hold(op_kind, shape, dtype, out_dtype) -> None:
    """SoftmaxClusterSplitKernel refuses calls outside both of its plans."""
    call = SoftmaxCall(
        device=run_device(),
        shape=shape,
        axis=1,
        op_kind=op_kind,
        dtype=dtype,
        out_dtype=out_dtype,
    )
    assert SoftmaxClusterSplitKernel.refusal(call) is not None


@pytest.mark.parametrize(
    "dtype, low, shape",
    [
        pytest.param(torch.float32, -100.0, (264, 65536), id="float32", marks=pytest.mark.smoke),
        pytest.param(torch.bfloat16, -88.0, (264, 65536), id="bfloat16", marks=pytest.mark.smoke),
        # Four rows split across a cluster each: every warp but the first holds only
        # ``low``, and its factor against the row maximum is subnormal on its own.
        pytest.param(
            torch.bfloat16, -88.0, (4, 102400), id="cluster-split", marks=pytest.mark.smoke
        ),
        # Rows a group of a warp's lanes holds, and rows one CTA holds in registers: both
        # exponentiate with an exp2 that flushes a subnormal result to zero.
        pytest.param(torch.float32, -100.0, (1, 512), id="warp-rows", marks=pytest.mark.full),
        pytest.param(torch.bfloat16, -88.0, (1, 2048), id="held-row", marks=pytest.mark.full),
    ],
)
def test_subnormal_probabilities_survive(dtype, low: float, shape: tuple) -> None:
    """A probability below the smallest normal float comes back as torch returns it.

    Every row is one zero and the rest ``low``, so all but one probability is subnormal
    in float32. A tolerance check cannot see them flushed to zero, so they are compared
    exactly against torch.
    """
    x = torch.full(shape, low, dtype=dtype, device=run_device())
    x[:, 0] = 0
    y = SoftmaxFwdOp(dim=-1)(x)
    expected = F.softmax(x, dim=-1)
    assert torch.equal(y[:, 1:] != 0, expected[:, 1:] != 0)
    torch.testing.assert_close(y, expected, rtol=1e-2, atol=0)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "shape, dtype",
    [
        # Rows a group of a warp's lanes holds.
        pytest.param((4, 512), torch.float32, id="warp-rows"),
        # Rows one CTA holds in registers.
        pytest.param((8, 2048), torch.float16, id="held-row"),
    ],
)
def test_register_held_rows_special_values(shape: tuple, dtype) -> None:
    """Rows held in registers return torch's result around special values.

    The ``-inf`` part of row 0 comes back zero; rows 1 to 3 come back NaN throughout, as
    torch returns them.
    """
    x = _special_value_rows(shape, dtype)
    compare_outputs(
        SoftmaxFwdOp(dim=-1)(x),
        F.softmax(x, dim=-1),
        softmax_verification(dtype, logarithmic=False),
    )


# The L2 policy of Softmax's input-row loads (``keeps_input_rows``): evict-last where a kernel
# reads each input row once into one CTA's registers and writes an output row for it, on an
# input past L2; the default load on an input within L2. The inputs past L2 take 128 MB, past
# the L2 of any supported device.


def _launched_sources(monkeypatch) -> list:
    """The CUDA source of every program launched or built from here on."""
    sources = []
    call, build = JITKernel.__call__, CompiledProgram.__init__

    def launched(self, *args, **kwargs):
        sources.append(self.get_kernel_source())
        return call(self, *args, **kwargs)

    def built(self, jit):
        sources.append(jit.get_kernel_source())
        build(self, jit)

    monkeypatch.setattr(JITKernel, "__call__", launched)
    monkeypatch.setattr(CompiledProgram, "__init__", built)
    return sources


def _run_and_capture(monkeypatch, shape, dtype, kernel) -> list:
    """The CUDA sources of the programs Softmax runs on *shape*, which must be *kernel*'s;
    the output is checked against the reference."""
    sources = _launched_sources(monkeypatch)
    test = SoftmaxTest(shape, dtype)
    op = SoftmaxFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())
    assert [type(k).__name__ for k in op.iter_kernels()] == [kernel]
    assert sources
    return sources


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "shape, dtype, kernel",
    [
        pytest.param((131072, 512), torch.float16, "SoftmaxWarpRowsKernel", id="warp-rows"),
        pytest.param((32768, 2048), torch.bfloat16, "SoftmaxKernel", id="held-row"),
    ],
)
def test_softmax_rows_past_l2_load_evict_last(monkeypatch, shape, dtype, kernel) -> None:
    """Every 16-byte input-row load, on an input past L2, is evict-last in L2."""
    for source in _run_and_capture(monkeypatch, shape, dtype, kernel):
        reads = [line for line in source.splitlines() if re.search(r"\bx\[|\(x \+ ", line)]
        assert reads
        assert all("tileops::load16_evict_last(" in line for line in reads)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "shape, dtype, kernel",
    [
        pytest.param((1024, 512), torch.float16, "SoftmaxWarpRowsKernel", id="warp-rows"),
        pytest.param((1024, 2048), torch.bfloat16, "SoftmaxKernel", id="held-row"),
    ],
)
def test_softmax_rows_within_l2_keep_the_default_load(monkeypatch, shape, dtype, kernel) -> None:
    """On an input within L2, the input rows load without an L2 eviction hint."""
    assert not any(
        "evict_last" in source for source in _run_and_capture(monkeypatch, shape, dtype, kernel)
    )
