"""SumFwdOp, MeanFwdOp, AminFwdOp, AmaxFwdOp, ProdFwdOp, StdFwdOp, VarFwdOp and VarMeanFwdOp
over one int dim; the variance ops also over a tuple and ``None``. Other ``dim`` forms are in
``test_reduction_dims.py``."""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops._ops.reduction import (
    AmaxFwdOp,
    AminFwdOp,
    MeanFwdOp,
    ProdFwdOp,
    StdFwdOp,
    SumFwdOp,
    VarFwdOp,
    VarMeanFwdOp,
    VectorNormFwdOp,
)
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.reduction import (
    ExtremaNanCase,
    ProdWorkload,
    ReduceCase,
    VarianceConstantCase,
    WelfordCase,
    _reduction_reference,
    reduction_verification,
)


class ReduceTest(ReduceCase, TestBase):
    pass


class ProdTest(ProdWorkload, TestBase):
    """Parameterized test helper for prod op (uses small-range inputs)."""

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        super().__init__((m, n), dtype)


class WelfordTest(WelfordCase, TestBase):
    pass


class ExtremaNanTest(ExtremaNanCase, TestBase):
    pass


class VarianceConstantTest(VarianceConstantCase, TestBase):
    pass


@pytest.mark.smoke
@pytest.mark.parametrize("kind", ["var", "std", "var_mean"])
def test_variance_constant_offset(kind) -> None:
    """A finite common offset does not poison the first partial merge."""

    axes = (0, 2)
    test = VarianceConstantTest((2, 3, 16), axes, kind, 1e20)
    op = {"var": VarFwdOp, "std": StdFwdOp, "var_mean": VarMeanFwdOp}[kind](dim=axes, correction=0)
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize("kind", ["amax", "amin"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape,axes",
    [
        pytest.param((3, 257), (1,), marks=pytest.mark.smoke, id="row-tail"),
        pytest.param((3, 256), (1,), marks=pytest.mark.full, id="row-vector"),
        pytest.param((3, 32769), (1,), marks=pytest.mark.full, id="row-tiled"),
        pytest.param((257, 3), (0,), marks=pytest.mark.full, id="leading"),
        pytest.param((4, 24, 2048), (0, 2), marks=pytest.mark.full, id="edge-fold"),
        pytest.param((17, 24, 2048), (0, 2), marks=pytest.mark.full, id="edge-passes"),
    ],
)
def test_extrema_preserves_nan(shape, axes, dtype, kind) -> None:
    """NaNs survive row, leading and edge reductions, including across partials."""

    test = ExtremaNanTest(shape, dtype, axes, kind)
    op = AmaxFwdOp(dim=axes) if kind == "amax" else AminFwdOp(dim=axes)
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "shape,dtype,kind",
    [
        pytest.param(
            (1024, 256), torch.float16, "amax", marks=pytest.mark.smoke, id="split-tail-f16"
        ),
        pytest.param(
            (1024, 256), torch.float32, "amin", marks=pytest.mark.smoke, id="split-tail-f32"
        ),
        pytest.param((1024, 512), torch.float16, "amax", marks=pytest.mark.full, id="split-f16"),
        pytest.param((1024, 512), torch.float32, "amin", marks=pytest.mark.full, id="rows-f32"),
        pytest.param(
            (1025, 512), torch.float16, "amax", marks=pytest.mark.full, id="split-ragged-f16"
        ),
        pytest.param(
            (1025, 512), torch.float32, "amin", marks=pytest.mark.full, id="ragged-rows-f32"
        ),
        pytest.param((2, 524288), torch.float16, "amax", marks=pytest.mark.full, id="vector-f16"),
        pytest.param((2, 524288), torch.float32, "amin", marks=pytest.mark.full, id="vector-f32"),
        pytest.param((2048, 4096), torch.float16, "amax", marks=pytest.mark.full, id="cluster-f16"),
        pytest.param(
            (1000, 4104), torch.float16, "amin", marks=pytest.mark.full, id="cluster-ragged-f16"
        ),
    ],
)
def test_extrema_leading_preserves_nan(shape, dtype, kind) -> None:
    """NaNs survive leading-axis reductions split across row slices (a column tail, even and
    short last slices), the one-launch vector walk over whole 16-byte tiles at two element
    widths, and the thread-block cluster that merges NaNs held by different CTAs (SM90)."""

    test = ExtremaNanTest(shape, dtype, (0,), kind)
    op = AmaxFwdOp(dim=0) if kind == "amax" else AminFwdOp(dim=0)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_extrema_tuned_preserves_nan() -> None:
    """Tuned mode builds and times every candidate; NaNs survive the one it keeps.

    The row is off the copy alignment, so it reads its end masked, and the odd row
    count leaves a multi-row candidate's last block short.
    """

    test = ExtremaNanTest((37, 1001), torch.float16, (1,), "amax")
    op = AmaxFwdOp(dim=1)
    op.request_tune()
    test.check(op, *test.gen_inputs())


_OPS = {
    "sum": SumFwdOp,
    "mean": MeanFwdOp,
    "amin": AminFwdOp,
    "amax": AmaxFwdOp,
    "prod": ProdFwdOp,
    "std": StdFwdOp,
    "var": VarFwdOp,
    "var_mean": VarMeanFwdOp,
}
#: One op per kernel family (simple, product, Welford) carries the shape boundaries.
_FAMILIES = ("sum", "prod", "var")


def _row_workload(op_kind: str, m: int, n: int, dtype: torch.dtype) -> TestBase:
    if op_kind == "prod":
        return ProdTest(m, n, dtype)
    if op_kind in ("std", "var", "var_mean"):
        return WelfordTest(m, n, dtype, op_kind, correction=1)
    return ReduceTest(m, n, dtype, op_kind)


def _row_cases() -> list:
    """Every op at every dtype is smoke; the families' row-shape boundaries follow."""
    smoke, full = pytest.mark.smoke, pytest.mark.full
    cases = [
        pytest.param(
            k,
            128,
            512,
            dtype,
            marks=[smoke, pytest.mark.packaging(family="reduction")]
            if (k, dtype) == ("sum", torch.float16)
            else smoke,
        )
        for k in _OPS
        for dtype in (torch.float16, torch.float32, torch.bfloat16)
    ]
    # N past one tile: the tiled path.
    cases += [pytest.param(k, 64, 32768, torch.bfloat16, marks=smoke) for k in _FAMILIES]
    for k in _FAMILIES:
        cases += [
            pytest.param(k, 256, 4096, torch.bfloat16, marks=full),
            pytest.param(k, 128, 300, torch.float16, marks=full, id=f"{k}-unaligned-n"),
            pytest.param(k, 129, 512, torch.float16, marks=full, id=f"{k}-tail-m"),
            pytest.param(k, 64, 32769, torch.bfloat16, marks=full, id=f"{k}-tiled-tail"),
        ]
    # Mean divides a padded row by its unpadded N.
    cases.append(pytest.param("mean", 128, 300, torch.float16, marks=full, id="mean-unaligned-n"))
    return cases


@pytest.mark.parametrize("op_kind, m, n, dtype", _row_cases())
def test_reduce_rows(op_kind: str, m: int, n: int, dtype: torch.dtype) -> None:
    test = _row_workload(op_kind, m, n, dtype)
    test.check(_OPS[op_kind](dim=-1), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "m, n, threads", [(4, 11008, 512), (1024, 8200, 1024)], ids=["512-threads", "1024-threads"]
)
def test_sum_wide_fold_tail(m: int, n: int, threads: int) -> None:
    """A row long enough for a wide fold that ends in a partial step.

    2752 fp32 vectors a row on 4 rows: five full steps of 512 threads, then a guarded
    step in which 192 threads load. 2050 vectors a row on 1024 rows: two full steps of
    1024 threads, then a guarded step in which 2 threads load.
    """

    op = SumFwdOp(dim=-1)
    test = ReduceTest(m, n, torch.float32, "sum")
    test.check(op, *test.gen_inputs())
    (entry,) = op._entries_built("reduce").values()
    assert entry.config["threads"] == threads


@pytest.mark.smoke
def test_reduce_untiled_autotune_unaligned_n() -> None:
    """Every untiled candidate builds at an N_padded off the pass width.

    N_padded=7936 neither divides nor is a multiple of threads * 16 bytes / elem_bytes, so
    ``layout_ok`` must leave only block_m 1; the row still fits the register budget, so it
    is not tiled.
    """

    m, n, dtype = 8, 7935, torch.float16
    test = ReduceTest(m, n, dtype, "sum")
    op = SumFwdOp(dim=-1)
    op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "op_kind",
    [
        pytest.param("sum", marks=pytest.mark.smoke),
        # Welford builds a different tiled kernel with two shared buffers.
        pytest.param("var", marks=pytest.mark.full),
    ],
)
def test_reduce_tiled_autotune(op_kind: str) -> None:
    """``request_tune()`` must build and time every tiled candidate.

    N is not a power of two: a power-of-two N_padded lets ``compute_tile_n``
    fall back on an exact divisor, which hides a mis-derived tile_n.
    """

    m, n, dtype = 4, 39999, torch.float16
    if op_kind == "sum":
        test = ReduceTest(m, n, dtype, "sum")
        op = SumFwdOp(dim=-1)
        op.request_tune()
    else:
        test = WelfordTest(m, n, dtype, "var", correction=1)
        op = VarFwdOp(dim=-1)
        op.request_tune()
    test.check(op, *test.gen_inputs())


def _check(op, x: torch.Tensor, **params) -> None:
    """Compare every output of *op* on *x* with the reduction oracle at *params*."""
    out = op(x)
    ref = _reduction_reference(type(op).__name__, x, params)
    pairs = zip(out, ref, strict=True) if isinstance(ref, tuple) else [(out, ref)]
    for got, want in pairs:
        assert got.shape == want.shape, f"shape {got.shape} vs ref {want.shape}"
        assert got.dtype == want.dtype, f"dtype {got.dtype} vs ref {want.dtype}"
        compare_outputs(got, want, reduction_verification(want.dtype))


def _input(op_kind: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Gaussian values; near one for a product, so it stays finite."""
    if op_kind == "prod":
        return torch.rand(*shape, dtype=dtype, device=run_device()) * 0.01 + 0.99
    return torch.randn(*shape, dtype=dtype, device=run_device())


def _dim_cases() -> list:
    """Every op at both dtypes on a row is smoke; leading, middle and keepdim per family."""
    cases = [
        pytest.param(k, shape, -1, False, dtype, marks=pytest.mark.smoke)
        for k in _OPS
        for shape, dtype in (((128, 512), torch.float16), ((4, 32, 512), torch.bfloat16))
    ]
    for k in _FAMILIES:
        cases += [
            pytest.param(k, (4, 32, 512), 0, False, torch.float16, marks=pytest.mark.full),
            pytest.param(k, (4, 32, 512), 1, False, torch.float16, marks=pytest.mark.full),
            pytest.param(k, (4, 32, 512), -1, True, torch.bfloat16, marks=pytest.mark.full),
        ]
    return cases


@pytest.mark.parametrize("op_kind, shape, dim, keepdim, dtype", _dim_cases())
def test_reduce_dim_and_keepdim(
    op_kind: str, shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype
) -> None:
    _check(
        _OPS[op_kind](dim=dim, keepdim=keepdim),
        _input(op_kind, shape, dtype),
        dim=dim,
        keepdim=keepdim,
        correction=None,
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_kind, view",
    [("sum", "strided"), ("sum", "1d"), ("var", "1d")],
    ids=["sum-strided", "sum-1d", "var-1d"],
)
def test_reduce_views(op_kind: str, view: str) -> None:
    """A strided view and a 1-D input reduce along their last axis."""
    if view == "strided":
        x = _input(op_kind, (128, 1024), torch.float16)[:, :512]
    else:
        x = _input(op_kind, (512,), torch.float16)
    _check(_OPS[op_kind](dim=-1), x, dim=-1, correction=None)


@pytest.mark.smoke
def test_reduce_view_off_vector_boundary() -> None:
    """A contiguous view whose storage starts off a 16-byte boundary reduces correctly."""

    flat = torch.rand(4 * 4096 + 1, dtype=torch.float16, device=run_device()) * 0.001 + 1
    x = flat[1:].view(4, 4096)
    assert x.is_contiguous() and x.data_ptr() % 16

    for op, ref in ((SumFwdOp(dim=-1), torch.sum), (ProdFwdOp(dim=-1), torch.prod)):
        y = op(x)
        compare_outputs(
            y,
            ref(x.float(), dim=-1).half(),
            reduction_verification(torch.float16),
        )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "name, in_dtype, out_dtype",
    [
        ("sum", torch.bfloat16, torch.float32),
        ("prod", torch.bfloat16, torch.float32),
        ("l2", torch.bfloat16, torch.float32),
        ("sum", torch.float32, torch.float16),
    ],
    ids=["sum-widen", "prod-widen", "l2-widen", "sum-narrow"],
)
def test_dtype_casts_the_input_before_reducing(name, in_dtype, out_dtype) -> None:
    """``dtype`` casts the input first and is the output's, as in torch."""

    op, ref_fn = {
        "sum": (SumFwdOp(dim=-1, dtype=out_dtype), lambda x: torch.sum(x, -1, dtype=out_dtype)),
        "prod": (ProdFwdOp(-1, dtype=out_dtype), lambda x: torch.prod(x, -1, dtype=out_dtype)),
        "l2": (
            VectorNormFwdOp(2, dim=-1, dtype=out_dtype),
            lambda x: torch.linalg.vector_norm(x, 2, -1, dtype=out_dtype),
        ),
    }[name]
    x = torch.rand(8, 512, device=run_device()).mul(0.01).add(0.995).to(in_dtype)
    got = op(x)
    assert got.dtype == out_dtype
    compare_outputs(got, ref_fn(x), reduction_verification(out_dtype))


_VARIANCE_DIMS = [
    pytest.param(-1, id="dim=int"),
    pytest.param((0, 2), id="dim=tuple"),
    pytest.param(None, id="dim=None"),
]
_VARIANCE_OPS = ["var", "std", "var_mean"]


def _check_variance(op_kind: str, x: torch.Tensor, dim, keepdim: bool, correction) -> None:
    op = _OPS[op_kind](dim=dim, correction=correction, keepdim=keepdim)
    _check(op, x, dim=dim, keepdim=keepdim, correction=correction)


@pytest.mark.parametrize("op_kind", _VARIANCE_OPS)
@pytest.mark.parametrize("dim", _VARIANCE_DIMS)
@pytest.mark.parametrize(
    "keepdim",
    [
        pytest.param(False, id="keepdim=False", marks=pytest.mark.smoke),
        pytest.param(True, id="keepdim=True", marks=pytest.mark.full),
    ],
)
def test_the_variance_output_shape_matches_torch(op_kind, dim, keepdim) -> None:
    """The two axes that pick branches, crossed: ``dim=None, keepdim=False`` is 0-D."""
    torch.manual_seed(0)
    x = torch.randn(4, 8, 256, dtype=torch.float16, device=run_device())
    _check_variance(op_kind, x, dim, keepdim, correction=1)


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", _VARIANCE_OPS)
@pytest.mark.parametrize("dim", [-1, (0, 2)], ids=["rows", "edge"])
def test_a_zero_correction_matches_torch(op_kind, dim) -> None:
    """The one ``correction`` that differs in kind: the denominator is ``N``, not ``N - c``;
    the rows kernel and the edge-axis merge each bake it in."""
    torch.manual_seed(0)
    x = torch.randn(4, 8, 256, dtype=torch.float16, device=run_device())
    _check_variance(op_kind, x, dim, keepdim=False, correction=0)


@pytest.mark.parametrize(
    "op_kind, shape, dim",
    [
        # The edge-axis path merges Welford partials.
        pytest.param("var", (4, 8, 256), (0, 2), marks=pytest.mark.smoke, id="edge"),
        # Padded rows, one tile and several: the pad is masked out of the centered sum.
        pytest.param("var", (4, 8, 255), -1, marks=pytest.mark.full, id="row"),
        pytest.param("var_mean", (4, 8, 255), -1, marks=pytest.mark.full, id="row-pair"),
        pytest.param("std", (4, 33000), -1, marks=pytest.mark.full, id="tiled-row"),
        pytest.param("var_mean", (4, 33000), -1, marks=pytest.mark.full, id="tiled-row-pair"),
    ],
)
def test_variance_keeps_a_large_mean_fp16(op_kind, shape, dim) -> None:
    """A mean far from zero, where a naive sum of squares would cancel."""
    torch.manual_seed(0)
    x = (torch.randn(*shape, dtype=torch.float16, device=run_device()) + 60.0).half()
    _check_variance(op_kind, x, dim, keepdim=False, correction=1)


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", _VARIANCE_OPS)
@pytest.mark.parametrize("dim", _VARIANCE_DIMS)
def test_variance_over_an_unaligned_innermost_dim_matches_torch(op_kind, dim) -> None:
    """255 flushes the masked-load boundary that a tile-multiple extent skips."""
    torch.manual_seed(0)
    x = torch.randn(4, 8, 255, dtype=torch.float16, device=run_device())
    _check_variance(op_kind, x, dim, keepdim=False, correction=1)


@pytest.mark.smoke
@pytest.mark.filterwarnings("ignore:.*degrees of freedom:UserWarning")
@pytest.mark.parametrize("correction", [0.5, 2, 3])
@pytest.mark.parametrize("op_cls, ref_fn", [(VarFwdOp, torch.var), (StdFwdOp, torch.std)])
def test_a_fractional_or_excess_variance_correction_matches_torch(
    op_cls, ref_fn, correction
) -> None:
    """torch divides by ``max(0, n - correction)``: a spread over no degrees of freedom is
    ``inf``, including one too small for the storage dtype, and no spread is NaN."""
    x = torch.tensor([[0.0, 1e-4], [1.0, 1.0]], dtype=torch.float16, device=run_device())
    got = op_cls(dim=1, correction=correction)(x)
    compare_outputs(
        got,
        ref_fn(x, dim=1, correction=correction),
        reduction_verification(x.dtype),
    )


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_the_arch_check_asks_about_the_input_s_device(monkeypatch) -> None:
    """Not whichever device is current: the two differ on a mixed-architecture host.

    A homogeneous host cannot show the wrong answer, so what is asserted is which device
    was asked about.
    """
    import tileops._utils as utils

    asked: list = []
    real = utils.device_facts

    def recording(index=None):
        asked.append(index)
        return real(index)

    # The call record carries the architecture the kernel is checked against.
    monkeypatch.setattr(utils, "device_facts", recording)

    x = torch.randn(4, 8, dtype=torch.float16, device=run_device())
    SumFwdOp(dim=-1)(x)

    assert asked, "the kernel declares supported_archs, so the call record must have probed"
    assert all(i == x.device.index for i in asked), asked
