"""Correctness tests for the 8 basic reduce ops.

Covers: SumFwdOp, MeanFwdOp, AminFwdOp, AmaxFwdOp, ProdFwdOp, StdFwdOp, VarFwdOp, VarMeanFwdOp.
Each op reduces along one dim of a 1D-4D input; the variance ops also over a tuple of dims
and ``dim=None``. Other ``dim`` forms are in ``test_reduction_dims.py``.
"""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
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
    reduction_verification,
)


class ReduceBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(
                    128,
                    512,
                    torch.float16,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="reduction")],
                ),
                pytest.param(128, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(256, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-aligned N
                pytest.param(128, 300, torch.float16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(129, 512, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class ReduceTiledFixture(FixtureBase):
    """Large-N cases that exercise the tiled reduce path (N > MAX_SINGLE_TILE_COLS).

    One representative op per kernel family: sum (simple), prod, var (welford).
    """

    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(64, 32768, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(64, 32769, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class ReduceNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Reduce1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(512, torch.float16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class BesselFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, correction",
            [
                pytest.param(128, 512, torch.float16, 0, marks=pytest.mark.smoke),
            ],
        ),
    ]


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
        pytest.param((1024, 512), torch.float32, "amin", marks=pytest.mark.full, id="split-f32"),
        pytest.param(
            (1025, 512), torch.float16, "amax", marks=pytest.mark.full, id="split-ragged-f16"
        ),
        pytest.param(
            (1025, 512), torch.float32, "amin", marks=pytest.mark.full, id="split-ragged-f32"
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
    """NaNs survive leading-axis reductions split across row slices, the vector walk and
    the cluster's one launch.

    The row slices split a column tail, divide the rows, or leave the last slice short;
    columns of whole 16-byte-vector tiles take the one-launch vector walk. A 2-byte and a
    4-byte element are walked to different depths. Rows of whole vectors wide enough to
    fill the device fold in one launch across a thread-block cluster, which merges NaNs
    held by different CTAs.
    """

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


@ReduceBasicFixture
def test_sum_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = ReduceTest(m, n, dtype, "sum")
    op = SumFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@ReduceTiledFixture
def test_sum_tiled(m: int, n: int, dtype: torch.dtype) -> None:
    test = ReduceTest(m, n, dtype, "sum")
    op = SumFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@ReduceTiledFixture
def test_prod_tiled(m: int, n: int, dtype: torch.dtype) -> None:
    test = ProdTest(m, n, dtype)
    op = ProdFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@ReduceTiledFixture
def test_var_tiled(m: int, n: int, dtype: torch.dtype) -> None:
    test = WelfordTest(m, n, dtype, "var", correction=1)
    op = VarFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


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
    """Every untiled candidate must build at an N_padded off the pass width.

    N_padded=7936 neither divides nor is a multiple of
    threads * 128-bit / elem_bytes, so ``layout_ok`` excludes every block_m
    above 1 and the candidate list must not offer them.  With the serial row
    loop such a fragment would in fact build; the envelope stays because it
    also covers the residue that does not.  The width also has to stay inside the
    register budget, or the row is tiled and there is no untiled candidate list.
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


@ReduceNonContigFixture
def test_sum_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    x_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    x = x_full[:, :n]
    op = SumFwdOp(dim=-1)
    ref = x.contiguous().float().sum(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, reduction_verification(ref.dtype))


@ReduceBasicFixture
def test_mean_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = ReduceTest(m, n, dtype, "mean")
    op = MeanFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@ReduceBasicFixture
def test_amin_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = ReduceTest(m, n, dtype, "amin")
    op = AminFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@ReduceBasicFixture
def test_amax_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = ReduceTest(m, n, dtype, "amax")
    op = AmaxFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@ReduceBasicFixture
def test_prod_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = ProdTest(m, n, dtype)
    op = ProdFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@ReduceBasicFixture
def test_std_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = WelfordTest(m, n, dtype, "std", correction=1)
    op = StdFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@BesselFixture
def test_std_bessel(m: int, n: int, dtype: torch.dtype, correction: int) -> None:
    test = WelfordTest(m, n, dtype, "std", correction=correction)
    op = StdFwdOp(correction=correction, dim=-1)
    test.check(op, *test.gen_inputs())


@ReduceBasicFixture
def test_var_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = WelfordTest(m, n, dtype, "var", correction=1)
    op = VarFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@BesselFixture
def test_var_bessel(m: int, n: int, dtype: torch.dtype, correction: int) -> None:
    test = WelfordTest(m, n, dtype, "var", correction=correction)
    op = VarFwdOp(correction=correction, dim=-1)
    test.check(op, *test.gen_inputs())


@ReduceBasicFixture
def test_var_mean_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = WelfordTest(m, n, dtype, "var_mean", correction=1)
    op = VarMeanFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@BesselFixture
def test_var_mean_bessel(m: int, n: int, dtype: torch.dtype, correction: int) -> None:
    test = WelfordTest(m, n, dtype, "var_mean", correction=correction)
    op = VarMeanFwdOp(correction=correction, dim=-1)
    test.check(op, *test.gen_inputs())


# 1D input


@Reduce1DFixture
def test_sum_1d(n: int, dtype: torch.dtype) -> None:
    x = torch.randn(n, dtype=dtype, device=run_device())
    op = SumFwdOp(dim=-1)
    ref = x.float().sum(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y.view_as(ref), ref, reduction_verification(ref.dtype))


@Reduce1DFixture
def test_var_1d(n: int, dtype: torch.dtype) -> None:
    x = torch.randn(n, dtype=dtype, device=run_device())
    op = VarFwdOp(dim=-1)
    ref = x.float().var(dim=-1, correction=1).to(dtype)
    y = op(x)

    compare_outputs(y.view_as(ref), ref, reduction_verification(ref.dtype))


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


# Spec-conformant tests (dim + keepdim interface)


class SpecReduceFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, keepdim, dtype",
            [
                # Each spec test also has fp16 full cases; keep one fp16 smoke case so the
                # collection contract covers every dtype at the smoke tier.
                pytest.param((128, 512), -1, False, torch.float16, marks=pytest.mark.smoke),
                pytest.param((4, 32, 512), -1, False, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param((4, 32, 512), 0, False, torch.float16, marks=pytest.mark.full),
                pytest.param((4, 32, 512), 1, False, torch.float16, marks=pytest.mark.full),
                pytest.param((4, 32, 512), -1, True, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@SpecReduceFixture
def test_sum_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: reduction along arbitrary dim (0, 1, -1) for 2D/3D tensors."""

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = SumFwdOp(dim=dim, keepdim=keepdim)
    ref = torch.sum(x.float(), dim=dim, keepdim=keepdim).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@SpecReduceFixture
def test_mean_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: MeanFwdOp with dim + keepdim."""

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = MeanFwdOp(dim=dim, keepdim=keepdim)
    ref = torch.mean(x.float(), dim=dim, keepdim=keepdim).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@SpecReduceFixture
def test_amax_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: AmaxFwdOp with dim + keepdim."""

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = AmaxFwdOp(dim=dim, keepdim=keepdim)
    ref = torch.amax(x.float(), dim=dim, keepdim=keepdim).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@SpecReduceFixture
def test_amin_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: AminFwdOp with dim + keepdim."""

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = AminFwdOp(dim=dim, keepdim=keepdim)
    ref = torch.amin(x.float(), dim=dim, keepdim=keepdim).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@SpecReduceFixture
def test_prod_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: ProdFwdOp with dim + keepdim."""

    x = torch.rand(*shape, dtype=dtype, device=run_device()) * 0.01 + 0.99
    op = ProdFwdOp(dim=dim, keepdim=keepdim)
    ref = torch.prod(x.float(), dim=dim, keepdim=keepdim).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@SpecReduceFixture
def test_var_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: VarFwdOp with dim + keepdim + correction."""

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = VarFwdOp(dim=dim, keepdim=keepdim)
    ref = torch.var(x.float(), dim=dim, keepdim=keepdim, correction=1).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@SpecReduceFixture
def test_std_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: StdFwdOp with dim + keepdim + correction."""

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = StdFwdOp(dim=dim, keepdim=keepdim)
    ref = torch.std(x.float(), dim=dim, keepdim=keepdim, correction=1).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@SpecReduceFixture
def test_var_mean_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: VarMeanFwdOp with dim + keepdim."""

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = VarMeanFwdOp(dim=dim, keepdim=keepdim)
    ref_var = torch.var(x.float(), dim=dim, keepdim=keepdim, correction=1).to(dtype)
    ref_mean = torch.mean(x.float(), dim=dim, keepdim=keepdim).to(dtype)
    var_out, mean_out = op(x)

    assert var_out.shape == ref_var.shape, f"var shape mismatch: {var_out.shape} vs {ref_var.shape}"
    assert mean_out.shape == ref_mean.shape, "mean shape mismatch"
    compare_outputs(var_out, ref_var, reduction_verification(ref_var.dtype))
    compare_outputs(mean_out, ref_mean, reduction_verification(ref_mean.dtype))


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


# Variance reductions against torch.var / torch.std / torch.var_mean. The three share a
# forward, so they share a case table. The dim shape and keepdim each pick a branch, so they
# are crossed; correction is a constant the kernel bakes in, where only zero and nonzero differ.


_VARIANCE_DIMS = [
    pytest.param(-1, id="dim=int"),
    pytest.param((0, 2), id="dim=tuple"),
    pytest.param(None, id="dim=None"),
]


def _ref_var(x, dim, keepdim, correction):
    return (torch.var(x.float(), dim=dim, keepdim=keepdim, correction=correction).to(x.dtype),)


def _ref_std(x, dim, keepdim, correction):
    return (torch.std(x.float(), dim=dim, keepdim=keepdim, correction=correction).to(x.dtype),)


def _ref_var_mean(x, dim, keepdim, correction):
    var, mean = torch.var_mean(x.float(), dim=dim, keepdim=keepdim, correction=correction)
    return var.to(x.dtype), mean.to(x.dtype)


#: Each op with the reference it must match. Every reference returns a tuple, so one
#: comparison serves the two single-output ops and the one that returns a pair.
_VARIANCE_OPS = [
    pytest.param(VarFwdOp, _ref_var, id="var"),
    pytest.param(StdFwdOp, _ref_std, id="std"),
    pytest.param(VarMeanFwdOp, _ref_var_mean, id="var-mean"),
]


def _check_variance(op_cls, ref_fn, x, dim, keepdim, correction) -> None:
    """Run *op_cls* against its reference and compare every output it returns."""
    out = op_cls(dim=dim, correction=correction, keepdim=keepdim)(x)
    got = out if isinstance(out, tuple) else (out,)
    want = ref_fn(x, dim, keepdim, correction)
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape, f"shape {g.shape} vs ref {w.shape}"
        assert g.dtype == w.dtype, f"dtype {g.dtype} vs ref {w.dtype}"
        compare_outputs(g, w, reduction_verification(w.dtype))


@pytest.mark.parametrize("op_cls, ref_fn", _VARIANCE_OPS)
@pytest.mark.parametrize("dim", _VARIANCE_DIMS)
@pytest.mark.parametrize(
    "keepdim",
    [
        pytest.param(False, id="keepdim=False", marks=pytest.mark.smoke),
        pytest.param(True, id="keepdim=True", marks=pytest.mark.full),
    ],
)
def test_the_variance_output_shape_matches_torch(op_cls, ref_fn, dim, keepdim) -> None:
    """The two axes that pick branches, crossed: ``dim=None, keepdim=False`` is 0-D."""
    shape = (4, 8, 256)
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float16, device=run_device())

    _check_variance(op_cls, ref_fn, x, dim, keepdim, correction=1)


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls, ref_fn", _VARIANCE_OPS)
def test_a_zero_correction_on_edge_axes_matches_torch(op_cls, ref_fn) -> None:
    """The one ``correction`` that differs in kind: the denominator is ``N``, not ``N - c``.

    ``dim=(0, 2)`` bakes the correction into the edge-axis merge; the rows kernel's
    denominator is test_*_bessel's.
    """
    dim = (0, 2)
    shape = (4, 8, 256)
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float16, device=run_device())

    _check_variance(op_cls, ref_fn, x, dim=dim, keepdim=False, correction=0)


@pytest.mark.parametrize(
    "op_cls, ref_fn, shape, dim",
    [
        # The edge-axis path merges Welford partials.
        pytest.param(VarFwdOp, _ref_var, (4, 8, 256), (0, 2), marks=pytest.mark.smoke, id="edge"),
        # Padded rows, one tile and several: the pad is masked out of the centered sum.
        pytest.param(VarFwdOp, _ref_var, (4, 8, 255), -1, marks=pytest.mark.full, id="row"),
        pytest.param(
            VarMeanFwdOp, _ref_var_mean, (4, 8, 255), -1, marks=pytest.mark.full, id="row-pair"
        ),
        pytest.param(StdFwdOp, _ref_std, (4, 33000), -1, marks=pytest.mark.full, id="tiled-row"),
        pytest.param(
            VarMeanFwdOp,
            _ref_var_mean,
            (4, 33000),
            -1,
            marks=pytest.mark.full,
            id="tiled-row-pair",
        ),
    ],
)
def test_variance_keeps_a_large_mean_fp16(op_cls, ref_fn, shape, dim) -> None:
    """A mean far from zero, where a naive sum of squares would cancel."""
    torch.manual_seed(0)
    x = (torch.randn(*shape, dtype=torch.float16, device=run_device()) + 60.0).half()

    _check_variance(op_cls, ref_fn, x, dim=dim, keepdim=False, correction=1)


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls, ref_fn", _VARIANCE_OPS)
@pytest.mark.parametrize("dim", _VARIANCE_DIMS)
def test_variance_over_an_unaligned_innermost_dim_matches_torch(op_cls, ref_fn, dim) -> None:
    """255 flushes the masked-load boundary that a tile-multiple extent skips."""
    unaligned_shape = (4, 8, 255)
    torch.manual_seed(0)
    x = torch.randn(*unaligned_shape, dtype=torch.float16, device=run_device())

    _check_variance(op_cls, ref_fn, x, dim, keepdim=False, correction=1)


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
