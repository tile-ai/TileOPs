"""How the reduction ops read dim: a list or tuple of axes, None and the default, an
empty sequence, negative and duplicate axes, and an innermost extent no tile divides once the
reduced axes move to the end.

A single int dim is covered per op in test_reduce.py, test_softmax.py,
test_logical_reduce.py and test_vector_norm.py.
"""

from __future__ import annotations

import math

import pytest
import torch

from tileops._ops.reduction import (
    AllFwdOp,
    AmaxFwdOp,
    AminFwdOp,
    AnyFwdOp,
    CountNonzeroFwdOp,
    LogSumExpFwdOp,
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
    LogSumExpCase,
    _reduction_reference,
    reduction_tolerance,
    reduction_verification,
    vector_norm_verification,
)

_OPS = {
    "sum": (SumFwdOp, {}),
    "mean": (MeanFwdOp, {}),
    "amax": (AmaxFwdOp, {}),
    "amin": (AminFwdOp, {}),
    "var": (VarFwdOp, {}),
    "std": (StdFwdOp, {}),
    "var_mean": (VarMeanFwdOp, {}),
    "logsumexp": (LogSumExpFwdOp, {}),
    "l1": (VectorNormFwdOp, {"ord": 1}),
    "l2": (VectorNormFwdOp, {"ord": 2}),
    "inf": (VectorNormFwdOp, {"ord": math.inf}),
    "all": (AllFwdOp, {}),
    "any": (AnyFwdOp, {}),
    "count_nonzero": (CountNonzeroFwdOp, {}),
}
_LOGICAL = ("all", "any", "count_nonzero")
#: One op per kernel family carries the shape and keepdim boundaries.
_FAMILIES = ("sum", "var", "logsumexp", "l2")


def _build(name: str, **kwargs):
    op_cls, extra = _OPS[name]
    if name == "count_nonzero":
        kwargs.pop("keepdim", None)
    return op_cls(**extra, **kwargs)


def _input(name: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Gaussian values; ``{-1, 0, 1}`` for a logical reduction, so it sees both truths."""
    if name in _LOGICAL:
        return torch.randint(-1, 2, shape, device=run_device()).to(dtype)
    return torch.randn(*shape, dtype=dtype, device=run_device())


def _check(name: str, x: torch.Tensor, dim, keepdim: bool = False) -> None:
    """*name* over *dim* on *x* against the reduction oracle, shape, dtype and values."""
    op_cls, extra = _OPS[name]
    y = _build(name, dim=dim, keepdim=keepdim)(x)
    # torch.amax and torch.logsumexp take no dim=None.
    axes = list(range(x.ndim)) if dim is None else dim
    ref = _reduction_reference(
        op_cls.__name__, x, dict(extra, dim=axes, keepdim=keepdim, correction=None)
    )
    if name == "logsumexp":
        verification = LogSumExpCase(tuple(x.shape), x.dtype).verification(x)
    elif op_cls is VectorNormFwdOp:
        verification = vector_norm_verification(x.dtype)
    else:
        verification = None
    pairs = zip(y, ref, strict=True) if isinstance(ref, tuple) else [(y, ref)]
    for got, want in pairs:
        assert got.shape == want.shape, f"{name}: shape {got.shape} vs ref {want.shape}"
        assert got.dtype == want.dtype, f"{name}: dtype {got.dtype} vs ref {want.dtype}"
        compare_outputs(got, want, verification or reduction_verification(want.dtype))


def _multidim_cases() -> list:
    """Every arithmetic op on two leading axes is smoke; each family also takes keepdim, two
    middle axes and the first and last; the logical ops take their read dtypes."""
    shape3, shape4 = (4, 32, 256), (2, 4, 8, 256)
    smoke, full = pytest.mark.smoke, pytest.mark.full
    cases = [
        pytest.param(k, shape3, [0, 1], False, torch.float16, marks=smoke)
        for k in _OPS
        if k not in _LOGICAL
    ]
    cases += [
        pytest.param(k, shape3, [0, 1], False, dtype, marks=smoke)
        for k in _LOGICAL
        for dtype in (torch.float32, torch.bool, torch.complex64)
    ]
    for k in _FAMILIES:
        cases += [
            pytest.param(k, shape3, [0, 1], True, torch.float16, marks=full),
            pytest.param(k, shape4, [1, 2], False, torch.float16, marks=full),
            pytest.param(k, shape4, [0, 3], False, torch.float16, marks=full),
        ]
    cases += [
        pytest.param(k, shape3, [0, 1], True, torch.float32, marks=full) for k in ("all", "any")
    ]
    return cases


@pytest.mark.parametrize("name, shape, dims, keepdim, dtype", _multidim_cases())
def test_a_list_of_axes_matches_torch(
    name: str, shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype
) -> None:
    _check(name, _input(name, shape, dtype), dims, keepdim)


def _dim_none_cases() -> list:
    """Every op at fp16 is smoke (fp32 for the variance ops, bool for all and any, whose other
    dtypes are test_reduce.py's and test_logical_reduce.py's conformance cells); fp32, whose
    4-byte element changes the flattened row's vector width, and keepdim per family."""
    shape = (4, 8, 256)
    smoke, full = pytest.mark.smoke, pytest.mark.full
    first = {"var": torch.float32, "std": torch.float32, "var_mean": torch.float32}
    first.update(all=torch.bool, any=torch.bool)
    cases = [
        pytest.param(k, shape, False, first.get(k, torch.float16), marks=smoke)
        for k in _OPS
        if k not in ("logsumexp", "count_nonzero")
    ]
    cases += [pytest.param(k, shape, False, torch.float32, marks=smoke) for k in ("sum", "l2")]
    cases += [pytest.param(k, shape, True, torch.float16, marks=full) for k in ("sum", "l2")]
    return cases


@pytest.mark.parametrize("name, shape, keepdim, dtype", _dim_none_cases())
def test_dim_none_reduces_every_axis(
    name: str, shape: tuple, keepdim: bool, dtype: torch.dtype
) -> None:
    _check(name, _input(name, shape, dtype), None, keepdim)


@pytest.mark.smoke
def test_mean_edge_axes_fp16_keeps_fp32_intermediates() -> None:
    """A mean over edge axes must not narrow its partial sums to the storage dtype.

    fp16 rows of 100.0 sum to 1740800 per row — past fp16's max — so any pass
    that casts an undivided sum back to fp16 answers inf instead of 100. Seventeen
    segments a row is past what the one-launch fold takes, so the edge kernel runs.
    """

    x = torch.full((17, 8, 1024), 100.0, dtype=torch.float16, device=run_device())
    y = MeanFwdOp(dim=[0, 2])(x)
    assert torch.isfinite(y).all(), "edge-axes mean overflowed an intermediate"
    compare_outputs(y, torch.full_like(y, 100.0), reduction_verification(y.dtype))


@pytest.mark.smoke
def test_prod_multidim_rejected() -> None:
    """ProdFwdOp narrows ``dim`` to ``int`` per its manifest signature, so
    the multi-dim (``list[int]`` / ``tuple[int, ...]``) overload is rejected
    at construction time; so is an empty sequence."""

    for dim in ([0, 1], (0, 1), [], None):
        with pytest.raises(ValueError, match="ProdFwdOp: dim = "):
            ProdFwdOp(dim=dim)


@pytest.mark.smoke
def test_logsumexp_edge_axes_special_values() -> None:
    """Own-layout edge-axis logsumexp preserves -inf and NaN row semantics."""

    x = torch.randn(4, 32, 256, dtype=torch.float16, device=run_device())
    x[:, 0, :] = float("-inf")
    x[2, 1, 7] = float("nan")
    y = LogSumExpFwdOp(dim=[0, 2])(x)
    ref = torch.logsumexp(x.float(), dim=[0, 2]).to(x.dtype)
    assert y[0].item() == float("-inf")
    assert torch.isnan(y[1])
    finite = torch.isfinite(ref)
    compare_outputs(y[finite], ref[finite], LogSumExpCase(tuple(x.shape), x.dtype).verification(x))


@pytest.mark.smoke
def test_negative_dims_accepted() -> None:
    """Negative dims should be normalized and produce correct results."""

    x = torch.randn(4, 8, 256, dtype=torch.float16, device=run_device())
    op = SumFwdOp(dim=[-1, 0], keepdim=False)
    ref = torch.sum(x.float(), dim=[0, 2], keepdim=False).to(torch.float16)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification(ref.dtype))


@pytest.mark.smoke
def test_duplicate_dims_raises() -> None:
    """Duplicate dims (after normalization) must raise ValueError at op level."""

    x = torch.randn(4, 8, 256, dtype=torch.float16, device=run_device())
    op = SumFwdOp(dim=[1, 1], keepdim=False)
    with pytest.raises(ValueError, match="unique_axes"):
        op(x)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_edge_axis_reduce_returns_the_storage_dtype(dtype: torch.dtype) -> None:
    """The edge-axis columns pass writes the result dtype itself, with no cast after.

    Nothing downstream converts, so a pass left writing fp32 reaches the caller
    as fp32. Seventeen segments a row is past what the one-launch fold takes.
    """

    x = torch.randn(17, 32, 512, dtype=dtype, device=run_device())
    for op_cls, ref in (
        (SumFwdOp, lambda z: torch.sum(z.float(), dim=[0, 2])),
        (MeanFwdOp, lambda z: torch.mean(z.float(), dim=[0, 2])),
        (AmaxFwdOp, lambda z: torch.amax(z.float(), dim=[0, 2])),
    ):
        out = op_cls(dim=[0, 2])(x)
        assert out.dtype == dtype, f"{op_cls.__name__} returned {out.dtype}"
        compare_outputs(out, ref(x).to(dtype), reduction_verification(dtype))

    counted = CountNonzeroFwdOp(dim=[0, 2])(x)
    assert counted.dtype == torch.int64
    compare_outputs(
        counted,
        torch.count_nonzero(x, dim=[0, 2]),
        reduction_verification(torch.int64),
    )


@pytest.mark.smoke
@pytest.mark.parametrize("shape", [(769, 516)], ids=["ragged-split"])
def test_leading_axis_split_reduces_every_row(shape):
    """A split that does not divide the reduced extent still sums every row.

    Rows of 516 fp16 are not whole 16-byte vectors, which the cluster pass refuses, so they
    take the split launches on every board.
    """

    x = torch.randn(shape, dtype=torch.float16, device=run_device())
    compare_outputs(SumFwdOp(dim=0)(x), x.sum(dim=0), reduction_verification(x.dtype))


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_cluster_split_follows_the_sm_count() -> None:
    """The cluster is the largest power of two keeping one CTA of one L2 line per SM, and
    the one launch stands down where the portable cap leaves half the SMs idle, where rows
    are not whole 16-byte vectors, and before thread-block clusters."""
    from tileops._kernels.reduction._primitives import down_rows_cluster

    # 4096 bf16 columns are 64 lines: 2 CTAs each fill 128 of 132 SMs.
    assert down_rows_cluster(2048, 4096, "bfloat16", 90, 132) == 2
    # 128 fp32 lines already hold a CTA per SM.
    assert down_rows_cluster(2048, 4096, "float32", 90, 132) == 1
    # 12 lines take the portable cap of 8: 96 CTAs.
    assert down_rows_cluster(2048, 768, "bfloat16", 90, 132) == 8
    # 8 lines at the cap reach 64 of 132 SMs.
    assert down_rows_cluster(2048, 512, "bfloat16", 90, 132) == 0
    assert down_rows_cluster(2048, 4100, "bfloat16", 90, 132) == 0
    assert down_rows_cluster(2048, 4096, "bfloat16", 80, 108) == 0


@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param((2048, 4096), torch.bfloat16, marks=pytest.mark.smoke, id="cluster-2"),
        pytest.param(
            (1001, 4104), torch.float16, marks=pytest.mark.smoke, id="ragged-rows-and-line"
        ),
        pytest.param((1024, 512), torch.float32, marks=pytest.mark.smoke, id="cluster-8"),
        pytest.param(
            (1001, 4104), torch.float32, marks=pytest.mark.smoke, id="cluster-1-partial-line"
        ),
        pytest.param(
            (3, 65536), torch.bfloat16, marks=pytest.mark.full, id="fewer-rows-than-groups"
        ),
        pytest.param((2048, 2048), torch.bfloat16, marks=pytest.mark.full, id="cluster-4"),
        pytest.param(
            (2, 128256), torch.bfloat16, marks=pytest.mark.full, id="band-wider-than-threads"
        ),
    ],
)
def test_leading_axis_cluster_reduces_every_row_and_column(shape, dtype) -> None:
    """The one-launch cluster pass folds every row and column into each op's result.

    The ragged shape gives the second CTA of the cluster one row fewer than the first,
    leaves its row groups' last step short and its last line partial. Its fp32 twin spans
    129 lines, so each CTA walks every row alone and the last CTA stores 8 of its 32
    columns. The 3-row one takes a 2-CTA cluster of four row groups each 16 lines wide: the
    first CTA reads two rows and the second one, so most groups get no row. The 2-row one
    takes two row groups of 32 lines, so each CTA finishes more columns than it has threads.
    """
    # A sentinel row follows the reduced rows in memory: a read past them shows.
    rows = torch.rand(shape[0] + 1, *shape[1:], dtype=dtype, device=run_device()) * 0.04 + 0.98
    rows[-1] = 100.0
    x = rows[:-1]
    for op, ref in (
        (SumFwdOp(dim=0), torch.sum(x.float(), dim=0)),
        (MeanFwdOp(dim=0), torch.mean(x.float(), dim=0)),
        (AmaxFwdOp(dim=0), torch.amax(x.float(), dim=0)),
        (AminFwdOp(dim=0), torch.amin(x.float(), dim=0)),
    ):
        compare_outputs(op(x), ref.to(dtype), reduction_verification(dtype))
    compare_outputs(
        ProdFwdOp(dim=0)(x),
        torch.prod(x.float(), dim=0).to(dtype),
        reduction_verification(dtype, product=True),
    )


@pytest.mark.smoke
def test_leading_axis_view_off_the_vector_boundary_reduces() -> None:
    """A view that starts between 16-byte vectors, which the cluster pass reads whole,
    reduces to the same sums."""
    flat = torch.randn(2048 * 4096 + 1, dtype=torch.bfloat16, device=run_device())
    x = flat[1:].view(2048, 4096)
    compare_outputs(
        SumFwdOp(dim=0)(x), x.float().sum(dim=0).bfloat16(), reduction_verification(x.dtype)
    )


@pytest.mark.smoke
def test_edge_columns_pass_in_one_cluster_launch() -> None:
    """The edge kernels' columns pass over enough fp32 partials takes the one launch, with
    and without a square-root epilogue."""
    x = torch.randn(40, 4096, 160, dtype=torch.float16, device=run_device())
    compare_outputs(
        SumFwdOp(dim=[0, 2])(x),
        torch.sum(x.float(), dim=[0, 2]).to(x.dtype),
        reduction_verification(x.dtype),
    )
    compare_outputs(
        VectorNormFwdOp(ord=2, dim=[0, 2])(x),
        torch.linalg.vector_norm(x.float(), 2, (0, 2)).to(x.dtype),
        vector_norm_verification(x.dtype),
    )


def _vector_tiles_kept(dtype: torch.dtype) -> int:
    """The fewest kept columns the one-launch vector down-rows pass takes."""
    from tileops._kernels.constants import VECTOR_ACCESS_BYTES
    from tileops._kernels.reduction._primitives import _LEADING_POLICY

    vec = VECTOR_ACCESS_BYTES // dtype.itemsize
    return _LEADING_POLICY.min_vector_blocks * _LEADING_POLICY.vector_threads * vec


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_leading_axis_vector_tiles_reduce_every_column(dtype: torch.dtype) -> None:
    """At the fewest vector tiles every column of every row reaches the result."""

    kept = _vector_tiles_kept(dtype)
    x = torch.rand(4, 2, kept // 2, dtype=dtype, device=run_device()) * 0.5 + 0.75
    for op, ref in (
        (ProdFwdOp(dim=0), torch.prod(x.float(), dim=0)),
        (MeanFwdOp(dim=0), torch.mean(x.float(), dim=0)),
        (AmaxFwdOp(dim=0), torch.amax(x.float(), dim=0)),
    ):
        compare_outputs(op(x), ref.to(dtype), reduction_verification(dtype))


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_vector_tiles_take_whole_tiles_from_the_minimum() -> None:
    """The vector pass takes the minimum of whole tiles, not one tile fewer or a ragged width;
    the logical and vector-norm callers' dtypes and epilogue reduce exactly on it, and the
    logical callers' dtypes on split rows, which an SM90 board folds across a cluster."""
    from tileops._kernels.reduction._primitives import (
        _LEADING_POLICY,
        _vector_tiles,
        down_rows_launcher,
    )
    from tileops._utils import device_facts

    kept = _vector_tiles_kept(torch.float16)
    tile = kept // _LEADING_POLICY.min_vector_blocks
    assert _vector_tiles(kept, "float16")
    assert not _vector_tiles(kept - tile, "float16")
    assert not _vector_tiles(kept + tile // 2, "float16")

    device = run_device()
    flags = torch.randint(
        0, 2, (3, _vector_tiles_kept(torch.int8)), dtype=torch.int8, device=device
    )
    counts = torch.randint(0, 9, (3, _vector_tiles_kept(torch.float32)), device=device).float()
    # Split rows of the logical callers' dtypes: an SM90 board folds them across a cluster.
    rare = (torch.rand(300, 4096, device=device) > 0.995).to(torch.int8)
    tallies = torch.randint(0, 9, (300, 4096), device=device).float()
    facts = device_facts(device.index)
    for op, dtypes, epilogue, x, splits, ref in (
        ("amin", ("int8", "int8"), "", flags, 1, flags.amin(0)),
        ("sum", ("float32", "int64"), "", counts, 1, counts.sum(0).long()),
        ("sum", ("float32", "float16"), "sqrt", counts, 1, counts.sum(0).sqrt().half()),
        ("amax", ("int8", "int8"), "", rare, 2, rare.amax(0)),
        ("amin", ("int8", "int8"), "", 1 - rare, 2, (1 - rare).amin(0)),
        ("sum", ("float32", "int64"), "", tallies, 2, tallies.sum(0).long()),
    ):
        torch.testing.assert_close(
            down_rows_launcher(
                *x.shape,
                op,
                *dtypes,
                0.0,
                splits,
                epilogue,
                arch=facts.arch,
                sm_count=facts.sm_count,
            )(x),
            ref,
        )


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "lead, folds", [(16, True), (17, False)], ids=["row-in-flight", "row-past-in-flight"]
)
def test_edge_axes_route_by_the_row_a_block_holds_in_flight(lead: int, folds: bool) -> None:
    """A row a 512-thread block holds in flight, 16 loads a thread, folds in one launch.

    Each row is ``lead`` segments of 1000 fp16 (125 vectors), so every segment ends in a
    partial step. One segment more takes the edge kernels' two passes.
    """

    dtype = torch.float16
    x = torch.randn(lead, 8, 1000, dtype=dtype, device=run_device())
    for op, ref, verification in (
        (SumFwdOp(dim=[0, 2]), torch.sum(x.float(), dim=[0, 2]), reduction_verification),
        (MeanFwdOp(dim=[0, 2]), torch.mean(x.float(), dim=[0, 2]), reduction_verification),
        (AmaxFwdOp(dim=[0, 2]), torch.amax(x.float(), dim=[0, 2]), reduction_verification),
        (AminFwdOp(dim=[0, 2]), torch.amin(x.float(), dim=[0, 2]), reduction_verification),
        (
            VectorNormFwdOp(ord=2, dim=[0, 2]),
            torch.linalg.vector_norm(x.float(), 2, (0, 2)),
            vector_norm_verification,
        ),
    ):
        out = op(x)
        (entry,) = op._entries_built("reduce").values()
        name = type(entry).__name__
        assert (name == "ReduceFoldKernel") == folds, f"{type(op).__name__} took {name}"
        compare_outputs(out, ref.to(dtype), verification(dtype))


# Logical reduce ops: all, any, count_nonzero


# Vector norm ops: l1, l2, inf


@pytest.mark.smoke
@pytest.mark.parametrize("name", [k for k in _OPS if k not in ("logsumexp", "l1", "l2", "inf")])
def test_the_default_dim_reduces_every_axis(name: str) -> None:
    """The manifest declares ``default: null`` on ``dim``: the op built without one reduces
    every axis."""
    x = _input(name, (2, 4, 8), torch.float16)
    got = _OPS[name][0]()(x)
    want = _build(name, dim=None)(x)
    got, want = (got, want) if isinstance(want, tuple) else ((got,), (want,))
    assert [(g.shape, g.dtype) for g in got] == [(w.shape, w.dtype) for w in want]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "name, empty_dim, keepdim",
    [
        ("sum", [], False),
        ("mean", (), True),
        ("amin", [], False),
        ("amax", [], False),
        ("count_nonzero", [], False),
        ("std", [], False),
        ("var", [], False),
        ("var_mean", [], False),
    ],
)
def test_an_empty_dim_reduces_every_axis(name: str, empty_dim, keepdim: bool) -> None:
    x = torch.randn(2, 3, 4, dtype=torch.float16, device=run_device())
    got = _build(name, dim=empty_dim, keepdim=keepdim)(x)
    want = _build(name, dim=None, keepdim=keepdim)(x)
    got, want = (got, want) if isinstance(want, tuple) else ((got,), (want,))
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape
        torch.testing.assert_close(g, w, **reduction_tolerance(torch.float16))


@pytest.mark.smoke
def test_logsumexp_empty_dim_rejects() -> None:
    x = torch.randn(2, 3, 4, dtype=torch.float16, device=run_device())
    op = LogSumExpFwdOp(dim=[], keepdim=False)
    with pytest.raises(ValueError, match="an empty dim is rejected"):
        op(x)


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls", [AllFwdOp, AnyFwdOp], ids=["AllFwdOp", "AnyFwdOp"])
@pytest.mark.parametrize("empty_dim", [[], ()])
def test_an_empty_dim_is_a_noop_for_all_and_any(op_cls: type, empty_dim) -> None:
    """The spec's ``dim=[]`` contract: the output is ``x.bool()`` in the input's shape."""
    x = _input("all", (2, 4, 8), torch.float16)
    y = op_cls(dim=empty_dim)(x)
    assert y.shape == x.shape
    assert y.dtype == torch.bool
    compare_outputs(y, x.bool(), reduction_verification(torch.bool))


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls", [AllFwdOp, AnyFwdOp], ids=["AllFwdOp", "AnyFwdOp"])
def test_empty_dim_noop_answers_without_a_target(op_cls: type) -> None:
    """``dim=[]`` reduces nothing, so it needs no kernel and no device of any kind.

    The op computes the degenerate answer itself. Nothing about it is a target's to
    serve, so there is nobody to refuse a CPU tensor: which devices a *kernel* runs on is
    that kernel's statement, and this call reaches none. The same op with ``dim=-1`` does
    reach one and is refused there — that asymmetry is the edge of what the installed
    targets cover, not an inconsistency in the op.
    """
    x = torch.randint(-1, 2, (2, 4, 8)).to(torch.float16)  # cpu
    out = op_cls(dim=[])(x)

    assert out.device == x.device
    assert out.dtype == torch.bool
    compare_outputs(out, x != 0, reduction_verification(torch.bool))


@pytest.mark.smoke
@pytest.mark.parametrize("name", ["sum", "mean", "amax", "amin"])
@pytest.mark.parametrize(
    "dim", [pytest.param((0, 2), id="dim=tuple"), pytest.param(None, id="dim=None")]
)
def test_arithmetic_reduce_unaligned_innermost(name: str, dim) -> None:
    """An innermost extent of 255, off every tile multiple, takes the masked-load boundary
    whose fill is each op's identity; ``dim=-1`` there is test_reduce.py's ``(128, 300)``."""
    torch.manual_seed(0)
    _check(name, torch.randn(4, 8, 255, dtype=torch.float16, device=run_device()), dim)
