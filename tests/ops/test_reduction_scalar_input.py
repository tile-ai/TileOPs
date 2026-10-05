"""Spec-conformance tests for scalar (0-D) reduction inputs.

A 0-D input is answered by the op itself — every family's kernel is undefined at that
extent — so what is under test is the closed form each one returns and its agreement with
PyTorch.

``keepdim`` cannot add an axis to a 0-D result and the element type reaches no branch, so
neither is crossed with ``dim``; each is swept once. ``dim`` is crossed with nothing but
carries every form ``_validate_scalar_dim`` accepts.

The axis ops (softmax, logsumexp, argmax, vector norm) answer a 0-D input the same way and
are checked against PyTorch once each.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from tileops.manifest import load_adts, load_manifest
from tileops.manifest.values import convert
from workloads.device import run_device, run_device_available
from workloads.numerics import compare_outputs
from workloads.reduction import reduction_verification

pytestmark = pytest.mark.skipif(
    not run_device_available(), reason="the run device is not available"
)

_MANIFEST = load_manifest()
_ADTS = load_adts()

_FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
_DTYPE_IDS = ["fp16", "bf16", "fp32"]
_DIMS = [None, 0, -1, (), []]
_DIM_IDS = ["dim=None", "dim=0", "dim=-1", "dim=()", "dim=[]"]

#: A 0-D input and the reference each op must match on it. ``prod`` takes no ``None``
#: dim, so it is asked about an int one.
_ARITHMETIC = [
    pytest.param("SumFwdOp", torch.sum, id="sum"),
    pytest.param("MeanFwdOp", torch.mean, id="mean"),
    pytest.param("AmaxFwdOp", torch.amax, id="amax"),
    pytest.param("AminFwdOp", torch.amin, id="amin"),
]
_WELFORD = ["VarFwdOp", "StdFwdOp", "VarMeanFwdOp"]
_LOGICAL = [
    pytest.param("AllFwdOp", torch.all, torch.bool, id="all"),
    pytest.param("AnyFwdOp", torch.any, torch.bool, id="any"),
    pytest.param("CountNonzeroFwdOp", torch.count_nonzero, torch.int64, id="count-nonzero"),
]


def _op(name: str, **kwargs):
    import tileops.ops.reduction.logical_reduce as logical
    import tileops.ops.reduction.reduce as reduce_ops

    module = logical if hasattr(logical, name) else reduce_ops
    return getattr(module, name)(**kwargs)


def _as_tuple(value):
    return value if isinstance(value, tuple) else (value,)


def _welford_ref(name: str, x, dim, keepdim):
    fn = {"VarFwdOp": torch.var, "StdFwdOp": torch.std, "VarMeanFwdOp": torch.var_mean}[name]
    out = fn(x.float(), dim=dim, keepdim=keepdim, correction=1)
    return tuple(t.to(x.dtype) for t in _as_tuple(out))


@pytest.mark.smoke
@pytest.mark.parametrize("op_name, torch_fn", _ARITHMETIC)
@pytest.mark.parametrize("dim", _DIMS, ids=_DIM_IDS)
def test_an_arithmetic_reduction_of_one_element_is_that_element(op_name, torch_fn, dim) -> None:
    x = torch.tensor(1.5, dtype=torch.float16, device=run_device())

    y = _op(op_name, dim=dim)(x)

    ref = torch_fn(x.float(), dim=dim).to(x.dtype)
    assert y.shape == ref.shape, f"{op_name} dim={dim}: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype, scalar=True))


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim=False", "keepdim=True"])
def test_the_scalar_path_honours_dtype_and_keepdim(dtype, keepdim) -> None:
    """Swept rather than crossed: neither reaches a branch ``dim`` does not."""
    x = torch.tensor(1.5, dtype=dtype, device=run_device())

    y = _op("SumFwdOp", dim=None, keepdim=keepdim)(x)

    ref = torch.sum(x.float(), dim=None, keepdim=keepdim).to(dtype)
    assert y.shape == ref.shape
    assert y.dtype == dtype
    compare_outputs(y, ref, reduction_verification((ref).dtype, scalar=True))


@pytest.mark.smoke
@pytest.mark.parametrize("dim", [0, -1], ids=["dim=0", "dim=-1"])
def test_prod_of_one_element_is_that_element(dim) -> None:
    """``ProdFwdOp`` narrows ``dim`` to an int, so it is asked about the two it takes."""
    x = torch.tensor(1.5, dtype=torch.float16, device=run_device())

    y = _op("ProdFwdOp", dim=dim)(x)

    ref = torch.prod(x.float(), dim=dim).to(x.dtype)
    assert y.shape == ref.shape
    compare_outputs(y, ref, reduction_verification((ref).dtype, scalar=True))


def _warns_dof(fn) -> tuple[object, bool]:
    """Call *fn*; return its result and whether it warned about degrees of freedom."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, any(
        issubclass(w.category, UserWarning) and "degrees of freedom" in str(w.message)
        for w in caught
    )


@pytest.mark.smoke
@pytest.mark.parametrize("op_name", _WELFORD)
@pytest.mark.parametrize("dim", _DIMS, ids=_DIM_IDS)
def test_a_welford_reduction_of_one_element_matches_torch(op_name, dim) -> None:
    """``correction=1`` over one element is undefined: PyTorch warns and calls it ``nan``."""
    x = torch.tensor(1.5, dtype=torch.float16, device=run_device())

    got, op_warned = _warns_dof(lambda: _as_tuple(_op(op_name, dim=dim)(x)))
    want, ref_warned = _warns_dof(lambda: _welford_ref(op_name, x, dim, False))

    assert op_warned == ref_warned, f"{op_name} dim={dim}: warned {op_warned}, torch {ref_warned}"
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape, f"{op_name} dim={dim}: {g.shape} vs {w.shape}"
        compare_outputs(g, w, reduction_verification((w).dtype, scalar=True))


@pytest.mark.smoke
@pytest.mark.filterwarnings("ignore:.*degrees of freedom:UserWarning")
@pytest.mark.parametrize("op_name", _WELFORD)
@pytest.mark.parametrize(
    ("shape", "dim"),
    [((1,), None), ((1,), 0), ((2, 1), -1)],
    ids=["1d-full", "1d-axis", "2d-inner-axis"],
)
def test_a_reduction_with_no_degrees_of_freedom_matches_torch(op_name, shape, dim) -> None:
    """A length-1 axis with ``correction=1``: the kernel cannot be built for it."""
    x = torch.ones(shape, dtype=torch.float32, device=run_device()).cumsum(0)

    got = _as_tuple(_op(op_name, dim=dim)(x))

    for g, w in zip(got, _welford_ref(op_name, x, dim, False), strict=True):
        compare_outputs(g, w, reduction_verification((w).dtype, scalar=True))


@pytest.mark.smoke
@pytest.mark.parametrize("op_name, torch_fn, out_dtype", _LOGICAL)
@pytest.mark.parametrize("dim", _DIMS, ids=_DIM_IDS)
@pytest.mark.parametrize("value", [0.0, 1.5], ids=["zero", "nonzero"])
def test_a_logical_reduction_of_one_element_matches_torch(
    op_name, torch_fn, out_dtype, dim, value
) -> None:
    """Both truth values, because the predicate is what these ops compute."""
    x = torch.tensor(value, dtype=torch.float16, device=run_device())

    y = _op(op_name, dim=dim)(x)

    ref = torch_fn(x, dim=dim)
    assert y.dtype == out_dtype, f"{op_name}: {y.dtype}"
    assert y.shape == ref.shape, f"{op_name} dim={dim}: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype, scalar=True))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dim",
    [[0, 0], [0, -1], [-1, -1], [-1, 0], (0, -1)],
    ids=["dim=[0,0]", "dim=[0,-1]", "dim=[-1,-1]", "dim=[-1,0]", "dim=(0,-1)"],
)
def test_an_aliasing_dim_on_a_scalar_is_refused_as_torch_refuses_it(dim) -> None:
    x = torch.tensor(1.5, dtype=torch.float32, device=run_device())
    with pytest.raises(RuntimeError, match="appears multiple times"):
        torch.sum(x, dim=list(dim))
    with pytest.raises(ValueError, match="unique_axes"):
        _op("SumFwdOp", dim=list(dim))(x)


@pytest.mark.smoke
@pytest.mark.filterwarnings("ignore:.*degrees of freedom:UserWarning")
@pytest.mark.parametrize("op_name", ["VarFwdOp", "VarMeanFwdOp"])
def test_a_welford_scalar_keeps_autograd_history(op_name) -> None:
    """The ``nan`` fast path returns a tensor backward can run through, as PyTorch's does."""
    x = torch.tensor(0.5, dtype=torch.float32, device=run_device(), requires_grad=True)

    got = _as_tuple(_op(op_name, dim=None)(x))

    assert got[0].shape == () and torch.isnan(got[0]).item(), op_name
    assert all(g.requires_grad and g.grad_fn is not None for g in got), op_name


@pytest.mark.smoke
@pytest.mark.parametrize(
    "name, ref",
    [
        ("SoftmaxFwdOp", lambda x: torch.softmax(x, 0)),
        ("LogSoftmaxFwdOp", lambda x: torch.log_softmax(x, 0)),
        ("LogSumExpFwdOp", lambda x: torch.logsumexp(x, 0)),
        ("ArgmaxFwdOp", lambda x: torch.argmax(x, 0)),
        ("VectorNormFwdOp", lambda x: torch.linalg.vector_norm(x, 2, 0)),
    ],
)
def test_a_scalar_input_to_an_axis_op_matches_torch(name: str, ref) -> None:
    import tileops.reduction as reduction

    x = torch.tensor(-1.5, dtype=torch.float32, device=run_device())
    want = ref(x)
    compare_outputs(
        getattr(reduction, name)(dim=0)(x), want, reduction_verification(want.dtype, scalar=True)
    )


#: How each op's manifest formula prices a scalar: one element read, then what it
#: writes. ``all``/``any`` write one byte, ``count_nonzero`` one int64, the rest one
#: element of the input dtype; ``var_mean`` writes two.
_SCALAR_WRITE_BYTES = {
    "SumFwdOp": lambda e: e,
    "MeanFwdOp": lambda e: e,
    "AmaxFwdOp": lambda e: e,
    "AminFwdOp": lambda e: e,
    "ProdFwdOp": lambda e: e,
    "VarFwdOp": lambda e: e,
    "StdFwdOp": lambda e: e,
    "VarMeanFwdOp": lambda e: 2 * e,
    "AllFwdOp": lambda e: 1,
    "AnyFwdOp": lambda e: 1,
    "CountNonzeroFwdOp": lambda e: 8,
}

#: Every dim form a scalar reduction can be given, the singleton sequences included: each
#: one reaches ``dim % x.ndim`` in the manifest formula by a different branch.
_SCALAR_ROOFLINE_DIMS = [None, 0, -1, (), [], [0], [-1], (0,), (-1,)]
_SCALAR_ROOFLINE_DIM_IDS = [
    "dim=None", "dim=0", "dim=-1", "dim=()", "dim=[]",
    "dim=[0]", "dim=[-1]", "dim=(0,)", "dim=(-1,)",
]  # fmt: skip


def _dim_legal(op_name: str, dim) -> bool:
    """Whether the op's manifest ``dim`` type admits *dim*."""
    try:
        convert(dim, _MANIFEST[op_name]["signature"]["params"]["dim"]["type"], _ADTS)
    except ValueError:
        return False
    return True


_SCALAR_ROOFLINE_CASES = [
    pytest.param(op_name, dim, id=f"{dim_id}-{op_name}")
    for dim, dim_id in zip(_SCALAR_ROOFLINE_DIMS, _SCALAR_ROOFLINE_DIM_IDS, strict=True)
    for op_name in sorted(_SCALAR_WRITE_BYTES)
    if _dim_legal(op_name, dim)
]


@pytest.mark.smoke
@pytest.mark.filterwarnings("ignore:.*degrees of freedom:UserWarning")
@pytest.mark.parametrize("op_name, dim", _SCALAR_ROOFLINE_CASES)
def test_the_scalar_path_prices_one_element_whatever_dim_names(op_name, dim) -> None:
    """A 0-D input has no axis to reduce, so every legal ``dim`` form prices one element.

    The manifest formulas take ``dim % x.ndim``, which is a division by zero at this
    extent; the entries carry the guard that makes it one element instead.
    """
    op = _op(op_name, dim=dim)
    x = torch.tensor(3.0, device=run_device(), dtype=torch.float32)
    op(x)

    _, nbytes = op.eval_roofline()
    assert nbytes == x.element_size() + _SCALAR_WRITE_BYTES[op_name](x.element_size())
