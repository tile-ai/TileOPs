"""Spec-conformance tests for scalar (0-D) reduction inputs.

A 0-D input is answered by the op itself — every family's kernel is undefined at that
extent — so what is under test is the closed form each one returns and its agreement with
PyTorch.

``keepdim`` cannot add an axis to a 0-D result and the element type reaches no branch, so
neither is crossed with ``dim``; each is swept once. ``dim`` is crossed with nothing but
carries every form ``_validate_scalar_dim`` accepts.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

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
    x = torch.tensor(1.5, dtype=torch.float16, device="cuda")

    y = _op(op_name, dim=dim)(x)

    ref = torch_fn(x.float(), dim=dim).to(x.dtype)
    assert y.shape == ref.shape, f"{op_name} dim={dim}: {y.shape} vs {ref.shape}"
    torch.testing.assert_close(y, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim=False", "keepdim=True"])
def test_the_scalar_path_honours_dtype_and_keepdim(dtype, keepdim) -> None:
    """Swept rather than crossed: neither reaches a branch ``dim`` does not."""
    x = torch.tensor(1.5, dtype=dtype, device="cuda")

    y = _op("SumFwdOp", dim=None, keepdim=keepdim)(x)

    ref = torch.sum(x.float(), dim=None, keepdim=keepdim).to(dtype)
    assert y.shape == ref.shape
    assert y.dtype == dtype
    torch.testing.assert_close(y, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", [0, -1], ids=["dim=0", "dim=-1"])
def test_prod_of_one_element_is_that_element(dim) -> None:
    """``ProdFwdOp`` narrows ``dim`` to an int, so it is asked about the two it takes."""
    x = torch.tensor(1.5, dtype=torch.float16, device="cuda")

    y = _op("ProdFwdOp", dim=dim)(x)

    ref = torch.prod(x.float(), dim=dim).to(x.dtype)
    assert y.shape == ref.shape
    torch.testing.assert_close(y, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.smoke
# One element leaves correction=1 without degrees of freedom, so both sides warn. That is
# the contract under test.
@pytest.mark.filterwarnings("ignore:.*degrees of freedom:UserWarning")
@pytest.mark.parametrize("op_name", _WELFORD)
@pytest.mark.parametrize("dim", _DIMS, ids=_DIM_IDS)
def test_a_welford_reduction_of_one_element_matches_torch(op_name, dim) -> None:
    """``correction=1`` over one element is undefined, and PyTorch calls that ``nan``."""
    x = torch.tensor(1.5, dtype=torch.float16, device="cuda")

    got = _as_tuple(_op(op_name, dim=dim)(x))

    for g, w in zip(got, _welford_ref(op_name, x, dim, False), strict=True):
        assert g.shape == w.shape, f"{op_name} dim={dim}: {g.shape} vs {w.shape}"
        torch.testing.assert_close(g, w, atol=1e-4, rtol=1e-4, equal_nan=True)


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
    x = torch.ones(shape, dtype=torch.float32, device="cuda").cumsum(0)

    got = _as_tuple(_op(op_name, dim=dim)(x))

    for g, w in zip(got, _welford_ref(op_name, x, dim, False), strict=True):
        torch.testing.assert_close(g, w, atol=1e-4, rtol=1e-4, equal_nan=True)


@pytest.mark.smoke
@pytest.mark.parametrize("op_name, torch_fn, out_dtype", _LOGICAL)
@pytest.mark.parametrize("dim", _DIMS, ids=_DIM_IDS)
@pytest.mark.parametrize("value", [0.0, 1.5], ids=["zero", "nonzero"])
def test_a_logical_reduction_of_one_element_matches_torch(
    op_name, torch_fn, out_dtype, dim, value
) -> None:
    """Both truth values, because the predicate is what these ops compute."""
    x = torch.tensor(value, dtype=torch.float16, device="cuda")

    y = _op(op_name, dim=dim)(x)

    ref = torch_fn(x, dim=dim)
    assert y.dtype == out_dtype, f"{op_name}: {y.dtype}"
    assert y.shape == ref.shape, f"{op_name} dim={dim}: {y.shape} vs {ref.shape}"
    torch.testing.assert_close(y, ref, atol=0, rtol=0)
