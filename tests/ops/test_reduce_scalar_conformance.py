"""Spec-conformance tests for scalar (0-D) reduction inputs.

Covers the reduction families whose manifest/API contract accepts scalar
inputs and must match PyTorch on 0-D tensors. This is the missing piece that
lets the shared reduction base promote `implemented` from a shape-limited
claim to the full PyTorch scalar contract.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

_FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
_DIMS = [None, 0, -1, (), []]
_DIM_IDS = ["dim=None", "dim=0", "dim=-1", "dim=()", "dim=[]"]


def _make_scalar(dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(1.5, dtype=dtype, device="cuda")


def _make_zero_scalar(dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(0.0, dtype=dtype, device="cuda")


@pytest.mark.smoke
@pytest.mark.parametrize("dim", _DIMS, ids=_DIM_IDS)
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim=False", "keepdim=True"])
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=["fp16", "bf16", "fp32"])
def test_scalar_arithmetic_reductions(dim, keepdim: bool, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import AmaxFwdOp, AminFwdOp, MeanFwdOp, SumFwdOp

    x = _make_scalar(dtype)
    cases = [
        (SumFwdOp, torch.sum),
        (MeanFwdOp, torch.mean),
        (AmaxFwdOp, torch.amax),
        (AminFwdOp, torch.amin),
    ]

    for op_cls, torch_fn in cases:
        op = op_cls(dtype=dtype, dim=dim, keepdim=keepdim)
        y = op(x)
        ref = torch_fn(x.float(), dim=dim, keepdim=keepdim).to(dtype)
        assert y.shape == ref.shape, (
            f"{op_cls.__name__} scalar dim={dim} keepdim={keepdim} dtype={dtype}: "
            f"shape {y.shape} vs ref {ref.shape}"
        )
        torch.testing.assert_close(y, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", [0, -1], ids=["dim=0", "dim=-1"])
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim=False", "keepdim=True"])
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=["fp16", "bf16", "fp32"])
def test_scalar_prod_reduction(dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import ProdFwdOp

    x = _make_scalar(dtype)
    op = ProdFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    y = op(x)
    ref = torch.prod(x.float(), dim=dim, keepdim=keepdim).to(dtype)
    assert y.shape == ref.shape, (
        f"ProdFwdOp scalar dim={dim} keepdim={keepdim} dtype={dtype}: "
        f"shape {y.shape} vs ref {ref.shape}"
    )
    torch.testing.assert_close(y, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", _DIMS, ids=_DIM_IDS)
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim=False", "keepdim=True"])
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=["fp16", "bf16", "fp32"])
def test_scalar_welford_reductions(dim, keepdim: bool, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import StdFwdOp, VarFwdOp, VarMeanFwdOp

    x = _make_scalar(dtype)
    cases = [
        (VarFwdOp, lambda t, *, dim, keepdim: torch.var(t.float(), dim=dim, keepdim=keepdim, correction=1).to(dtype)),
        (StdFwdOp, lambda t, *, dim, keepdim: torch.std(t.float(), dim=dim, keepdim=keepdim, correction=1).to(dtype)),
    ]

    for op_cls, ref_fn in cases:
        op = op_cls(dtype=dtype, dim=dim, keepdim=keepdim)
        y = op(x)
        ref = ref_fn(x, dim=dim, keepdim=keepdim)
        assert y.shape == ref.shape, (
            f"{op_cls.__name__} scalar dim={dim} keepdim={keepdim} dtype={dtype}: "
            f"shape {y.shape} vs ref {ref.shape}"
        )
        torch.testing.assert_close(y, ref, atol=1e-4, rtol=1e-4, equal_nan=True)

    op = VarMeanFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    var_y, mean_y = op(x)
    ref_var, ref_mean = torch.var_mean(
        x.float(), dim=dim, keepdim=keepdim, correction=1,
    )
    ref_var = ref_var.to(dtype)
    ref_mean = ref_mean.to(dtype)
    assert var_y.shape == ref_var.shape
    assert mean_y.shape == ref_mean.shape
    torch.testing.assert_close(var_y, ref_var, atol=1e-4, rtol=1e-4, equal_nan=True)
    torch.testing.assert_close(mean_y, ref_mean, atol=1e-4, rtol=1e-4, equal_nan=True)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("shape", "dim"),
    [((1,), None), ((1,), 0), ((2, 1), -1)],
    ids=["shape=(1,),dim=None", "shape=(1,),dim=0", "shape=(2,1),dim=-1"],
)
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim=False", "keepdim=True"])
def test_invalid_dof_welford_reductions_match_pytorch(
    shape: tuple[int, ...], dim, keepdim: bool,
) -> None:
    from tileops.ops.reduction.reduce import StdFwdOp, VarFwdOp, VarMeanFwdOp

    if shape == (2, 1):
        x = torch.tensor([[1.5], [2.5]], dtype=torch.float32, device="cuda")
    else:
        x = torch.tensor([1.5], dtype=torch.float32, device="cuda")

    for op_cls, torch_fn in [
        (VarFwdOp, torch.var),
        (StdFwdOp, torch.std),
    ]:
        op = op_cls(dtype=torch.float32, dim=dim, keepdim=keepdim)
        y = op(x)
        ref = torch_fn(x, dim=dim, keepdim=keepdim, correction=1)
        torch.testing.assert_close(y, ref, atol=1e-4, rtol=1e-4, equal_nan=True)

    op = VarMeanFwdOp(dtype=torch.float32, dim=dim, keepdim=keepdim)
    var_y, mean_y = op(x)
    ref_var, ref_mean = torch.var_mean(x, dim=dim, keepdim=keepdim, correction=1)
    torch.testing.assert_close(var_y, ref_var, atol=1e-4, rtol=1e-4, equal_nan=True)
    torch.testing.assert_close(mean_y, ref_mean, atol=1e-4, rtol=1e-4, equal_nan=True)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", _DIMS, ids=_DIM_IDS)
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=["fp16", "bf16", "fp32"])
@pytest.mark.parametrize(
    "make_input", [_make_zero_scalar, _make_scalar], ids=["zero", "nonzero"],
)
def test_scalar_logical_and_count_reductions(
    dim, dtype: torch.dtype, make_input,
) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp
    from tileops.ops.reduction.any_op import AnyFwdOp
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    x = make_input(dtype)

    for op_cls, torch_fn, out_dtype in [
        (AllFwdOp, torch.all, torch.bool),
        (AnyFwdOp, torch.any, torch.bool),
        (CountNonzeroFwdOp, torch.count_nonzero, torch.int64),
    ]:
        op = op_cls(dtype=dtype, dim=dim)
        y = op(x)
        ref = torch_fn(x, dim=dim)
        assert y.dtype == out_dtype, f"{op_cls.__name__} scalar dtype {y.dtype}"
        assert y.shape == ref.shape, (
            f"{op_cls.__name__} scalar dim={dim} dtype={dtype}: "
            f"shape {y.shape} vs ref {ref.shape}"
        )
        torch.testing.assert_close(y, ref, atol=0, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
