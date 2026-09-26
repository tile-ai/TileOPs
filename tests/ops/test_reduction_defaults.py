"""Regression tests for reduction-op constructor defaults and empty-dim semantics.

Pins two manifest-conformance invariants for the reduction op family:

1. For the ten ops whose manifest declares ``default: null`` on ``dim``
   (Sum/Mean/Amax/Amin/Var/Std/VarMean/All/Any/CountNonzero), constructing
   the op with only ``dtype=`` performs a full reduction (output shape
   equals ``torch.<op>(x).shape``).

2. ``AllFwdOp`` / ``AnyFwdOp`` honor the spec's ``dim=[]`` / ``dim=()``
   no-op contract: output shape equals the input shape, output dtype is
   ``bool``, and values equal ``x.bool()``.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


_FLOAT_SHAPE = (2, 4, 8)
_LOGICAL_SHAPE = (2, 4, 8)


def _make_float(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape, dtype=dtype, device="cuda")


def _make_logical(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    # values in {-1, 0, 1} so .bool() has both T and F.
    return (torch.randint(-1, 2, shape, device="cuda")).to(dtype)


# default dim=None for the ten ops -> full reduction on 3-D input


@pytest.mark.smoke
def test_sum_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import SumFwdOp

    x = _make_float(_FLOAT_SHAPE, torch.float16)
    op = SumFwdOp()
    y = op(x)
    assert y.shape == torch.sum(x).shape


@pytest.mark.smoke
def test_mean_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import MeanFwdOp

    x = _make_float(_FLOAT_SHAPE, torch.float16)
    op = MeanFwdOp()
    y = op(x)
    assert y.shape == torch.mean(x).shape


@pytest.mark.smoke
def test_amax_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import AmaxFwdOp

    x = _make_float(_FLOAT_SHAPE, torch.float16)
    op = AmaxFwdOp()
    y = op(x)
    assert y.shape == torch.amax(x).shape


@pytest.mark.smoke
def test_amin_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import AminFwdOp

    x = _make_float(_FLOAT_SHAPE, torch.float16)
    op = AminFwdOp()
    y = op(x)
    assert y.shape == torch.amin(x).shape


@pytest.mark.smoke
def test_var_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import VarFwdOp

    x = _make_float(_FLOAT_SHAPE, torch.float16)
    op = VarFwdOp()
    y = op(x)
    assert y.shape == torch.var(x).shape


@pytest.mark.smoke
def test_std_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import StdFwdOp

    x = _make_float(_FLOAT_SHAPE, torch.float16)
    op = StdFwdOp()
    y = op(x)
    assert y.shape == torch.std(x).shape


@pytest.mark.smoke
def test_var_mean_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import VarMeanFwdOp

    x = _make_float(_FLOAT_SHAPE, torch.float16)
    op = VarMeanFwdOp()
    var_out, mean_out = op(x)
    ref_var, ref_mean = torch.var_mean(x)
    assert var_out.shape == ref_var.shape
    assert mean_out.shape == ref_mean.shape


@pytest.mark.smoke
def test_all_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    x = _make_logical(_LOGICAL_SHAPE, torch.float16)
    op = AllFwdOp()
    y = op(x)
    assert y.shape == torch.all(x.bool()).shape
    assert y.dtype == torch.bool


@pytest.mark.smoke
def test_any_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    x = _make_logical(_LOGICAL_SHAPE, torch.float16)
    op = AnyFwdOp()
    y = op(x)
    assert y.shape == torch.any(x.bool()).shape
    assert y.dtype == torch.bool


@pytest.mark.smoke
def test_count_nonzero_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    x = _make_logical(_LOGICAL_SHAPE, torch.float16)
    op = CountNonzeroFwdOp()
    y = op(x)
    assert y.shape == torch.count_nonzero(x).shape
    assert y.dtype == torch.int64


# AllFwdOp/AnyFwdOp dim=[] / dim=() noop contract


@pytest.mark.smoke
@pytest.mark.parametrize("empty_dim", [[], ()])
def test_all_empty_dim_noop(empty_dim) -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    x = _make_logical(_LOGICAL_SHAPE, torch.float16)
    op = AllFwdOp(dim=empty_dim)
    y = op(x)
    assert y.shape == x.shape
    assert y.dtype == torch.bool
    assert torch.equal(y, x.bool())


@pytest.mark.smoke
@pytest.mark.parametrize("empty_dim", [[], ()])
def test_any_empty_dim_noop(empty_dim) -> None:
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    x = _make_logical(_LOGICAL_SHAPE, torch.float16)
    op = AnyFwdOp(dim=empty_dim)
    y = op(x)
    assert y.shape == x.shape
    assert y.dtype == torch.bool
    assert torch.equal(y, x.bool())


@pytest.mark.smoke
@pytest.mark.parametrize("op_name", ["AllFwdOp", "AnyFwdOp"])
def test_empty_dim_noop_answers_without_a_target(op_name: str) -> None:
    """``dim=[]`` reduces nothing, so it needs no kernel and no device of any kind.

    The op computes the degenerate answer itself. Nothing about it is a target's to
    serve, so there is nobody to refuse a CPU tensor: which devices a *kernel* runs on is
    that kernel's statement, and this call reaches none. The same op with ``dim=-1`` does
    reach one and is refused there — that asymmetry is the edge of what the installed
    targets cover, not an inconsistency in the op.
    """
    import tileops.ops.reduction.logical_reduce as logical_reduce

    x = (torch.randint(-1, 2, _LOGICAL_SHAPE)).to(torch.float16)  # cpu
    op = getattr(logical_reduce, op_name)(dim=[])

    out = op(x)

    assert out.device == x.device
    assert out.dtype == torch.bool
    assert torch.equal(out, x != 0)


# A kernel's architecture check reads the device the op handed over


@pytest.mark.smoke
def test_the_arch_check_asks_about_the_input_s_device(monkeypatch) -> None:
    """Not whichever device is current: the two differ on a mixed-architecture host.

    A homogeneous host cannot show the wrong answer, so what is asserted is which device
    was asked about.
    """
    import tileops.utils as utils
    from tileops.ops.reduction.reduce import SumFwdOp

    asked: list = []
    real = utils.get_sm_version

    def recording(index=None):
        asked.append(index)
        return real(index)

    monkeypatch.setattr(utils, "get_sm_version", recording)

    x = torch.randn(4, 8, dtype=torch.float16, device="cuda")
    SumFwdOp(dim=-1)(x)

    assert asked, "the kernel declares supported_archs, so it must have probed"
    assert all(i == x.device.index for i in asked), asked
