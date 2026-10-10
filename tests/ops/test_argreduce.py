"""Correctness tests for argreduce ops (argmax, argmin).

Covers: ArgmaxFwdOp, ArgminFwdOp.
Each op reduces along a configurable dim and returns int64 indices.
Uses exact match (torch.equal) instead of allclose.
"""

import math
from typing import cast

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.reduction import ArgreduceCase, reduction_verification


def _call(op, x: torch.Tensor) -> torch.Tensor:
    """Invoke a single-output argreduce op and narrow the return to ``Tensor``.

    The shared ``OpBase.__call__`` is typed as ``Union[Tensor, tuple]`` to
    accommodate ops with multiple outputs.  Argreduce ops always return a
    single ``Tensor``; this helper keeps the call sites well-typed without
    sprinkling ``cast(...)`` everywhere.
    """
    return cast(torch.Tensor, op(x))


class ArgreduceBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(256, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-aligned N (non-pow2 last dim)
                pytest.param(128, 300, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 300, torch.bfloat16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(129, 512, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class ArgreduceNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Argreduce3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 64, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Argreduce3DDim0Fixture(FixtureBase):
    """dim=0 reduction on 3D tensors — small outermost dim triggers
    the TileLang layout constraint (N << N_padded)."""

    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(4, 8, 256, torch.float16, marks=pytest.mark.smoke),
                pytest.param(4, 8, 256, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4, 8, 256, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Argreduce4DFixture(FixtureBase):
    PARAMS = [
        (
            "b0, b1, b2, n, dtype",
            [
                pytest.param(2, 4, 8, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 4, 8, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Argreduce4DDim0Fixture(FixtureBase):
    """dim=0 reduction on 4D tensors — regression coverage for 3D+ contract."""

    PARAMS = [
        (
            "b0, b1, b2, n, dtype",
            [
                pytest.param(2, 4, 8, 256, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 4, 8, 256, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Argreduce1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class SpecArgreduceFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, keepdim, dtype",
            [
                pytest.param((128, 512), -1, False, torch.float16, marks=pytest.mark.smoke),
                pytest.param((4, 32, 512), -1, False, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param((128, 512), -1, True, torch.float16, marks=pytest.mark.full),
                pytest.param((512, 4, 32), 0, False, torch.float16, marks=pytest.mark.full),
                pytest.param((4, 32, 512), 1, False, torch.float16, marks=pytest.mark.full),
                pytest.param((4, 32, 512), -1, True, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class ArgreduceTest(ArgreduceCase, TestBase):
    pass


@ArgreduceBasicFixture
def test_argmax_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    test = ArgreduceTest(m, n, dtype, "argmax")
    op = ArgmaxFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@ArgreduceNonContigFixture
def test_argmax_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    x = x_full[:, :n]
    op = ArgmaxFwdOp(dim=-1)
    ref = x.contiguous().argmax(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@Argreduce3DFixture
def test_argmax_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = ArgmaxFwdOp(dim=-1)
    ref = x.argmax(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@Argreduce4DFixture
def test_argmax_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = ArgmaxFwdOp(dim=-1)
    ref = x.argmax(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@Argreduce1DFixture
def test_argmax_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(n, dtype=dtype, device=run_device())
    op = ArgmaxFwdOp(dim=-1)
    ref = x.argmax(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    compare_outputs(y.view_as(ref), ref, reduction_verification((ref).dtype))


@Argreduce3DDim0Fixture
def test_argmax_3d_dim0(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Argmax along dim=0 on 3D tensors (outermost-dim reduction)."""
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = ArgmaxFwdOp(dim=0)
    ref = x.argmax(dim=0)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@Argreduce3DDim0Fixture
def test_argmax_3d_dim0_keepdim(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Argmax along dim=0 with keepdim=True on 3D tensors."""
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = ArgmaxFwdOp(dim=0, keepdim=True)
    ref = x.argmax(dim=0, keepdim=True)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@Argreduce4DDim0Fixture
def test_argmax_4d_dim0(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    """Argmax along dim=0 on 4D tensors (outermost-dim reduction, 3D+ regression)."""
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = ArgmaxFwdOp(dim=0)
    ref = x.argmax(dim=0)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@Argreduce4DDim0Fixture
def test_argmax_4d_dim0_keepdim(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    """Argmax along dim=0 with keepdim=True on 4D tensors."""
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = ArgmaxFwdOp(dim=0, keepdim=True)
    ref = x.argmax(dim=0, keepdim=True)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@SpecArgreduceFixture
def test_argmax_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: ArgmaxFwdOp with dim + keepdim."""
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = ArgmaxFwdOp(dim=dim, keepdim=keepdim)
    ref = x.argmax(dim=dim, keepdim=keepdim)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@ArgreduceBasicFixture
def test_argmin_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    test = ArgreduceTest(m, n, dtype, "argmin")
    op = ArgminFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@ArgreduceNonContigFixture
def test_argmin_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    x = x_full[:, :n]
    op = ArgminFwdOp(dim=-1)
    ref = x.contiguous().argmin(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@Argreduce3DFixture
def test_argmin_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = ArgminFwdOp(dim=-1)
    ref = x.argmin(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@Argreduce4DFixture
def test_argmin_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = ArgminFwdOp(dim=-1)
    ref = x.argmin(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@Argreduce1DFixture
def test_argmin_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(n, dtype=dtype, device=run_device())
    op = ArgminFwdOp(dim=-1)
    ref = x.argmin(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    compare_outputs(y.view_as(ref), ref, reduction_verification((ref).dtype))


@Argreduce3DDim0Fixture
def test_argmin_3d_dim0(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Argmin along dim=0 on 3D tensors (outermost-dim reduction)."""
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = ArgminFwdOp(dim=0)
    ref = x.argmin(dim=0)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@Argreduce3DDim0Fixture
def test_argmin_3d_dim0_keepdim(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Argmin along dim=0 with keepdim=True on 3D tensors."""
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = ArgminFwdOp(dim=0, keepdim=True)
    ref = x.argmin(dim=0, keepdim=True)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@Argreduce4DDim0Fixture
def test_argmin_4d_dim0(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    """Argmin along dim=0 on 4D tensors (outermost-dim reduction, 3D+ regression)."""
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = ArgminFwdOp(dim=0)
    ref = x.argmin(dim=0)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@Argreduce4DDim0Fixture
def test_argmin_4d_dim0_keepdim(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    """Argmin along dim=0 with keepdim=True on 4D tensors."""
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = ArgminFwdOp(dim=0, keepdim=True)
    ref = x.argmin(dim=0, keepdim=True)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@SpecArgreduceFixture
def test_argmin_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: ArgminFwdOp with dim + keepdim."""
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = ArgminFwdOp(dim=dim, keepdim=keepdim)
    ref = x.argmin(dim=dim, keepdim=keepdim)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    compare_outputs(y, ref, reduction_verification((ref).dtype))


# Regression: the two zeros compare equal, so the lower index wins


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("op_name", ["argmax", "argmin"])
def test_argreduce_signed_zero_breaks_to_lower_index(op_name: str, dtype: torch.dtype) -> None:
    """A row of alternating -0.0 and +0.0 is one tie, which index 0 has to win."""
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp, ArgminFwdOp

    x = torch.zeros(8, 4096, dtype=dtype, device=run_device())
    x[:, ::2] = -0.0
    op = ArgmaxFwdOp(dim=-1) if op_name == "argmax" else ArgminFwdOp(dim=-1)
    y = _call(op, x)
    compare_outputs(y, torch.zeros_like(y), reduction_verification((torch.zeros_like(y)).dtype))


# Regression: multidim dim must be rejected for argreduce ops


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls_path, dim",
    [
        ("tileops.ops.reduction.argreduce.ArgmaxFwdOp", [0, 1]),
        ("tileops.ops.reduction.argreduce.ArgminFwdOp", [0, 1]),
        ("tileops.ops.reduction.argreduce.ArgmaxFwdOp", (0, 1)),
        ("tileops.ops.reduction.argreduce.ArgminFwdOp", (0, 1)),
    ],
)
def test_argreduce_rejects_multidim(op_cls_path: str, dim) -> None:
    """Argreduce ops only support scalar dim or None; list/tuple must raise."""
    import importlib

    module_path, cls_name = op_cls_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    op_cls = getattr(mod, cls_name)

    with pytest.raises((TypeError, ValueError)):
        op_cls(dim=dim)


# dim=None (full-tensor reduction) tests


class ArgreduceDimNoneFixture(FixtureBase):
    """Full-tensor reduction (dim=None).

    Coverage rationale (testing.md §Test case policy):
      - dtype dispatch: one 2D shape across {fp16, bf16, fp32}.
      - ndim shape branches (flatten path): 1D / 3D / 4D in fp16 only;
        ndim and dtype are not crossed since the flatten code path is
        dtype-independent.
      - One non-aligned flat size as full coverage (tail handling).
    """

    PARAMS = [
        (
            "shape, dtype",
            [
                # dtype dispatch on a single 2D shape
                pytest.param((16, 64), torch.float16, marks=pytest.mark.smoke),
                pytest.param((16, 64), torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param((16, 64), torch.float32, marks=pytest.mark.smoke),
                # ndim shape coverage (flatten path is dtype-agnostic)
                pytest.param((512,), torch.float16, marks=pytest.mark.smoke),
                pytest.param((4, 8, 32), torch.float16, marks=pytest.mark.smoke),
                pytest.param((2, 4, 8, 16), torch.float16, marks=pytest.mark.smoke),
                # Non-aligned flat size (tail handling)
                pytest.param((10, 30), torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@ArgreduceDimNoneFixture
def test_argmax_dim_none(shape: tuple, dtype: torch.dtype) -> None:
    """ArgmaxFwdOp(dim=None) matches torch.argmax(x); covers keepdim={False, True}."""
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    ref_flat = torch.argmax(x)

    y = _call(ArgmaxFwdOp(dim=None), x)
    assert y.dtype == torch.int64
    assert y.shape == ref_flat.shape, f"shape mismatch: {y.shape} vs {ref_flat.shape}"
    compare_outputs(y, ref_flat, reduction_verification((ref_flat).dtype))

    y_keep = _call(ArgmaxFwdOp(dim=None, keepdim=True), x)
    expected_shape = tuple(1 for _ in shape)
    assert y_keep.shape == expected_shape, (
        f"keepdim shape mismatch: {y_keep.shape} vs {expected_shape}"
    )
    compare_outputs(y_keep.reshape(()), ref_flat, reduction_verification((ref_flat).dtype))


@ArgreduceDimNoneFixture
def test_argmin_dim_none(shape: tuple, dtype: torch.dtype) -> None:
    """ArgminFwdOp(dim=None) matches torch.argmin(x); covers keepdim={False, True}."""
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    ref_flat = torch.argmin(x)

    y = _call(ArgminFwdOp(dim=None), x)
    assert y.dtype == torch.int64
    assert y.shape == ref_flat.shape, f"shape mismatch: {y.shape} vs {ref_flat.shape}"
    compare_outputs(y, ref_flat, reduction_verification((ref_flat).dtype))

    y_keep = _call(ArgminFwdOp(dim=None, keepdim=True), x)
    expected_shape = tuple(1 for _ in shape)
    assert y_keep.shape == expected_shape, (
        f"keepdim shape mismatch: {y_keep.shape} vs {expected_shape}"
    )
    compare_outputs(y_keep.reshape(()), ref_flat, reduction_verification((ref_flat).dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("op_kind", "dtype"),
    [("argmax", torch.float16), ("argmin", torch.bfloat16)],
)
def test_argreduce_large_n(op_kind: str, dtype: torch.dtype) -> None:
    """The multi-CTA path supports the LM-head workload without tiling skips."""
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp, ArgminFwdOp

    x = torch.randn(4, 102400, dtype=dtype, device=run_device())
    op_cls = ArgmaxFwdOp if op_kind == "argmax" else ArgminFwdOp
    op = op_cls(dim=-1)
    ref = getattr(torch, op_kind)(x, dim=-1)
    y = _call(op, x)
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", ["argmax", "argmin"])
def test_argreduce_first_index_and_nan_semantics(op_kind: str) -> None:
    """Pair reduction preserves PyTorch's first-index and NaN behavior."""
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp, ArgminFwdOp

    x = torch.tensor(
        [
            [1.0, float("nan"), 3.0, float("nan"), 3.0],
            [2.0, 2.0, -1.0, -1.0, 0.0],
        ],
        dtype=torch.float32,
        device=run_device(),
    )
    op_cls = ArgmaxFwdOp if op_kind == "argmax" else ArgminFwdOp
    op = op_cls(dim=-1)
    ref = getattr(torch, op_kind)(x, dim=-1)
    y = _call(op, x)
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", ["argmax", "argmin"])
@pytest.mark.parametrize("n", [8, 4096, 40960])
def test_argreduce_ties_between_two_nans(op_kind: str, n: int) -> None:
    """Two NaNs tie, so the lower index wins — on every reduction layout.

    No float comparison can express this: every comparison between two NaNs is
    false, so an ordering that leans on them returns whichever lane merged
    first. The three row lengths select the warp, CTA and multi-CTA paths, and
    the NaNs sit far enough apart to land in different lanes.
    """
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp, ArgminFwdOp

    x = torch.arange(1, n + 1, dtype=torch.float32, device=run_device())
    first, second = n // 4, n // 2
    x[first] = float("nan")
    x[second] = float("nan")

    op = (ArgmaxFwdOp if op_kind == "argmax" else ArgminFwdOp)(dim=-1)
    ref = getattr(torch, op_kind)(x)
    got = _call(op, x)
    assert got.item() == ref.item() == first, (
        f"{op_kind} n={n}: got {got.item()}, torch {ref.item()}, expected {first}"
    )


@pytest.mark.parametrize(
    "tune",
    [pytest.param(False, marks=pytest.mark.smoke), pytest.param(True, marks=pytest.mark.full)],
)
def test_argmax_wide_rows_under_tuning(tune: bool) -> None:
    """Rows wide enough to split across CTAs; the extremes sit in late chunks."""
    from tileops.ops.reduction import ArgmaxFwdOp

    x = torch.randn(3, 65536, dtype=torch.float16, device=run_device())
    x[0, 65000] = 100.0
    x[1, 40000] = 100.0
    x[2, 64000] = float("nan")
    op = ArgmaxFwdOp(dim=-1)
    if tune:
        op.request_tune()
    compare_outputs(
        op(x),
        torch.argmax(x, dim=-1),
        reduction_verification(torch.int64),
    )


@pytest.mark.parametrize(
    "shape, dim",
    [
        pytest.param((64, 8192), -1, marks=pytest.mark.smoke, id="rows"),
        pytest.param((4, 128, 4096), 0, marks=pytest.mark.full, id="strided"),
    ],
)
def test_argmax_contiguous_input_off_a_vector_boundary(shape: tuple, dim: int) -> None:
    """A contiguous input starting one element past a 16-byte boundary matches torch."""
    from tileops.ops.reduction import ArgmaxFwdOp

    x = torch.randn(math.prod(shape) + 1, dtype=torch.float16, device=run_device())[1:]
    x = x.view(shape)
    compare_outputs(
        _call(ArgmaxFwdOp(dim=dim), x),
        torch.argmax(x, dim=dim),
        reduction_verification(torch.int64),
    )
