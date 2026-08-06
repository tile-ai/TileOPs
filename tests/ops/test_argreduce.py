"""Correctness tests for argreduce ops (argmax, argmin).

Covers: ArgmaxFwdOp, ArgminFwdOp.
Each op reduces along a configurable dim and returns int64 indices.
Uses exact match (torch.equal) instead of allclose.
"""

from typing import cast

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from workloads.reduction import ArgmaxWorkload


def _call(op, x: torch.Tensor) -> torch.Tensor:
    """Invoke a single-output argreduce op and narrow the return to ``Tensor``.

    The shared ``OpBase.__call__`` is typed as ``Union[Tensor, tuple]`` to
    accommodate ops with multiple outputs.  Argreduce ops always return a
    single ``Tensor``; this helper keeps the call sites well-typed without
    sprinkling ``cast(...)`` everywhere.
    """
    return cast(torch.Tensor, op(x))

# Fixtures


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


# TestBase helpers — inherit gen_inputs() from workload classes


class ArgreduceTest(ArgmaxWorkload, TestBase):
    """Parameterized test helper for argreduce ops."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind

    def ref_program(self, *inputs: torch.Tensor) -> torch.Tensor:
        (x,) = inputs
        if self.op_kind == "argmax":
            return x.argmax(dim=-1)
        elif self.op_kind == "argmin":
            return x.argmin(dim=-1)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


def _exact_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Exact match comparison using torch.equal."""
    assert output.dtype == torch.int64, f"Expected int64, got {output.dtype}"
    assert output_ref.dtype == torch.int64, f"Expected ref int64, got {output_ref.dtype}"
    assert torch.equal(output, output_ref), (
        f"Indices mismatch.\n"
        f"  output:     {output[:10]}...\n"
        f"  output_ref: {output_ref[:10]}...\n"
        f"  mismatches: {(output != output_ref).sum().item()} / {output.numel()}"
    )


# ArgmaxFwdOp tests


@ArgreduceBasicFixture
def test_argmax_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    test = ArgreduceTest(m, n, dtype, "argmax")
    op = ArgmaxFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@ArgreduceNonContigFixture
def test_argmax_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]
    op = ArgmaxFwdOp(dim=-1)
    ref = x.contiguous().argmax(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"non-contig argmax mismatch: {(y != ref).sum().item()}"


@Argreduce3DFixture
def test_argmax_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = ArgmaxFwdOp(dim=-1)
    ref = x.argmax(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D argmax mismatch: {(y != ref).sum().item()}"


@Argreduce4DFixture
def test_argmax_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = ArgmaxFwdOp(dim=-1)
    ref = x.argmax(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D argmax mismatch: {(y != ref).sum().item()}"


@Argreduce1DFixture
def test_argmax_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(n, dtype=dtype, device="cuda")
    op = ArgmaxFwdOp(dim=-1)
    ref = x.argmax(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert torch.equal(y.view_as(ref), ref), "1D argmax mismatch"


@Argreduce3DDim0Fixture
def test_argmax_3d_dim0(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Argmax along dim=0 on 3D tensors (outermost-dim reduction)."""
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = ArgmaxFwdOp(dim=0)
    ref = x.argmax(dim=0)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D dim=0 argmax mismatch: {(y != ref).sum().item()}"


@Argreduce3DDim0Fixture
def test_argmax_3d_dim0_keepdim(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Argmax along dim=0 with keepdim=True on 3D tensors."""
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = ArgmaxFwdOp(dim=0, keepdim=True)
    ref = x.argmax(dim=0, keepdim=True)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D dim=0 keepdim argmax mismatch: {(y != ref).sum().item()}"


@Argreduce4DDim0Fixture
def test_argmax_4d_dim0(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    """Argmax along dim=0 on 4D tensors (outermost-dim reduction, 3D+ regression)."""
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = ArgmaxFwdOp(dim=0)
    ref = x.argmax(dim=0)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D dim=0 argmax mismatch: {(y != ref).sum().item()}"


@Argreduce4DDim0Fixture
def test_argmax_4d_dim0_keepdim(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    """Argmax along dim=0 with keepdim=True on 4D tensors."""
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = ArgmaxFwdOp(dim=0, keepdim=True)
    ref = x.argmax(dim=0, keepdim=True)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D dim=0 keepdim argmax mismatch: {(y != ref).sum().item()}"


@SpecArgreduceFixture
def test_argmax_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: ArgmaxFwdOp with dim + keepdim."""
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = ArgmaxFwdOp(dim=dim, keepdim=keepdim)
    ref = x.argmax(dim=dim, keepdim=keepdim)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"spec dim={dim} argmax mismatch: {(y != ref).sum().item()}"


# ArgminFwdOp tests


@ArgreduceBasicFixture
def test_argmin_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    test = ArgreduceTest(m, n, dtype, "argmin")
    op = ArgminFwdOp(dim=-1)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@ArgreduceNonContigFixture
def test_argmin_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]
    op = ArgminFwdOp(dim=-1)
    ref = x.contiguous().argmin(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"non-contig argmin mismatch: {(y != ref).sum().item()}"


@Argreduce3DFixture
def test_argmin_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = ArgminFwdOp(dim=-1)
    ref = x.argmin(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D argmin mismatch: {(y != ref).sum().item()}"


@Argreduce4DFixture
def test_argmin_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = ArgminFwdOp(dim=-1)
    ref = x.argmin(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D argmin mismatch: {(y != ref).sum().item()}"


@Argreduce1DFixture
def test_argmin_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(n, dtype=dtype, device="cuda")
    op = ArgminFwdOp(dim=-1)
    ref = x.argmin(dim=-1)
    y = _call(op, x)
    assert y.dtype == torch.int64
    assert torch.equal(y.view_as(ref), ref), "1D argmin mismatch"


@Argreduce3DDim0Fixture
def test_argmin_3d_dim0(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Argmin along dim=0 on 3D tensors (outermost-dim reduction)."""
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = ArgminFwdOp(dim=0)
    ref = x.argmin(dim=0)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D dim=0 argmin mismatch: {(y != ref).sum().item()}"


@Argreduce3DDim0Fixture
def test_argmin_3d_dim0_keepdim(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Argmin along dim=0 with keepdim=True on 3D tensors."""
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = ArgminFwdOp(dim=0, keepdim=True)
    ref = x.argmin(dim=0, keepdim=True)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D dim=0 keepdim argmin mismatch: {(y != ref).sum().item()}"


@Argreduce4DDim0Fixture
def test_argmin_4d_dim0(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    """Argmin along dim=0 on 4D tensors (outermost-dim reduction, 3D+ regression)."""
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = ArgminFwdOp(dim=0)
    ref = x.argmin(dim=0)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D dim=0 argmin mismatch: {(y != ref).sum().item()}"


@Argreduce4DDim0Fixture
def test_argmin_4d_dim0_keepdim(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    """Argmin along dim=0 with keepdim=True on 4D tensors."""
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = ArgminFwdOp(dim=0, keepdim=True)
    ref = x.argmin(dim=0, keepdim=True)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D dim=0 keepdim argmin mismatch: {(y != ref).sum().item()}"


@SpecArgreduceFixture
def test_argmin_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: ArgminFwdOp with dim + keepdim."""
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = ArgminFwdOp(dim=dim, keepdim=keepdim)
    ref = x.argmin(dim=dim, keepdim=keepdim)
    y = _call(op, x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"spec dim={dim} argmin mismatch: {(y != ref).sum().item()}"


# Regression: multidim dim must be rejected for argreduce ops


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls_path, dim", [
    ("tileops.ops.reduction.argreduce.ArgmaxFwdOp", [0, 1]),
    ("tileops.ops.reduction.argreduce.ArgminFwdOp", [0, 1]),
    ("tileops.ops.reduction.argreduce.ArgmaxFwdOp", (0, 1)),
    ("tileops.ops.reduction.argreduce.ArgminFwdOp", (0, 1)),
])
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

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    ref_flat = torch.argmax(x)

    y = _call(ArgmaxFwdOp(dim=None), x)
    assert y.dtype == torch.int64
    assert y.shape == ref_flat.shape, f"shape mismatch: {y.shape} vs {ref_flat.shape}"
    assert torch.equal(y, ref_flat), f"dim=None argmax mismatch on shape={shape} dtype={dtype}"

    y_keep = _call(ArgmaxFwdOp(dim=None, keepdim=True), x)
    expected_shape = tuple(1 for _ in shape)
    assert y_keep.shape == expected_shape, f"keepdim shape mismatch: {y_keep.shape} vs {expected_shape}"
    assert torch.equal(y_keep.reshape(()), ref_flat), (
        f"dim=None keepdim argmax value mismatch on shape={shape} dtype={dtype}"
    )


@ArgreduceDimNoneFixture
def test_argmin_dim_none(shape: tuple, dtype: torch.dtype) -> None:
    """ArgminFwdOp(dim=None) matches torch.argmin(x); covers keepdim={False, True}."""
    from tileops.ops.reduction.argreduce import ArgminFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    ref_flat = torch.argmin(x)

    y = _call(ArgminFwdOp(dim=None), x)
    assert y.dtype == torch.int64
    assert y.shape == ref_flat.shape, f"shape mismatch: {y.shape} vs {ref_flat.shape}"
    assert torch.equal(y, ref_flat), f"dim=None argmin mismatch on shape={shape} dtype={dtype}"

    y_keep = _call(ArgminFwdOp(dim=None, keepdim=True), x)
    expected_shape = tuple(1 for _ in shape)
    assert y_keep.shape == expected_shape, f"keepdim shape mismatch: {y_keep.shape} vs {expected_shape}"
    assert torch.equal(y_keep.reshape(()), ref_flat), (
        f"dim=None keepdim argmin value mismatch on shape={shape} dtype={dtype}"
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("op_kind", "dtype"),
    [("argmax", torch.float16), ("argmin", torch.bfloat16)],
)
def test_argreduce_large_n(op_kind: str, dtype: torch.dtype) -> None:
    """The multi-CTA path supports the LM-head workload without tiling skips."""
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp, ArgminFwdOp

    x = torch.randn(4, 102400, dtype=dtype, device="cuda")
    op_cls = ArgmaxFwdOp if op_kind == "argmax" else ArgminFwdOp
    op = op_cls(dim=-1)
    ref = getattr(torch, op_kind)(x, dim=-1)
    y = _call(op, x)
    assert torch.equal(y, ref), f"large-N {op_kind} mismatch: {(y != ref).sum().item()}"


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
        device="cuda",
    )
    op_cls = ArgmaxFwdOp if op_kind == "argmax" else ArgminFwdOp
    op = op_cls(dim=-1)
    ref = getattr(torch, op_kind)(x, dim=-1)
    y = _call(op, x)
    assert torch.equal(y, ref)


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

    x = torch.arange(1, n + 1, dtype=torch.float32, device="cuda")
    first, second = n // 4, n // 2
    x[first] = float("nan")
    x[second] = float("nan")

    op = (ArgmaxFwdOp if op_kind == "argmax" else ArgminFwdOp)(dim=-1)
    ref = getattr(torch, op_kind)(x)
    got = _call(op, x)
    assert got.item() == ref.item() == first, (
        f"{op_kind} n={n}: got {got.item()}, torch {ref.item()}, expected {first}"
    )


@pytest.mark.smoke
@pytest.mark.parametrize("ctas_per_row", [31, 33, 64])
def test_argreduce_multicta_reduces_every_partial(ctas_per_row: int) -> None:
    """A split wider than a warp must still see every partial.

    The final pass assigns partials to lanes; reading one each would drop
    everything past lane 31 and return an index from the wrong chunk. Splits
    are chosen by the tuner, so nothing bounds them to a warp.
    """
    from tileops.kernels.reduction import argreduce as kernels

    M, N = 3, 65536
    x = torch.randn(M, N, dtype=torch.float16, device="cuda")
    # Put the extremes in high chunks, which only a full sweep of the partials
    # can reach, and a NaN in another so the tie rules run there too.
    chunk = N // ctas_per_row
    x[0, min(N - 1, chunk * (ctas_per_row - 1) + 7)] = 100.0
    x[1, min(N - 1, chunk * (ctas_per_row // 2) + 3)] = -100.0
    x[2, min(N - 1, chunk * (ctas_per_row - 2) + 1)] = float("nan")

    partial = kernels._argreduce_multicta_partial_kernel(M, N, "argmax", "float16")
    final = kernels._argreduce_multicta_final_kernel(M, N, "argmax", ctas_per_row)
    values, indices = partial(256, ctas_per_row)(x)
    got = final()(values, indices)
    torch.testing.assert_close(got, torch.argmax(x, dim=-1))


@pytest.mark.smoke
@pytest.mark.parametrize("shape, dim, expect_strided", [
    ((4, 128, 4096), 0, True),      # short strided axis: read it in place
    ((1, 32768, 8), 1, False),      # long strided axis: transposing wins
])
def test_argreduce_strided_axis_crossover(shape, dim, expect_strided) -> None:
    """A strided axis is read in place only while walking it stays cheap.

    Output-parallel gives one thread the whole axis, so choosing it on
    contiguity alone makes a long axis orders of magnitude slower.
    """
    from tileops.ops.reduction.argreduce import ArgmaxFwdOp

    x = torch.randn(*shape, device="cuda", dtype=torch.float16)
    op = ArgmaxFwdOp(dim=dim)
    torch.testing.assert_close(_call(op, x), torch.argmax(x, dim=dim))
    strategies = {k.strategy for k in op._kernel_cache.values()}
    assert ("output" in strategies) is expect_strided, strategies


@pytest.mark.smoke
@pytest.mark.parametrize("m, n, inner_stride, strategy", [
    (4, 1024, 1, "warp"),
    (4, 8192, 1, "cta"),
    (4, 65536, 1, "multi_cta"),
    (4096, 4, 4096, "output"),
])
def test_argreduce_tuning_space_matches_its_kernel(
    m: int, n: int, inner_stride: int, strategy: str,
) -> None:
    """A strategy may only offer knobs its own kernel takes.

    The four layouts are built from different JIT signatures, so one shared
    config space hands at least one of them a parameter it would reject.
    """
    from tileops.kernels.reduction.argreduce import ArgreduceKernel

    kernel = ArgreduceKernel(m, n, "argmax", torch.float16, inner_stride=inner_stride)
    assert kernel.strategy == strategy
    accepted = set(kernel.kernel.signature.parameters)
    assert set(kernel.default_config) <= accepted
    for candidate in kernel.autotune_configs:
        assert set(candidate) <= accepted, (
            f"{strategy}: candidate {candidate} names a knob outside {accepted}"
        )
    assert kernel.default_config in kernel.autotune_configs, (
        "tuning cannot be worse than not tuning: the default must be a candidate"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
