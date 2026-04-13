"""Correctness tests for argreduce ops (argmax, argmin).

Covers: ArgmaxFwdOp, ArgminFwdOp.
Each op reduces along a configurable dim and returns int64 indices.
Uses exact match (torch.equal) instead of allclose.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from workloads.argreduce import ArgmaxTest as _ArgmaxWorkload

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class ArgreduceBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.full),
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
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class Argreduce3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 64, 512, torch.bfloat16, marks=pytest.mark.full),
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
                pytest.param(4, 8, 256, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(4, 8, 256, torch.float32, marks=pytest.mark.full),
            ],
        ),
    ]


class Argreduce4DFixture(FixtureBase):
    PARAMS = [
        (
            "b0, b1, b2, n, dtype",
            [
                pytest.param(2, 4, 8, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 4, 8, 512, torch.bfloat16, marks=pytest.mark.full),
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
                pytest.param(2, 4, 8, 256, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class Argreduce1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(512, torch.float32, marks=pytest.mark.full),
                pytest.param(512, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class SpecArgreduceFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, keepdim, dtype",
            [
                pytest.param((128, 512), -1, False, torch.float16, marks=pytest.mark.smoke),
                pytest.param((128, 512), -1, True, torch.float16, marks=pytest.mark.full),
                pytest.param((512, 4, 32), 0, False, torch.float16, marks=pytest.mark.full),
                pytest.param((4, 32, 512), 1, False, torch.float16, marks=pytest.mark.full),
                pytest.param((4, 32, 512), -1, False, torch.bfloat16, marks=pytest.mark.full),
                pytest.param((4, 32, 512), -1, True, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# TestBase helpers — inherit gen_inputs() from workload classes
# ---------------------------------------------------------------------------


class ArgreduceTest(_ArgmaxWorkload, TestBase):
    """Parameterized test helper for argreduce ops."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# ArgmaxFwdOp tests
# ---------------------------------------------------------------------------


@ArgreduceBasicFixture
def test_argmax_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    test = ArgreduceTest(m, n, dtype, "argmax")
    op = ArgmaxFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@ArgreduceNonContigFixture
def test_argmax_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]
    op = ArgmaxFwdOp(dtype=dtype)
    ref = x.contiguous().argmax(dim=-1)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"non-contig argmax mismatch: {(y != ref).sum().item()}"


@Argreduce3DFixture
def test_argmax_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = ArgmaxFwdOp(dtype=dtype)
    ref = x.argmax(dim=-1)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D argmax mismatch: {(y != ref).sum().item()}"


@Argreduce4DFixture
def test_argmax_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = ArgmaxFwdOp(dtype=dtype)
    ref = x.argmax(dim=-1)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D argmax mismatch: {(y != ref).sum().item()}"


@Argreduce1DFixture
def test_argmax_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(n, dtype=dtype, device="cuda")
    op = ArgmaxFwdOp(dtype=dtype)
    ref = x.argmax(dim=-1)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y.view_as(ref), ref), "1D argmax mismatch"


@Argreduce3DDim0Fixture
def test_argmax_3d_dim0(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Argmax along dim=0 on 3D tensors (outermost-dim reduction)."""
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = ArgmaxFwdOp(dtype=dtype, dim=0)
    ref = x.argmax(dim=0)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D dim=0 argmax mismatch: {(y != ref).sum().item()}"


@Argreduce3DDim0Fixture
def test_argmax_3d_dim0_keepdim(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Argmax along dim=0 with keepdim=True on 3D tensors."""
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = ArgmaxFwdOp(dtype=dtype, dim=0, keepdim=True)
    ref = x.argmax(dim=0, keepdim=True)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D dim=0 keepdim argmax mismatch: {(y != ref).sum().item()}"


@Argreduce4DDim0Fixture
def test_argmax_4d_dim0(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    """Argmax along dim=0 on 4D tensors (outermost-dim reduction, 3D+ regression)."""
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = ArgmaxFwdOp(dtype=dtype, dim=0)
    ref = x.argmax(dim=0)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D dim=0 argmax mismatch: {(y != ref).sum().item()}"


@Argreduce4DDim0Fixture
def test_argmax_4d_dim0_keepdim(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    """Argmax along dim=0 with keepdim=True on 4D tensors."""
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = ArgmaxFwdOp(dtype=dtype, dim=0, keepdim=True)
    ref = x.argmax(dim=0, keepdim=True)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D dim=0 keepdim argmax mismatch: {(y != ref).sum().item()}"


@SpecArgreduceFixture
def test_argmax_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: ArgmaxFwdOp with dim + keepdim."""
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = ArgmaxFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    ref = x.argmax(dim=dim, keepdim=keepdim)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"spec dim={dim} argmax mismatch: {(y != ref).sum().item()}"


# ---------------------------------------------------------------------------
# ArgminFwdOp tests
# ---------------------------------------------------------------------------


@ArgreduceBasicFixture
def test_argmin_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmin import ArgminFwdOp

    test = ArgreduceTest(m, n, dtype, "argmin")
    op = ArgminFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@ArgreduceNonContigFixture
def test_argmin_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]
    op = ArgminFwdOp(dtype=dtype)
    ref = x.contiguous().argmin(dim=-1)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"non-contig argmin mismatch: {(y != ref).sum().item()}"


@Argreduce3DFixture
def test_argmin_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = ArgminFwdOp(dtype=dtype)
    ref = x.argmin(dim=-1)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D argmin mismatch: {(y != ref).sum().item()}"


@Argreduce4DFixture
def test_argmin_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = ArgminFwdOp(dtype=dtype)
    ref = x.argmin(dim=-1)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D argmin mismatch: {(y != ref).sum().item()}"


@Argreduce1DFixture
def test_argmin_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(n, dtype=dtype, device="cuda")
    op = ArgminFwdOp(dtype=dtype)
    ref = x.argmin(dim=-1)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y.view_as(ref), ref), "1D argmin mismatch"


@Argreduce3DDim0Fixture
def test_argmin_3d_dim0(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Argmin along dim=0 on 3D tensors (outermost-dim reduction)."""
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = ArgminFwdOp(dtype=dtype, dim=0)
    ref = x.argmin(dim=0)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D dim=0 argmin mismatch: {(y != ref).sum().item()}"


@Argreduce3DDim0Fixture
def test_argmin_3d_dim0_keepdim(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Argmin along dim=0 with keepdim=True on 3D tensors."""
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = ArgminFwdOp(dtype=dtype, dim=0, keepdim=True)
    ref = x.argmin(dim=0, keepdim=True)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D dim=0 keepdim argmin mismatch: {(y != ref).sum().item()}"


@Argreduce4DDim0Fixture
def test_argmin_4d_dim0(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    """Argmin along dim=0 on 4D tensors (outermost-dim reduction, 3D+ regression)."""
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = ArgminFwdOp(dtype=dtype, dim=0)
    ref = x.argmin(dim=0)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D dim=0 argmin mismatch: {(y != ref).sum().item()}"


@Argreduce4DDim0Fixture
def test_argmin_4d_dim0_keepdim(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    """Argmin along dim=0 with keepdim=True on 4D tensors."""
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = ArgminFwdOp(dtype=dtype, dim=0, keepdim=True)
    ref = x.argmin(dim=0, keepdim=True)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D dim=0 keepdim argmin mismatch: {(y != ref).sum().item()}"


@SpecArgreduceFixture
def test_argmin_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: ArgminFwdOp with dim + keepdim."""
    from tileops.ops.reduction.argmin import ArgminFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = ArgminFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    ref = x.argmin(dim=dim, keepdim=keepdim)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"spec dim={dim} argmin mismatch: {(y != ref).sum().item()}"


# ---------------------------------------------------------------------------
# Regression: multidim dim must be rejected for argreduce ops
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls_path, dim", [
    ("tileops.ops.reduction.argmax.ArgmaxFwdOp", [0, 1]),
    ("tileops.ops.reduction.argmin.ArgminFwdOp", [0, 1]),
    ("tileops.ops.reduction.argmax.ArgmaxFwdOp", None),
    ("tileops.ops.reduction.argmin.ArgminFwdOp", None),
])
def test_argreduce_rejects_multidim(op_cls_path: str, dim) -> None:
    """Argreduce ops only support scalar dim; list/tuple/None must raise."""
    import importlib

    module_path, cls_name = op_cls_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    op_cls = getattr(mod, cls_name)

    with pytest.raises((TypeError, ValueError)):
        op_cls(dtype=torch.float16, dim=dim)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
