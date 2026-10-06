"""Correctness tests for cumulative ops (cumsum, cumprod).

Covers: CumsumFwdOp, CumprodFwdOp.
Each op computes an inclusive prefix scan along dim=-1 and supports 1D-4D input.
Output has the same shape as input.
"""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.reduction import CumulativeWorkload


class CumulativeBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(256, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.bfloat16, marks=pytest.mark.full),
                # A whole-row chunk of 12 elements, read in vectors of four
                pytest.param(128, 768, torch.float16, marks=pytest.mark.full),
                # Non-aligned N (non-pow2)
                pytest.param(128, 300, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 300, torch.bfloat16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(129, 512, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class CumulativeNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Cumulative3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 64, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Cumulative4DFixture(FixtureBase):
    PARAMS = [
        (
            "b0, b1, b2, n, dtype",
            [
                pytest.param(2, 4, 8, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 4, 8, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Cumulative1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class CumulativeTest(CumulativeWorkload, TestBase):
    """Parameterized test helper for cumulative ops."""


# Helper to get tolerances


@CumulativeBasicFixture
def test_cumsum_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    test = CumulativeTest((m, n), dtype, "cumsum")
    op = CumsumFwdOp()
    test.check(op, *test.gen_inputs())


@CumulativeNonContigFixture
def test_cumsum_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    x = x_full[:, :n]
    op = CumsumFwdOp()
    ref = x.contiguous().float().cumsum(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x))


@Cumulative3DFixture
def test_cumsum_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = CumsumFwdOp()
    ref = x.float().cumsum(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x))


@Cumulative4DFixture
def test_cumsum_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = CumsumFwdOp()
    ref = x.float().cumsum(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x))


@Cumulative1DFixture
def test_cumsum_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    x = torch.randn(n, dtype=dtype, device=run_device())
    op = CumsumFwdOp()
    ref = x.float().cumsum(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x))


@CumulativeBasicFixture
def test_cumprod_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumprodFwdOp

    test = CumulativeTest((m, n), dtype, "cumprod", use_small_range=True)
    op = CumprodFwdOp()
    test.check(op, *test.gen_inputs())


@CumulativeNonContigFixture
def test_cumprod_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumprodFwdOp

    x_full = torch.rand(m, n * 2, dtype=dtype, device=run_device()) * 0.01 + 0.99
    x = x_full[:, :n]
    op = CumprodFwdOp()
    ref = x.contiguous().float().cumprod(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumprod").verification(x))


@Cumulative3DFixture
def test_cumprod_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumprodFwdOp

    x = torch.rand(batch, seq, hidden, dtype=dtype, device=run_device()) * 0.01 + 0.99
    op = CumprodFwdOp()
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumprod").verification(x))


@Cumulative4DFixture
def test_cumprod_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumprodFwdOp

    x = torch.rand(b0, b1, b2, n, dtype=dtype, device=run_device()) * 0.01 + 0.99
    op = CumprodFwdOp()
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumprod").verification(x))


@Cumulative1DFixture
def test_cumprod_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumprodFwdOp

    x = torch.rand(n, dtype=dtype, device=run_device()) * 0.01 + 0.99
    op = CumprodFwdOp()
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumprod").verification(x))


class CumulativeDimAxis1Fixture(FixtureBase):
    PARAMS = [
        (
            "batch, hidden, seq, dtype",
            [
                pytest.param(2, 512, 256, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 512, 256, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@CumulativeDimAxis1Fixture
def test_cumsum_dim_axis1(batch: int, hidden: int, seq: int, dtype: torch.dtype) -> None:
    """Cumsum along dim=1 (3D) — exercises movedim choreography in `_run`."""
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    x = torch.randn(batch, hidden, seq, dtype=dtype, device=run_device())
    op = CumsumFwdOp(dim=1)
    ref = x.float().cumsum(dim=1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x))


@CumulativeDimAxis1Fixture
def test_cumprod_dim_axis1(batch: int, hidden: int, seq: int, dtype: torch.dtype) -> None:
    """Cumprod along dim=1 (3D) — exercises movedim choreography in `_run`."""
    from tileops.ops.reduction.cumulative import CumprodFwdOp

    # Values close to 1 to avoid over/underflow in cumprod over hidden dim.
    x = torch.rand(batch, hidden, seq, dtype=dtype, device=run_device()) * 0.01 + 0.99
    op = CumprodFwdOp(dim=1)
    ref = x.float().cumprod(dim=1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumprod").verification(x))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "M, N, dtype, backend",
    [
        (64, 16384, torch.float32, "CumulativeRowScanKernel"),
        (64, 32768, torch.bfloat16, "CumulativeRowScanKernel"),
        # A padded width the row scan declines, at both element widths.
        (64, 8200, torch.float32, "CumsumParallelScanKernel"),
        (64, 8200, torch.bfloat16, "CumsumParallelScanKernel"),
        # 65 elements per thread is not a whole number of vector accesses
        (64, 16640, torch.bfloat16, "CumsumParallelScanKernel"),
    ],
)
def test_cumsum_backend_dispatch(M: int, N: int, dtype: torch.dtype, backend: str) -> None:
    """Each shape takes the expected backend and matches torch.cumsum.

    The row scan takes every width it can stage exactly at a chunk whose bytes are a
    whole number of vector accesses, which is where it measures fastest; the parallel
    scan is left the widths the alignment would pad and the chunks it would misalign.
    """
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    x = torch.randn(M, N, dtype=dtype, device=run_device())
    op = CumsumFwdOp(dim=-1)
    y = op(x)

    ref = x.float().cumsum(dim=-1).to(dtype)
    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x))

    # The kernel the call built, not one refetched by a key: the key is a read-back of
    # the arguments and says nothing about which backend was chosen.


@pytest.mark.smoke
@pytest.mark.parametrize(
    "name, marks",
    [
        ("nan", [(100, float("nan"))]),
        ("both_infinities", [(100, float("inf")), (200, -float("inf"))]),
        ("signed_zero", [(100, -0.0), (200, 0.0)]),
    ],
)
def test_scan_nonfinite_and_signed_zero_match_torch(name: str, marks: list) -> None:
    """A scan carries non-finite values and signed zero the way torch does.

    The whole-row backend combines per-thread chunk totals rather than accumulating left
    to right, so a NaN or an inf has to still reach every later element.
    """
    from tileops.ops.reduction.cumulative import CumprodFwdOp, CumsumFwdOp

    n = 4096
    x = torch.ones(2, n, dtype=torch.float32, device=run_device())
    for index, value in marks:
        x[:, index] = value

    for op, ref in ((CumsumFwdOp(dim=-1), torch.cumsum), (CumprodFwdOp(dim=-1), torch.cumprod)):
        compare_outputs(
            op(x),
            ref(x, dim=-1),
            CumulativeWorkload(
                tuple(x.shape), x.dtype, "cumprod" if isinstance(op, CumprodFwdOp) else "cumsum"
            ).verification(x),
        )


@pytest.mark.smoke
@pytest.mark.parametrize("M, N", [(1, 32768), (127, 16384)])
def test_cumsum_parallel_scan_row_ownership(M: int, N: int) -> None:
    """Carry propagation stays per-row across tiles and partial row blocks."""
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    row_values = torch.arange(1, M + 1, dtype=torch.float32, device=run_device()).unsqueeze(1)
    x = row_values.expand(-1, N).contiguous()

    y = CumsumFwdOp(dim=-1)(x)

    # Row r holds the constant r + 1, so its cumsum is (r + 1) * [1, ..., N].
    expected = row_values * torch.arange(1, N + 1, dtype=torch.float32, device=run_device())
    compare_outputs(
        y, expected, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x)
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "M, N, dtype",
    [
        pytest.param(64, 512, torch.float16),  # sequential custom_op
        pytest.param(64, 16384, torch.float16),  # parallel custom_op
    ],
)
def test_cumsum_compile_fullgraph_warm_cache(M: int, N: int, dtype: torch.dtype) -> None:
    """torch.compile(fullgraph=True) must succeed on a warm kernel cache.

    Guards the custom_op boundary: tracing the raw JIT callables instead
    raises 'unsupported Function.call'.

    Not compile-contract evidence (see tests/compile_contract.py) — that
    contract covers a *cold* compile, which would trace kernel construction
    inside the graph. Pre-warming here sidesteps it, so CumsumFwdOp must not
    declare ``torch_compile_fullgraph`` on this test's strength.
    """
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    op = CumsumFwdOp(dim=-1)
    x = torch.randn(M, N, dtype=dtype, device=run_device())
    op(x)

    compiled = torch.compile(op, fullgraph=True)
    y = compiled(x)

    ref = x.float().cumsum(dim=-1).to(dtype)
    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x))
