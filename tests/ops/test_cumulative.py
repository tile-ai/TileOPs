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
                # A whole-row chunk of 33 elements, too odd for any vector width
                pytest.param(128, 8448, torch.float16, marks=pytest.mark.full),
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


@pytest.mark.smoke
@pytest.mark.parametrize("rows", [3, 5, 7, 15])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cumsum_parallel_odd_rows(rows: int, dtype: torch.dtype) -> None:
    """A partial row block must admit a layout for the parallel prefix scan."""
    from tileops.reduction import CumsumFwdOp

    test = CumulativeTest((rows, 8193), dtype, "cumsum")
    test.check(CumsumFwdOp(dim=-1), *test.gen_inputs())


# Helper to get tolerances


@CumulativeBasicFixture
def test_cumsum_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops._ops.reduction.cumulative import CumsumFwdOp

    test = CumulativeTest((m, n), dtype, "cumsum")
    op = CumsumFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@CumulativeNonContigFixture
def test_cumsum_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops._ops.reduction.cumulative import CumsumFwdOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    x = x_full[:, :n]
    op = CumsumFwdOp(dim=-1)
    ref = x.contiguous().float().cumsum(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x))


@Cumulative3DFixture
def test_cumsum_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops._ops.reduction.cumulative import CumsumFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = CumsumFwdOp(dim=-1)
    ref = x.float().cumsum(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x))


@Cumulative4DFixture
def test_cumsum_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops._ops.reduction.cumulative import CumsumFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = CumsumFwdOp(dim=-1)
    ref = x.float().cumsum(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x))


@Cumulative1DFixture
def test_cumsum_1d(n: int, dtype: torch.dtype) -> None:
    from tileops._ops.reduction.cumulative import CumsumFwdOp

    x = torch.randn(n, dtype=dtype, device=run_device())
    op = CumsumFwdOp(dim=-1)
    ref = x.float().cumsum(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x))


@CumulativeBasicFixture
def test_cumprod_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops._ops.reduction.cumulative import CumprodFwdOp

    test = CumulativeTest((m, n), dtype, "cumprod", use_small_range=True)
    op = CumprodFwdOp(dim=-1)
    test.check(op, *test.gen_inputs())


@CumulativeNonContigFixture
def test_cumprod_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops._ops.reduction.cumulative import CumprodFwdOp

    x_full = torch.rand(m, n * 2, dtype=dtype, device=run_device()) * 0.01 + 0.99
    x = x_full[:, :n]
    op = CumprodFwdOp(dim=-1)
    ref = x.contiguous().float().cumprod(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumprod").verification(x))


@Cumulative3DFixture
def test_cumprod_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops._ops.reduction.cumulative import CumprodFwdOp

    x = torch.rand(batch, seq, hidden, dtype=dtype, device=run_device()) * 0.01 + 0.99
    op = CumprodFwdOp(dim=-1)
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumprod").verification(x))


@Cumulative4DFixture
def test_cumprod_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops._ops.reduction.cumulative import CumprodFwdOp

    x = torch.rand(b0, b1, b2, n, dtype=dtype, device=run_device()) * 0.01 + 0.99
    op = CumprodFwdOp(dim=-1)
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumprod").verification(x))


@Cumulative1DFixture
def test_cumprod_1d(n: int, dtype: torch.dtype) -> None:
    from tileops._ops.reduction.cumulative import CumprodFwdOp

    x = torch.rand(n, dtype=dtype, device=run_device()) * 0.01 + 0.99
    op = CumprodFwdOp(dim=-1)
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
    from tileops._ops.reduction.cumulative import CumsumFwdOp

    x = torch.randn(batch, hidden, seq, dtype=dtype, device=run_device())
    op = CumsumFwdOp(dim=1)
    ref = x.float().cumsum(dim=1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x))


@CumulativeDimAxis1Fixture
def test_cumprod_dim_axis1(batch: int, hidden: int, seq: int, dtype: torch.dtype) -> None:
    """Cumprod along dim=1 (3D) — exercises movedim choreography in `_run`."""
    from tileops._ops.reduction.cumulative import CumprodFwdOp

    # Values close to 1 to avoid over/underflow in cumprod over hidden dim.
    x = torch.rand(batch, hidden, seq, dtype=dtype, device=run_device()) * 0.01 + 0.99
    op = CumprodFwdOp(dim=1)
    ref = x.float().cumprod(dim=1).to(dtype)
    y = op(x)

    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumprod").verification(x))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "M, N, dtype, offset",
    [
        # Widths the row scan stages exactly, at a chunk of whole vector accesses.
        pytest.param(64, 16384, torch.float32, 0, id="row-scan-fp32"),
        pytest.param(64, 32768, torch.bfloat16, 0, id="row-scan-bf16"),
        # A width the alignment would pad, which the parallel scan takes.
        pytest.param(64, 8200, torch.float32, 0, id="parallel-scan-padded-fp32"),
        pytest.param(64, 8200, torch.bfloat16, 0, id="parallel-scan-padded-bf16"),
        # 65 elements a thread is not a whole number of vector accesses.
        pytest.param(64, 16640, torch.bfloat16, 0, id="parallel-scan-misaligned-chunk"),
        # A contiguous input starting one element past a 16-byte boundary.
        pytest.param(64, 16384, torch.bfloat16, 1, id="row-scan-off-a-vector-boundary"),
        # Few rows of 64 KiB, which the segmented scan takes from the whole-row scan, and
        # one row more, which stays whole.
        pytest.param(16, 32768, torch.bfloat16, 0, id="few-long-rows-segmented"),
        pytest.param(17, 32768, torch.bfloat16, 0, id="few-long-rows-whole"),
    ],
)
def test_cumsum_matches_torch_on_each_scan(M: int, N: int, dtype: torch.dtype, offset: int) -> None:
    """Rows of the widths each scan serves match torch.cumsum."""
    from tileops._ops.reduction.cumulative import CumsumFwdOp

    x = torch.randn(M * N + offset, dtype=dtype, device=run_device())[offset:].view(M, N)
    y = CumsumFwdOp(dim=-1)(x)

    ref = x.float().cumsum(dim=-1).to(dtype)
    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "name, marks",
    [
        ("nan", [(100, float("nan"))]),
        ("both_infinities", [(100, float("inf")), (3000, -float("inf"))]),
        ("signed_zero", [(100, -0.0), (3000, 0.0)]),
    ],
)
@pytest.mark.parametrize(
    "n",
    [
        pytest.param(4096, id="whole-row"),
        # The marks fall in different segments of a row with a partial last segment.
        pytest.param(16385, id="segments"),
    ],
)
def test_scan_nonfinite_and_signed_zero_match_torch(name: str, marks: list, n: int) -> None:
    """A scan carries non-finite values and signed zero the way torch does.

    The whole-row backend combines per-thread chunk totals rather than accumulating left
    to right, and the segmented one sums the totals of earlier segments, so a NaN or an
    inf has to still reach every later element and no earlier one.
    """
    from tileops._ops.reduction.cumulative import CumprodFwdOp, CumsumFwdOp

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


@pytest.mark.parametrize(
    "M, N",
    [
        # Segments that end on the row's end.
        pytest.param(1, 65536, marks=pytest.mark.smoke),
        # A partial last segment on every row.
        pytest.param(127, 16385, marks=pytest.mark.smoke),
        # More segments ahead of the last one than a block has threads.
        pytest.param(1, 2**19 + 7, marks=pytest.mark.smoke),
        # More segments than a grid's second dimension holds.
        pytest.param(1, 2**24, marks=pytest.mark.full),
    ],
)
def test_cumsum_parallel_scan_row_ownership(M: int, N: int) -> None:
    """Every segment of a row adds the totals of that row's earlier segments, and only those."""
    from tileops._ops.reduction.cumulative import CumsumFwdOp

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
    from tileops._ops.reduction.cumulative import CumsumFwdOp

    op = CumsumFwdOp(dim=-1)
    x = torch.randn(M, N, dtype=dtype, device=run_device())
    op(x)

    compiled = torch.compile(op, fullgraph=True)
    y = compiled(x)

    ref = x.float().cumsum(dim=-1).to(dtype)
    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x))


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", ["cumsum", "cumprod"])
@pytest.mark.parametrize(
    "shape, in_dtype, out_dtype",
    [
        pytest.param((64, 1024), torch.bfloat16, torch.float32, id="row-scan"),
        # A 4-byte output halves the whole-row chunk to 32 elements.
        pytest.param((64, 4096), torch.bfloat16, torch.float32, id="row-scan-wide-output"),
        pytest.param((64, 1024), torch.float16, torch.float32, id="row-scan-fp16"),
        # Unaligned columns and a partial row block.
        pytest.param((129, 300), torch.bfloat16, torch.float32, id="tiled-scan"),
        # cumsum takes the parallel scan here, cumprod the tiled scan.
        pytest.param((64, 8200), torch.bfloat16, torch.float32, id="parallel-scan"),
        # The float32 output makes these rows 64 KiB: cumsum segments them, cumprod keeps
        # the whole-row scan.
        pytest.param((16, 16384), torch.bfloat16, torch.float32, id="few-long-rows-widening"),
    ],
)
def test_dtype_casts_the_input_before_the_scan(
    op_kind: str, shape: tuple, in_dtype: torch.dtype, out_dtype: torch.dtype
) -> None:
    """``dtype`` casts the input before the scan and sets the output dtype, as in torch."""
    from tileops._ops.reduction.cumulative import CumprodFwdOp, CumsumFwdOp

    workload = CumulativeWorkload(shape, in_dtype, op_kind, out_dtype=out_dtype)
    (x,) = workload.gen_inputs()
    op_cls = CumsumFwdOp if op_kind == "cumsum" else CumprodFwdOp
    y = op_cls(dim=-1, dtype=out_dtype)(x)
    assert y.dtype == out_dtype
    compare_outputs(y, workload.torch_program(x), workload.verification(x.to(out_dtype)))
