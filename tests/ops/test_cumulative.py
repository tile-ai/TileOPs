"""CumsumFwdOp and CumprodFwdOp: an inclusive prefix scan along ``dim`` of a 1D-4D input."""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops._ops.reduction.cumulative import CumprodFwdOp, CumsumFwdOp
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.reduction import CumprodExceptionalWorkload, CumulativeWorkload

_OPS = {"cumsum": CumsumFwdOp, "cumprod": CumprodFwdOp}


class CumulativeTest(CumulativeWorkload, TestBase):
    pass


def _input(op_kind: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Gaussian values; near one for a product, so it stays finite."""
    if op_kind == "cumprod":
        return torch.rand(*shape, dtype=dtype, device=run_device()) * 0.01 + 0.99
    return torch.randn(*shape, dtype=dtype, device=run_device())


def _check(op_kind: str, x: torch.Tensor, dim: int = -1) -> None:
    y = _OPS[op_kind](dim=dim)(x)
    ref = getattr(x.float(), op_kind)(dim=dim).to(x.dtype)
    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, op_kind).verification(x))


def _row_cases() -> list:
    cases = [
        pytest.param(k, 128, 512, dtype, marks=pytest.mark.smoke)
        for k in _OPS
        for dtype in (torch.float32, torch.float16, torch.bfloat16)
    ]
    for k in _OPS:
        cases += [
            pytest.param(k, 256, 4096, torch.float16, marks=pytest.mark.full),
            # A whole-row chunk of 33 elements, too odd for any vector width.
            pytest.param(k, 128, 8448, torch.float16, marks=pytest.mark.full),
            pytest.param(k, 128, 300, torch.float16, marks=pytest.mark.full),
        ]
    return cases


@pytest.mark.parametrize("op_kind, m, n, dtype", _row_cases())
def test_cumulative_rows(op_kind: str, m: int, n: int, dtype: torch.dtype) -> None:
    test = CumulativeTest((m, n), dtype, op_kind, use_small_range=op_kind == "cumprod")
    test.check(_OPS[op_kind](dim=-1), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", list(_OPS))
@pytest.mark.parametrize("view", ["strided", "1d", "4d", "dim1"])
def test_cumulative_views(op_kind: str, view: str) -> None:
    """A strided view, a 1-D row, leading axes folded into rows, and a scan along dim=1."""
    shape = {"strided": (128, 1024), "1d": (512,), "4d": (2, 4, 8, 512), "dim1": (2, 512, 256)}
    x = _input(op_kind, shape[view], torch.float16)
    if view == "strided":
        x = x[:, :512]
    _check(op_kind, x, 1 if view == "dim1" else -1)


@pytest.mark.parametrize(
    "M, N, dtype, offset",
    [
        # Widths the row scan stages exactly, at a chunk of whole vector accesses.
        pytest.param(64, 16384, torch.float32, 0, id="row-scan-fp32", marks=pytest.mark.smoke),
        pytest.param(64, 32768, torch.bfloat16, 0, id="row-scan-bf16", marks=pytest.mark.smoke),
        # Few rows with a partial last segment.
        pytest.param(3, 8193, torch.float16, 0, id="odd-rows-ragged", marks=pytest.mark.smoke),
        # A width the alignment would pad, which the parallel scan takes.
        pytest.param(
            64, 8200, torch.float32, 0, id="parallel-scan-padded-fp32", marks=pytest.mark.full
        ),
        pytest.param(
            64, 8200, torch.bfloat16, 0, id="parallel-scan-padded-bf16", marks=pytest.mark.full
        ),
        # 65 elements a thread is not a whole number of vector accesses.
        pytest.param(
            64,
            16640,
            torch.bfloat16,
            0,
            id="parallel-scan-misaligned-chunk",
            marks=pytest.mark.full,
        ),
        # A contiguous input starting one element past a 16-byte boundary.
        pytest.param(
            64,
            16384,
            torch.bfloat16,
            1,
            id="row-scan-off-a-vector-boundary",
            marks=pytest.mark.full,
        ),
        # Few rows of 64 KiB, which the segmented scan takes from the whole-row scan.
        pytest.param(
            16, 32768, torch.bfloat16, 0, id="few-long-rows-segmented", marks=pytest.mark.full
        ),
    ],
)
def test_cumsum_matches_torch_on_each_scan(M: int, N: int, dtype: torch.dtype, offset: int) -> None:
    """Rows of the widths each scan serves match torch.cumsum."""
    x = torch.randn(M * N + offset, dtype=dtype, device=run_device())[offset:].view(M, N)
    y = CumsumFwdOp(dim=-1)(x)

    ref = x.float().cumsum(dim=-1).to(dtype)
    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x))


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
        pytest.param(4096, id="whole-row", marks=pytest.mark.smoke),
        # The marks fall in different segments of a row with a partial last segment.
        pytest.param(16385, id="segments", marks=pytest.mark.full),
    ],
)
def test_scan_nonfinite_and_signed_zero_match_torch(name: str, marks: list, n: int) -> None:
    """A scan carries non-finite values and signed zero the way torch does.

    The whole-row backend combines per-thread chunk totals rather than accumulating left
    to right, and the segmented one combines the totals of earlier segments, so a NaN or an
    inf has to still reach every later element and no earlier one.
    """
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
    row_values = torch.arange(1, M + 1, dtype=torch.float32, device=run_device()).unsqueeze(1)
    x = row_values.expand(-1, N).contiguous()

    y = CumsumFwdOp(dim=-1)(x)

    # Row r holds the constant r + 1, so its cumsum is (r + 1) * [1, ..., N].
    expected = row_values * torch.arange(1, N + 1, dtype=torch.float32, device=run_device())
    compare_outputs(
        y, expected, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x)
    )


@pytest.mark.parametrize(
    "M, N, dtype, out_dtype",
    [
        # Rows one block cannot stage, with a partial last segment.
        pytest.param(5, 32769, torch.float32, None, id="unstaged-f32", marks=pytest.mark.smoke),
        pytest.param(5, 16385, torch.bfloat16, None, id="unstaged-bf16", marks=pytest.mark.smoke),
        pytest.param(5, 8193, torch.float16, None, id="unstaged-f16", marks=pytest.mark.smoke),
        # Rows of 64 KiB through the float32 output, taken from the whole-row scan.
        pytest.param(
            5,
            16384,
            torch.bfloat16,
            torch.float32,
            id="few-long-rows-widening",
            marks=pytest.mark.smoke,
        ),
        # More segments ahead of the last one than a block has threads.
        pytest.param(5, 2**19 + 7, torch.float32, None, id="many-segments", marks=pytest.mark.full),
    ],
)
def test_cumprod_segments_carry_exceptional_values(
    M: int, N: int, dtype: torch.dtype, out_dtype: torch.dtype | None
) -> None:
    """Segment totals combine into each later segment's product as a serial cumprod would.

    A zero, NaN or inf, and an over- or underflow, reached in one segment reach every later
    element of the row and no earlier one, with the sign of the running product.
    """
    workload = CumprodExceptionalWorkload((M, N), dtype, out_dtype=out_dtype)
    (x,) = workload.gen_inputs()
    y = CumprodFwdOp(dim=-1, dtype=out_dtype)(x)
    compare_outputs(y, workload.ref_program(x), workload.verification(x.to(y.dtype)))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "M, N, dtype",
    [
        pytest.param(64, 512, torch.float16),  # sequential custom_op
        pytest.param(64, 16384, torch.float16),  # parallel custom_op
    ],
)
def test_cumsum_compile_fullgraph_warm_cache(
    M: int, N: int, dtype: torch.dtype, isolated_dynamo
) -> None:
    """``torch.compile(fullgraph=True)`` succeeds on a warm kernel cache: the custom_op
    boundary keeps the raw JIT callables out of the trace. A warm compile is not the cold
    compile-contract evidence ``torch_compile_fullgraph`` needs."""
    op = CumsumFwdOp(dim=-1)
    x = torch.randn(M, N, dtype=dtype, device=run_device())
    op(x)

    compiled = torch.compile(op, fullgraph=True)
    y = compiled(x)

    ref = x.float().cumsum(dim=-1).to(dtype)
    compare_outputs(y, ref, CumulativeWorkload(tuple(x.shape), x.dtype, "cumsum").verification(x))


@pytest.mark.parametrize("op_kind", ["cumsum", "cumprod"])
@pytest.mark.parametrize(
    "shape, in_dtype, out_dtype",
    [
        pytest.param(
            (64, 1024), torch.bfloat16, torch.float32, id="row-scan", marks=pytest.mark.smoke
        ),
        pytest.param(
            (64, 1024), torch.float16, torch.float32, id="row-scan-fp16", marks=pytest.mark.smoke
        ),
        # A 4-byte output halves the whole-row chunk to 32 elements.
        pytest.param(
            (64, 4096),
            torch.bfloat16,
            torch.float32,
            id="row-scan-wide-output",
            marks=pytest.mark.full,
        ),
        # Unaligned columns and a partial row block.
        pytest.param(
            (129, 300), torch.bfloat16, torch.float32, id="tiled-scan", marks=pytest.mark.full
        ),
        pytest.param(
            (64, 8200), torch.bfloat16, torch.float32, id="parallel-scan", marks=pytest.mark.full
        ),
        # The float32 output makes these rows 64 KiB, which the segmented scan takes.
        pytest.param(
            (16, 16384),
            torch.bfloat16,
            torch.float32,
            id="few-long-rows-widening",
            marks=pytest.mark.full,
        ),
    ],
)
def test_dtype_casts_the_input_before_the_scan(
    op_kind: str, shape: tuple, in_dtype: torch.dtype, out_dtype: torch.dtype
) -> None:
    """``dtype`` casts the input before the scan and sets the output dtype, as in torch."""
    workload = CumulativeWorkload(shape, in_dtype, op_kind, out_dtype=out_dtype)
    (x,) = workload.gen_inputs()
    y = _OPS[op_kind](dim=-1, dtype=out_dtype)(x)
    assert y.dtype == out_dtype
    compare_outputs(y, workload.torch_program(x), workload.verification(x.to(out_dtype)))


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.sm90
def test_cumsum_cluster_exchanges_the_totals_of_all_eight_blocks() -> None:
    from tileops._kernels.reduction.cumulative import CumsumClusterScanKernel
    from tileops._utils.utils import get_sm_count
    from tileops.backend import BUILTIN

    if get_sm_count() < 128:
        pytest.skip("The eight-block cluster kernel requires at least 128 SMs")
    rows, width = 24, 65536
    storage = torch.empty(rows * width + 1, dtype=torch.float32, device=run_device())
    op = CumsumFwdOp(dim=-1, target=BUILTIN)
    for offset in (0, 1):
        x = storage[offset : offset + rows * width].view(rows, width)
        x.zero_()
        values = torch.arange(rows * 8, dtype=x.dtype, device=x.device).view(rows, 8) + 1
        x[:, ::8192] = values
        x[:, 8191::8192] = -2 * values
        expected = torch.cumsum(x, dim=-1)
        assert torch.equal(op(x), expected)
        assert [type(k) for k in op.iter_kernels()] == [CumsumClusterScanKernel]
