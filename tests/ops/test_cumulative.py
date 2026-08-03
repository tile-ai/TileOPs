"""Correctness tests for cumulative ops (cumsum, cumprod).

Covers: CumsumFwdOp, CumprodFwdOp.
Each op computes an inclusive prefix scan along dim=-1 and supports 1D-4D input.
Output has the same shape as input.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from workloads.reduction import CumulativeWorkload

# Fixtures


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


# TestBase helpers


class CumulativeTest(CumulativeWorkload, TestBase):
    """Parameterized test helper for cumulative ops."""

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        if self.op_kind == "cumsum":
            return x_f32.cumsum(dim=-1).to(x.dtype)
        elif self.op_kind == "cumprod":
            return x_f32.cumprod(dim=-1).to(x.dtype)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


# Helper to get tolerances


def _tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    return {"atol": 1e-2, "rtol": 1e-2}


def _cumprod_tol(dtype: torch.dtype) -> dict:
    """Tolerances for cumprod tests (more numerically sensitive)."""
    if dtype == torch.float32:
        return {"atol": 1e-3, "rtol": 1e-3}
    return {"atol": 5e-2, "rtol": 5e-2}


# CumsumFwdOp tests


@CumulativeBasicFixture
def test_cumsum_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    test = CumulativeTest((m, n), dtype, "cumsum")
    op = CumsumFwdOp()
    test.check(op, *test.gen_inputs(), **_tol(dtype))


@CumulativeNonContigFixture
def test_cumsum_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]
    op = CumsumFwdOp()
    ref = x.contiguous().float().cumsum(dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@Cumulative3DFixture
def test_cumsum_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = CumsumFwdOp()
    ref = x.float().cumsum(dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"3D max err: {(y - ref).abs().max()}"


@Cumulative4DFixture
def test_cumsum_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = CumsumFwdOp()
    ref = x.float().cumsum(dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"4D max err: {(y - ref).abs().max()}"


@Cumulative1DFixture
def test_cumsum_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    x = torch.randn(n, dtype=dtype, device="cuda")
    op = CumsumFwdOp()
    ref = x.float().cumsum(dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"1D cumsum max err: {(y - ref).abs().max()}"


@pytest.mark.smoke
def test_cumsum_dynamic_shape_kernel_cache() -> None:
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    op = CumsumFwdOp()
    x1 = torch.randn(4, 8, dtype=torch.float16, device="cuda")
    x2 = torch.randn(5, 8, dtype=torch.float16, device="cuda")

    op(x1)
    assert len(op._kernel_cache) == 1
    op(x1)
    assert len(op._kernel_cache) == 1
    op(x2)
    assert len(op._kernel_cache) == 2


# CumprodFwdOp tests


@CumulativeBasicFixture
def test_cumprod_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumprodFwdOp

    test = CumulativeTest((m, n), dtype, "cumprod", use_small_range=True)
    op = CumprodFwdOp()
    test.check(op, *test.gen_inputs(), **_cumprod_tol(dtype))


@CumulativeNonContigFixture
def test_cumprod_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumprodFwdOp

    x_full = torch.rand(m, n * 2, dtype=dtype, device="cuda") * 0.01 + 0.99
    x = x_full[:, :n]
    op = CumprodFwdOp()
    ref = x.contiguous().float().cumprod(dim=-1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@Cumulative3DFixture
def test_cumprod_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumprodFwdOp

    x = torch.rand(batch, seq, hidden, dtype=dtype, device="cuda") * 0.01 + 0.99
    op = CumprodFwdOp()
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), f"3D cumprod max err: {(y - ref).abs().max()}"


@Cumulative4DFixture
def test_cumprod_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumprodFwdOp

    x = torch.rand(b0, b1, b2, n, dtype=dtype, device="cuda") * 0.01 + 0.99
    op = CumprodFwdOp()
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), f"4D cumprod max err: {(y - ref).abs().max()}"


@Cumulative1DFixture
def test_cumprod_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumulative import CumprodFwdOp

    x = torch.rand(n, dtype=dtype, device="cuda") * 0.01 + 0.99
    op = CumprodFwdOp()
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), f"1D cumprod max err: {(y - ref).abs().max()}"


class CumulativeDimAxis1Fixture(FixtureBase):
    PARAMS = [
        ("batch, hidden, seq, dtype", [
            pytest.param(2, 512, 256, torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 512, 256, torch.bfloat16, marks=pytest.mark.smoke),
        ]),
    ]


@CumulativeDimAxis1Fixture
def test_cumsum_dim_axis1(
    batch: int, hidden: int, seq: int, dtype: torch.dtype
) -> None:
    """Cumsum along dim=1 (3D) — exercises movedim choreography in `_run`."""
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    x = torch.randn(batch, hidden, seq, dtype=dtype, device="cuda")
    op = CumsumFwdOp(dim=1)
    ref = x.float().cumsum(dim=1).to(dtype)
    y = op(x)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    assert torch.allclose(y, ref, atol=atol, rtol=atol), \
        f"cumsum dim=1 max err: {(y - ref).abs().max()}"


@CumulativeDimAxis1Fixture
def test_cumprod_dim_axis1(
    batch: int, hidden: int, seq: int, dtype: torch.dtype
) -> None:
    """Cumprod along dim=1 (3D) — exercises movedim choreography in `_run`."""
    from tileops.ops.reduction.cumulative import CumprodFwdOp

    # Values close to 1 to avoid over/underflow in cumprod over hidden dim.
    x = torch.rand(batch, hidden, seq, dtype=dtype, device="cuda") * 0.01 + 0.99
    op = CumprodFwdOp(dim=1)
    ref = x.float().cumprod(dim=1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), \
        f"cumprod dim=1 max err: {(y - ref).abs().max()}"


@pytest.mark.smoke
@pytest.mark.parametrize(
    "M, N, dtype, parallel",
    [
        (64, 16384, torch.float32, True),      # block_n=128
        (64, 32768, torch.bfloat16, True),     # block_n=256
        (32, 16384, torch.float16, True),      # fp16 intermediate
        (64, 8200, torch.float32, True),       # N % block_n != 0: masked tail
        (64, 8192, torch.bfloat16, False),     # N boundary
        (128, 16384, torch.bfloat16, False),   # M boundary
    ],
)
def test_cumsum_backend_dispatch(M: int, N: int, dtype: torch.dtype, parallel: bool) -> None:
    """Each shape takes the expected backend and matches torch.cumsum."""
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    x = torch.randn(M, N, dtype=dtype, device="cuda")
    op = CumsumFwdOp(dim=-1)
    y = op(x)

    ref = x.float().cumsum(dim=-1).to(dtype)
    assert torch.allclose(y, ref, **_tol(dtype)), \
        f"({M}, {N}) {dtype}: max_diff={torch.abs(y - ref).max()}"

    kernel = op._get_kernel(M, N, dtype, x.device.index)
    assert kernel.use_parallel == parallel, f"({M}, {N}): unexpected backend"
    if parallel:
        assert kernel.config["block_n"] == (256 if N > 16384 else 128)


@pytest.mark.smoke
@pytest.mark.parametrize("M, N", [(1, 32768), (127, 16384)])
def test_cumsum_parallel_scan_row_ownership(M: int, N: int) -> None:
    """Carry propagation stays per-row across tiles and partial row blocks."""
    from tileops.ops.reduction.cumulative import CumsumFwdOp

    row_values = torch.arange(1, M + 1, dtype=torch.float32, device="cuda").unsqueeze(1)
    x = row_values.expand(-1, N).contiguous()

    y = CumsumFwdOp(dim=-1)(x)

    # Row r holds the constant r + 1, so its cumsum is (r + 1) * [1, ..., N].
    expected = row_values * torch.arange(1, N + 1, dtype=torch.float32, device="cuda")
    assert torch.allclose(y, expected, atol=1e-3, rtol=1e-3), \
        f"({M}, {N}): max_diff={torch.abs(y - expected).max()}"


@pytest.mark.smoke
@pytest.mark.parametrize(
    "M, N, dtype",
    [
        pytest.param(64, 512, torch.float16),    # sequential custom_op
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
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    op(x)

    compiled = torch.compile(op, fullgraph=True)
    y = compiled(x)

    ref = x.float().cumsum(dim=-1).to(dtype)
    assert torch.allclose(y, ref, **_tol(dtype)), \
        f"Compiled output mismatch for shape ({M},{N}): max_diff={torch.abs(y - ref).max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
