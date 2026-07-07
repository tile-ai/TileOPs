import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GemmFp8Op, GemmOp
from tileops.utils import expand_fp8_scale
from workloads.gemm import GemmFp8Workload, GemmWorkload


class GemmTest(GemmWorkload, TestBase):
    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.trans_a:
            a = a.T
        if self.trans_b:
            b = b.T
        return torch.matmul(a, b)


class GemmFp8Test(GemmFp8Workload, TestBase):
    def _expand_scale(self, scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        return expand_fp8_scale(scale, rows, cols)

    def ref_program(self, *inputs: torch.Tensor) -> torch.Tensor:
        a, b, scale_a, scale_b = inputs[:4]
        bias = inputs[4] if len(inputs) == 5 else None
        a_f = a.float() * self._expand_scale(scale_a, self.m, self.k)
        b_f = b.float() * self._expand_scale(scale_b, self.n, self.k)
        out = torch.matmul(a_f, b_f.T)
        if bias is not None:
            out = out + bias.float()
        return out.to(self.out_dtype)


class GemmFixture(FixtureBase):
    PARAMS = [
        ("m, n, k, dtype, trans_a, trans_b, tune", [
            pytest.param(
                1024, 1024, 1024, torch.float16, False, False, False,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
                id="smoke-fp16-square",
            ),
            pytest.param(
                1024, 1024, 1024, torch.bfloat16, False, False, False,
                marks=pytest.mark.smoke,
                id="smoke-bf16-square",
            ),
            pytest.param(
                1, 1024, 1024, torch.float16, False, True, False,
                marks=pytest.mark.full,
                id="full-fp16-trans-b-small-m",
            ),
            pytest.param(
                128, 2112, 4096, torch.float16, False, True, False,
                marks=pytest.mark.full,
                id="full-fp16-nt-dense-ws",
            ),
            pytest.param(
                1, 7168, 16384, torch.float16, False, True, True,
                marks=pytest.mark.full,
                id="full-fp16-tuned-wide",
            ),
            pytest.param(
                1, 18432, 7168, torch.float16, False, True, False,
                marks=pytest.mark.full,
                id="full-fp16-tuned-wide-alt",
            ),
            pytest.param(
                1024, 1, 1024, torch.float16, False, False, False,
                marks=pytest.mark.full,
                id="full-fp16-thin-n",
            ),
            pytest.param(
                7168, 1, 16384, torch.float16, False, False, False,
                marks=pytest.mark.full,
                id="full-fp16-tuned-thin-n",
            ),
            pytest.param(
                18432, 1, 7168, torch.float16, False, False, False,
                marks=pytest.mark.full,
                id="full-fp16-tuned-thin-n-alt",
            ),
            pytest.param(
                1, 1024, 1024, torch.bfloat16, False, True, False,
                marks=pytest.mark.full,
                id="full-bf16-trans-b-small-m",
            ),
            pytest.param(
                1, 7168, 16384, torch.bfloat16, False, True, False,
                marks=pytest.mark.full,
                id="full-bf16-tuned-wide",
            ),
            pytest.param(
                1, 18432, 7168, torch.bfloat16, False, True, False,
                marks=pytest.mark.full,
                id="full-bf16-tuned-wide-alt",
            ),
            pytest.param(
                1024, 1, 1024, torch.bfloat16, False, False, False,
                marks=pytest.mark.full,
                id="full-bf16-thin-n",
            ),
            pytest.param(
                7168, 1, 16384, torch.bfloat16, False, False, False,
                marks=pytest.mark.full,
                id="full-bf16-tuned-thin-n",
            ),
            pytest.param(
                18432, 1, 7168, torch.bfloat16, False, False, False,
                marks=pytest.mark.full,
                id="full-bf16-tuned-thin-n-alt",
            ),
        ]),
    ]


class GemvBoundaryFixture(FixtureBase):
    """GEMV cases with non-aligned n/k to exercise partial-tile paths."""
    PARAMS = [
        ("n, k, dtype, tune", [
            # lhs_row: m=1, trans_b=True — non-aligned n
            pytest.param(3000, 1024, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(3000, 1024, torch.bfloat16, False, marks=pytest.mark.smoke),
            # lhs_row: non-aligned k
            pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
            # rhs_col: n=1 — non-aligned m (mapped to gemv n param)
            pytest.param(3001, 1024, torch.float16, False, marks=pytest.mark.full),
        ]),
    ]


class GemmFp8Fixture(FixtureBase):
    PARAMS = [
        ("m, n, k, dtype, scale_mode, out_dtype, bias", [
            pytest.param(
                128, 128, 128, torch.float8_e4m3fn, "per_tensor", torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-fp8-e4m3-per-tensor",
            ),
            pytest.param(
                128, 256, 256, torch.float8_e4m3fn, "block128", torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-fp8-e4m3-block128",
            ),
            pytest.param(
                128, 128, 128, torch.float8_e5m2, "per_tensor", torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-fp8-e5m2-per-tensor",
            ),
            pytest.param(
                4096, 256, 256, torch.float8_e4m3fn, "block128", torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-fp8-e4m3-block128-large-m",
            ),
            pytest.param(
                8, 256, 128, torch.float8_e4m3fn, "per_tensor", torch.float16, True,
                marks=pytest.mark.full,
                id="full-fp8-e4m3-per-tensor-small-m-bias",
            ),
            pytest.param(
                1, 256, 128, torch.float8_e4m3fn, "per_tensor", torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-fp8-e4m3-per-tensor-gemv",
            ),
        ]),
    ]


@GemmFixture
def test_gemm(m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool, trans_b: bool,
              tune: bool) -> None:
    test = GemmTest(m, n, k, dtype, trans_a, trans_b)
    op = GemmOp(trans_a=trans_a, trans_b=trans_b, tune=tune)
    if dtype == torch.float16:
        tolerances = {"atol": 1e-3, "rtol": 1e-3}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, *test.gen_inputs(), **tolerances)

@GemmFp8Fixture
def test_gemm_fp8(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    scale_mode: str,
    out_dtype: torch.dtype,
    bias: bool,
) -> None:
    test = GemmFp8Test(m, n, k, dtype, scale_mode, out_dtype=out_dtype, bias=bias)
    op = GemmFp8Op(out_dtype=out_dtype)
    inputs = test.gen_inputs()
    if dtype != torch.float8_e4m3fn:
        with pytest.raises(ValueError, match="only supports torch.float8_e4m3fn"):
            op(*inputs)
        return
    test.check(op, *inputs, atol=2e-2, rtol=2e-2)


@GemvBoundaryFixture
def test_gemv_boundary_lhs_row(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    """GEMV lhs_row path (m=1, trans_b=True) with non-aligned n or k."""
    test = GemmTest(1, n, k, dtype, trans_a=False, trans_b=True)
    op = GemmOp(trans_a=False, trans_b=True, tune=tune)
    tolerances = {"atol": 1e-2, "rtol": 1e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@GemvBoundaryFixture
def test_gemv_boundary_rhs_col(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    """GEMV rhs_col path (n=1, no transpose) with non-aligned m or k."""
    m = n  # reuse fixture's n as the non-aligned m dimension
    test = GemmTest(m, 1, k, dtype, trans_a=False, trans_b=False)
    op = GemmOp(trans_a=False, trans_b=False, tune=tune)
    tolerances = {"atol": 1e-2, "rtol": 1e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
