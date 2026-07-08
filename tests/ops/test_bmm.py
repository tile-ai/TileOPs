import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import BmmFwdOp
from workloads.bmm import BmmWorkload


class BmmTest(BmmWorkload, TestBase):
    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a, b)


class BmmFixture(FixtureBase):
    PARAMS = [
        ("batch, m, n, k, dtype, tune", [
            pytest.param(
                4, 128, 128, 128, torch.float16, False,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
                id="smoke-fp16-b4-128",
            ),
            pytest.param(
                4, 128, 128, 128, torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-bf16-b4-128",
            ),
            pytest.param(
                8, 512, 512, 512, torch.float16, False,
                marks=pytest.mark.full,
                id="full-fp16-b8-512",
            ),
            pytest.param(
                8, 512, 512, 512, torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-bf16-b8-512",
            ),
            pytest.param(
                16, 256, 256, 256, torch.float16, False,
                marks=pytest.mark.full,
                id="full-fp16-b16-256",
            ),
            pytest.param(
                1, 1024, 1024, 1024, torch.float16, False,
                marks=pytest.mark.full,
                id="full-fp16-b1-1k",
            ),
            pytest.param(
                32, 128, 512, 128, torch.float16, False,
                marks=pytest.mark.full,
                id="full-fp16-b32-mha-qk",
            ),
            pytest.param(
                8, 128, 128, 2048, torch.float16, False,
                marks=pytest.mark.full,
                id="full-fp16-b8-mha-pv",
            ),
            pytest.param(
                8, 128, 128, 2048, torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-bf16-b8-mha-pv",
            ),
            pytest.param(
                32, 256, 256, 1024, torch.bfloat16, False,
                marks=pytest.mark.full,
                id="full-bf16-b32-moe",
            ),
            pytest.param(
                4, 200, 300, 128, torch.float16, False,
                marks=pytest.mark.full,
                id="full-fp16-b4-mn-nonaligned",
            ),
        ]),
    ]


@BmmFixture
def test_bmm(batch: int, m: int, n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    test = BmmTest(batch, m, n, k, dtype)
    op = BmmFwdOp(tune=tune)
    if dtype == torch.float16:
        tolerances = {"atol": 1e-3, "rtol": 1e-3}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@pytest.mark.smoke
def test_bmm_batch_mismatch_raises() -> None:
    op = BmmFwdOp()
    a = torch.randn(4, 16, 16, device="cuda", dtype=torch.float16)
    b = torch.randn(5, 16, 16, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="batch dim mismatch"):
        op(a, b)


@pytest.mark.smoke
def test_bmm_contraction_mismatch_raises() -> None:
    op = BmmFwdOp()
    a = torch.randn(4, 16, 32, device="cuda", dtype=torch.float16)
    b = torch.randn(4, 16, 16, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="contraction dim mismatch"):
        op(a, b)


@pytest.mark.smoke
def test_bmm_rank_mismatch_raises() -> None:
    op = BmmFwdOp()
    a = torch.randn(16, 16, device="cuda", dtype=torch.float16)
    b = torch.randn(4, 16, 16, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="strict 3D"):
        op(a, b)


@pytest.mark.smoke
def test_bmm_dtype_mismatch_raises() -> None:
    op = BmmFwdOp()
    a = torch.randn(4, 16, 16, device="cuda", dtype=torch.float16)
    b = torch.randn(4, 16, 16, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        op(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
