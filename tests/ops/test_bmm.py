import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import BmmFp8Op, BmmFwdOp
from workloads.bmm import BmmFp8Workload, BmmWorkload


class BmmTest(BmmWorkload, TestBase):
    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a, b)


class BmmFp8Test(BmmFp8Workload, TestBase):
    def ref_program(self, *inputs: torch.Tensor) -> torch.Tensor:
        a, b, scale_a, scale_b = inputs
        # per_tensor: 0-D fp32 scalars (matching flashinfer.bmm_fp8).
        assert scale_a.dim() == 0 and scale_b.dim() == 0, (
            f"BmmFp8Test only supports per-tensor scales, got "
            f"{tuple(scale_a.shape)} / {tuple(scale_b.shape)}"
        )
        # b is [B, K, N] (torch.bmm layout, matching BmmFp8Op).
        a_f = a.float() * scale_a
        b_f = b.float() * scale_b
        out = torch.bmm(a_f, b_f)
        return out.to(self.out_dtype)


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


@pytest.mark.smoke
def test_bmm_b_dtype_change_after_valid_call_raises() -> None:
    op = BmmFwdOp()
    a = torch.randn(4, 16, 16, device="cuda", dtype=torch.float16)
    b_ok = torch.randn(4, 16, 16, device="cuda", dtype=torch.float16)
    op(a, b_ok)  # populate the fast path
    b_bad = torch.randn(4, 16, 16, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        op(a, b_bad)


@pytest.mark.smoke
def test_bmm_k_not_multiple_of_16_raises() -> None:
    """K must be a multiple of 16 (manifest shape_rules + op precondition)."""
    op = BmmFwdOp()
    a = torch.randn(4, 16, 24, device="cuda", dtype=torch.float16)
    b = torch.randn(4, 24, 16, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="multiple of 16"):
        op(a, b)


class BmmFp8Fixture(FixtureBase):
    # Only *supported* dtype combinations belong in the main fixture; a
    # rejected dtype (e5m2) is exercised by its own dedicated negative
    # test below so that each parametrised case has a single purpose.
    PARAMS = [
        ("batch, m, n, k, dtype, out_dtype", [
            pytest.param(
                4, 128, 128, 128, torch.float8_e4m3fn,
                torch.bfloat16,
                marks=pytest.mark.smoke,
                id="smoke-fp8-b4-per-tensor",
            ),
            pytest.param(
                8, 128, 256, 128, torch.float8_e4m3fn,
                torch.float16,
                marks=pytest.mark.full,
                id="full-fp8-b8-per-tensor",
            ),
            pytest.param(
                16, 128, 128, 2048, torch.float8_e4m3fn,
                torch.bfloat16,
                marks=pytest.mark.full,
                id="full-fp8-b16-mha-pv-per-tensor",
            ),
        ]),
    ]


@BmmFp8Fixture
def test_bmm_fp8(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> None:
    test = BmmFp8Test(batch, m, n, k, dtype, out_dtype=out_dtype)
    op = BmmFp8Op(out_dtype=out_dtype)
    inputs = test.gen_inputs()
    test.check(op, *inputs, atol=2e-2, rtol=2e-2)


@pytest.mark.smoke
def test_bmm_fp8_rejects_e5m2() -> None:
    """BmmFp8Op advertises fp8_e4m3fn only; e5m2 inputs must be rejected.

    Kept separate from the main fixture so that ``test_bmm_fp8`` carries a
    single purpose (correctness on supported dtypes) and this test carries
    the other (dtype-guard on unsupported dtypes).
    """
    test = BmmFp8Test(4, 128, 128, 128, torch.float8_e5m2)
    op = BmmFp8Op(out_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="only supports torch.float8_e4m3fn"):
        op(*test.gen_inputs())


@pytest.mark.smoke
def test_bmm_fp8_rejects_unsupported_scale_grids() -> None:
    """Per-tensor is 0-D; every other rank must be rejected."""
    batch, m, n, k = 2, 128, 256, 256
    test = BmmFp8Test(batch, m, n, k, torch.float8_e4m3fn)
    a, b, _, _ = test.gen_inputs()
    op = BmmFp8Op()

    scale_k = k // 128
    # Legacy 2-D per-row block128 grid -- rejected now that per_tensor is
    # the only supported scale layout (0-D scalars).
    with pytest.raises(ValueError, match="supports scale shapes"):
        op(
            a,
            b,
            torch.ones((m, scale_k), device="cuda", dtype=torch.float32),
            torch.ones((n, scale_k), device="cuda", dtype=torch.float32),
        )

    # 2-D 1x1 -- also rejected (per_tensor requires rank-0, not rank-2).
    with pytest.raises(ValueError, match="supports scale shapes"):
        op(
            a,
            b,
            torch.ones((1, 1), device="cuda", dtype=torch.float32),
            torch.ones((1, 1), device="cuda", dtype=torch.float32),
        )

    # Legacy per-batch ``(B, 1, 1)`` shape -- rejected now that per-tensor
    # is 0-D (matches flashinfer.bmm_fp8's global A_scale/B_scale).
    with pytest.raises(ValueError, match="supports scale shapes"):
        op(
            a,
            b,
            torch.ones((batch, 1, 1), device="cuda", dtype=torch.float32),
            torch.ones((batch, 1, 1), device="cuda", dtype=torch.float32),
        )


@pytest.mark.smoke
def test_bmm_fp8_revalidates_cached_signature_dtypes() -> None:
    test = BmmFp8Test(
        2,
        128,
        128,
        128,
        torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
    )
    a, b, scale_a, scale_b = test.gen_inputs()
    op = BmmFp8Op(out_dtype=torch.bfloat16)
    op(a, b, scale_a, scale_b)

    with pytest.raises(ValueError, match="expects b dtype"):
        op(a, b.to(torch.float8_e5m2), scale_a, scale_b)

    with pytest.raises(ValueError, match="scale_a and scale_b"):
        op(a, b, scale_a.to(torch.float16), scale_b)


@pytest.mark.smoke
def test_bmm_fp8_batch_mismatch_raises() -> None:
    op = BmmFp8Op()
    a = torch.randn(4, 128, 128, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(5, 128, 128, device="cuda").to(torch.float8_e4m3fn)
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="batch dim mismatch"):
        op(a, b, scale_a, scale_b)


@pytest.mark.smoke
def test_bmm_fp8_contraction_mismatch_raises() -> None:
    op = BmmFp8Op()
    a = torch.randn(4, 128, 128, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(4, 64, 96, device="cuda").to(torch.float8_e4m3fn)
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="contraction dim mismatch"):
        op(a, b, scale_a, scale_b)


@pytest.mark.smoke
def test_bmm_fp8_rank_mismatch_raises() -> None:
    op = BmmFp8Op()
    a = torch.randn(128, 128, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(4, 128, 128, device="cuda").to(torch.float8_e4m3fn)
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="strict 3D"):
        op(a, b, scale_a, scale_b)


@pytest.mark.smoke
def test_bmm_fp8_k_not_multiple_of_32_raises() -> None:
    op = BmmFp8Op()
    a = torch.randn(4, 128, 48, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(4, 48, 128, device="cuda").to(torch.float8_e4m3fn)
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="multiple of 32"):
        op(a, b, scale_a, scale_b)


@pytest.mark.smoke
def test_bmm_fp8_scale_dtype_change_after_valid_call_raises() -> None:
    op = BmmFp8Op()
    a = torch.randn(2, 128, 128, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(2, 128, 128, device="cuda").to(torch.float8_e4m3fn)
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    op(a, b, scale_a, scale_b)  # populate the fast path
    with pytest.raises(ValueError, match="scale_a and scale_b"):
        op(a, b, scale_a.to(torch.float16), scale_b)


@pytest.mark.smoke
def test_bmm_fp8_accepts_nk_layout_when_k_ne_n() -> None:
    batch, m, n, k = 4, 128, 256, 128
    test = BmmFp8Test(batch, m, n, k, torch.float8_e4m3fn)
    a, b_kn, scale_a, scale_b = test.gen_inputs()  # b_kn is [B, K, N]
    # NK-layout: [B, N, K], K-innermost, contiguous.  ``transpose+
    # contiguous`` materialises the physical NK buffer so its shape[2]
    # unambiguously carries K.
    b_nk = b_kn.transpose(-2, -1).contiguous()
    assert b_nk.shape == (batch, n, k)
    op = BmmFp8Op(out_dtype=torch.bfloat16)
    out_kn = op(a, b_kn, scale_a, scale_b).clone()
    # New op instance to avoid the KN cached signature masking layout
    # dispatch bugs.
    op_nk = BmmFp8Op(out_dtype=torch.bfloat16)
    out_nk = op_nk(a, b_nk, scale_a, scale_b)
    # Numerically identical: same kernel, same buffer bits, just no
    # internal DtoD copy on the NK path.
    torch.testing.assert_close(out_kn, out_nk, atol=0.0, rtol=0.0)


@pytest.mark.smoke
def test_bmm_fp8_nk_view_when_k_eq_n() -> None:
    """When K == N, shape is ambiguous; a non-contiguous NK view forces
    the fast path (``stride(-2) == 1`` tie-breaker).
    """
    batch, m, n, k = 4, 128, 128, 128  # K == N
    test = BmmFp8Test(batch, m, n, k, torch.float8_e4m3fn)
    a, b_kn, scale_a, scale_b = test.gen_inputs()  # contiguous [B, K, N]
    # ``transpose`` alone (no contiguous) yields the NK view; shape becomes
    # [B, N, K] but stride is (K*N, 1, N) so stride(-2) == 1, marking the
    # buffer as K-innermost so the op treats it as NK-input.
    b_nk_view = b_kn.transpose(-2, -1)
    assert b_nk_view.shape == (batch, n, k)
    assert b_nk_view.stride(-2) == 1
    op_kn = BmmFp8Op(out_dtype=torch.bfloat16)
    op_nk = BmmFp8Op(out_dtype=torch.bfloat16)
    out_kn = op_kn(a, b_kn, scale_a, scale_b).clone()
    out_nk = op_nk(a, b_nk_view, scale_a, scale_b)
    torch.testing.assert_close(out_kn, out_nk, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
