import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from workloads.rms_norm import RMSNormTest as _RMSNormTestWorkload


class RMSNormTest(_RMSNormTestWorkload, TestBase):
    def ref_program(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return ((x_f32 / rms) * weight.float()).to(x.dtype)


class RMSNormFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype, tune", [
            # Standard aligned shapes (AC required)
            pytest.param(1024, 4096, torch.float16, False, marks=[pytest.mark.smoke, pytest.mark.packaging]),
            pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
            pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(8192, 8192, torch.float16, False, marks=pytest.mark.full),
            pytest.param(8192, 8192, torch.bfloat16, False, marks=pytest.mark.full),
            # Non-aligned N (AC required)
            pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2048, 5120, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2048, 5120, torch.bfloat16, False, marks=pytest.mark.full),
            # Tail-M: M not divisible by block_m (proves T.copy partial block safety)
            pytest.param(1025, 4096, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1025, 4096, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@RMSNormFixture
def test_rms_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = RMSNormTest(m, n, dtype)
    op = RMSNormFwdOp(normalized_shape=(n,), dtype=dtype)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    rtol = atol
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class RMSNormNonContigFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
        ]),
    ]


@RMSNormNonContigFixture
def test_rms_norm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]  # non-contiguous slice
    weight = torch.randn(n, dtype=dtype, device="cuda")

    op = RMSNormFwdOp(normalized_shape=(n,), dtype=dtype)

    # Reference on contiguous copy
    eps = 1e-6
    x_ref = x.contiguous()
    x_f32 = x_ref.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = ((x_f32 / rms) * weight.float()).to(dtype)

    y = op(x, weight)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    assert torch.allclose(y, y_ref, atol=atol, rtol=atol), \
        f"Non-contiguous test failed, max err: {(y - y_ref).abs().max()}"


class RMSNorm3DFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype", [
            pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
        ]),
    ]


@RMSNorm3DFixture
def test_rms_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    weight = torch.randn(hidden, dtype=dtype, device="cuda")

    op = RMSNormFwdOp(normalized_shape=(hidden,), dtype=dtype)

    # Reference
    eps = 1e-6
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = ((x_f32 / rms) * weight.float()).to(dtype)

    y = op(x, weight)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    assert torch.allclose(y, y_ref, atol=atol, rtol=atol), \
        f"3D test failed, max err: {(y - y_ref).abs().max()}"




class RMSNormStridedFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, hidden, hidden_full, dtype", [
            # Sliced trailing axis: leading dims contiguous, inner stride 1,
            # but stride_M (== hidden_full) > N. Exercises the stride-aware
            # zero-copy kernel path.
            pytest.param(2, 4, 4096, 8192, torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 4, 4096, 8192, torch.bfloat16, marks=pytest.mark.smoke),
        ]),
    ]


@RMSNormStridedFixture
def test_rms_norm_strided_zero_copy(
    batch: int, heads: int, hidden: int, hidden_full: int, dtype: torch.dtype,
) -> None:
    """3-D input sliced on the trailing axis; the op feeds a strided view to
    the kernel without an intervening ``contiguous()`` copy."""
    x_full = torch.randn(batch, heads, hidden_full, dtype=dtype, device="cuda")
    x = x_full[..., :hidden]
    weight = torch.randn(hidden, dtype=dtype, device="cuda")

    assert not x.is_contiguous()
    assert x.stride(-1) == 1
    assert x.stride(-2) == hidden_full

    op = RMSNormFwdOp(normalized_shape=(hidden,), dtype=dtype)

    eps = 1e-6
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = ((x_f32 / rms) * weight.float()).to(dtype)

    y = op(x, weight)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    assert torch.allclose(y, y_ref, atol=atol, rtol=atol), (
        f"strided test failed, max err: {(y - y_ref).abs().max()}"
    )
    # The kernel cache key encodes stride_M; a non-default stride proves the
    # strided path was taken.
    cache_keys = list(op._kernel_cache.keys())
    assert any(k[1] == hidden_full for k in cache_keys), (
        f"expected stride_M={hidden_full} kernel cache entry, got {cache_keys}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
