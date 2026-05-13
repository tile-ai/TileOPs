"""Tests for fp8 dtype rejection in independent elementwise kernels.

After narrowing `_FLOAT_DTYPES` to drop fp8, the independent elementwise
ops no longer advertise fp8 in their kernel SUPPORTED_DTYPES. The only
remaining contract worth asserting at this layer is that ops continue to
reject fp8 dtypes (negative path) and accept the manifest-declared
non-fp8 dtypes (positive path).
"""

import pytest
import torch


@pytest.mark.smoke
@pytest.mark.parametrize(
    "bad_dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2],
)
def test_where_rejects_fp8_dtype(bad_dtype: torch.dtype) -> None:
    """WhereFwdOp must reject fp8 dtypes at construction (manifest contract)."""
    from tileops.ops.elementwise import WhereFwdOp

    shape = (4, 8)
    with pytest.raises((ValueError, TypeError)):
        WhereFwdOp(
            condition=shape, input=shape, other=shape, dtype=bad_dtype,
        )


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_where_accepts_manifest_dtypes(dtype: torch.dtype) -> None:
    """WhereFwdOp constructs and runs for every manifest-declared dtype."""
    from tileops.ops.elementwise import WhereFwdOp

    shape = (4, 8)
    cond = torch.randint(0, 2, shape, device="cuda").bool()
    inp = torch.randn(shape, device="cuda", dtype=dtype)
    other = torch.randn(shape, device="cuda", dtype=dtype)
    op = WhereFwdOp(condition=shape, input=shape, other=shape, dtype=dtype)
    out = op(cond, inp, other)
    ref = torch.where(cond, inp, other)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
