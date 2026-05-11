"""Tests for fp8 dtype rejection in elementwise kernels.

After narrowing `_FLOAT_DTYPES` to drop fp8, the elementwise / rope / dropout
kernels no longer advertise fp8 in `SUPPORTED_DTYPES`. The tests here are
sentinel checks that float and bitwise kernels correctly reject fp8 inputs
at the kernel layer (exercising `SUPPORTED_DTYPES`, not `Op._validate_dtypes`).
"""

import pytest
import torch

_N = 1024 * 16


@pytest.mark.smoke
def test_float_unary_kernel_rejects_fp8():
    """ReluFwdKernel raises ValueError for fp8 (not in narrowed _FLOAT_DTYPES)."""
    from tileops.kernels.elementwise import ReluFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        ReluFwdKernel(N_total=_N, dtype=torch.float8_e4m3fn)


@pytest.mark.smoke
def test_bitwise_kernel_rejects_fp8():
    """BitwiseNotFwdKernel raises ValueError for fp8 (not in _BITWISE_DTYPES)."""
    from tileops.kernels.elementwise import BitwiseNotFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        BitwiseNotFwdKernel(N_total=_N, dtype=torch.float8_e4m3fn)


@pytest.mark.smoke
def test_binary_bitwise_kernel_rejects_fp8():
    """BitwiseAndFwdKernel raises ValueError for fp8 (not in _BITWISE_DTYPES)."""
    from tileops.kernels.elementwise import BitwiseAndFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        BitwiseAndFwdKernel(
            N_total=_N, dtype=torch.float8_e4m3fn,
            coalesced_shape=(_N,), a_strides=(1,), b_strides=(1,),
            a_numel=_N, b_numel=_N,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
