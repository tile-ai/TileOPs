import math

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import FFTC2CFwdOp
from workloads.fft import FFTWorkload


class FFTTest(FFTWorkload, TestBase):
    pass


# One smallest or boundary case per execution structure. The selection test in
# test_fft_c2c.py covers every plan-table entry without allocating its tensors.
_CORRECTNESS_CASES = (
    pytest.param(2, torch.complex64, (), marks=pytest.mark.smoke, id="tiny-lower"),
    pytest.param(64, torch.complex64, (), marks=pytest.mark.smoke, id="warp8-lower"),
    pytest.param(4096, torch.complex128, (), marks=pytest.mark.smoke, id="three-pass-upper-c128"),
    pytest.param(32768, torch.complex64, (3,), marks=pytest.mark.smoke, id="two-factor-lower"),
    pytest.param(32, torch.complex128, (3,), marks=pytest.mark.full, id="tiny-upper-c128"),
    pytest.param(128, torch.complex128, (3,), marks=pytest.mark.full, id="warp8-upper-c128"),
    pytest.param(256, torch.complex64, (2, 4), marks=pytest.mark.full, id="packed-16x16"),
    pytest.param(512, torch.complex128, (3,), marks=pytest.mark.full, id="packed-8x8x8-c128"),
    pytest.param(1024, torch.complex64, (2, 4), marks=pytest.mark.full, id="three-pass-lower"),
    pytest.param(8192, torch.complex128, (3,), marks=pytest.mark.full, id="four-pass-lower"),
    pytest.param(16384, torch.complex64, (), marks=pytest.mark.full, id="one-cta-upper"),
    pytest.param(
        16384,
        torch.complex128,
        (),
        marks=pytest.mark.full,
        id="decomposed-dtype-boundary",
    ),
    pytest.param(1 << 25, torch.complex64, (), marks=pytest.mark.full, id="three-factor-lower"),
)


class FFTFixture(FixtureBase):
    PARAMS = [("n, dtype, batch_shape", _CORRECTNESS_CASES)]


@FFTFixture
def test_fft_c2c(n: int, dtype: torch.dtype, batch_shape: tuple) -> None:
    batch = math.prod(batch_shape) if batch_shape else 1
    # Allow for input, scratch, output, reference, and allocator overlap.
    need = 5 * batch * n * (8 if dtype == torch.complex64 else 16)
    free, _total = torch.cuda.mem_get_info()
    if need > free:
        pytest.skip(f"n={n} {dtype} needs {need >> 20} MiB free, device has {free >> 20} MiB")
    test = FFTTest(n, dtype, batch_shape=batch_shape)
    op = FFTC2CFwdOp()
    if dtype == torch.complex64:
        tolerances = {"atol": 1e-4, "rtol": 1e-4}
    else:
        tolerances = {"atol": 1e-8, "rtol": 1e-8}
    if n >= 1 << 15:
        # FFT output magnitude, and therefore absolute error, scales with sqrt(n).
        scale = math.sqrt(n / (1 << 20))
        tolerances = (
            {"atol": 6e-3 * scale, "rtol": 1e-4}
            if dtype == torch.complex64
            else {"atol": 2e-11 * scale, "rtol": 1e-8}
        )
    test.check(op, *test.gen_inputs(), **tolerances)


@pytest.mark.smoke
def test_fft_batch_above_grid_y_limit() -> None:
    """The symbolic batch is grid.x, so batch may exceed CUDA grid.y's 65535 limit."""
    n = 64
    x = torch.randn(65536, n, device="cuda", dtype=torch.complex64)

    got = FFTC2CFwdOp()(x)

    torch.testing.assert_close(got, torch.fft.fft(x), atol=1e-4, rtol=1e-4)


@pytest.mark.smoke
def test_fft_lazy_conjugate_input() -> None:
    """A conjugate view keeps its conj bit through contiguous(); view_as_real rejects it."""
    x = torch.randn(4, 64, device="cuda", dtype=torch.complex64).conj()

    got = FFTC2CFwdOp()(x)

    torch.testing.assert_close(got, torch.fft.fft(x), atol=1e-4, rtol=1e-4)
