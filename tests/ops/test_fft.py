
import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.kernel_base import Kernel
from tileops.ops import FFTC2COp
from workloads.fft import FFTWorkload


class FFTTest(FFTWorkload, TestBase):
    pass


class FFTFixture(FixtureBase):
    PARAMS = [
        ("n, dtype, tune, batch_shape", [
            pytest.param(64, torch.complex64, False, (), marks=pytest.mark.smoke),
            pytest.param(64, torch.complex128, False, (), marks=pytest.mark.smoke),
            pytest.param(64, torch.complex64, False, (4,), marks=pytest.mark.smoke),
            pytest.param(128, torch.complex64, False, (), marks=pytest.mark.full),
            pytest.param(256, torch.complex64, False, (8,), marks=pytest.mark.full),
            pytest.param(512, torch.complex64, False, (16,), marks=pytest.mark.full),
            pytest.param(1024, torch.complex64, False, (), marks=pytest.mark.full),
            pytest.param(1024, torch.complex64, False, (2, 4), marks=pytest.mark.full),
            pytest.param(128, torch.complex128, False, (4,), marks=pytest.mark.full),
        ]),
    ]


@FFTFixture
def test_fft_c2c(n: int, dtype: torch.dtype, tune: bool, batch_shape: tuple) -> None:
    test = FFTTest(n, dtype, batch_shape=batch_shape)
    op = FFTC2COp(tune=tune)
    if dtype == torch.complex64:
        tolerances = {"atol": 1e-4, "rtol": 1e-4}
    else:
        tolerances = {"atol": 1e-8, "rtol": 1e-8}
    test.check(op, *test.gen_inputs(), **tolerances)


@pytest.mark.parametrize(
    "batch_shape",
    [
        pytest.param((), id="manifest-c64-b1", marks=pytest.mark.smoke),
        pytest.param((64,), id="manifest-c64-b64", marks=pytest.mark.full),
    ],
)
def test_fft_manifest_c64_rows(batch_shape: tuple) -> None:
    test = FFTTest(4096, torch.complex64, batch_shape=batch_shape)
    test.check(FFTC2COp(), *test.gen_inputs(), atol=1e-4, rtol=1e-4)


@pytest.mark.smoke
def test_fft_manifest_c128_row() -> None:
    test = FFTTest(4096, torch.complex128, batch_shape=(64,))
    test.check(FFTC2COp(), *test.gen_inputs(), atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize(
    ("n", "dtype", "batch_shape", "layout"),
    [
        pytest.param(1, torch.complex64, (), "contiguous", id="zero-stage", marks=pytest.mark.smoke),
        pytest.param(2, torch.complex128, (), "contiguous", id="single-stage", marks=pytest.mark.smoke),
        pytest.param(8, torch.complex64, (2, 3), "contiguous", id="leading-dims", marks=pytest.mark.full),
        pytest.param(8, torch.complex64, (2, 3), "strided", id="strided-view", marks=pytest.mark.full),
        pytest.param(8, torch.complex64, (), "conjugate", id="conjugate-view", marks=pytest.mark.full),
    ],
)
def test_fft_boundary_and_layout_paths(
    n: int,
    dtype: torch.dtype,
    batch_shape: tuple,
    layout: str,
) -> None:
    test = FFTTest(n, dtype, batch_shape=batch_shape, layout=layout)
    tolerance = 1e-4 if dtype is torch.complex64 else 1e-8
    test.check(FFTC2COp(), *test.gen_inputs(), atol=tolerance, rtol=tolerance)


class _SplitInputKernel(Kernel):
    """Kernel-map probe for the established four-tensor input contract."""

    def __init__(self, n, batch_size, dtype, tune=False):
        super().__init__()
        self.n = n
        self.batch_size = batch_size
        self.dtype = dtype

    def forward(self, x_real, x_imag, lut_real, lut_imag):
        assert x_real.is_contiguous()
        assert x_imag.is_contiguous()
        assert lut_real.shape == (self.n - 1,)
        assert lut_imag.shape == (self.n - 1,)
        return torch.stack((x_real, x_imag), dim=-1)


@pytest.mark.smoke
def test_fft_kernel_map_override_keeps_split_inputs() -> None:
    x = torch.randn(2, 8, device="cuda", dtype=torch.complex64).conj()
    op = FFTC2COp(kernel_map={"fft_c2c_kernel": _SplitInputKernel})

    torch.testing.assert_close(op(x), x)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
