
import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import FFTC2COp
from workloads.fft import FFTTest as _FFTTestWorkload


class FFTTest(_FFTTestWorkload, TestBase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(x, dim=-1)


class FFTFixture(FixtureBase):
    PARAMS = [
        ("n, dtype, tune, batch_shape", [
            pytest.param(64, torch.complex64, False, (), marks=pytest.mark.smoke),
            pytest.param(64, torch.complex128, False, (), marks=pytest.mark.smoke),
            pytest.param(64, torch.complex64, False, (4,), marks=pytest.mark.full),
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
    op = FFTC2COp(n, dtype=dtype, tune=tune)
    if dtype == torch.complex64:
        tolerances = {"atol": 1e-4, "rtol": 1e-4}
    else:
        tolerances = {"atol": 1e-8, "rtol": 1e-8}
    test.check(op, *test.gen_inputs(), **tolerances)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
