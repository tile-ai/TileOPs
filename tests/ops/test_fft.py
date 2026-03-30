from typing import Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import FFTC2COp


class FFTFixture(FixtureBase):
    PARAMS = [
        ("n, dtype, tune, batch_shape", [
            pytest.param(64, torch.complex64, False, (), marks=pytest.mark.smoke),
            pytest.param(64, torch.complex64, False, (4,), marks=pytest.mark.smoke),
            pytest.param(128, torch.complex64, False, (), marks=pytest.mark.full),
            pytest.param(256, torch.complex64, False, (8,), marks=pytest.mark.full),
            pytest.param(512, torch.complex64, False, (16,), marks=pytest.mark.full),
            pytest.param(1024, torch.complex64, False, (), marks=pytest.mark.full),
            pytest.param(1024, torch.complex64, False, (2, 4), marks=pytest.mark.full),
            pytest.param(64, torch.complex128, False, (), marks=pytest.mark.full),
            pytest.param(128, torch.complex128, False, (4,), marks=pytest.mark.full),
        ]),
    ]


class FFTTest(TestBase):

    def __init__(self, n: int, dtype: torch.dtype, batch_shape: tuple = ()):
        self.n = n
        self.dtype = dtype
        self.batch_shape = batch_shape

    def gen_inputs(self) -> Tuple[torch.Tensor]:
        x = torch.randn(*self.batch_shape, self.n, device='cuda', dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(x, dim=-1)


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
