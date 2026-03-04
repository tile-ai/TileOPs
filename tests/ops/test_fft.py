from typing import Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import FFTC2COp


class FFTFixture(FixtureBase):
    PARAMS = [
        ("n, dtype, tune", [
            (64, torch.complex64, False),
            (128, torch.complex64, False),
            (256, torch.complex64, False),
            (512, torch.complex64, False),
            (1024, torch.complex64, False),
            (64, torch.complex128, False),
            (128, torch.complex128, False),
        ]),
    ]


class FFTTest(TestBase):

    def __init__(self, n: int, dtype: torch.dtype):
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor]:
        # Generate random complex input
        x = torch.randn(self.n, device='cuda', dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        # Use PyTorch's FFT as reference
        return torch.fft.fft(x, dim=-1)


@FFTFixture
def test_fft_c2c(n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FFTTest(n, dtype)
    op = FFTC2COp(n, dtype=dtype, tune=tune)
    if dtype == torch.complex64:
        tolerances = {"atol": 1e-4, "rtol": 1e-4}
    else:
        tolerances = {"atol": 1e-8, "rtol": 1e-8}
    test.check(op, *test.gen_inputs(), **tolerances)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
