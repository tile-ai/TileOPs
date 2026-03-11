from typing import Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import FFTC2CLUTOp


class FFTLUTFixture(FixtureBase):
    PARAMS = [
        ("n, dtype, tune", [
            pytest.param(64, torch.complex64, False, marks=pytest.mark.smoke),
            pytest.param(128, torch.complex64, False, marks=pytest.mark.full),
            pytest.param(256, torch.complex64, False, marks=pytest.mark.full),
            pytest.param(512, torch.complex64, False, marks=pytest.mark.full),
            pytest.param(1024, torch.complex64, False, marks=pytest.mark.full),
            pytest.param(64, torch.complex128, False, marks=pytest.mark.full),
            pytest.param(128, torch.complex128, False, marks=pytest.mark.full),
        ]),
    ]


class FFTLUTTest(TestBase):

    def __init__(self, n: int, dtype: torch.dtype):
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor]:
        x = torch.randn(self.n, device='cuda', dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(x, dim=-1)


@FFTLUTFixture
def test_fft_c2c_lut(n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FFTLUTTest(n, dtype)
    op = FFTC2CLUTOp(n, dtype=dtype, tune=tune)
    if dtype == torch.complex64:
        tolerances = {"atol": 1e-4, "rtol": 1e-4}
    else:
        tolerances = {"atol": 1e-8, "rtol": 1e-8}
    test.check(op, *test.gen_inputs(), **tolerances)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
