import math

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.fft_c2c import FFT_PLANS
from tileops.ops import FFTC2CFwdOp
from workloads.fft import FFTWorkload


class FFTTest(FFTWorkload, TestBase):
    pass


# The records the smoke tier checks the numbers of: both dtypes of one packed
# length, the longest single-CTA length, and the shortest decomposed one. Every
# other record is a full-tier case.
_SMOKE = {(64, "complex64"), (64, "complex128"), (4096, "complex64"), (32768, "complex64")}

# A decomposed call holds its input, its scratch, its output and the reference at
# once, so from 2**22 up -- 32 MB a transform already at complex64 -- the cases
# run unbatched. Below that the batch shape rotates over the three forms the op's
# flattening of leading dimensions has to handle.
_BATCH_SHAPES = ((), (3,), (2, 4))
_UNBATCHED_FROM = 1 << 22


def _cases() -> list:
    """One case per record in the service table, plus the n = 1 identity.

    Each (length, dtype) is its own plan with its own index maps, so a
    wrong one is wrong numbers rather than a slow kernel, and reading the cases
    off FFT_PLANS is what makes a new length impossible to add without one. The
    smoke cases lead, as the tier check requires.
    """
    smoke, full = [], []
    for index, (n, name) in enumerate(sorted(FFT_PLANS)):
        batch_shape = () if n >= _UNBATCHED_FROM else _BATCH_SHAPES[index % len(_BATCH_SHAPES)]
        is_smoke = (n, name) in _SMOKE
        (smoke if is_smoke else full).append(
            pytest.param(
                n,
                getattr(torch, name),
                batch_shape,
                marks=pytest.mark.smoke if is_smoke else pytest.mark.full,
                id=f"n{n}-{name}",
            )
        )
    identity = [
        pytest.param(1, dtype, (3,), marks=pytest.mark.full, id=f"n1-{name}")
        for dtype, name in ((torch.complex64, "complex64"), (torch.complex128, "complex128"))
    ]
    return smoke + identity + full


class FFTFixture(FixtureBase):
    PARAMS = [("n, dtype, batch_shape", _cases())]


@FFTFixture
def test_fft_c2c(n: int, dtype: torch.dtype, batch_shape: tuple) -> None:
    batch = math.prod(batch_shape) if batch_shape else 1
    # Input, scratch, output and the reference, and the caching allocator holds
    # the run's buffers while the comparison builds its own. The longest lengths
    # are 4 GB a transform, so a device that cannot hold five of them skips
    # rather than reporting an out-of-memory error as a failed transform.
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
        # A transform's outputs carry sqrt(n) times the input's scale, so a fixed
        # absolute tolerance is a progressively tighter relative one as n grows,
        # and this one is scaled by sqrt(n) to stay one line rather than many. At
        # n = 2**20 on unit-variance input this op and cuFFT differ by at most
        # 1.8e-3 (complex64) and 5.1e-12 (complex128) -- each dtype's rounding
        # floor against a float64 reference, not a disagreement about the answer.
        # The anchors leave about 3x headroom over that, and the measured
        # difference tracks the same sqrt(n) at every other length.
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
    n = 4096
    x = torch.randn(70000, n, device="cuda", dtype=torch.complex64)

    got = FFTC2CFwdOp()(x)

    torch.testing.assert_close(got, torch.fft.fft(x), atol=1e-4, rtol=1e-4)
