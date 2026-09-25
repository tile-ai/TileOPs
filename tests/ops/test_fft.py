import math
from types import SimpleNamespace

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.constants import MAX_BLOCK_THREADS
from tileops.kernels.fft import FFT_PLANS, FFTC2CCall, FFTC2CDecomposedKernel
from tileops.ops import FFTC2CFwdOp
from workloads.fft import FFTWorkload


class FFTTest(FFTWorkload, TestBase):
    pass


# One smallest or boundary case per execution structure. The range test below
# covers every plan-table entry without allocating its tensors.
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


@pytest.mark.parametrize(
    "dtype",
    (
        pytest.param(torch.complex64, marks=pytest.mark.smoke, id="c64"),
        pytest.param(torch.complex128, marks=pytest.mark.smoke, id="c128"),
    ),
)
def test_every_power_of_two_through_2_28_has_a_kernel(dtype: torch.dtype) -> None:
    """The manifest's upper bound: 2**28 is served and 2**29 is not."""
    op = FFTC2CFwdOp()
    for exponent in range(1, 30):
        call = FFTC2CCall(n=1 << exponent, dtype=dtype, arch=90, sm_count=1)
        if exponent == 29:
            with pytest.raises(ValueError, match="no implementation serves"):
                op.select_kernel(call)
        else:
            op.select_kernel(call)


@pytest.mark.smoke
def test_fft_n1_is_an_out_of_place_identity() -> None:
    x = torch.randn(3, 1, device="cuda", dtype=torch.complex64)

    got = FFTC2CFwdOp()(x)

    torch.testing.assert_close(got, x, atol=0, rtol=0)
    assert got.data_ptr() != x.data_ptr()


@pytest.mark.smoke
@pytest.mark.parametrize(
    "shape, dtype, device, message",
    (
        pytest.param((), torch.complex64, "cuda", "at least 1D", id="rank-zero"),
        pytest.param((0,), torch.complex64, "cuda", "positive power of 2", id="zero-length"),
        pytest.param((3,), torch.complex64, "cuda", "positive power of 2", id="non-power-of-two"),
        pytest.param(
            (2,), torch.float32, "cuda", "complex64 or complex128", id="unsupported-dtype"
        ),
        pytest.param((2,), torch.complex64, "cpu", "CUDA tensor", id="cpu-input"),
    ),
)
def test_fft_rejects_inputs_outside_its_lower_boundaries(
    shape: tuple, dtype: torch.dtype, device: str, message: str
) -> None:
    x = torch.randn(shape, dtype=dtype, device=device)

    with pytest.raises(ValueError, match=message):
        FFTC2CFwdOp()(x)


@pytest.mark.smoke
def test_tune_configures_every_kernel_of_a_four_step_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a decomposed plan tuned nothing and silently kept its defaults."""
    tuned: list = []

    def fake_tune(self, _builder, candidates, seed_config, **_kwargs):
        # Any candidate but the default, so keeping the defaults cannot pass.
        won = next(c for c in candidates if c != seed_config)
        tuned.append(won)
        return SimpleNamespace(config=won)

    monkeypatch.setattr(FFTC2CDecomposedKernel, "tune_jit_kernel", fake_tune)
    # The shortest decomposed length: two kernels.
    x = torch.randn(2, 1 << 14, device="cuda", dtype=torch.complex128)
    op = FFTC2CFwdOp(tune=True)
    got = op(x)

    assert len(tuned) == len(op.kernel.plan.factors)
    assert op.kernel.config == {
        "tile": tuple(c["tw"] for c in tuned),
        "pad": tuple(tuple(c[k] for k in ("row", "grp") if k in c) for c in tuned),
    }
    torch.testing.assert_close(got, torch.fft.fft(x), atol=1e-8, rtol=1e-8)


@pytest.mark.smoke
def test_every_plan_serves_sm80_and_sm90_within_the_block_limits() -> None:
    """CI runs on Hopper, whose larger limits would hide a plan other cards cannot run."""
    # The four-pass kernel sizes its strides from the length; 137 KB exceeds sm_86's 99 KB.
    assert 86 not in FFT_PLANS[16384, "complex64"].archs
    for (n, dtype_str), plan in FFT_PLANS.items():
        where = f"{n} {dtype_str}"
        assert {80, 90} <= set(plan.archs), where
        if not plan.decomposed:
            assert n < 1024 or n // 16 <= MAX_BLOCK_THREADS, where
            continue
        for factor in plan.factors:
            assert not FFT_PLANS[factor, dtype_str].decomposed, f"{where}: factor {factor}"
        for index, tile in enumerate(plan.tile):
            _nf, lanes, extent, _twrows, _r = plan.geometry(index)
            assert tile >= 1 and extent % tile == 0, f"{where} kernel {index}"
            assert tile * lanes <= MAX_BLOCK_THREADS, f"{where} kernel {index}"
