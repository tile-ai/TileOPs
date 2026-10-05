import math
from types import SimpleNamespace

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.kernels.constants import MAX_BLOCK_THREADS
from tileops.kernels.fft import FFT_NARROW_PLANS, FFT_PLANS, FFTC2CCall, FFTC2CFourStepKernel
from tileops.ops import FFTC2CFwdOp
from workloads.device import run_device
from workloads.fft import FFTWorkload
from workloads.numerics import compare_outputs


class FFTTest(FFTWorkload, TestBase):
    pass


# One smallest or boundary case per execution structure. The range test below
# covers every plan-table entry without allocating its tensors.
_CORRECTNESS_CASES = (
    pytest.param(
        2,
        torch.complex64,
        (),
        marks=[pytest.mark.smoke, pytest.mark.packaging(family="fft")],
        id="tiny-lower",
    ),
    pytest.param(64, torch.complex64, (), marks=pytest.mark.smoke, id="warp8-lower"),
    pytest.param(4096, torch.complex128, (), marks=pytest.mark.smoke, id="three-pass-upper-c128"),
    pytest.param(32768, torch.complex64, (3,), marks=pytest.mark.smoke, id="two-factor-lower"),
    pytest.param(32, torch.complex128, (3,), marks=pytest.mark.full, id="tiny-upper-c128"),
    pytest.param(128, torch.complex128, (3,), marks=pytest.mark.full, id="warp8-upper-c128"),
    pytest.param(256, torch.complex64, (2, 4), marks=pytest.mark.full, id="packed-16x16"),
    pytest.param(512, torch.complex128, (3,), marks=pytest.mark.full, id="packed-8x8x8-c128"),
    pytest.param(1024, torch.complex64, (2, 4), marks=pytest.mark.full, id="three-pass-lower"),
    pytest.param(2048, torch.complex64, (3,), marks=pytest.mark.full, id="three-pass-radix8"),
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
    # Asks for 80 GiB free; the nightly job runs alone on its GPU.
    pytest.param(
        16384,
        torch.complex128,
        (65536,),
        marks=pytest.mark.nightly,
        id="decomposed-batch-above-grid-y-limit",
    ),
)


class FFTFixture(FixtureBase):
    PARAMS = [("n, dtype, batch_shape", _CORRECTNESS_CASES)]


@pytest.mark.cuda_only
@FFTFixture
def test_fft_c2c(n: int, dtype: torch.dtype, batch_shape: tuple) -> None:
    batch = math.prod(batch_shape) if batch_shape else 1
    need = 5 * batch * n * (8 if dtype == torch.complex64 else 16)
    free, _total = torch.cuda.mem_get_info(run_device())
    if need > free:
        pytest.skip(f"n={n} {dtype} needs {need >> 20} MiB free, device has {free >> 20} MiB")
    test = FFTTest(n, dtype, batch_shape=batch_shape)
    op = FFTC2CFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_fft_batch_above_grid_y_limit() -> None:
    """The symbolic batch is grid.x, so batch may exceed CUDA grid.y's 65535 limit."""
    n = 64
    x = torch.randn(65536, n, device=run_device(), dtype=torch.complex64)

    got = FFTC2CFwdOp()(x)

    workload = FFTWorkload(x.shape[-1], x.dtype)
    compare_outputs(got, workload.ref_program(x), workload.verification(x))


@pytest.mark.smoke
def test_fft_lazy_conjugate_input() -> None:
    """A conjugate view keeps its conj bit through contiguous(); view_as_real rejects it."""
    x = torch.randn(4, 64, device=run_device(), dtype=torch.complex64).conj()

    got = FFTC2CFwdOp()(x)

    workload = FFTWorkload(x.shape[-1], x.dtype)
    compare_outputs(got, workload.ref_program(x), workload.verification(x))


@pytest.mark.cuda_only
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
        call = FFTC2CCall(n=1 << exponent, dtype=dtype)
        if exponent == 29:
            with pytest.raises(ValueError, match="no implementation serves"):
                op.select_implementation("fft_c2c", call)
        else:
            op.select_implementation("fft_c2c", call)


@pytest.mark.smoke
def test_fft_n1_is_an_out_of_place_identity() -> None:
    x = torch.randn(3, 1, device=run_device(), dtype=torch.complex64)

    got = FFTC2CFwdOp()(x)

    torch.testing.assert_close(got, x, atol=0, rtol=0)
    assert got.data_ptr() != x.data_ptr()


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_tune_configures_every_kernel_of_a_four_step_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a decomposed plan tuned nothing and silently kept its defaults."""
    tuned: list = []

    def fake_tune(self, _builder, candidates, seed_config, **_kwargs):
        # Any candidate but the default, so keeping the defaults cannot pass.
        won = next(c for c in candidates if c != seed_config)
        tuned.append(won)
        return SimpleNamespace(config=won)

    monkeypatch.setattr(FFTC2CFourStepKernel, "tune_jit_kernel", fake_tune)
    # The shortest decomposed length: two kernels.
    x = torch.randn(2, 1 << 14, device=run_device(), dtype=torch.complex128)
    op = FFTC2CFwdOp(tune=True)
    got = op(x)

    workload = FFTWorkload(x.shape[-1], x.dtype)
    compare_outputs(got, workload.ref_program(x), workload.verification(x))


@pytest.mark.smoke
def test_every_plan_fits_the_blocks_of_the_architectures_it_serves() -> None:
    """CI runs on Hopper, whose larger limits would hide a plan other cards cannot run."""
    # The four-pass kernel sizes its strides from the length; 137 KB exceeds sm_86's 99 KB,
    # which is the reason FFT_NARROW_PLANS carries a second record for this length.
    assert 86 not in FFT_PLANS[16384, "complex64"].archs
    for (n, dtype_str), plan in FFT_PLANS.items():
        assert {80, 90} <= set(plan.archs), f"{n} {dtype_str}"
    for (n, dtype_str), plan in FFT_NARROW_PLANS.items():
        # An empty record serves nothing and makes ``smem_cap`` reduce over no architecture.
        assert plan.archs, f"{n} {dtype_str}"
    for (n, dtype_str), plan in [*FFT_PLANS.items(), *FFT_NARROW_PLANS.items()]:
        where = f"{n} {dtype_str}"
        if not plan.decomposed:
            assert n < 1024 or n // 16 <= MAX_BLOCK_THREADS, where
            continue
        for factor in plan.factors:
            assert not FFT_PLANS[factor, dtype_str].decomposed, f"{where}: factor {factor}"
        for index, tile in enumerate(plan.tile):
            _nf, lanes, extent, _twrows, _r = plan.geometry(index)
            assert tile >= 1 and extent % tile == 0, f"{where} kernel {index}"
            assert tile * lanes <= MAX_BLOCK_THREADS, f"{where} kernel {index}"
