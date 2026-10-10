import math
from types import SimpleNamespace

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.kernels.fft import FFTC2CCall, FFTC2CFourStepKernel
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
    # Asks for 112 GiB free; the nightly job runs alone on its GPU.
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
    # The check holds up to six copies of the output at once; one more is headroom.
    need = 7 * batch * n * (8 if dtype == torch.complex64 else 16)
    free, _total = torch.cuda.mem_get_info(run_device())
    if need > free:
        pytest.skip(f"n={n} {dtype} needs {need >> 20} MiB free, device has {free >> 20} MiB")
    test = FFTTest(n, dtype, batch_shape=batch_shape)
    op = FFTC2CFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.sm89
@pytest.mark.smoke
@pytest.mark.parametrize("n", [16384, 1 << 22])
def test_fft_c2c_in_99_kb_of_shared_memory(n: int) -> None:
    """SM89 gives a block 99 KB of opt-in shared memory, less than SM80 and SM90 do, so these
    lengths run on decompositions of their own."""
    test = FFTTest(n, torch.complex64)
    test.check(FFTC2CFwdOp(), *test.gen_inputs())


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


@pytest.mark.in_tree_kernels
@pytest.mark.cuda_only
@pytest.mark.parametrize(
    "dtype",
    (
        pytest.param(torch.complex64, marks=pytest.mark.smoke, id="c64"),
        pytest.param(torch.complex128, marks=pytest.mark.smoke, id="c128"),
    ),
)
def test_every_power_of_two_through_2_28_has_a_kernel(dtype: torch.dtype) -> None:
    """Every length in the manifest's domain, 2 through 2**28, has an implementation."""
    op = FFTC2CFwdOp()
    for exponent in range(1, 29):
        op.key_for("fft_c2c", FFTC2CCall(n=1 << exponent, dtype=dtype))


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
