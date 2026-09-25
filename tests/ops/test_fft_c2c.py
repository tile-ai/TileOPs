"""Service range, input contract, tuning and architecture bounds of the FFT op."""

from types import SimpleNamespace

import pytest
import torch

import tileops.kernels.fft_c2c as fft_c2c
from tileops.kernels.fft_c2c import (
    _MAX_BLOCK_SMEM,
    _MAX_THREADS,
    FFT_PLANS,
    FFTC2CDecomposedKernel,
    _four_step_geometry,
    _four_step_smem_reals,
    _one_cta_smem_reals,
)
from tileops.kernels.fft_call_spec import FFTC2CCall
from tileops.ops import FFTC2CFwdOp


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
    n, dtype = 1 << 14, torch.complex128  # the shortest decomposed length: two kernels
    launched: dict = {}
    run = fft_c2c._fft_decomposed_run

    def spy(builders, plan, tile, pad, *args):
        launched["tile"], launched["pad"] = tile, pad
        return run(builders, plan, tile, pad, *args)

    monkeypatch.setattr(fft_c2c, "_fft_decomposed_run", spy)
    tuned: list = []

    def fake_tune(self, _builder, candidates, seed_config, **_kwargs):
        # Any candidate but the default, so a launch on the defaults cannot pass.
        won = next(c for c in candidates if c != seed_config)
        tuned.append(won)
        return SimpleNamespace(config=won)

    monkeypatch.setattr(FFTC2CDecomposedKernel, "tune_jit_kernel", fake_tune)

    x = torch.randn(2, n, device="cuda", dtype=dtype)
    op = FFTC2CFwdOp(tune=True)
    got = op(x)

    assert len(tuned) == len(op.kernel.plan.factors)
    assert launched == {
        "tile": tuple(c["tw"] for c in tuned),
        "pad": tuple(tuple(c[k] for k in ("row", "grp") if k in c) for c in tuned),
    }
    torch.testing.assert_close(got, torch.fft.fft(x), atol=1e-8, rtol=1e-8)


@pytest.mark.smoke
def test_every_record_stays_inside_the_bounds_its_config_space_assumes() -> None:
    """Every default config satisfies its thread, grid, and shared-memory bounds."""
    for (n, dtype_str), plan in FFT_PLANS.items():
        where = f"{n} {dtype_str}"
        itemsize = 4 if dtype_str == "complex64" else 8
        assert plan.archs, where
        if not plan.decomposed:
            row, grp = plan.pad[0]
            reals = _one_cta_smem_reals(plan, n, row, grp)
            assert reals * itemsize <= _MAX_BLOCK_SMEM, where
            if n >= 1024:
                assert n // 16 <= _MAX_THREADS, where
            continue
        for factor in plan.factors:
            assert (factor, dtype_str) in FFT_PLANS, f"{where}: factor {factor} is not served"
            assert not FFT_PLANS[factor, dtype_str].decomposed, f"{where}: factor {factor}"
        for index, tile in enumerate(plan.tile):
            _nf, lanes, extent, _twrows, _r = _four_step_geometry(plan, index)
            assert tile >= 1 and extent % tile == 0, f"{where} kernel {index}"
            assert tile * lanes <= _MAX_THREADS, f"{where} kernel {index}"
            reals = _four_step_smem_reals(plan, index, tile, plan.pad[index])
            assert reals * itemsize <= _MAX_BLOCK_SMEM, f"{where} kernel {index}"
