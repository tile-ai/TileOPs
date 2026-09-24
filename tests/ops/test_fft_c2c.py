"""Selection, service-table and tuning sentinels for the FFT kernels."""

import math
import warnings
from types import SimpleNamespace

import pytest
import torch

import tileops.kernels.fft_c2c as fft_c2c
from tileops.kernels.fft_c2c import (
    _MAX_BLOCK_SMEM,
    _MAX_THREADS,
    FFT_PLANS,
    FFTC2CDecomposedKernel,
    FFTC2COneCTAKernel,
    _four_step_candidates,
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
def test_every_power_of_two_through_2_28_selects_the_kernel_its_record_names(
    dtype: torch.dtype,
) -> None:
    """Check every supported power and the first one above the upper bound."""
    op = FFTC2CFwdOp()
    name = str(dtype).split(".")[-1]
    for exponent in range(1, 30):
        n = 1 << exponent
        plan = FFT_PLANS.get((n, name))
        expected_launches = (
            1
            if exponent < 14 or (exponent == 14 and dtype == torch.complex64)
            else 2
            if exponent <= 24
            else 3
            if exponent <= 28
            else 0
        )
        call = FFTC2CCall(n=n, dtype=dtype, arch=90, sm_count=1)
        if expected_launches == 0:
            assert plan is None, f"n={n} {name} unexpectedly has a plan"
            with pytest.raises(ValueError, match="no implementation serves"):
                op.select_kernel(call)
            continue

        assert plan is not None, f"n={n} {name} has no plan"
        assert len(plan.factors) == expected_launches, f"n={n} {name}"
        expected = FFTC2CDecomposedKernel if plan.decomposed else FFTC2COneCTAKernel
        assert op.select_kernel(call) is expected, f"n={n} {name}"


@pytest.mark.smoke
def test_every_plan_record_describes_as_many_kernels_as_it_launches() -> None:
    """All per-kernel fields agree with the record's factor count."""
    for (n, name), plan in FFT_PLANS.items():
        where = f"({n}, {name})"
        assert math.prod(plan.factors) == n, where
        assert len(plan.radix) == len(plan.factors), where
        assert len(plan.pad) == len(plan.factors), where
        assert len(plan.builders) == len(plan.factors), where
        assert len(plan.tile) == (len(plan.factors) if plan.decomposed else 0), where
        assert len(plan.twiddle_exp) == len(plan.factors) - 1, where
        assert plan.archs, where


@pytest.mark.smoke
def test_single_tone_peak_is_bin_7() -> None:
    n = 4096
    k = torch.arange(n, dtype=torch.float64, device="cuda")
    x = torch.exp(2j * torch.pi * 7 * k / n).to(torch.complex64)

    got = FFTC2CFwdOp()(x)
    peak = int(got.abs().argmax())
    sidelobe = torch.cat((got[:peak], got[peak + 1 :])).abs().max() / got[peak].abs()

    assert peak == 7
    assert sidelobe < 1e-4


@pytest.mark.parametrize(
    "dtype",
    (
        pytest.param(torch.complex64, marks=pytest.mark.smoke, id="c64"),
        pytest.param(torch.complex128, marks=pytest.mark.smoke, id="c128"),
    ),
)
def test_fft_n1_is_an_out_of_place_identity(dtype: torch.dtype) -> None:
    assert (1, str(dtype).split(".")[-1]) not in FFT_PLANS
    x = torch.randn(3, 1, device="cuda", dtype=dtype)

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
    """Regression: every factor receives and launches its own tuned config."""
    n, dtype = 1 << 14, torch.complex128  # the shortest decomposed length: two kernels
    launched: dict = {}
    run = fft_c2c._fft_decomposed_run

    def spy(builders, plan, tile, pad, *args):
        launched["tile"], launched["pad"] = tile, pad
        return run(builders, plan, tile, pad, *args)

    monkeypatch.setattr(fft_c2c, "_fft_decomposed_run", spy)
    tuned_candidates: list[list[dict]] = []

    def fake_tune(self, _builder, candidates, **_kwargs):
        tuned_candidates.append(candidates)
        return SimpleNamespace(config=candidates[-1])

    monkeypatch.setattr(FFTC2CDecomposedKernel, "tune_jit_kernel", fake_tune)

    x = torch.randn(2, n, device="cuda", dtype=dtype)
    op = FFTC2CFwdOp(tune=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = op(x)
    assert [str(w.message) for w in caught if "autotune_configs" in str(w.message)] == []

    kernel = op.kernel
    assert isinstance(kernel, FFTC2CDecomposedKernel)
    plan = kernel.plan
    assert len(tuned_candidates) == len(plan.factors)
    assert len(kernel.config["tile"]) == len(plan.factors)
    assert len(kernel.config["pad"]) == len(plan.factors)
    tuned = zip(kernel.config["tile"], kernel.config["pad"], strict=True)
    for index, (tile, pad) in enumerate(tuned):
        offered = _four_step_candidates(plan, index, 8)
        assert len(offered) > 1, f"kernel {index} was offered nothing to choose between"
        assert {"tw": tile, **dict(zip(("row", "grp"), pad, strict=False))} in offered

    assert launched == {"tile": kernel.config["tile"], "pad": kernel.config["pad"]}
    torch.testing.assert_close(got, torch.fft.fft(x), atol=1e-8, rtol=1e-8)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "n, tunes",
    (
        pytest.param(512, False, id="one-config"),
        pytest.param(1024, True, id="several-configs"),
    ),
)
def test_a_one_cta_plan_tunes_only_where_it_has_a_choice(
    n: int, tunes: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fixed-layout plan skips tuning; a multi-candidate plan invokes it."""
    swept: list = []

    def fake_tune(self, _builder, candidates, **_kwargs):
        swept.append(self.n)
        return SimpleNamespace(config=candidates[-1])

    monkeypatch.setattr(FFTC2COneCTAKernel, "tune_jit_kernel", fake_tune)

    x = torch.randn(2, n, device="cuda", dtype=torch.complex64)
    op = FFTC2CFwdOp(tune=True)
    got = op(x)

    kernel = op.kernel
    assert isinstance(kernel, FFTC2COneCTAKernel)
    offered = kernel.autotune_configs
    assert (len(offered) > 1) is tunes, f"{n} offers {len(offered)} configs"
    assert bool(swept) is tunes, f"{n} tuned {swept}"
    assert kernel.config in offered
    torch.testing.assert_close(got, torch.fft.fft(x), atol=1e-4, rtol=1e-4)


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


@pytest.mark.smoke
@pytest.mark.parametrize(
    "n, dtype, config, message",
    (
        pytest.param(
            4096,
            torch.complex64,
            {"row": 271, "grp": 17},
            "strides must be at or above",
            id="row-one-below-floor",
        ),
        pytest.param(
            4096,
            torch.complex64,
            {"row": 272, "grp": 16},
            "strides must be at or above",
            id="group-one-below-floor",
        ),
        pytest.param(
            4096,
            torch.complex64,
            {"row": 10000, "grp": 10000},
            "shared memory exceeds",
            id="over-shared",
        ),
        pytest.param(4096, torch.complex64, {"row": 272}, "config keys", id="missing-key"),
        pytest.param(
            512,
            torch.complex64,
            {"row": 16, "grp": 2},
            "only config it runs",
            id="plan-takes-no-config",
        ),
        pytest.param(
            1 << 20,
            torch.complex128,
            {"tile": (999, 999), "pad": ((67, 5), (65, 5))},
            "must divide its grid axis",
            id="tile-indivisible",
        ),
        pytest.param(
            1 << 20,
            torch.complex128,
            {"tile": (32, 4), "pad": ((67, 5), (65, 5))},
            "threads exceeds",
            id="tile-over-thread-limit",
        ),
        pytest.param(
            1 << 20,
            torch.complex128,
            {"tile": (4, 4), "pad": ((64, 5), (65, 5))},
            "strides must be at or above",
            id="decomposed-row-one-below-floor",
        ),
        pytest.param(
            1 << 20,
            torch.complex128,
            {"tile": (4, 4), "pad": ((67,), (65, 5))},
            "expected 2 shared strides",
            id="decomposed-pad-arity",
        ),
        pytest.param(
            1 << 20,
            torch.complex128,
            {"tile": (4,), "pad": ((67, 5), (65, 5))},
            "one tile and one pad per factor",
            id="one-tile-short",
        ),
    ),
)
def test_a_config_outside_the_bounds_is_refused_rather_than_launched(
    n: int, dtype: torch.dtype, config: dict, message: str
) -> None:
    """Reject public configs outside structural bounds before device launch."""
    cls = FFTC2CDecomposedKernel if FFT_PLANS[n, str(dtype)[6:]].decomposed else FFTC2COneCTAKernel
    with pytest.raises(ValueError, match=message):
        cls(n, dtype, config=config)


@pytest.mark.smoke
def test_the_config_each_record_carries_is_accepted() -> None:
    """The refusal above must not reject what the plan table itself names."""
    for (n, dtype_str), plan in FFT_PLANS.items():
        dtype = torch.complex64 if dtype_str == "complex64" else torch.complex128
        if plan.decomposed:
            FFTC2CDecomposedKernel(n, dtype, config={"tile": plan.tile, "pad": plan.pad})
        else:
            row, grp = plan.pad[0]
            FFTC2COneCTAKernel(n, dtype, config={"row": row, "grp": grp})
