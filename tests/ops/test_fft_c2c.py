"""Selection, service-table and tuning sentinels for the FFT kernels."""

import math
import warnings

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
    """Selection over the whole promised range, against the table that decides it.

    One check per length from 2 to 2**28: a record must exist, and the class the
    op selects must be the one that record's factor count implies -- one factor
    is a single CTA, more than one is the decomposition. That holds three
    contracts in one sweep. The table leaves no length in the range uncovered,
    which is what lets the op promise every power of two. The two regions stay
    disjoint, since a call both claimed would make ``select_kernel`` raise rather
    than prefer one. And 16384, the one length whose two dtypes fall on different
    sides, is split by dtype and not by length.

    Selection only: the transforms themselves are checked by ``test_fft_c2c``.
    """
    op = FFTC2CFwdOp()
    name = str(dtype).split(".")[-1]
    wrong = []
    for exponent in range(1, 29):
        n = 1 << exponent
        plan = FFT_PLANS.get((n, name))
        if plan is None:
            wrong.append((n, "no record in FFT_PLANS"))
            continue
        expected = FFTC2CDecomposedKernel if plan.decomposed else FFTC2COneCTAKernel
        call = FFTC2CCall(n=n, dtype=dtype, device_index=0, device=torch.device("cuda:0"))
        try:
            selected = op.select_kernel(call)
        except ValueError as exc:
            wrong.append((n, str(exc)))
            continue
        if selected is not expected:
            wrong.append((n, f"{selected.__name__}, expected {expected.__name__}"))
    assert wrong == []


@pytest.mark.smoke
def test_every_plan_record_describes_as_many_kernels_as_it_launches() -> None:
    """The per-kernel fields of a record agree with its factor count.

    Every builder and every consumer indexes these records by kernel, so a
    record whose fields disagree would launch one kernel with another's strides.
    ``tile`` is the exception by design: a single-CTA plan takes its packing from
    its length and passes no width.
    """
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
    x = torch.randn(3, 1, device="cuda", dtype=dtype)

    got = FFTC2CFwdOp()(x)

    torch.testing.assert_close(got, x, atol=0, rtol=0)
    assert got.data_ptr() != x.data_ptr()


@pytest.mark.smoke
def test_tune_sweeps_every_kernel_of_a_four_step_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """``tune=True`` on a decomposed length tunes each of its kernels.

    Regression for a composite kernel that declared no ``autotune_configs``: the
    base class warned once and silently kept ``default_config``, and
    ``self.kernel`` being a tuple of builders meant the inherited single-kernel
    sweep could not have run on it anyway. This holds the three parts of the fix
    -- no such warning, one winner per kernel drawn from that kernel's own
    candidate list, and those winners being what the launch path is handed.
    """
    n, dtype = 1 << 14, torch.complex128  # the shortest decomposed length: two kernels
    launched: dict = {}
    run = fft_c2c._fft_decomposed_run

    def spy(builders, plan, tile, pad, *args):
        launched["tile"], launched["pad"] = tile, pad
        return run(builders, plan, tile, pad, *args)

    monkeypatch.setattr(fft_c2c, "_fft_decomposed_run", spy)

    x = torch.randn(2, n, device="cuda", dtype=dtype)
    op = FFTC2CFwdOp(tune=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = op(x)
    assert [str(w.message) for w in caught if "autotune_configs" in str(w.message)] == []

    kernel = op.kernel
    assert isinstance(kernel, FFTC2CDecomposedKernel)
    plan = kernel.plan
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
    "n, sweeps",
    (
        pytest.param(512, False, id="one-config"),
        pytest.param(1024, True, id="several-configs"),
    ),
)
def test_a_one_cta_plan_sweeps_only_where_it_has_a_choice(
    n: int, sweeps: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``tune=True`` pays for a sweep only where the record leaves something to pick.

    A plan below 1024, and a four-pass plan, takes its shared strides from its
    length, so its candidate list holds one entry. Timing it would compile the
    kernel a second time and fill a 1024-transform input buffer to confirm the
    only option there was, so that entry is kept without a sweep.
    """
    swept: list = []
    original = FFTC2COneCTAKernel.tune_jit_kernel

    def spy(self, *args, **kwargs):
        swept.append(self.n)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FFTC2COneCTAKernel, "tune_jit_kernel", spy)

    x = torch.randn(2, n, device="cuda", dtype=torch.complex64)
    op = FFTC2CFwdOp(tune=True)
    got = op(x)

    kernel = op.kernel
    assert isinstance(kernel, FFTC2COneCTAKernel)
    offered = kernel.autotune_configs
    assert (len(offered) > 1) is sweeps, f"{n} offers {len(offered)} configs"
    assert bool(swept) is sweeps, f"{n} swept {swept}"
    assert kernel.config in offered
    torch.testing.assert_close(got, torch.fft.fft(x), atol=1e-4, rtol=1e-4)


@pytest.mark.smoke
def test_every_record_stays_inside_the_bounds_its_config_space_assumes() -> None:
    """The bounds the candidate lists are generated from, checked on the records.

    ``_four_step_candidates`` and ``autotune_configs`` offer only configs inside
    these, but the record's own default never passes through either, so nothing
    else checks it. A record added with a tile that does not divide its grid
    axis, or that asks a block for more shared memory than the narrowest
    architecture it claims can give, would otherwise fail at launch.
    """
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
    "n, dtype, config",
    (
        pytest.param(4096, torch.complex64, {"row": 0, "grp": 0}, id="strides-below-floor"),
        pytest.param(4096, torch.complex64, {"row": 10000, "grp": 10000}, id="over-shared"),
        pytest.param(4096, torch.complex64, {"row": -1, "grp": -1}, id="negative"),
        pytest.param(4096, torch.complex64, {"row": 272}, id="missing-key"),
        pytest.param(512, torch.complex64, {"row": 16, "grp": 2}, id="plan-takes-no-config"),
        pytest.param(
            1 << 20,
            torch.complex128,
            {"tile": (999, 999), "pad": ((67, 5), (65, 5))},
            id="tile-indivisible",
        ),
        pytest.param(
            1 << 20, torch.complex128, {"tile": (0, 0), "pad": ((67, 5), (65, 5))}, id="tile-zero"
        ),
        pytest.param(
            1 << 20,
            torch.complex128,
            {"tile": (4,), "pad": ((67, 5), (65, 5))},
            id="one-tile-short",
        ),
    ),
)
def test_a_config_outside_the_bounds_is_refused_rather_than_launched(
    n: int, dtype: torch.dtype, config: dict
) -> None:
    """``config`` is public, so its whole domain is reachable, not just the sweep's.

    Every case here reached the device before this check existed and came back
    as ``CUDA error: an illegal memory access was encountered``, which says
    nothing about what was wrong and leaves the context unusable.
    """
    cls = FFTC2CDecomposedKernel if FFT_PLANS[n, str(dtype)[6:]].decomposed else FFTC2COneCTAKernel
    with pytest.raises(ValueError):
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
