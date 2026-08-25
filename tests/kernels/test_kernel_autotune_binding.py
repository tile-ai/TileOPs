import inspect

import pytest

import tileops.kernels.kernel_base as kernel_base
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import BlockConfigPlanner, RowTiledAutotuneMixin

pytestmark = pytest.mark.smoke


class _FakeVar:
    """A PrimFunc parameter, which carries its name like a TIR ``Var``."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeBuffer:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype


class _FakePrimFunc:
    """A PrimFunc taking no tensor parameters."""

    params: tuple = ()
    buffer_map: dict = {}


_X, _SIZES, _OUT = _FakeVar("x"), _FakeVar("sizes"), _FakeVar("out")


class _FakeIntPrimFunc:
    """A PrimFunc taking one float input, one int input, and a float output."""

    params = (_X, _SIZES, _OUT)
    buffer_map = {
        _X: _FakeBuffer("float16"),
        _SIZES: _FakeBuffer("int32"),
        _OUT: _FakeBuffer("float16"),
    }


class _FakeJit:
    signature = inspect.signature(lambda block_m, threads: None)
    out_idx = None

    @staticmethod
    def get_tir(**config):
        return _FakePrimFunc()


class _FakeAliasedJit:
    signature = inspect.signature(lambda threads_arg, npt_arg: None)


class _TunedKernel:
    config = {"block_m": 64, "threads": 128}


class _KernelWithRequiredTunables(Kernel):
    def __init__(self) -> None:
        super().__init__()
        self.kernel = _FakeJit()

    @property
    def default_config(self) -> dict:
        return {"block_m": 128, "threads": 256, "tile_n": 0}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_m": 64, "threads": 128},
            {"block_m": 128, "threads": 256},
        ]

    def forward(self):  # pragma: no cover - not needed for this unit test
        raise NotImplementedError


def test_autotune_seeds_required_jit_params_from_default_config(monkeypatch):
    calls: dict[str, object] = {}

    def fake_autotune(**autotune_kwargs):
        calls["autotune_kwargs"] = autotune_kwargs

        def decorate(kernel):
            calls["kernel"] = kernel

            def wrapped(**kwargs):
                calls["initial_kwargs"] = kwargs
                kernel.signature.bind(**kwargs)
                return _TunedKernel()

            return wrapped

        return decorate

    monkeypatch.setattr(kernel_base, "autotune", fake_autotune)

    kernel = _KernelWithRequiredTunables()
    kernel.autotune()

    assert calls["initial_kwargs"] == {"block_m": 128, "threads": 256}
    assert "tile_n" not in calls["initial_kwargs"]
    assert kernel.config == _TunedKernel.config


def test_autotune_group_seed_is_filtered_to_jit_signature():
    kernel = _KernelWithRequiredTunables()

    captured = kernel._call_autotuned_kernel(
        lambda **kwargs: kwargs,
        kernel.kernel,
        {"block_m": 1, "threads": 128, "tile_n": 4096},
    )

    assert captured == {"block_m": 1, "threads": 128}


def test_autotune_initial_kwargs_support_common_jit_param_aliases():
    kernel = _KernelWithRequiredTunables()

    captured = kernel._call_autotuned_kernel(
        lambda **kwargs: kwargs,
        _FakeAliasedJit(),
        {"threads": 128, "num_per_thread": 4},
    )

    assert captured == {"threads_arg": 128, "npt_arg": 4}


class _FakeIntJit(_FakeJit):
    out_idx = [-1]

    @staticmethod
    def get_tir(**config):
        return _FakeIntPrimFunc()


class _KernelWithIntInputs(_KernelWithRequiredTunables):
    """A kernel whose JIT builder takes an integer tensor input."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel = _FakeIntJit()


def test_autotune_refuses_integer_inputs_no_supplier_would_supply():
    """Random integer metadata makes every candidate time a collapsed kernel."""
    with pytest.raises(ValueError, match="integer tensor inputs sizes"):
        _KernelWithIntInputs().autotune()


def test_autotune_accepts_random_integer_inputs_when_declared(monkeypatch):
    """A kernel whose integer inputs are data says so and tunes as before."""

    def fake_autotune(**autotune_kwargs):
        def decorate(kernel):
            def wrapped(**kwargs):
                return _TunedKernel()

            return wrapped

        return decorate

    monkeypatch.setattr(kernel_base, "autotune", fake_autotune)
    kernel = _KernelWithIntInputs()
    kernel.autotune_accepts_random_int_inputs = True
    kernel.autotune()

    assert kernel.config == _TunedKernel.config


# tile_n search space for the row-tiled reduction kernels


class _RowTiled(RowTiledAutotuneMixin):
    """The mixin's declared inputs, without building a kernel to get them."""

    _MAX_TILE_N_CANDIDATES = 3

    def __init__(self, n_padded: int, elem_bytes: int, smem_budget: int) -> None:
        self.N_padded = n_padded
        self._elem_bytes = elem_bytes
        self._smem_budget = smem_budget
        self._planner = BlockConfigPlanner(n_padded, elem_bytes, smem_budget)


def test_row_that_fits_untiled_at_every_thread_count_offers_no_tile():
    """A cheap fragment does not pay for the extra compilation."""
    assert _RowTiled(4096, 2, 227 * 1024)._tile_n_candidates() == [0]


def test_row_that_fits_untiled_only_at_the_most_threads_offers_tiles():
    """tile_n is baked in and reused across every thread count the sweep tries.

    A 32768-column bf16 row holds 64 elements per thread at 512 threads and 256 at
    128, so admitting it untiled leaves the sweep free to run it over budget with no
    tiled candidate to fall to.
    """
    assert _RowTiled(32768, 2, 227 * 1024)._tile_n_candidates() == [0, 32768, 16384]
