import inspect

import pytest

import tileops.kernels.kernel_base as kernel_base
from tileops.kernels.kernel_base import Kernel

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
