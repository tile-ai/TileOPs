import inspect

import pytest

import tileops.kernels.kernel_base as kernel_base
from tileops.kernels.kernel_base import Kernel

pytestmark = pytest.mark.smoke


class _FakeBinder:
    """Mimics TileLang's _JITArgumentBinder fields used by _seed_autotune_binder."""

    def __init__(self, param_names):
        self.param_names = tuple(param_names)
        self.defaults = ()
        self.required_indices = tuple(range(len(param_names)))


class _FakeFunc:
    def __init__(self, param_names):
        self._argument_binder = _FakeBinder(param_names)


class _FakeJit:
    """Stand-in for a @tilelang.jit kernel: exposes .func._argument_binder and .signature."""

    def __init__(self, param_names):
        self.func = _FakeFunc(param_names)
        self.signature = inspect.Signature(
            [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in param_names]
        )


class _TunedKernel:
    config = {"block_m": 64, "threads": 128}


class _KernelWithRequiredTunables(Kernel):
    def __init__(self, param_names=("block_m", "threads")) -> None:
        super().__init__()
        self.kernel = _FakeJit(param_names)

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


def test_autotune_runs_wrapper_with_no_args_and_seeds_binder(monkeypatch):
    """autotune() must NOT pass tunable args (that makes TileLang 0.1.11 skip
    tuning); it seeds the JIT binder defaults from default_config instead."""
    calls: dict[str, object] = {}

    def fake_autotune(**autotune_kwargs):
        def decorate(kernel):
            def wrapped(*args, **kwargs):
                calls["args"] = args
                calls["kwargs"] = kwargs
                return _TunedKernel()

            return wrapped

        return decorate

    monkeypatch.setattr(kernel_base, "autotune", fake_autotune)

    kernel = _KernelWithRequiredTunables()
    kernel.autotune()

    # The autotuned wrapper is called with no tunable arguments ...
    assert calls["args"] == ()
    assert calls["kwargs"] == {}
    # ... and the binder defaults were seeded from default_config (signature-filtered).
    binder = kernel.kernel.func._argument_binder
    assert binder.defaults == (128, 256)  # block_m, threads; tile_n is not a JIT param
    assert binder.required_indices == ()
    assert kernel.config == _TunedKernel.config


def test_seed_binder_uses_passed_config_filtered_to_signature():
    kernel = _KernelWithRequiredTunables()

    Kernel._seed_autotune_binder(kernel.kernel, {"block_m": 1, "threads": 128, "tile_n": 4096})

    binder = kernel.kernel.func._argument_binder
    assert binder.defaults == (1, 128)  # tile_n filtered out (not in signature)
    assert binder.required_indices == ()


def test_seed_binder_supports_common_jit_param_aliases():
    kernel = _KernelWithRequiredTunables(param_names=("threads_arg", "npt_arg"))

    Kernel._seed_autotune_binder(kernel.kernel, {"threads": 128, "num_per_thread": 4})

    binder = kernel.kernel.func._argument_binder
    assert binder.defaults == (128, 4)  # threads_arg<-threads, npt_arg<-num_per_thread
    assert binder.required_indices == ()
