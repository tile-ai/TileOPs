import inspect

import pytest

import tileops.kernels.kernel_base as kernel_base
from tileops.kernels.kernel_base import Kernel

pytestmark = pytest.mark.smoke


class _FakeJit:
    signature = inspect.signature(lambda block_m, threads: None)


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
