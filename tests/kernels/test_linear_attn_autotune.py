"""Sweep behaviour of the shared delta-rule forward autotune helper.

The real sweep compiles fifteen kernels, so the selection logic is checked here
against stub JIT builders: which candidates reach the autotuner, which sub-kernel
winner lands in which merged key, and what happens when a sub-kernel cannot be
tuned. ``tests/ops/test_deltanet_fwd.py`` covers the same helper on the device.
"""

import inspect

import pytest

from tileops.kernels import linear_attn_autotune as la
from tileops.kernels.deltanet import deltanet_fwd
from tileops.kernels.gated_deltanet import gated_deltanet_fwd

pytestmark = pytest.mark.smoke


class _StubJit:
    """Stands in for a ``@tilelang.jit`` builder with tunable launch params."""

    def __init__(self, name: str, **build_kwargs: int) -> None:
        self.name = name
        self.build_kwargs = build_kwargs
        if name == "o":
            self.signature = inspect.signature(lambda threads: None)
        else:
            self.signature = inspect.signature(lambda num_stages, threads: None)


class _StubTuned:
    def __init__(self, config: dict | None, latency: float | None) -> None:
        self.config = config
        self.latency = latency


class _StubKernel:
    """A delta-rule forward kernel with the sweep's inputs and nothing else."""

    def __init__(self, chunk_size: int = 64, dim_v: int = 64) -> None:
        self.batch, self.head, self.seq_len = 2, 2, 128
        self.chunk_size, self.dim_k, self.dim_v = chunk_size, 64, dim_v
        self.dtype_str = "bfloat16"
        self.sweeps: list[tuple[str, dict, tuple[dict, ...]]] = []
        #: ``(sub-kernel name, block_v) -> (config, latency)`` the stub reports.
        self.results: dict[tuple[str, int], tuple[dict | None, float | None]] = {}

    @property
    def default_config(self) -> dict:
        return {
            "fused_num_stages": 2,
            "fused_threads": 256,
            "h_num_stages": 2,
            "h_threads": 256,
            "h_block_v": 32 if self.chunk_size >= 64 else 0,
            "o_threads": 256,
        }

    def tune_jit_kernel(self, jit_kernel, configs, warmup, rep, seed_config):
        assert seed_config is configs[0], "the sweep must seed from its first candidate"
        self.sweeps.append((jit_kernel.name, jit_kernel.build_kwargs, tuple(configs)))
        key = (jit_kernel.name, jit_kernel.build_kwargs.get("block_v", 0))
        config, latency = self.results.get(key, (configs[-1], 1.0))
        return _StubTuned(config, latency)


def _run(kernel: _StubKernel) -> dict:
    return la.tune_delta_rule_fwd(
        kernel,
        fused_builder=lambda *shape: _StubJit("fused"),
        h_builder=lambda *shape, block_v: _StubJit("h", block_v=block_v),
        o_builder=lambda *shape: _StubJit("o"),
        warmup=1,
        rep=1,
    )


def test_every_candidate_is_swept_and_the_winner_lands_in_its_key() -> None:
    kernel = _StubKernel()
    kernel.results = {
        ("fused", 0): ({"num_stages": 1, "threads": 128}, 3.0),
        ("h", 0): ({"num_stages": 1, "threads": 256}, 2.0),
        ("h", 32): ({"num_stages": 2, "threads": 128}, 5.0),
        ("o", 0): ({"threads": 64}, 1.0),
    }

    config = _run(kernel)

    swept = {(name, build["block_v"] if build else 0): configs
             for name, build, configs in kernel.sweeps}
    assert swept[("fused", 0)] == la.PIPELINE_CONFIGS
    assert swept[("h", 0)] == la.PIPELINE_CONFIGS
    assert swept[("h", 32)] == la.PIPELINE_CONFIGS
    assert swept[("o", 0)] == la.OUTPUT_CONFIGS
    assert config == {
        "fused_num_stages": 1,
        "fused_threads": 128,
        # block_v=0 measured 2.0 ms against 5.0 ms for block_v=32.
        "h_num_stages": 1,
        "h_threads": 256,
        "h_block_v": 0,
        "o_threads": 64,
    }


def test_fastest_v_tile_width_wins() -> None:
    """Widths are compared on latency, not on candidate or default order."""
    kernel = _StubKernel()
    kernel.results = {
        ("h", 0): ({"num_stages": 1, "threads": 128}, 4.0),
        ("h", 32): ({"num_stages": 2, "threads": 256}, 0.5),
    }

    config = _run(kernel)

    assert config["h_block_v"] == 32
    assert (config["h_num_stages"], config["h_threads"]) == (2, 256)


def test_untunable_sub_kernel_falls_back_to_default_config() -> None:
    """A sub-kernel the autotuner cannot tune must not leave the config unbuildable."""
    kernel = _StubKernel()
    kernel.results = {
        ("fused", 0): (None, None),
        ("h", 0): (None, None),
        ("h", 32): (None, None),
        ("o", 0): (None, None),
    }

    config = _run(kernel)

    assert config == kernel.default_config


def test_selected_config_is_always_a_declared_candidate() -> None:
    kernel = _StubKernel()
    declared = la.delta_rule_fwd_autotune_configs(kernel.dim_v, kernel.chunk_size)

    assert _run(kernel) in declared
    kernel.results = {("fused", 0): (None, None), ("h", 0): (None, None),
                      ("h", 32): (None, None), ("o", 0): (None, None)}
    assert _run(kernel) in declared


def test_v_tiling_is_not_offered_below_the_minimum_chunk_size() -> None:
    """A tiled recurrence the kernel does not build must stay out of the sweep."""
    kernel = _StubKernel(chunk_size=32)

    config = _run(kernel)

    assert la.h_block_v_candidates(64, 32) == (0,)
    assert [build.get("block_v") for name, build, _ in kernel.sweeps if name == "h"] == [0]
    assert config["h_block_v"] == 0


def test_v_tile_width_must_divide_dim_v() -> None:
    assert la.h_block_v_candidates(48, 64) == (0,)
    assert la.h_block_v_candidates(64, 64) == (0, 32)


def test_both_forward_kernels_share_one_sweep_implementation() -> None:
    """Neither kernel may carry its own copy of the sweep."""
    assert deltanet_fwd.tune_delta_rule_fwd is la.tune_delta_rule_fwd
    assert gated_deltanet_fwd.tune_delta_rule_fwd is la.tune_delta_rule_fwd
    assert (deltanet_fwd.delta_rule_fwd_autotune_configs
            is gated_deltanet_fwd.delta_rule_fwd_autotune_configs)


@pytest.mark.parametrize(
    "kernel_cls",
    [deltanet_fwd.DeltaNetFwdKernel, gated_deltanet_fwd.GatedDeltaNetFwdKernel],
)
def test_tune_true_reaches_the_sweep(monkeypatch, kernel_cls) -> None:
    """``init_config`` must not fall back for want of declared candidates."""
    calls: list[str] = []
    monkeypatch.setattr(
        kernel_cls, "autotune", lambda self, **kwargs: calls.append(type(self).__name__)
    )

    kernel = kernel_cls(
        batch=1, head=1, seq_len=128, chunk_size=64, dim_k=64, dim_v=64,
        dtype="bfloat16", tune=True,
    )

    assert calls == [kernel_cls.__name__]
    assert kernel.autotune_configs == la.delta_rule_fwd_autotune_configs(64, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
