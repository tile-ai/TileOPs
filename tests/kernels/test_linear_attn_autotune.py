"""Sweep behaviour of the shared delta-rule forward autotune helper.

The real sweep compiles fifteen kernels, so the selection logic is checked here
against stub JIT builders: which candidates reach the autotuner, which sub-kernel
winner lands in which merged key, and what happens when a sub-kernel cannot be
tuned. ``tests/ops/test_deltanet_fwd.py`` covers the same helper on the device.
"""

import inspect
import sys

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
        # h_block_v deliberately ignores dim_v divisibility, so a sweep that
        # took its fallback width from here would return an unbuildable one.
        return {
            "fused_num_stages": 2,
            "fused_threads": 256,
            "h_num_stages": 2,
            "h_threads": 256,
            "h_block_v": 32 if self.chunk_size >= 64 else 0,
            "o_threads": 256,
        }

    def tune_jit_kernel(self, jit_kernel, configs, warmup, rep, seed_config, supply_prog):
        assert seed_config is configs[0], "the sweep must seed from its first candidate"
        assert supply_prog is None, "a sub-kernel sweep must not inherit the kernel's supplier"
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
    """An autotuner that returns without a result leaves every key at its default."""
    # dim_v=64 offers two widths, so the untuned preference (32) differs from
    # the first that compiled (0) and the choice between them is visible.
    kernel = _StubKernel(chunk_size=64, dim_v=64)
    kernel.results = {
        ("fused", 0): (None, None),
        ("h", 0): (None, None),
        ("h", 32): (None, None),
        ("o", 0): (None, None),
    }

    config = _run(kernel)

    assert la.h_block_v_candidates(64) == (0, 32), "both widths must compile for this to bite"
    assert config == {**kernel.default_config, "h_block_v": la.default_h_block_v(64, 64)}
    assert config["h_block_v"] == 32


def test_selected_config_is_always_a_declared_candidate() -> None:
    kernel = _StubKernel()
    declared = la.delta_rule_fwd_autotune_configs(kernel.dim_v)

    assert _run(kernel) in declared
    kernel.results = {("fused", 0): (None, None), ("h", 0): (None, None),
                      ("h", 32): (None, None), ("o", 0): (None, None)}
    assert _run(kernel) in declared


def test_short_chunks_still_sweep_the_tiled_width() -> None:
    """The chunk preference steers the untuned default, not the sweep.

    A width the recurrence can build stays measurable at any chunk length; only
    the default declines it below the threshold.
    """
    kernel = _StubKernel(chunk_size=32)

    _run(kernel)

    assert [build.get("block_v") for name, build, _ in kernel.sweeps if name == "h"] == [0, 32]


def test_untunable_fallback_width_is_buildable_when_dim_v_is_indivisible() -> None:
    """The fallback width must come from the candidates, not from default_config.

    At dim_v=48 a tiled width of 32 gives the recurrence a V grid of
    ``48 // 32 == 1`` tile, covering 32 of 48 columns. The width must therefore
    stay out of the returned config even when no candidate could be tuned.
    """
    kernel = _StubKernel(chunk_size=64, dim_v=48)
    kernel.results = {("fused", 0): (None, None), ("h", 0): (None, None),
                      ("o", 0): (None, None)}

    config = _run(kernel)

    assert kernel.default_config["h_block_v"] == 32, "stub must diverge for this to bite"
    assert config["h_block_v"] == 0
    assert config in la.delta_rule_fwd_autotune_configs(48)


def test_a_shape_with_no_buildable_width_is_refused_by_both_entry_points() -> None:
    """The default and the declared set must refuse the same shapes.

    An empty config list reads as "tunable, with nothing to try": ``[]`` is not
    ``None``, so ``init_config`` would skip its warning and walk into the sweep.
    """
    assert la.h_block_v_candidates(8) == ()  # below the minimum gemm N extent
    with pytest.raises(ValueError, match="no buildable"):
        la.default_h_block_v(8, 64)
    with pytest.raises(ValueError, match="no buildable"):
        la.delta_rule_fwd_autotune_configs(8)


def test_widths_resolving_to_one_tile_are_a_single_candidate() -> None:
    """At dim_v=32 both widths give a 32-column tile, so building both is waste."""
    assert la.h_block_v_candidates(32) == (0,)
    assert {config["h_block_v"] for config in la.delta_rule_fwd_autotune_configs(32)} == {0}


def test_a_width_that_fails_to_tune_does_not_sink_the_sweep() -> None:
    """Compilation happens inside the autotuner, which raises when none survives.

    That failure is this width's, so it is dropped with a warning and the other
    width still wins — the sweep aborting would waste a usable width.
    """
    kernel = _StubKernel(chunk_size=64)
    inner = kernel.tune_jit_kernel

    def fail_wide_tile(jit_kernel, *args, **kwargs):
        if jit_kernel.build_kwargs.get("block_v") == 32:
            raise RuntimeError("Auto-tuning failed: No configuration successfully compiled")
        return inner(jit_kernel, *args, **kwargs)

    kernel.tune_jit_kernel = fail_wide_tile

    with pytest.warns(UserWarning, match="unavailable"):
        config = _run(kernel)

    assert config["h_block_v"] == 0
    assert config in la.delta_rule_fwd_autotune_configs(kernel.dim_v)


def test_a_width_that_compiled_is_used_even_when_another_failed() -> None:
    """The untuned preference only counts if that width compiled.

    At chunk 64 the preference is 32. If 32's sweep raises and 0's returns
    without a measurement, 0 still compiled — the autotuner would have raised
    otherwise — so it answers, rather than naming 32 or refusing outright.
    """
    kernel = _StubKernel(chunk_size=64)
    kernel.results = {("h", 0): (None, None)}
    inner = kernel.tune_jit_kernel

    def fail_wide_tile(jit_kernel, *args, **kwargs):
        if jit_kernel.build_kwargs.get("block_v") == 32:
            raise RuntimeError("Auto-tuning failed: No configuration successfully compiled")
        return inner(jit_kernel, *args, **kwargs)

    kernel.tune_jit_kernel = fail_wide_tile

    assert la.default_h_block_v(64, 64) == 32, "the preference must diverge for this to bite"
    with pytest.warns(UserWarning, match="unavailable"):
        config = _run(kernel)

    assert config["h_block_v"] == 0


def test_every_failure_reaches_the_raised_error() -> None:
    """Warnings can be filtered away, so the error carries every cause itself."""
    kernel = _StubKernel(chunk_size=64)
    inner = kernel.tune_jit_kernel

    def fail_each_differently(jit_kernel, *args, **kwargs):
        if jit_kernel.name == "h":
            raise RuntimeError(f"smem overflow at block_v={jit_kernel.build_kwargs['block_v']}")
        return inner(jit_kernel, *args, **kwargs)

    kernel.tune_jit_kernel = fail_each_differently

    with pytest.warns(UserWarning), pytest.raises(RuntimeError) as e:
        _run(kernel)

    assert "block_v=0" in str(e.value)
    assert "block_v=32" in str(e.value)


def test_the_raised_error_bounds_a_long_failure_text() -> None:
    """A compile failure can carry a whole log; the error must stay readable."""
    kernel = _StubKernel(chunk_size=64)
    inner = kernel.tune_jit_kernel
    log = "header line\n" + "x" * 5000

    def fail_with_a_log(jit_kernel, *args, **kwargs):
        if jit_kernel.name == "h":
            raise RuntimeError(log)
        return inner(jit_kernel, *args, **kwargs)

    kernel.tune_jit_kernel = fail_with_a_log

    with pytest.warns(UserWarning), pytest.raises(RuntimeError) as e:
        _run(kernel)

    message = str(e.value)
    widths = len(la.h_block_v_candidates(kernel.dim_v))
    assert "\n" not in message
    # One bounded summary per width, so the bound scales with the candidate set
    # rather than with however many widths happen to be offered today.
    assert len(message) < 300 * widths
    assert "x" * 300 not in message, "the log body must not survive whole"
    assert log in str(e.value.__cause__), "the chained cause keeps the whole text"


def test_no_width_tuning_reports_the_failure_with_its_cause() -> None:
    """A width is not the caller's to choose, so a config naming one is no answer.

    A failure that is not width-specific looks the same from here, so it is not
    lost either: it is what the raised error chains.
    """
    kernel = _StubKernel(chunk_size=64)
    inner = kernel.tune_jit_kernel

    def fail_every_width(jit_kernel, *args, **kwargs):
        if jit_kernel.name == "h":
            raise RuntimeError("device lost")
        return inner(jit_kernel, *args, **kwargs)

    kernel.tune_jit_kernel = fail_every_width

    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="no recurrence V-tile") as e:
        _run(kernel)

    assert isinstance(e.value.__cause__, RuntimeError)
    assert "device lost" in str(e.value.__cause__)


def test_default_h_block_v_prefers_a_tiled_width_and_stays_declared() -> None:
    """Always declared: it is what ``init_config`` falls back to."""
    assert la.default_h_block_v(64, 64) == 32
    assert la.default_h_block_v(96, 64) == 32  # 32 divides 96
    assert la.default_h_block_v(48, 64) == 0  # 32 does not divide 48
    assert la.default_h_block_v(64, 32) == 0  # short chunk prefers no tiling
    for dim_v, chunk_size in ((64, 64), (96, 64), (48, 64), (64, 32)):
        declared = {c["h_block_v"] for c in la.delta_rule_fwd_autotune_configs(dim_v)}
        assert la.default_h_block_v(dim_v, chunk_size) in declared


@pytest.mark.parametrize(
    "kernel_cls",
    [deltanet_fwd.DeltaNetFwdKernel, gated_deltanet_fwd.GatedDeltaNetFwdKernel],
)
def test_default_config_width_is_one_the_kernel_builds(kernel_cls) -> None:
    """Both kernels draw their untuned width from the shared candidates.

    dim_v=48 is the shape that regressed silently: a tiled width of 32 gives
    one tile covering 32 of 48 columns. The width rules themselves are checked
    on the helper; this checks the kernels are wired to them.
    """
    kernel = kernel_cls(
        batch=1, head=1, seq_len=128, chunk_size=64, dim_k=64, dim_v=48,
        dtype="bfloat16",
    )

    assert kernel.config["h_block_v"] in la.h_block_v_candidates(48)
    assert kernel.config in kernel.autotune_configs
    # Nothing else fails if a kernel grows its own copy of the sweep.
    assert sys.modules[kernel_cls.__module__].tune_delta_rule_fwd is la.tune_delta_rule_fwd


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
    assert kernel.autotune_configs == la.delta_rule_fwd_autotune_configs(64)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
