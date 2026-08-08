"""Autotune sweep shared by the chunked delta-rule forward kernels.

DeltaNet and Gated DeltaNet forward each run as three sub-kernels — the fused
w/u preparation, the state recurrence, and the output projection — and each
sub-kernel carries its own launch config. The sweep tunes them independently and
merges the winners into the single flat config the wrapped kernel reads.
"""

import itertools
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tileops.kernels.v_tile import resolve_block_v

__all__ = [
    "H_BLOCK_V_WIDTHS",
    "OUTPUT_CONFIGS",
    "PIPELINE_CONFIGS",
    "TILED_DEFAULT_MIN_CHUNK_SIZE",
    "default_h_block_v",
    "delta_rule_fwd_autotune_configs",
    "h_block_v_candidates",
    "tune_delta_rule_fwd",
]

#: ``num_stages`` / ``threads`` candidates for the two matmul-heavy sub-kernels.
PIPELINE_CONFIGS: Tuple[Dict[str, int], ...] = tuple(
    {"num_stages": num_stages, "threads": threads}
    for num_stages in (1, 2)
    for threads in (128, 256)
)

#: ``threads`` candidates for the output projection, which takes no pipeline depth.
OUTPUT_CONFIGS: Tuple[Dict[str, int], ...] = tuple(
    {"threads": threads} for threads in (64, 128, 256)
)

#: V-tile widths the recurrence may be built with. 0 means one tile spanning dim_v.
#: 16 is absent because it loses too much precision in fp16.
H_BLOCK_V_WIDTHS: Tuple[int, ...] = (0, 32)

#: Chunk length at or above which the untuned recurrence prefers a tiled width.
#: Carried over from the per-kernel defaults this module replaced, which stated
#: no reason for it. No builder enforces it, so it stays a default preference
#: and does not narrow what the sweep may try.
TILED_DEFAULT_MIN_CHUNK_SIZE: int = 64


def h_block_v_candidates(dim_v: int) -> Tuple[int, ...]:
    """Return the recurrence V-tile widths buildable at this value dimension.

    Whether a width is buildable is ``resolve_block_v``'s question — minimum
    gemm N extent and divisibility — so ask it rather than restate its rules
    here, which is what keeps the candidate set from drifting from what the
    recurrence will accept. Widths resolving to the same tile are one
    candidate: at ``dim_v == 32`` both 0 and 32 give a 32-column tile and the
    same generated kernel, so offering both would build and time it twice.

    Args:
        dim_v: Value dimension.
    """
    buildable = []
    resolved = set()
    for block_v in H_BLOCK_V_WIDTHS:
        try:
            width = resolve_block_v(dim_v, block_v)
        except ValueError:
            continue
        if width in resolved:
            continue
        resolved.add(width)
        buildable.append(block_v)
    return tuple(buildable)


def _require_h_block_v_candidates(dim_v: int) -> Tuple[int, ...]:
    """Return the buildable widths, so both callers refuse the same shapes."""
    candidates = h_block_v_candidates(dim_v)
    if not candidates:
        raise ValueError(
            f"no buildable recurrence V-tile width for dim_v={dim_v}; "
            f"widths are {H_BLOCK_V_WIDTHS} and none satisfies resolve_block_v")
    return candidates


def default_h_block_v(dim_v: int, chunk_size: int) -> int:
    """Return the V-tile width the recurrence runs with when it is not tuned.

    The narrowest buildable tiled width, since tiling keeps the recurrence's
    state in shared memory; short chunks take none. Drawn from the candidates
    so the untuned config stays inside the declared set.

    Args:
        dim_v: Value dimension.
        chunk_size: Chunk length.

    Raises:
        ValueError: if no V-tile width is buildable at *dim_v*.
    """
    candidates = _require_h_block_v_candidates(dim_v)
    tiled = [block_v for block_v in candidates if block_v]
    if tiled and chunk_size >= TILED_DEFAULT_MIN_CHUNK_SIZE:
        return min(tiled)
    return 0 if 0 in candidates else min(tiled)


def delta_rule_fwd_autotune_configs(dim_v: int) -> List[Dict[str, int]]:
    """Return every merged config the sweep can select at this value dimension.

    The three sub-kernels are tuned independently, so the reachable set is their
    product. Declaring it is what makes ``init_config(tune=True)`` reach
    ``autotune`` instead of falling back to ``default_config``, and it gives a
    test a set to check the selected config against. Chunk length does not enter:
    it steers the untuned default, not what the sweep may build.

    Args:
        dim_v: Value dimension.

    Raises:
        ValueError: if no V-tile width is buildable at *dim_v*, so the shape is
            refused here rather than declaring an empty set that reads as
            "tunable, with nothing to try".
    """
    return [
        {
            "fused_num_stages": fused["num_stages"],
            "fused_threads": fused["threads"],
            "h_num_stages": recurrence["num_stages"],
            "h_threads": recurrence["threads"],
            "h_block_v": block_v,
            "o_threads": output["threads"],
        }
        for fused, recurrence, block_v, output in itertools.product(
            PIPELINE_CONFIGS,
            PIPELINE_CONFIGS,
            _require_h_block_v_candidates(dim_v),
            OUTPUT_CONFIGS,
        )
    ]


def _tune_sub_kernel(
    kernel,
    label: str,
    jit_kernel: Callable,
    configs: Sequence[Dict[str, int]],
    warmup: int,
    rep: int,
) -> Tuple[Optional[Dict[str, int]], Optional[float]]:
    """Sweep one sub-kernel and return its ``(winning config, latency)``.

    Either element is ``None`` when TileLang's autotuner produced no result,
    which is how a sub-kernel whose candidates all fail to build reports back.
    """
    print(f"Autotuning {label} ({len(configs)} configs)...")
    # supply_prog=None, not the kernel's: a whole-kernel supplier is written
    # against the forward's inputs, which are not this sub-kernel's.
    tuned = kernel.tune_jit_kernel(
        jit_kernel,
        list(configs),
        warmup=warmup,
        rep=rep,
        seed_config=configs[0],
        supply_prog=None,
    )
    config = getattr(tuned, "config", None)
    latency = getattr(tuned, "latency", None)
    print(f"  Best: {config}")
    return config, latency


def _tuned_value(config: Optional[Dict[str, int]], key: str, fallback: Any) -> Any:
    """Return the tuned value for *key*, or *fallback* if the sweep found none."""
    if config is None or config.get(key) is None:
        return fallback
    return config[key]


def tune_delta_rule_fwd(
    kernel,
    fused_builder: Callable[..., Callable],
    h_builder: Callable[..., Callable],
    o_builder: Callable[..., Callable],
    warmup: int = 10,
    rep: int = 10,
) -> Dict[str, int]:
    """Return the merged best config for a three-sub-kernel delta-rule forward.

    The recurrence's V-tile width changes the generated kernel, so TileLang's
    autotuner cannot sweep it as a config key: each width is built and tuned
    separately, and the widths are then compared on the autotuner's own measured
    latency. A launch parameter the autotuner cannot tune keeps its
    ``default_config`` value. The V-tile width does not: it decides which kernel
    exists, so it is taken from the widths that produced a measurement, and from
    the first width offered when none did.

    Args:
        kernel: The forward kernel being tuned. Supplies the shape, the default
            config, and the per-sub-kernel sweep through ``tune_jit_kernel``.
        fused_builder: ``(*shape) -> jit`` for the fused w/u preparation.
        h_builder: ``(*shape, block_v=...) -> jit`` for the state recurrence.
        o_builder: ``(*shape) -> jit`` for the output projection.
        warmup: Warmup iterations per candidate.
        rep: Timed iterations per candidate.

    Returns:
        A config with the flat ``fused_`` / ``h_`` / ``o_`` keys the wrapped
        kernel reads, always a member of
        ``delta_rule_fwd_autotune_configs(kernel.dim_v)``.

    Raises:
        ValueError: if no V-tile width is buildable at this shape.
        RuntimeError: if every width's builder raised, since the sweep then has
            no width to name and the caller does not choose one.
    """
    shape = (
        kernel.batch,
        kernel.head,
        kernel.seq_len,
        kernel.chunk_size,
        kernel.dim_k,
        kernel.dim_v,
        kernel.dtype_str,
    )
    default = kernel.default_config
    print(f"Start autotuning {kernel.__class__.__name__}...")

    fused_config, _ = _tune_sub_kernel(
        kernel,
        "fused_prepare_compute_w_u",
        fused_builder(*shape),
        PIPELINE_CONFIGS,
        warmup,
        rep,
    )

    h_config: Optional[Dict[str, int]] = None
    h_block_v: Optional[int] = None
    #: Widths whose builder returned. Says nothing about the kernel compiling.
    offered: List[int] = []
    #: Widths the autotuner returned a measurement for — the only evidence
    #: that this width compiles and runs, since per-candidate build failures
    #: reach us as a missing measurement rather than an exception.
    measured: List[int] = []
    builder_error: Optional[Exception] = None
    best_latency = float("inf")
    for block_v in _require_h_block_v_candidates(kernel.dim_v):
        label = f"h_recurrence (block_v={block_v})" if block_v else "h_recurrence (no V tiling)"
        try:
            jit_kernel = h_builder(*shape, block_v=block_v)
        except Exception as exc:  # noqa: BLE001 - one width must not sink the rest
            # Only the builder call is guarded, and only this width is dropped.
            # Tuning is not: a failure there is the autotuner's or the device's,
            # not this width's, and reporting it as a dropped width would hide
            # it behind a config that looks tuned.
            builder_error = exc
            warnings.warn(  # noqa: B028
                f"{label} unavailable, dropping it from the sweep: "
                f"{type(exc).__name__}: {exc}")
            continue
        offered.append(block_v)
        config, latency = _tune_sub_kernel(
            kernel, label, jit_kernel, PIPELINE_CONFIGS, warmup, rep)
        if config is None or latency is None:
            continue
        measured.append(block_v)
        if latency < best_latency:
            h_config, h_block_v, best_latency = config, block_v, latency

    if not offered:
        raise RuntimeError(
            f"no recurrence V-tile width could be built for dim_v={kernel.dim_v}"
        ) from builder_error
    if h_block_v is None:
        # Nothing was measured, so no width is known to run. The untuned
        # preference is taken only on that evidence; without it the first
        # candidate is the safer answer, since candidate order puts the
        # untiled width first and that one builds whenever any width does.
        preferred = default_h_block_v(kernel.dim_v, kernel.chunk_size)
        h_block_v = preferred if preferred in measured else offered[0]

    o_config, _ = _tune_sub_kernel(
        kernel, "output_o", o_builder(*shape), OUTPUT_CONFIGS, warmup, rep
    )

    config = {
        "fused_num_stages": _tuned_value(
            fused_config, "num_stages", default["fused_num_stages"]),
        "fused_threads": _tuned_value(fused_config, "threads", default["fused_threads"]),
        "h_num_stages": _tuned_value(h_config, "num_stages", default["h_num_stages"]),
        "h_threads": _tuned_value(h_config, "threads", default["h_threads"]),
        "h_block_v": h_block_v,
        "o_threads": _tuned_value(o_config, "threads", default["o_threads"]),
    }
    print(f"{kernel.__class__.__name__} autotuned config: {config}")
    return config
