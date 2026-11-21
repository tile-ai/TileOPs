"""Autotune sweep for the chunked delta-rule forward kernel.

The DeltaNet forward runs as three sub-kernels — the fused w/u preparation, the
state recurrence, and the output projection — and each sub-kernel carries its own
launch config. The sweep tunes them independently and
merges the winners into the single flat config the wrapped kernel reads.
"""

import itertools
import warnings
from collections.abc import Callable, Sequence
from typing import Any

from tileops._kernels.autotune import TUNING_FATAL
from tileops._kernels.linear_attention.v_tile import (
    fit_launches,
    min_gemm_n_threads,
    resolve_block_v,
)

__all__ = [
    "OUTPUT_CONFIGS",
    "PIPELINE_CONFIGS",
    "chunk_gemm_configs",
    "default_h_block_v",
    "default_h_threads",
    "delta_rule_fwd_autotune_configs",
    "h_block_v_candidates",
    "tune_delta_rule_fwd",
]

# ``num_stages`` / ``threads`` candidates for the recurrence; the preparation takes their
# ``threads`` alone.
PIPELINE_CONFIGS: tuple[dict[str, int], ...] = tuple(
    {"num_stages": num_stages, "threads": threads}
    for num_stages in (1, 2)
    for threads in (128, 256)
)

# ``threads`` candidates for the output projection, which takes no pipeline depth.
OUTPUT_CONFIGS: tuple[dict[str, int], ...] = tuple(
    {"threads": threads} for threads in (64, 128, 256)
)


def chunk_gemm_configs(
    chunk_size: int, dim_v: int, dim_k: int, wgmma: bool, walk: bool = False
) -> tuple[tuple[dict[str, int], ...], tuple[dict[str, int], ...]]:
    """Preparation and output launches, each narrowed by ``gemm_threads`` over its GEMM outputs.

    Args:
        chunk_size: Chunk length.
        dim_v: Value dimension.
        dim_k: Key dimension.
        wgmma: Whether the target lowers the gemms to WGMMA.
        walk: Whether the preparation also writes the state walk's operands.
    """
    c, k, v = chunk_size, dim_k, dim_v
    fused = [(c, c), (c, k), (c, v)] + ([(k, k), (k, v)] if walk else [])
    return (
        fit_launches([{"threads": c["threads"]} for c in PIPELINE_CONFIGS], wgmma, *fused),
        fit_launches(OUTPUT_CONFIGS, wgmma, (c, c), (c, v)),
    )


def h_block_v_candidates(dim_v: int) -> tuple[int, ...]:
    """Return the recurrence V-tile widths worth offering at this dimension.

    The widths ``resolve_block_v`` admits, one per resolved tile. Whether a width compiles is
    known only once it is built, so the sweep drops a width that fails.

    Args:
        dim_v: Value dimension.
    """
    buildable = []
    resolved = set()
    # 0 means one tile spanning dim_v; 16 is absent because it loses too much precision in
    # fp16.
    for block_v in (0, 32):
        try:
            width = resolve_block_v(dim_v, block_v)
        except ValueError:
            continue
        if width in resolved:
            continue
        resolved.add(width)
        buildable.append(block_v)
    return tuple(buildable)


def _require_h_block_v_candidates(dim_v: int) -> tuple[int, ...]:
    """Return the admissible widths, so both callers refuse the same shapes."""
    candidates = h_block_v_candidates(dim_v)
    if not candidates:
        raise ValueError(
            f"no admissible recurrence V-tile width for dim_v={dim_v}: "
            "none of the candidate widths satisfies resolve_block_v"
        )
    return candidates


def default_h_block_v(dim_v: int, chunk_size: int) -> int:
    """Return the V-tile width the recurrence runs with when it is not tuned.

    The narrowest admissible tiled width, since tiling keeps the recurrence's
    state in shared memory; short chunks take none. Drawn from the candidates
    so the untuned config stays inside the declared set.

    Args:
        dim_v: Value dimension.
        chunk_size: Chunk length.

    Raises:
        ValueError: if no V-tile width is admissible at *dim_v*.
    """
    # Chunk length at or above which the untuned recurrence prefers a tiled width. No
    # builder enforces it, so it steers the default only, never the sweep.
    tiled_default_min_chunk_size = 64
    candidates = _require_h_block_v_candidates(dim_v)
    tiled = [block_v for block_v in candidates if block_v]
    if tiled and chunk_size >= tiled_default_min_chunk_size:
        return min(tiled)
    return 0 if 0 in candidates else min(tiled)


def _recurrence_configs(
    dim_v: int, block_v: int, chunk_size: int, dim_k: int, wgmma: bool
) -> tuple[dict[str, int], ...]:
    """Recurrence launches over a V tile: its chunk-by-width and key-by-width products."""
    width = resolve_block_v(dim_v, block_v)
    declared = [{**c, "threads": min_gemm_n_threads(c["threads"], width)} for c in PIPELINE_CONFIGS]
    return fit_launches(declared, wgmma, (chunk_size, width), (dim_k, width))


def delta_rule_fwd_autotune_configs(
    dim_v: int, chunk_size: int, dim_k: int, wgmma: bool
) -> list[dict[str, int]]:
    """Return every merged config the sweep can select at this value dimension.

    The three sub-kernels are tuned independently, so the reachable set is their
    product; ``request_tune()`` reaches ``autotune`` only when it is declared.

    Args:
        dim_v: Value dimension.
        chunk_size: Chunk length.
        dim_k: Key dimension.
        wgmma: Whether the target lowers the gemms to WGMMA.

    Raises:
        ValueError: if no V-tile width is admissible at *dim_v*.
    """
    fused_configs, output_configs = chunk_gemm_configs(chunk_size, dim_v, dim_k, wgmma)
    return [
        {
            "fused_threads": fused["threads"],
            "h_num_stages": recurrence["num_stages"],
            "h_threads": recurrence["threads"],
            "h_block_v": block_v,
            "o_threads": output["threads"],
        }
        for fused, block_v, output in itertools.product(
            fused_configs,
            _require_h_block_v_candidates(dim_v),
            output_configs,
        )
        for recurrence in _recurrence_configs(dim_v, block_v, chunk_size, dim_k, wgmma)
    ]


def default_h_threads(dim_v: int, block_v: int, chunk_size: int, dim_k: int, wgmma: bool) -> int:
    """Return the widest recurrence launch ``gemm_threads`` admits for this V tile.

    Args:
        dim_v: Value dimension.
        block_v: V-tile width, as ``default_h_block_v`` returns it.
        chunk_size: Chunk length.
        dim_k: Key dimension.
        wgmma: Whether the target lowers the gemms to WGMMA.
    """
    configs = _recurrence_configs(dim_v, block_v, chunk_size, dim_k, wgmma)
    return max(c["threads"] for c in configs)


def tune_delta_rule_fwd(
    kernel,
    fused_builder: Callable[..., Callable],
    h_builder: Callable[..., Callable],
    o_builder: Callable[..., Callable],
    warmup: int = 10,
    rep: int = 10,
) -> dict[str, int]:
    """Return the merged best config for a three-sub-kernel delta-rule forward.

    The recurrence's V-tile width changes the generated kernel, so each width is tuned
    separately and the widths are compared on the autotuner's latency. A launch parameter
    the autotuner cannot tune keeps its ``default_config`` value. A width whose sweep raised
    is dropped; any failure, a device fault included, is attributed to the width it hit.

    Args:
        kernel: The forward kernel being tuned. Supplies the shape, ``wgmma``, the
            default config, and the per-sub-kernel sweep through ``tune_jit_kernel``.
        fused_builder: ``(*shape) -> jit`` for the fused w/u preparation.
        h_builder: ``(*shape, block_v=...) -> jit`` for the state recurrence.
        o_builder: ``(*shape) -> jit`` for the output projection.
        warmup: Warmup iterations per candidate.
        rep: Timed iterations per candidate.

    Returns:
        A config with the flat ``fused_`` / ``h_`` / ``o_`` keys the wrapped
        kernel reads, always a member of
        ``delta_rule_fwd_autotune_configs(kernel.dim_v, kernel.chunk_size, kernel.dim_k,
        kernel.wgmma)``.

    Raises:
        ValueError: if no V-tile width is admissible at this shape.
        RuntimeError: if no width compiled, listing every failure and chaining
            the last, since the sweep then has no width to name.
    """

    def _tune_sub_kernel(
        label: str,
        jit_kernel: Callable,
        configs: Sequence[dict[str, int]],
        untuned: dict[str, int] | None,
    ) -> tuple[dict[str, int] | None, float | None]:
        """Sweep one sub-kernel and return its ``(winning config, latency)``.

        *untuned* is the sub-kernel's launch config in the untuned kernel, which
        the sweep times among *configs*; ``None`` where the untuned kernel does not
        build this sub-kernel.

        Either element is ``None`` when the seeded parameters read as already tuned and
        the autotuner compiles without timing. A sweep in which no candidate compiles raises.
        """
        # supply_prog=None: a whole-kernel supplier is written against the forward's inputs.
        # The candidates are copied because the autotuner may annotate the shared
        # module-level tuples in place.
        candidates = [dict(config) for config in configs]
        tuned = kernel.tune_jit_kernel(
            jit_kernel,
            candidates,
            warmup=warmup,
            rep=rep,
            seed_config=candidates[0] if untuned is None else untuned,
            supply_prog=None,
        )
        config = getattr(tuned, "config", None)
        latency = getattr(tuned, "latency", None)
        if config is not None and latency is None:
            # Without a latency every width would silently tie.
            warnings.warn(
                f"{label} tuned to {config} but reported no latency, "
                "so this sweep's result cannot be compared against any other",
                stacklevel=3,
            )
        return config, latency

    def _summarize(exc: Exception) -> str:
        """Return a bounded one-line form of *exc*.

        The label beside it names the width; the text is bounded because it may be a whole
        compiler log.
        """
        return f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}"

    def _tuned_value(config: dict[str, int] | None, key: str, fallback: Any) -> Any:
        """Return the tuned value for *key*, or *fallback* if the sweep found none."""
        if config is None or config.get(key) is None:
            return fallback
        return config[key]

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
    fused_configs, output_configs = chunk_gemm_configs(
        kernel.chunk_size, kernel.dim_v, kernel.dim_k, kernel.wgmma
    )
    candidates = _require_h_block_v_candidates(kernel.dim_v)  # before any sweep

    fused_config, _ = _tune_sub_kernel(
        "fused_prepare_compute_w_u",
        fused_builder(*shape),
        fused_configs,
        {"threads": default["fused_threads"]},
    )

    h_config: dict[str, int] | None = None
    h_block_v: int | None = None
    # Widths whose sweep returned, having compiled.
    compiled: list[int] = []
    failures: list[tuple[str, Exception]] = []
    best_latency = float("inf")
    h_untuned = {"num_stages": default["h_num_stages"], "threads": default["h_threads"]}
    for block_v in candidates:
        label = f"h_recurrence (block_v={block_v})" if block_v else "h_recurrence (no V tiling)"
        try:
            config, latency = _tune_sub_kernel(
                label,
                h_builder(*shape, block_v=block_v),
                _recurrence_configs(
                    kernel.dim_v, block_v, kernel.chunk_size, kernel.dim_k, kernel.wgmma
                ),
                h_untuned if block_v == default["h_block_v"] else None,
            )
        except TUNING_FATAL:
            raise
        except Exception as exc:  # one width must not sink the rest
            # The builder only wraps the kernel; the compile is inside the
            # autotuner, which raises when no candidate of this width survives.
            failures.append((label, exc))
            warnings.warn(
                f"{label} unavailable, dropping it from the sweep: {_summarize(exc)}",
                stacklevel=2,
            )
            continue
        compiled.append(block_v)
        if config is None or latency is None:
            continue
        if latency < best_latency:
            h_config, h_block_v, best_latency = config, block_v, latency

    if h_block_v is None:
        if not compiled:
            raise RuntimeError(
                f"no recurrence V-tile width tuned for dim_v={kernel.dim_v}: "
                + "; ".join(f"{label}: {_summarize(exc)}" for label, exc in failures)
            ) from failures[-1][1]
        # No width has a latency: take the untuned width if it compiled, at the default stage
        # count, with threads resolved for the chosen width.
        preferred = default_h_block_v(kernel.dim_v, kernel.chunk_size)
        h_block_v = preferred if preferred in compiled else compiled[0]

    o_config, _ = _tune_sub_kernel(
        "output_o", o_builder(*shape), output_configs, {"threads": default["o_threads"]}
    )

    config = {
        "fused_threads": _tuned_value(fused_config, "threads", default["fused_threads"]),
        "h_num_stages": _tuned_value(h_config, "num_stages", default["h_num_stages"]),
        "h_threads": _tuned_value(
            h_config,
            "threads",
            default_h_threads(
                kernel.dim_v, h_block_v, kernel.chunk_size, kernel.dim_k, kernel.wgmma
            ),
        ),
        "h_block_v": h_block_v,
        "o_threads": _tuned_value(o_config, "threads", default["o_threads"]),
    }
    return config
