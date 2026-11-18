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

# ``num_stages`` / ``threads`` candidates for the two matmul-heavy sub-kernels.
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
        fit_launches(PIPELINE_CONFIGS, wgmma, *fused),
        fit_launches(OUTPUT_CONFIGS, wgmma, (c, c), (c, v)),
    )


def h_block_v_candidates(dim_v: int) -> tuple[int, ...]:
    """Return the recurrence V-tile widths worth offering at this dimension.

    Admissibility is ``resolve_block_v``'s question — gemm N extent and
    divisibility — so ask it rather than restate it. It is not the whole of
    buildability: whether a width compiles also depends on the thread count and
    the layout the recurrence lands on, known only once it is built, which is
    why the sweep drops a width that fails rather than trusting this. Widths
    resolving to the same tile are one candidate: at ``dim_v == 32`` both 0 and
    32 give a 32-column tile and the same kernel.

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
    product. Declaring it is what makes ``request_tune()`` reach
    ``autotune`` at all.

    Args:
        dim_v: Value dimension.
        chunk_size: Chunk length.
        dim_k: Key dimension.
        wgmma: Whether the target lowers the gemms to WGMMA.

    Raises:
        ValueError: if no V-tile width is admissible at *dim_v*. An empty set
            would read as "tunable, with nothing to try".
    """
    fused_configs, output_configs = chunk_gemm_configs(chunk_size, dim_v, dim_k, wgmma)
    return [
        {
            "fused_num_stages": fused["num_stages"],
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

    The recurrence's V-tile width changes the generated kernel, so TileLang's
    autotuner cannot sweep it as a config key: each width is built and tuned
    separately, and the widths are then compared on the autotuner's own measured
    latency. A launch parameter the autotuner cannot tune keeps its
    ``default_config`` value. A width is kept only if its sweep returned, since
    the autotuner raises when none of a width's candidates compile; when every
    width failed, the failures are raised rather than answered. A failure that
    is not the width's own — a device fault mid-sweep — is indistinguishable
    from here and is attributed to that width, so it only surfaces if no width
    survives.

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
        kernel,
        label: str,
        jit_kernel: Callable,
        configs: Sequence[dict[str, int]],
        warmup: int,
        rep: int,
        untuned: dict[str, int] | None = None,
    ) -> tuple[dict[str, int] | None, float | None]:
        """Sweep one sub-kernel and return its ``(winning config, latency)``.

        *untuned* is the sub-kernel's launch config in the untuned kernel, which
        the sweep times among *configs*; ``None`` where the untuned kernel does not
        build this sub-kernel.

        Either element is ``None`` when the seeded parameters read as already
        tuned: the autotuner then skips the search and JIT-compiles the kernel
        directly, so it is built but never timed. That is not how every candidate
        failing to compile is reported — that raises — so a caller sweeping
        variants must catch, not test.
        """
        # supply_prog=None, not the kernel's: a whole-kernel supplier is written
        # against the forward's inputs, which are not this sub-kernel's. The
        # candidates are copied because the module-level tuples are shared by every
        # sweep in the process, and an autotuner that annotates one in place would
        # otherwise corrupt them for good.
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
            # A tuned config with no latency is not the skip path; it means the
            # attribute this comparison rests on has gone, and every width would
            # silently tie.
            warnings.warn(
                f"{label} tuned to {config} but reported no latency, "
                "so this sweep's result cannot be compared against any other",
                stacklevel=3,
            )
        return config, latency

    def _summarize(exc: Exception) -> str:
        """Return a bounded one-line form of *exc*.

        Which width failed is carried by the label beside this, not by the text:
        tilelang reports every failed sweep with the same sentence. The text is
        kept for everything else raised here, bounded because it may be a whole
        compiler log. Collapsed whole and bounded last: every attempt to bound it
        earlier — to a line, to a prefix — cost the diagnostic on some input, and
        the text is already in memory by the time this is called.
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
        kernel,
        "fused_prepare_compute_w_u",
        fused_builder(*shape),
        fused_configs,
        warmup,
        rep,
        {"num_stages": default["fused_num_stages"], "threads": default["fused_threads"]},
    )

    h_config: dict[str, int] | None = None
    h_block_v: int | None = None
    # Widths whose sweep returned, having compiled: the autotuner raises when
    # no candidate of a width compiles, and JIT-compiles directly when it skips
    # the search. Only a measurement, not membership here, says one is fast.
    compiled: list[int] = []
    failures: list[tuple[str, Exception]] = []
    best_latency = float("inf")
    h_untuned = {"num_stages": default["h_num_stages"], "threads": default["h_threads"]}
    for block_v in candidates:
        label = f"h_recurrence (block_v={block_v})" if block_v else "h_recurrence (no V tiling)"
        try:
            config, latency = _tune_sub_kernel(
                kernel,
                label,
                h_builder(*shape, block_v=block_v),
                _recurrence_configs(
                    kernel.dim_v, block_v, kernel.chunk_size, kernel.dim_k, kernel.wgmma
                ),
                warmup,
                rep,
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
        # Compiled but unmeasured: the untuned width answers if it is one. Its
        # stage count comes from the default. Threads must be resolved for the
        # chosen width, which can have fewer warp tiles than the untuned width.
        preferred = default_h_block_v(kernel.dim_v, kernel.chunk_size)
        h_block_v = preferred if preferred in compiled else compiled[0]

    o_config, _ = _tune_sub_kernel(
        kernel,
        "output_o",
        o_builder(*shape),
        output_configs,
        warmup,
        rep,
        {"threads": default["o_threads"]},
    )

    config = {
        "fused_num_stages": _tuned_value(fused_config, "num_stages", default["fused_num_stages"]),
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
