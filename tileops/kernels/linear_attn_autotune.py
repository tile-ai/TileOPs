"""Autotune sweep shared by the chunked delta-rule forward kernels.

DeltaNet and Gated DeltaNet forward each run as three sub-kernels — the fused
w/u preparation, the state recurrence, and the output projection — and each
sub-kernel carries its own launch config. The sweep tunes them independently and
merges the winners into the single flat config the wrapped kernel reads.
"""

import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "H_BLOCK_V_WIDTHS",
    "MIN_TILED_CHUNK_SIZE",
    "OUTPUT_CONFIGS",
    "PIPELINE_CONFIGS",
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

#: Smallest chunk size for which a tiled recurrence is offered. Below it the
#: kernels only build the untiled recurrence, so the sweep must not select a
#: tiled width the kernel cannot run.
MIN_TILED_CHUNK_SIZE: int = 64


def h_block_v_candidates(dim_v: int, chunk_size: int) -> Tuple[int, ...]:
    """Return the recurrence V-tile widths buildable at this shape.

    Args:
        dim_v: Value dimension; a tile width must divide it.
        chunk_size: Chunk length; tiled widths need ``MIN_TILED_CHUNK_SIZE``.
    """
    return tuple(
        block_v
        for block_v in H_BLOCK_V_WIDTHS
        if dim_v % (block_v or dim_v) == 0
        and (block_v == 0 or chunk_size >= MIN_TILED_CHUNK_SIZE)
    )


def default_h_block_v(dim_v: int, chunk_size: int) -> int:
    """Return the V-tile width the recurrence runs with when it is not tuned.

    Prefers the narrowest tiled width the shape allows, since tiling is what
    keeps the recurrence's state within shared memory, and falls back to no
    tiling when the shape supports no tiled width. Sharing
    ``h_block_v_candidates`` with the sweep is what keeps the untuned width and
    the tuned candidates from disagreeing: a width that divides ``dim_v`` for
    neither is one the recurrence would build with a V grid too short to cover
    every column.

    Args:
        dim_v: Value dimension.
        chunk_size: Chunk length.
    """
    tiled = [block_v for block_v in h_block_v_candidates(dim_v, chunk_size) if block_v]
    return min(tiled) if tiled else 0


def delta_rule_fwd_autotune_configs(dim_v: int, chunk_size: int) -> List[Dict[str, int]]:
    """Return every merged config the sweep can select at this shape.

    The three sub-kernels are tuned independently, so the reachable set is their
    product. Declaring it is what makes ``init_config(tune=True)`` reach
    ``autotune`` instead of falling back to ``default_config``, and it gives a
    test a set to check the selected config against.

    Args:
        dim_v: Value dimension.
        chunk_size: Chunk length.
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
            h_block_v_candidates(dim_v, chunk_size),
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
    tuned = kernel.tune_jit_kernel(
        jit_kernel, list(configs), warmup=warmup, rep=rep, seed_config=configs[0]
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
    latency. Any sub-kernel the autotuner cannot tune keeps its
    ``default_config`` value, so every key of the returned config is one the
    kernel can build and run.

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
        kernel reads. It is always a member of
        ``delta_rule_fwd_autotune_configs(kernel.dim_v, kernel.chunk_size)``.
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
    # Derived here rather than read from default_config: this is the one key
    # whose valid set depends on the shape, so taking it from the kernel would
    # let an untuned width the sweep never offers escape through the fallback.
    h_block_v = default_h_block_v(kernel.dim_v, kernel.chunk_size)
    best_latency = float("inf")
    for block_v in h_block_v_candidates(kernel.dim_v, kernel.chunk_size):
        label = f"h_recurrence (block_v={block_v})" if block_v else "h_recurrence (no V tiling)"
        config, latency = _tune_sub_kernel(
            kernel,
            label,
            h_builder(*shape, block_v=block_v),
            PIPELINE_CONFIGS,
            warmup,
            rep,
        )
        if config is None or latency is None:
            continue
        if latency < best_latency:
            h_config, h_block_v, best_latency = config, block_v, latency

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
