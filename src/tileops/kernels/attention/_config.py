"""Shared autotune search spaces for the attention kernels."""

import itertools

_BLOCK_M = (32, 64, 128)
_BLOCK_N = (32, 64, 128)
_NUM_STAGES = (1, 2, 3)
_THREADS = (128, 256)


def tile_stage_thread_configs() -> list[dict]:
    """The default GQA search space: block_m x block_n x num_stages x threads."""
    return [
        {"block_m": bm, "block_n": bn, "num_stages": ns, "threads": th}
        for bm, bn, ns, th in itertools.product(
            _BLOCK_M, _BLOCK_N, _NUM_STAGES, _THREADS,
        )
    ]
