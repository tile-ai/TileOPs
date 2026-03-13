"""Dropout kernel with Philox PRNG for deterministic random mask generation.

Implements inverted dropout: output = x * mask / (1 - p)
where mask ~ Bernoulli(1 - p), using cuRAND Philox4_32_10 for deterministic
replay (same seed + same thread layout = same mask).

Strategy: explicit_parallel (N elements per thread) with per-thread RNG state.
"""

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["DropoutKernel"]

_FLOAT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


def _make_dropout_kernel(N, dtype, p, seed, threads=256, num_per_thread=8):
    """Build a dropout kernel with Philox PRNG.

    Each thread initializes its own Philox state using (seed, thread_global_id)
    and generates uniform random floats. Elements where rand < p are dropped;
    surviving elements are scaled by 1/(1-p).

    Args:
        N: Total number of elements (flattened).
        dtype: TileLang dtype string.
        p: Drop probability in [0, 1].
        seed: Integer seed for Philox PRNG.
        threads: Threads per block.
        num_per_thread: Elements processed per thread.
    """
    block_size = threads * num_per_thread
    scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                T.rng_init(seed)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    idx = (bx * threads_arg + i) * npt_arg + j
                    rand_val = T.rng_rand_float()
                    keep = rand_val >= T.cast(p, "float32")
                    y[idx] = T.if_then_else(
                        keep,
                        x[idx] * T.cast(scale, x[idx].dtype),
                        T.cast(0.0, x[idx].dtype),
                    )

        return main

    return kernel


class DropoutKernel(Kernel):
    """Dropout kernel with Philox PRNG for deterministic mask generation.

    Applies inverted dropout: output = x * mask / (1 - p) where mask is
    Bernoulli(1 - p). Uses cuRAND Philox4_32_10 generator initialized with
    the given seed for deterministic replay.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype for input (float16, bfloat16, float32).
        p: Drop probability in [0, 1].
        seed: Integer seed for the Philox PRNG.
        config: Optional dict with "threads" and "num_per_thread".
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    SUPPORTED_DTYPES = _FLOAT_DTYPES

    def __init__(self, N_total, dtype, p=0.5, seed=0, config=None, tune=False):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self.N_total = N_total
        self.dtype = dtype
        self.p = p
        self.seed = seed
        self.kernel = self._build_kernel()
        self.init_config(config, tune)

    def _build_kernel(self):
        cfg = self.default_config
        return _make_dropout_kernel(
            self.N_total, self.dtype_str, self.p, self.seed,
            threads=cfg["threads"], num_per_thread=cfg["num_per_thread"],
        )

    @property
    def default_config(self) -> dict:
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, x):
        cfg = self.config
        return self.kernel(cfg["threads"], cfg["num_per_thread"])(x)
