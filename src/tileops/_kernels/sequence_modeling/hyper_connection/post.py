"""The manifold hyper-connection post mix: the block output and the residual streams mixed into
``n`` new streams, in registers."""

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.sequence_modeling.hyper_connection.call_spec import (
    HyperConnectionPostCall,
    HyperConnectionPostFwdInterface,
)

__all__ = ["HyperConnectionPostKernel", "stream_vector"]


def stream_vector(c: int) -> int:
    """Elements of the 16-byte bfloat16 vector a run spans, or the largest power of two
    dividing *c* below it."""
    vector = VECTOR_ACCESS_BYTES // 2
    while c % vector:
        vector //= 2
    return vector


def _load_run(run, column_runs: int, bounded: bool):
    """The run a thread loads for *run*: the row's last run in place of one past it, when
    *bounded*."""
    return T.min(run, column_runs - 1) if bounded else run


@target_cache(maxsize=32)
def _hyper_connection_post_kernel(n: int, c: int, mix_dtype: str, vector: int):
    @tilelang.jit
    def _hyper_connection_post_fwd_func(lanes, runs):
        if lanes < 1 or runs < 1:
            raise ValueError(f"lanes={lanes} and runs={runs} must be positive")
        column_runs = c // vector
        block_runs = lanes * runs
        column_blocks = T.ceildiv(column_runs, block_runs)
        ragged = column_runs % block_runs != 0
        # A row of n mixes loads as one vector where n is a power of two.
        mix_loop = T.vectorized if n & (n - 1) == 0 else T.unroll
        # The token count is a runtime extent, so a new one builds no program.
        rows = T.dynamic("rows")

        @T.macro
        def store(output, token, run, out_local, m):
            for j in T.unroll(n):
                for v in T.vectorized(vector):
                    output[token, j, run * vector + v] = out_local[m, j, v]

        @T.macro
        def mix(x, residual, output, token, first, lane, post_local, comb_local, bounded):
            # Thread lane holds runs first + m * lanes + lane: a warp's accesses to one stream
            # are contiguous for each m. A ragged CTA loads the row's last run in place of one
            # past it and drops its store.
            x_local = T.alloc_local((runs, vector), "bfloat16")
            res_local = T.alloc_local((runs, n, vector), "bfloat16")
            out_local = T.alloc_local((runs, n, vector), "bfloat16")
            for m in T.unroll(runs):
                run = _load_run(first + m * lanes + lane, column_runs, bounded)
                for v in T.vectorized(vector):
                    x_local[m, v] = x[token, run * vector + v]
                for i in T.unroll(n):
                    for v in T.vectorized(vector):
                        res_local[m, i, v] = residual[token, i, run * vector + v]
            # Each output element is one chain in float32: post times x, then each stream's
            # product added in stream order.
            for m in T.unroll(runs):
                for v in T.unroll(vector):
                    value = T.cast(x_local[m, v], "float32")
                    for j in T.unroll(n):
                        acc = T.alloc_var("float32")
                        acc = post_local[j] * value
                        for i in T.unroll(n):
                            acc = acc + comb_local[i * n + j] * T.cast(
                                res_local[m, i, v], "float32"
                            )
                        out_local[m, j, v] = T.cast(acc, "bfloat16")
            for m in T.unroll(runs):
                run = first + m * lanes + lane
                if bounded:
                    if run < column_runs:
                        store(output, token, run, out_local, m)
                else:
                    store(output, token, run, out_local, m)

        @T.prim_func
        def _hyper_connection_post_fwd_main(
            x: T.Tensor[(rows, c), T.bfloat16],
            residual: T.Tensor[(rows, n, c), T.bfloat16],
            post_layer_mix: T.Tensor[(rows, n), mix_dtype],
            comb_res_mix: T.Tensor[(rows, n, n), mix_dtype],
            output: T.Tensor[(rows, n, c), T.bfloat16],
        ):
            # A CTA owns one token's block_runs runs.
            with T.Kernel(rows, column_blocks, threads=lanes) as (token, pid_c):
                lane = T.get_thread_binding()
                # Every thread holds the token's mixes in float32.
                post_local = T.alloc_local((n,), "float32")
                comb_local = T.alloc_local((n * n,), "float32")
                for j in mix_loop(n):
                    post_local[j] = T.cast(post_layer_mix[token, j], "float32")
                for i in T.unroll(n):
                    for j in mix_loop(n):
                        comb_local[i * n + j] = T.cast(comb_res_mix[token, i, j], "float32")
                first = pid_c * block_runs
                # Only the last CTA of a ragged row checks its runs.
                if ragged:
                    if first + block_runs <= column_runs:
                        mix(x, residual, output, token, first, lane, post_local, comb_local, False)
                    else:
                        mix(x, residual, output, token, first, lane, post_local, comb_local, True)
                else:
                    mix(x, residual, output, token, first, lane, post_local, comb_local, False)

        return _hyper_connection_post_fwd_main

    return _hyper_connection_post_fwd_func


class HyperConnectionPostKernel(Kernel, HyperConnectionPostFwdInterface):
    """mHC post mix in registers; one program per regime serves any token count.

    A thread holds runs of one 16-byte column vector across all ``n`` residual streams and
    the block output, and writes the ``n`` mixed streams of those columns; a CTA covers
    ``lanes`` runs side by side of one token, ``runs`` times over. A large call gives a
    thread several runs, a small one a single run over more CTAs.

    Args:
        n: Residual streams.
        c: Columns of a stream.
        mix_dtype: Torch dtype of ``post_layer_mix`` and ``comb_res_mix``.
        config: Optional dict with the lanes of a CTA and the runs a thread holds in each
            launch regime, ``default_config``'s keys.
    """

    supported_archs: list[int] = [80, 89, 90]

    # The launch picks its program from the input size, so tuning has no candidate to time.
    autotune_configs = None

    @classmethod
    def refusal(cls, call: HyperConnectionPostCall) -> str | None:
        # comb_res_mix entries a thread holds, as float32 registers.
        max_comb_entries = 64
        if call.n * call.n > max_comb_entries:
            return (
                f"does not serve n={call.n}: a thread would hold {call.n * call.n} "
                f"comb_res_mix entries, past the register budget of {max_comb_entries}"
            )
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: HyperConnectionPostCall) -> Entry:
        return call, lambda: cls(call.n, call.c, call.mix_dtype)

    def __init__(self, n: int, c: int, mix_dtype: torch.dtype, config: dict | None = None):
        super().__init__()
        self.n = n
        self.c = c
        self.mix_dtype = mix_dtype
        self.vector = stream_vector(c)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # The threads of a CTA, and the runs each holds in the narrow and wide regimes. A
        # run is one 16-byte vector of a column range, across every stream. One run a thread
        # spreads a small call over the most CTAs; four give a thread 320 bytes in flight and
        # a CTA a whole 4096-column token.
        return {"lanes": 128, "narrow_runs": 1, "wide_runs": 4}

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        # The bytes of x, residual and output from which four runs a thread are faster:
        # between 256 and 512 DeepSeek-V4-Flash tokens.
        wide_bytes = 32 << 20
        moved = (x.numel() + 2 * residual.numel()) * residual.element_size()
        regime = "wide" if moved >= wide_bytes else "narrow"
        tokens = x.numel() // self.c
        output = torch.empty_like(residual)
        self.program(self._compile, regime)(
            x.view(tokens, self.c),
            residual.view(tokens, self.n, self.c),
            post_layer_mix.view(tokens, self.n),
            comb_res_mix.view(tokens, self.n, self.n),
            output.view(tokens, self.n, self.c),
        )
        return output

    def _compile(self, cfg: dict, regime: str):
        """The program of config *cfg* for a launch in *regime*."""
        lanes = cfg["lanes"]
        runs = cfg[f"{regime}_runs"]
        # A thread holds at most 128 residual elements, once loaded and once mixed, past
        # which the registers spill; and a CTA no more runs than a stream has.
        runs = max(1, min(runs, 128 // (self.n * self.vector)))
        column_runs = self.c // self.vector
        while runs > 1 and lanes * runs > column_runs:
            runs -= 1
        return _hyper_connection_post_kernel(
            self.n, self.c, self.dtype_to_str(self.mix_dtype), self.vector
        )(lanes, runs)
