"""The manifold hyper-connection head: a token's ``n`` residual streams collapsed into one under
weights computed from them, before the LM head."""

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.sequence_modeling.hyper_connection.call_spec import (
    HyperConnectionHeadCall,
    HyperConnectionHeadFwdInterface,
)
from tileops._kernels.sequence_modeling.hyper_connection.post import stream_vector
from tileops._utils import WARP_LANES

__all__ = ["HyperConnectionHeadKernel"]


def _head_weights(n: int, bt: int, threads: int, k_total: int, rms_norm_eps: float, hc_eps: float):
    """A macro totalling each thread's partial dots and square sums over the CTA and writing
    each token's stream weights to ``pre``."""
    warps = threads // WARP_LANES
    sums = n + 1

    @T.macro
    def weights(acc, red, pre, hc_scale, hc_base, tid):
        # acc[t * (n + 1) + m] is token t's partial dot with hc_fn row m, and
        # acc[t * (n + 1) + n] its partial square sum.
        for s in T.unroll(bt * sums):
            total = T.warp_reduce_sum(acc[s])
            if tid % WARP_LANES == 0:
                red[tid // WARP_LANES, s] = total
        T.sync_threads()
        if tid < bt * n:
            t = tid // n
            m = tid % n
            dot = T.alloc_var("float32")
            square = T.alloc_var("float32")
            dot = 0.0
            square = 0.0
            for w in T.unroll(warps):
                dot = dot + red[w, t * sums + m]
                square = square + red[w, t * sums + n]
            rsqrt = T.rsqrt(square / k_total + rms_norm_eps)
            pre[t, m] = T.sigmoid(dot * rsqrt * hc_scale[0] + hc_base[m]) + hc_eps
        T.sync_threads()

    return weights


def _accumulate(n: int, vector: int):
    """A macro adding one token's vector ``x`` of float32 values times each hc_fn row's
    vector ``fv`` to its dots, and its squares to its square sum."""

    @T.macro
    def accumulate(acc, base, x, fv):
        for v in T.unroll(vector):
            acc[base + n] = acc[base + n] + x[v] * x[v]
            for m in T.unroll(n):
                acc[base + m] = acc[base + m] + x[v] * fv[m, v]

    return accumulate


def _store_sum(n: int, vector: int):
    """A macro writing one output vector at *column* of token *row*: the ``[n, vector]``
    *streams* summed under the token's weights ``pre[t]``. A bounded store is dropped where
    *valid* is false."""

    @T.macro
    def store_sum(output, row, column, streams, pre, t, valid, bounded):
        y = T.alloc_local((vector,), T.bfloat16)
        for v in T.unroll(vector):
            total = T.alloc_var("float32")
            total = 0.0
            for i in T.unroll(n):
                total = total + pre[t, i] * T.cast(streams[i, v], "float32")
            y[v] = T.cast(total, T.bfloat16)
        if bounded:
            if valid:
                for v in T.vectorized(vector):
                    output[row, column + v] = y[v]
        else:
            for v in T.vectorized(vector):
                output[row, column + v] = y[v]

    return store_sum


@target_cache(maxsize=32)
def _hyper_connection_head_resident_kernel(
    n: int, c: int, vector: int, rms_norm_eps: float, hc_eps: float
):
    @tilelang.jit
    def _hyper_connection_head_resident_fwd_func(threads, bt):
        column_vectors = c // vector
        per_thread = -(-column_vectors // threads)
        ragged = column_vectors % threads != 0
        weights = _head_weights(n, bt, threads, n * c, rms_norm_eps, hc_eps)
        accumulate = _accumulate(n, vector)
        store_sum = _store_sum(n, vector)
        rows = T.dynamic("rows")

        @T.prim_func
        def _hyper_connection_head_resident_fwd_main(
            hidden_states: T.Tensor[(rows, n, c), T.bfloat16],
            hc_fn: T.Tensor[(n, n * c), T.float32],
            hc_scale: T.Tensor[(1,), T.float32],
            hc_base: T.Tensor[(n,), T.float32],
            output: T.Tensor[(rows, c), T.bfloat16],
        ):
            # A CTA owns bt tokens; thread tid holds column vectors j * threads + tid of
            # every stream of them, loaded once for both the dots and the weighted sum.
            with T.Kernel(T.ceildiv(rows, bt), threads=threads) as pid:
                tid = T.get_thread_binding()
                xs = T.alloc_local((bt, per_thread, n, vector), T.bfloat16)
                acc = T.alloc_local((bt * (n + 1),), "float32")
                fv = T.alloc_local((n, vector), "float32")
                x = T.alloc_local((vector,), "float32")
                xi = T.alloc_local((n, vector), T.bfloat16)
                red = T.alloc_shared((threads // WARP_LANES, bt * (n + 1)), "float32")
                pre = T.alloc_shared((bt, n), "float32")
                first = pid * bt
                # A token past the last loads the last; a column vector past the row loads
                # the row's last, and counts as zero.
                for t in T.unroll(bt):
                    for j in T.unroll(per_thread):
                        for i in T.unroll(n):
                            for v in T.vectorized(vector):
                                xs[t, j, i, v] = hidden_states[
                                    T.min(first + t, rows - 1),
                                    i,
                                    T.min(j * threads + tid, column_vectors - 1) * vector + v,
                                ]
                for s in T.unroll(bt * (n + 1)):
                    acc[s] = 0.0
                for i in T.unroll(n):
                    for j in T.unroll(per_thread):
                        for m in T.unroll(n):
                            for v in T.vectorized(vector):
                                fv[m, v] = hc_fn[
                                    m,
                                    i * c
                                    + T.min(j * threads + tid, column_vectors - 1) * vector
                                    + v,
                                ]
                        for t in T.unroll(bt):
                            for v in T.unroll(vector):
                                x[v] = T.cast(xs[t, j, i, v], "float32")
                                if ragged:
                                    x[v] = T.if_then_else(
                                        j * threads + tid < column_vectors, x[v], 0.0
                                    )
                            accumulate(acc, t * (n + 1), x, fv)
                weights(acc, red, pre, hc_scale, hc_base, tid)
                for t in T.unroll(bt):
                    if first + t < rows:
                        for j in T.unroll(per_thread):
                            for i in T.unroll(n):
                                for v in T.unroll(vector):
                                    xi[i, v] = xs[t, j, i, v]
                            store_sum(
                                output,
                                first + t,
                                (j * threads + tid) * vector,
                                xi,
                                pre,
                                t,
                                j * threads + tid < column_vectors,
                                ragged,
                            )

        return _hyper_connection_head_resident_fwd_main

    return _hyper_connection_head_resident_fwd_func


@target_cache(maxsize=32)
def _hyper_connection_head_streamed_kernel(
    n: int, c: int, vector: int, rms_norm_eps: float, hc_eps: float
):
    @tilelang.jit
    def _hyper_connection_head_streamed_fwd_func(threads, bt):
        k_vectors = n * c // vector
        k_steps = -(-k_vectors // threads)
        k_ragged = k_vectors % threads != 0
        column_vectors = c // vector
        column_steps = -(-column_vectors // threads)
        column_ragged = column_vectors % threads != 0
        weights = _head_weights(n, bt, threads, n * c, rms_norm_eps, hc_eps)
        accumulate = _accumulate(n, vector)
        store_sum = _store_sum(n, vector)
        rows = T.dynamic("rows")

        @T.prim_func
        def _hyper_connection_head_streamed_fwd_main(
            hidden_states: T.Tensor[(rows, n * c), T.bfloat16],
            hc_fn: T.Tensor[(n, n * c), T.float32],
            hc_scale: T.Tensor[(1,), T.float32],
            hc_base: T.Tensor[(n,), T.float32],
            output: T.Tensor[(rows, c), T.bfloat16],
        ):
            # A CTA owns bt tokens and walks their n * c values a vector a thread, each
            # hc_fn vector serving all bt tokens; the weighted sum reads the streams again.
            with T.Kernel(T.ceildiv(rows, bt), threads=threads) as pid:
                tid = T.get_thread_binding()
                acc = T.alloc_local((bt * (n + 1),), "float32")
                fv = T.alloc_local((n, vector), "float32")
                xl = T.alloc_local((bt, vector), T.bfloat16)
                x = T.alloc_local((vector,), "float32")
                xi = T.alloc_local((n, vector), T.bfloat16)
                red = T.alloc_shared((threads // WARP_LANES, bt * (n + 1)), "float32")
                pre = T.alloc_shared((bt, n), "float32")
                first = pid * bt
                for s in T.unroll(bt * (n + 1)):
                    acc[s] = 0.0
                for step in T.serial(k_steps):
                    # A vector past the last loads the last, and counts as zero.
                    k = T.min(step * threads + tid, k_vectors - 1) * vector
                    for m in T.unroll(n):
                        for v in T.vectorized(vector):
                            fv[m, v] = hc_fn[m, k + v]
                    for t in T.unroll(bt):
                        for v in T.vectorized(vector):
                            xl[t, v] = hidden_states[T.min(first + t, rows - 1), k + v]
                    for t in T.unroll(bt):
                        for v in T.unroll(vector):
                            x[v] = T.cast(xl[t, v], "float32")
                            if k_ragged:
                                x[v] = T.if_then_else(step * threads + tid < k_vectors, x[v], 0.0)
                        accumulate(acc, t * (n + 1), x, fv)
                weights(acc, red, pre, hc_scale, hc_base, tid)
                for t in T.unroll(bt):
                    if first + t < rows:
                        for step in T.serial(column_steps):
                            column = T.min(step * threads + tid, column_vectors - 1) * vector
                            for i in T.unroll(n):
                                for v in T.vectorized(vector):
                                    xi[i, v] = hidden_states[first + t, i * c + column + v]
                            store_sum(
                                output,
                                first + t,
                                column,
                                xi,
                                pre,
                                t,
                                step * threads + tid < column_vectors,
                                column_ragged,
                            )

        return _hyper_connection_head_streamed_fwd_main

    return _hyper_connection_head_streamed_fwd_func


class HyperConnectionHeadKernel(Kernel, HyperConnectionHeadFwdInterface):
    """mHC head collapse in one kernel; one program per token-count regime.

    A CTA reduces a few tokens' dots with ``hc_fn`` and square sums, then writes their
    weighted stream sums. With up to two tokens an SM, a thread keeps its columns of every
    stream in registers from the dots to the sum, where they fit in 128 values; otherwise a
    CTA streams one to four tokens, each ``hc_fn`` vector a thread loads serving all of
    them, and the sum reads the streams again, from L2.

    Args:
        n: Residual streams.
        c: Columns of a stream.
        rms_norm_eps: Epsilon added to the mean square.
        hc_eps: Epsilon added to each stream's weight.
        sm_count: SMs of the device, which the program is chosen against.
        config: Optional dict with the threads of a CTA, ``{"threads": ...}``.
    """

    supported_archs: list[int] = [80, 89, 90]

    # The launch picks its program from the token count, so tuning has no candidate to time.
    autotune_configs = None

    @classmethod
    def refusal(cls, call: HyperConnectionHeadCall) -> str | None:
        # A thread holds one hc_fn vector of each stream in registers.
        max_streams = 8
        if call.n > max_streams:
            return (
                f"does not serve n={call.n}: a thread would hold hc_fn vectors of more than "
                f"{max_streams} streams, past the register budget"
            )
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: HyperConnectionHeadCall) -> Entry:
        return call, lambda: cls(call.n, call.c, call.rms_norm_eps, call.hc_eps, call.sm_count)

    def __init__(
        self,
        n: int,
        c: int,
        rms_norm_eps: float,
        hc_eps: float,
        sm_count: int,
        config: dict | None = None,
    ):
        super().__init__()
        self.n = n
        self.c = c
        self.rms_norm_eps = rms_norm_eps
        self.hc_eps = hc_eps
        self.sm_count = sm_count
        self.vector = stream_vector(c)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # A thread reduces its share of a token's n * c values, which a warp shuffle and one
        # shared-memory pass then total.
        return {"threads": 256}

    def _resident_fits(self, bt: int) -> bool:
        """Whether a resident thread holds bt tokens' columns of every stream in registers:
        at most 128 bfloat16 values, past which they spill."""
        per_thread = -(-(self.c // self.vector) // self.config["threads"])
        return bt * self.n * per_thread * self.vector <= 128

    def forward(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        shape = (*hidden_states.shape[:-2], self.c)
        if self.n == 0:
            # No streams: each token's weighted sum is empty.
            return torch.zeros(shape, dtype=torch.bfloat16, device=hidden_states.device)
        tokens = hidden_states.numel() // (self.n * self.c)
        # (kind, tokens a CTA, most tokens an SM): the first program whose bound covers the
        # call serves it. More tokens a CTA share each hc_fn vector among more tokens; fewer
        # keep enough CTAs to fill the SMs, and a resident CTA, all of whose loads are in
        # flight at once, needs fewer of them than a streamed one.
        programs = (
            ("resident", 1, 1),
            ("resident", 2, 2),
            ("streamed", 1, 3),
            ("streamed", 2, 5),
            ("streamed", 4, None),
        )
        kind, bt = next(
            (kind, bt)
            for kind, bt, per_sm in programs
            if (per_sm is None or tokens <= per_sm * self.sm_count)
            and (kind == "streamed" or self._resident_fits(bt))
        )
        output = torch.empty(shape, dtype=torch.bfloat16, device=hidden_states.device)
        streams = hidden_states.view(tokens, self.n, self.c)
        self.program(self._compile, kind, bt)(
            streams if kind == "resident" else streams.view(tokens, self.n * self.c),
            hc_fn,
            hc_scale,
            hc_base,
            output.view(tokens, self.c),
        )
        return output

    def _compile(self, cfg: dict, kind: str, bt: int):
        """The program of config *cfg* collapsing *bt* tokens a CTA, *kind* resident or
        streamed."""
        threads = cfg["threads"]
        # The shuffle reduction takes whole warps, and a thread computes each weight.
        if threads % WARP_LANES or threads < bt * self.n:
            raise ValueError(f"threads={threads} must be whole warps, at least {bt * self.n}")
        build = (
            _hyper_connection_head_resident_kernel
            if kind == "resident"
            else _hyper_connection_head_streamed_kernel
        )
        return build(self.n, self.c, self.vector, self.rms_norm_eps, self.hc_eps)(threads, bt)
