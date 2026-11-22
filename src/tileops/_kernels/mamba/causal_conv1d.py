"""Depthwise causal short convolution of a Mamba-2 block: prefill over packed sequences, the
one-token decode step and draft verification, all with fused SiLU and a per-sequence
convolution state.

For a sequence occupying packed tokens ``[a, e)``, with ``h`` its ``W - 1`` preceding tokens
(``conv_state[s]`` where ``has_initial_state[s]``, zeros otherwise)::

    xp[c, :]       = concat(h[c, :], x[a:e, c])
    y[a + i, c]    = act(bias[c] + sum_k weight[c, k] * xp[c, i + k])
    conv_state[s] <- xp[c, e - a : e - a + W - 1]       (its last W - 1 tokens)

An empty sequence (``e == a``) keeps its state. Draft verification reads ``h`` from
column ``num_accepted_tokens[s] - 1`` of an ``L``-column state and keeps the last
``(W - 1) * dilation + q - 1`` tokens of history and its ``q`` tokens in the state's first
columns.
The products accumulate in float32 and the output rounds once to the input dtype.
"""

import math

import tilelang
import tilelang.language as T
import torch
from tvm import DataType

from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.constants import LOG2E
from tileops._kernels.elementwise._division import approx_fdiv
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.mamba.call_spec import (
    CausalConv1dCall,
    CausalConv1dFwdInterface,
    CausalConv1dUpdateCall,
    CausalConv1dUpdateFwdInterface,
)
from tileops._kernels.state_pool import state_row
from tileops._utils import WARP_LANES

__all__ = ["CausalConv1dKernel", "CausalConv1dUpdateKernel", "CausalConv1dUpdateRollbackKernel"]


def _activate(acc, silu: bool):
    """``acc / (1 + 2^(-acc log2 e))`` in float32 when *silu*.

    The divisor is at least one, inside ``approx_fdiv``'s range; past ``2**126`` its zero
    quotient is SiLU's limit.
    """
    if not silu:
        return acc
    return approx_fdiv(acc, T.float32(1.0) + T.exp2(-acc * T.float32(LOG2E)))


def _dense_state_refusal(call: "CausalConv1dCall | CausalConv1dUpdateCall") -> str | None:
    """Why a kernel holding one ``W - 1``-token state a sequence, in batch order or in the
    slot of a pool, and filters in the activation dtype does not serve *call*, or ``None``
    when it does."""
    if call.weight_dtype != call.dtype:
        return f"the filters are read in the activation dtype; a {call.weight_dtype} weight is not"
    if call.dilation != 1:
        return f"the filter taps are adjacent tokens; dilation {call.dilation} is not served"
    if call.state_len != call.width - 1:
        return f"a state is {call.width - 1} tokens wide; {call.state_len} columns are not served"
    return None


def _vector(dim: int, dtype: torch.dtype) -> int:
    """Channels a thread holds: the widest of one 16-byte vector, 4, 2 or 1 that divides *dim*."""
    widest = 16 // torch.empty((), dtype=dtype).element_size()
    return next(v for v in (widest, 4, 2, 1) if v <= widest and dim % v == 0)


@target_cache(maxsize=64)
def _causal_conv1d_kernel(
    dim: int,
    width: int,
    dtype: str,
    has_bias: bool,
    has_init: bool,
    silu: bool,
    pooled: bool,
):
    hist = width - 1

    # exp2 as one MUFU op, denormals flushed.
    @tilelang.jit(out_idx=[], compile_flags=[*CUDA_COMPILE_FLAGS, "--use_fast_math"])
    def _func(vec: int, warps: int, seg: int, steps: int, stages: int):
        total = T.dynamic("total")
        batch = T.dynamic("batch")
        pool = T.dynamic("pool")
        block_d = WARP_LANES * vec
        chunk = seg * steps
        # Keeps every index into part a constant of the unrolled step.
        if seg % width:
            raise ValueError(f"seg {seg} is not a multiple of the kernel width {width}")
        # cp.async moves 4, 8 or 16 bytes.
        async_copy = vec * DataType(dtype).bits >= 32

        @T.macro
        def stage(x, ring, w, buf, t0, lane, c0, ce):
            """Issue the copies of this lane's channels of tokens [t0, t0 + seg) into ring slot buf."""
            for i in T.unroll(seg):
                j = t0 + i
                live = T.And(c0 < dim, j < ce)
                if async_copy:
                    # A slot past the chunk zero-fills; its address stays in bounds.
                    T.ptx_cp_async(
                        T.access_ptr(ring[w, buf, i, lane * vec], "w", vec),
                        T.access_ptr(
                            x[T.if_then_else(live, j, 0), T.if_then_else(live, c0, 0)], "r", vec
                        ),
                        vec,
                        live,
                    )
                else:
                    for v in T.unroll(vec):
                        ring[w, buf, i, lane * vec + v] = T.if_then_else(
                            live,
                            x[T.if_then_else(live, j, 0), T.if_then_else(live, c0, 0) + v],
                            T.cast(0, dtype),
                        )
            if async_copy:
                T.ptx_commit_group()

        @T.macro
        def restart(part, hold, wf, bf, n):
            """Restart the sums of outputs n .. n + W - 2 from the W-1 tokens before n in hold.

            Output n + d reads hold[d + k] through tap k for k < W - 1 - d; the taps on token n
            and later arrive with those tokens.
            """
            for d in T.unroll(hist):
                for v in T.unroll(vec):
                    part[((n + d) % width) * vec + v] = bf[v]
                for k in T.unroll(hist - d):
                    for v in T.unroll(vec):
                        part[((n + d) % width) * vec + v] += wf[v * width + k] * T.cast(
                            hold[(d + k) * vec + v], "float32"
                        )

        @T.macro
        def arrive(part, prev, wf, bf, out, n):
            """Add the n-th token to the W outputs it feeds, then finish output n into out.

            Output n + m takes the token through tap W - 1 - m.
            """
            for v in T.unroll(vec):
                xv = T.cast(prev[v], "float32")
                for m in T.unroll(width):
                    part[((n + m) % width) * vec + v] += wf[v * width + hist - m] * xv
            for v in T.unroll(vec):
                out[v] = T.cast(_activate(part[(n % width) * vec + v], silu), dtype)
            for v in T.unroll(vec):
                part[(n % width) * vec + v] = bf[v]

        @T.macro
        def walk(x, weight, bias, conv_state, cu_seqlens, has_initial_state, slots, y):
            # A warp walks `chunk` packed tokens of block_d channels; each lane reads only the
            # ring slots it copied, so warps need no barrier.
            #
            # Only the warp whose chunk holds a sequence's first token reads or writes that
            # sequence's state: it computes the sequence's first W-1 outputs, even past its
            # chunk's end, and writes the final state last.
            with T.Kernel(
                T.ceildiv(dim, block_d),
                T.ceildiv(T.ceildiv(total, chunk), warps),
                threads=WARP_LANES * warps,
            ) as (bd, bc):
                ring = T.alloc_shared((warps, stages, seg, block_d), dtype)
                tx = T.get_thread_binding()
                lane = tx % WARP_LANES
                w = tx // WARP_LANES
                c0 = bd * block_d + lane * vec
                cs = (bc * warps + w) * chunk
                ce = T.min(cs + chunk, total)
                prev = T.alloc_local((vec,), dtype)
                # part[(n % W) * vec + v]: the running sum of the chunk's n-th output.
                part = T.alloc_local((width * vec,), "float32")
                hold = T.alloc_local((hist * vec,), dtype)
                wgt = T.alloc_local((vec * width,), dtype)
                wf = T.alloc_local((vec * width,), "float32")
                bia = T.alloc_local((vec,), dtype)
                bf = T.alloc_local((vec,), "float32")
                out = T.alloc_local((vec,), dtype)
                lo = T.alloc_var("int32")
                hi = T.alloc_var("int32")
                seq = T.alloc_var("int32")
                start = T.alloc_var("int32")
                end = T.alloc_var("int32")
                skip = T.alloc_var("int32")
                first = T.alloc_var("int32")
                if cs < total:
                    for q in T.unroll(stages - 1):
                        if q < steps:
                            stage(x, ring, w, q, cs + q * seg, lane, c0, ce)
                        elif async_copy:
                            T.ptx_commit_group()
                    if c0 < dim:
                        # The thread's vec x W weights are contiguous: wgt[v * width + k].
                        for q in T.unroll(width):
                            for v in T.vectorized(vec):
                                wgt[q * vec + v] = weight[c0 * width + q * vec + v]
                        for e in T.unroll(vec * width):
                            wf[e] = T.cast(wgt[e], "float32")
                        for v in T.unroll(vec):
                            bf[v] = T.float32(0)
                        if has_bias:
                            for v in T.vectorized(vec):
                                bia[v] = bias[c0 + v]
                            for v in T.unroll(vec):
                                bf[v] = T.cast(bia[v], "float32")
                    # The W-1 tokens before cs, read before the search to overlap the two.
                    if c0 < dim:
                        for k in T.unroll(hist):
                            j = cs - hist + k
                            for v in T.unroll(vec):
                                hold[k * vec + v] = T.cast(0, dtype)
                            if j >= 0:
                                for v in T.vectorized(vec):
                                    hold[k * vec + v] = x[j, c0 + v]
                    # The sequence holding cs: the last one starting at or before it.
                    lo = 0
                    hi = batch - 1
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if cu_seqlens[mid] <= cs:
                            lo = mid
                        else:
                            hi = mid - 1
                    seq = lo
                    start = cu_seqlens[seq]
                    end = cu_seqlens[seq + 1]
                    # The first W-1 outputs of a sequence starting in the chunk before are that
                    # chunk's.
                    skip = T.if_then_else(T.And(start < cs, cs - start < hist), start + hist, cs)
                    # The first sequence starting in [cs, ce), if any.
                    first = T.if_then_else(start == cs, seq, seq + 1)
                    # Tokens before the sequence's start are its state if it starts on cs, and
                    # otherwise feed only outputs the chunk before computes.
                    if c0 < dim:
                        for k in T.unroll(hist):
                            if cs - hist + k < start:
                                for v in T.unroll(vec):
                                    hold[k * vec + v] = T.cast(0, dtype)
                                if has_init and start == cs and has_initial_state[seq] != 0:
                                    for v in T.unroll(vec):
                                        hold[k * vec + v] = conv_state[
                                            state_row(slots, seq), c0 + v, k
                                        ]
                        restart(part, hold, wf, bf, 0)
                        for v in T.unroll(vec):
                            part[hist * vec + v] = bf[v]
                    for st in T.serial(steps):
                        t0 = cs + st * seg
                        buf = st % stages
                        if st + stages - 1 < steps:
                            stage(
                                x,
                                ring,
                                w,
                                (st + stages - 1) % stages,
                                t0 + (stages - 1) * seg,
                                lane,
                                c0,
                                ce,
                            )
                        elif async_copy:
                            T.ptx_commit_group()
                        if async_copy:
                            T.ptx_wait_group(stages - 1)
                        if c0 < dim:
                            for i in T.unroll(seg):
                                t = t0 + i
                                if t < ce:
                                    # A sequence starts at t; empty ones are skipped.
                                    if t >= end:
                                        while end <= t:
                                            seq = seq + 1
                                            start = cu_seqlens[seq]
                                            end = cu_seqlens[seq + 1]
                                        for e in T.unroll(hist * vec):
                                            hold[e] = T.cast(0, dtype)
                                        if has_init and has_initial_state[seq] != 0:
                                            for k in T.unroll(hist):
                                                for v in T.unroll(vec):
                                                    hold[k * vec + v] = conv_state[
                                                        state_row(slots, seq), c0 + v, k
                                                    ]
                                        restart(part, hold, wf, bf, i)
                                    for v in T.vectorized(vec):
                                        prev[v] = ring[w, buf, i, lane * vec + v]
                                    arrive(part, prev, wf, bf, out, i)
                                    if t >= skip:
                                        for v in T.vectorized(vec):
                                            y[t, c0 + v] = out[v]
                    if c0 < dim:
                        # The first W-1 outputs of a sequence starting in the chunk that run
                        # past its end.
                        if T.And(start >= cs, start + hist > ce):
                            for e in T.unroll(hist - 1):
                                t = ce + e
                                if T.And(t < start + hist, t < end):
                                    for v in T.vectorized(vec):
                                        prev[v] = x[t, c0 + v]
                                    arrive(part, prev, wf, bf, out, chunk + e)
                                    for v in T.vectorized(vec):
                                        y[t, c0 + v] = out[v]
                        # Final states of the sequences starting in [cs, ce). An empty sequence
                        # keeps its state.
                        seq = first
                        while seq < batch and cu_seqlens[seq] < ce:
                            start = cu_seqlens[seq]
                            end = cu_seqlens[seq + 1]
                            if end > start:
                                for p in T.unroll(hist):
                                    j = end - hist + p
                                    for v in T.unroll(vec):
                                        hold[p * vec + v] = T.cast(0, dtype)
                                    if j >= start:
                                        for v in T.unroll(vec):
                                            hold[p * vec + v] = x[j, c0 + v]
                                    elif has_init and has_initial_state[seq] != 0:
                                        for v in T.unroll(vec):
                                            hold[p * vec + v] = conv_state[
                                                state_row(slots, seq), c0 + v, end - start + p
                                            ]
                                for p in T.unroll(hist):
                                    for v in T.unroll(vec):
                                        conv_state[state_row(slots, seq), c0 + v, p] = hold[
                                            p * vec + v
                                        ]
                            seq = seq + 1

        @T.prim_func
        def main(
            x: T.Tensor((total, dim), dtype),
            weight: T.Tensor((dim * width,), dtype),
            bias: T.Tensor((dim,), dtype),
            conv_state: T.Tensor((pool if pooled else batch, dim, hist), dtype),
            cu_seqlens: T.Tensor((batch + 1,), "int32"),
            has_initial_state: T.Tensor((batch,), "uint8"),
            slots: T.Tensor((batch,) if pooled else (1,), "int32"),
            y: T.Tensor((total, dim), dtype),
        ):
            walk(
                x,
                weight,
                bias,
                conv_state,
                cu_seqlens,
                has_initial_state,
                slots if pooled else None,
                y,
            )

        return main

    return _func


class CausalConv1dKernel(Kernel, CausalConv1dFwdInterface):
    """Prefill over packed sequences, a warp a run of tokens of one channel strip.

    A warp walks ``seg * steps`` packed tokens of ``32 * vec`` channels through a
    ``stages``-slot cp.async ring in shared memory. The sequence count and lengths are runtime
    values: one build serves every batch.

    Args:
        dim: Channels, ``D``.
        width: Kernel width ``W``, 2 to 4.
        dtype: float16 or bfloat16.
        has_bias: Whether a bias is added.
        has_init: Whether ``has_initial_state`` is read; without it every sequence starts
            from zeros.
        silu: Whether SiLU follows the convolution.
        pooled: Whether each sequence's state is the pool row its slot names.
        config: Optional ``{"vec", "warps", "seg", "steps", "stages"}`` override.
        device_index: The device the kernel runs on.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def refusal(cls, call: CausalConv1dCall) -> str | None:
        return _dense_state_refusal(call) or super().refusal(call)

    @classmethod
    def entry_for(cls, call: CausalConv1dCall) -> Entry:
        args = (call.dim, call.width, call.dtype, call.has_bias, call.has_init, call.silu)
        index = call.device_index
        return (*args, call.pooled, index), lambda: cls(
            *args, device_index=index, pooled=call.pooled
        )

    def __init__(
        self,
        dim: int,
        width: int,
        dtype: torch.dtype,
        has_bias: bool,
        has_init: bool,
        silu: bool,
        config: dict | None = None,
        device_index: int | None = None,
        *,
        pooled: bool = False,
    ) -> None:
        super().__init__(device_index=device_index)
        self.dim, self.width, self.dtype = dim, width, dtype
        self.pooled = pooled
        self.kernel = _causal_conv1d_kernel(
            dim, width, self.dtype_str, has_bias, has_init, silu, pooled
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {
            "vec": _vector(self.dim, self.dtype),
            "warps": 4,
            "seg": self.width * (8 // self.width),
            "steps": 8,
            "stages": 2,
        }

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(**cfg)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        bias: torch.Tensor | None = None,
        has_initial_state: torch.Tensor | None = None,
        cache_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = torch.empty_like(x)
        batch = cu_seqlens.shape[0] - 1
        # An absent bias, has_initial_state or cache_indices is never read; a view of its
        # shape fills the slot.
        if has_initial_state is None:
            flags = cu_seqlens.view(torch.uint8)[:batch]
        else:
            flags = has_initial_state.view(torch.uint8)
        self.program(self._compile)(
            x,
            weight.view(-1),
            bias if bias is not None else x[0],
            conv_state,
            cu_seqlens,
            flags,
            cache_indices if self.pooled else cu_seqlens[:1],
            y,
        )
        return y


@target_cache(maxsize=64)
def _causal_conv1d_update_kernel(
    dim: int, width: int, dtype: str, has_bias: bool, silu: bool, vec: int, pooled: bool
):
    hist = width - 1

    @tilelang.jit(out_idx=[])
    def _func(threads: int):
        batch = T.dynamic("batch")
        pool = T.dynamic("pool")
        block_d = threads * vec

        @T.macro
        def step(x, weight, bias, conv_state, slots, y):
            # A CTA is block_d channels of one sequence; a thread holds vec of them, whose
            # vec x W weights and vec x (W-1) state elements are each contiguous.
            with T.Kernel(T.ceildiv(dim, block_d), batch, threads=threads) as (bd, b):
                c0 = bd * block_d + T.get_thread_binding() * vec
                # A state is indexed by its row, so a pool slot stays out of the flat offset and
                # the accesses stay 16-byte vectors.
                row = c0 * hist
                st = T.alloc_local((vec * hist,), dtype)
                wraw = T.alloc_local((vec * width,), dtype)
                xv = T.alloc_local((vec,), dtype)
                bv = T.alloc_local((vec,), dtype)
                acc = T.alloc_local((vec,), "float32")
                out = T.alloc_local((vec,), dtype)
                if c0 < dim:
                    for q in T.unroll(hist):
                        for v in T.vectorized(vec):
                            st[q * vec + v] = conv_state[state_row(slots, b), row + q * vec + v]
                    for q in T.unroll(width):
                        for v in T.vectorized(vec):
                            wraw[q * vec + v] = weight[c0 * width + q * vec + v]
                    for v in T.vectorized(vec):
                        xv[v] = x[b, c0 + v]
                    if has_bias:
                        for v in T.vectorized(vec):
                            bv[v] = bias[c0 + v]
                    for v in T.unroll(vec):
                        acc[v] = T.cast(bv[v], "float32") if has_bias else T.float32(0)
                        for k in T.unroll(hist):
                            acc[v] += T.cast(wraw[v * width + k], "float32") * T.cast(
                                st[v * hist + k], "float32"
                            )
                        acc[v] += T.cast(wraw[v * width + hist], "float32") * T.cast(
                            xv[v], "float32"
                        )
                        out[v] = T.cast(_activate(acc[v], silu), dtype)
                    for v in T.vectorized(vec):
                        y[b, c0 + v] = out[v]
                    # Shift each channel's window by one token.
                    for v in T.unroll(vec):
                        for k in T.unroll(hist - 1):
                            st[v * hist + k] = st[v * hist + k + 1]
                        st[v * hist + hist - 1] = xv[v]
                    for q in T.unroll(hist):
                        for v in T.vectorized(vec):
                            conv_state[state_row(slots, b), row + q * vec + v] = st[q * vec + v]

        @T.prim_func
        def main(
            x: T.Tensor((batch, dim), dtype),
            weight: T.Tensor((dim * width,), dtype),
            bias: T.Tensor((dim,), dtype),
            conv_state: T.Tensor((pool if pooled else batch, dim * hist), dtype),
            slots: T.Tensor((batch,) if pooled else (1,), "int32"),
            y: T.Tensor((batch, dim), dtype),
        ):
            step(x, weight, bias, conv_state, slots if pooled else None, y)

        return main

    return _func


class CausalConv1dUpdateKernel(Kernel, CausalConv1dUpdateFwdInterface):
    """One decode token per sequence, a CTA a run of channels of one sequence.

    Args:
        dim: Channels, ``D``.
        width: Kernel width ``W``, 2 to 4.
        dtype: float16 or bfloat16.
        has_bias: Whether a bias is added.
        silu: Whether SiLU follows the convolution.
        pooled: Whether each sequence's state is the pool row its slot names.
        config: Optional ``{"threads"}`` override.
        device_index: The device the kernel runs on.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def refusal(cls, call: CausalConv1dUpdateCall) -> str | None:
        if call.varlen:
            return "the kernel convolves one token a sequence; query_start_loc is not served"
        if call.rollback:
            return "the kernel shifts the state by one token; num_accepted_tokens is not served"
        return _dense_state_refusal(call) or super().refusal(call)

    @classmethod
    def entry_for(cls, call: CausalConv1dUpdateCall) -> Entry:
        args = (call.dim, call.width, call.dtype, call.has_bias, call.silu)
        index = call.device_index
        return (*args, call.pooled, index), lambda: cls(
            *args, device_index=index, pooled=call.pooled
        )

    def __init__(
        self,
        dim: int,
        width: int,
        dtype: torch.dtype,
        has_bias: bool,
        silu: bool,
        config: dict | None = None,
        device_index: int | None = None,
        *,
        pooled: bool = False,
    ) -> None:
        super().__init__(device_index=device_index)
        self.dim, self.width, self.dtype = dim, width, dtype
        self.pooled = pooled
        self.kernel = _causal_conv1d_update_kernel(
            dim, width, self.dtype_str, has_bias, silu, _vector(dim, dtype), pooled
        )
        self.init_config(config)
        # The slots an unpooled build leaves unread are declared one element wide.
        self._unread = torch.zeros(1, dtype=torch.int32, device=self._build_device)

    @property
    def default_config(self) -> dict:
        return {"threads": 128}

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(**cfg)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        bias: torch.Tensor | None = None,
        conv_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del num_accepted_tokens, query_start_loc
        y = torch.empty_like(x)
        # An absent bias is never read; a view of its shape fills the slot.
        self.program(self._compile)(
            x,
            weight.view(-1),
            bias if bias is not None else x[0],
            conv_state.view(conv_state.shape[0], -1),
            conv_state_indices if self.pooled else self._unread,
            y,
        )
        return y


def _rollback_block(dim: int) -> tuple[int, int]:
    """The threads of a draft-verification block and the channels a thread convolves.

    Two channels a thread: one 4-byte load of each 16-bit token, and a 128-thread block of 256
    channels whose state region is 3 KiB at six columns.
    """
    return 128, math.gcd(dim, 2)


@target_cache(maxsize=64)
def _causal_conv1d_rollback_kernel(
    dim: int,
    width: int,
    dilation: int,
    state_len: int,
    dtype: str,
    weight_dtype: str,
    has_bias: bool,
    silu: bool,
    pooled: bool,
    varlen: bool,
):
    span = (width - 1) * dilation
    # The most tokens a sequence verifies: its state keeps span + q - 1 columns.
    tokens = state_len - span + 1 if varlen else 1
    window = span + tokens
    # Elements of the state region one 16-byte vector moves; the region of every block
    # starts and ends on one.
    state_vec = math.gcd(dim * state_len, 16 // DataType(dtype).bytes)

    @tilelang.jit(out_idx=[])
    def _func(threads: int, vec: int):
        batch = T.dynamic("batch")
        pool = T.dynamic("pool")
        packed = T.dynamic("tokens")
        block_d = threads * vec
        region = block_d * state_len

        @T.macro
        def step(x, weight, bias, conv_state, slots, accepted, offsets, y):
            # A block is block_d channels of one sequence; their state rows are one contiguous
            # region, staged through shared memory in 16-byte vectors, and a thread convolves
            # vec adjacent channels.
            with T.Kernel(T.ceildiv(dim, block_d), batch, threads=threads) as (bd, b):
                tx = T.get_thread_binding()
                c_first = bd * block_d
                c0 = c_first + tx * vec
                held = T.min(block_d, dim - c_first) * state_len
                rows = T.alloc_shared((region,), dtype)
                win = T.alloc_local((vec * window,), "float32")
                wraw = T.alloc_local((vec * width,), weight_dtype)
                bv = T.alloc_local((vec,), weight_dtype)
                xv = T.alloc_local((tokens * vec,), dtype)
                acc = T.alloc_local((vec,), "float32")
                out = T.alloc_local((vec,), dtype)
                slot = state_row(slots if pooled else None, b)
                first = accepted[b] - 1
                start = offsets[b] if varlen else b
                length = offsets[b + 1] - start if varlen else 1
                # The tokens, filters and state rows are all read before the first barrier, so
                # their loads are in flight together.
                if c0 < dim:
                    for i in T.unroll(tokens):
                        if i < length:
                            for v in T.vectorized(vec):
                                xv[i * vec + v] = x[start + i, c0 + v]
                    for q in T.unroll(width):
                        for v in T.vectorized(vec):
                            wraw[q * vec + v] = weight[c0 * width + q * vec + v]
                    if has_bias:
                        for v in T.vectorized(vec):
                            bv[v] = bias[c0 + v]
                for i in T.serial(T.ceildiv(region, threads * state_vec)):
                    flat = (tx + i * threads) * state_vec
                    if flat < held:
                        for e in T.vectorized(state_vec):
                            rows[flat + e] = conv_state[slot, c_first * state_len + flat + e]
                T.sync_threads()
                if c0 < dim:
                    for v in T.unroll(vec):
                        for j in T.unroll(span):
                            win[v * window + j] = T.cast(
                                rows[(tx * vec + v) * state_len + first + j], "float32"
                            )
                        for i in T.unroll(tokens):
                            win[v * window + span + i] = T.cast(xv[i * vec + v], "float32")
                    for i in T.unroll(tokens):
                        if i < length:
                            for v in T.unroll(vec):
                                acc[v] = T.cast(bv[v], "float32") if has_bias else T.float32(0)
                                for k in T.unroll(width):
                                    acc[v] += (
                                        T.cast(wraw[v * width + k], "float32")
                                        * win[v * window + i + k * dilation]
                                    )
                                out[v] = T.cast(_activate(acc[v], silu), dtype)
                            for v in T.vectorized(vec):
                                y[start + i, c0 + v] = out[v]
                # The first span + q - 1 columns take the window after its first token; the
                # columns past them keep what they held, and a sequence with no token keeps
                # its state.
                if length > 0:
                    if c0 < dim:
                        for v in T.unroll(vec):
                            for j in T.unroll(state_len):
                                if j < span + length - 1:
                                    rows[(tx * vec + v) * state_len + j] = T.cast(
                                        win[v * window + j + 1], dtype
                                    )
                    T.sync_threads()
                    for i in T.serial(T.ceildiv(region, threads * state_vec)):
                        flat = (tx + i * threads) * state_vec
                        if flat < held:
                            for e in T.vectorized(state_vec):
                                conv_state[slot, c_first * state_len + flat + e] = rows[flat + e]

        @T.prim_func
        def main(
            x: T.Tensor((packed if varlen else batch, dim), dtype),
            weight: T.Tensor((dim * width,), weight_dtype),
            bias: T.Tensor((dim,), weight_dtype),
            conv_state: T.Tensor((pool if pooled else batch, dim * state_len), dtype),
            slots: T.Tensor((batch,) if pooled else (1,), "int32"),
            accepted: T.Tensor((batch,), "int32"),
            offsets: T.Tensor((batch + 1,) if varlen else (1,), "int32"),
            y: T.Tensor((packed if varlen else batch, dim), dtype),
        ):
            step(x, weight, bias, conv_state, slots, accepted, offsets, y)

        return main

    return _func


class CausalConv1dUpdateRollbackKernel(Kernel, CausalConv1dUpdateFwdInterface):
    """Draft verification: each sequence's tokens in turn from the state of its last accepted
    token, a CTA a run of channels of one sequence.

    The block stages the contiguous ``L``-column state rows of its channels in shared memory;
    a thread reads each of its channels' history from column ``num_accepted_tokens - 1`` and
    rewrites the row's first ``(W - 1) * dilation + q - 1`` columns, the others as they were,
    before the block writes the rows back.

    Args:
        dim: Channels, ``D``.
        width: Kernel width ``W``, 2 to 4.
        dilation: Tokens between adjacent filter taps.
        state_len: Columns of a state, ``L``.
        dtype: float16 or bfloat16.
        weight_dtype: The weight's and the bias's dtype, the activations' or float32.
        has_bias: Whether a bias is added.
        silu: Whether SiLU follows the convolution.
        pooled: Whether each sequence's state is the pool row its slot names.
        varlen: Whether ``query_start_loc`` packs several tokens a sequence.
        device_index: The device the kernel runs on.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def refusal(cls, call: CausalConv1dUpdateCall) -> str | None:
        if not call.rollback:
            return "verifies drafts; a call without num_accepted_tokens is not served"
        threads, vec = _rollback_block(call.dim)
        need = threads * vec * call.state_len * call.dtype.itemsize
        if need > call.smem_budget:
            return f"a block stages {need} bytes of state, past the {call.smem_budget} a block has"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: CausalConv1dUpdateCall) -> Entry:
        args = (
            call.dim, call.width, call.dilation, call.state_len, call.dtype, call.weight_dtype,
            call.has_bias, call.silu, call.pooled, call.varlen,
        )  # fmt: skip
        index = call.device_index
        return (*args, index), lambda: cls(*args, device_index=index)

    def __init__(
        self,
        dim: int,
        width: int,
        dilation: int,
        state_len: int,
        dtype: torch.dtype,
        weight_dtype: torch.dtype,
        has_bias: bool,
        silu: bool,
        pooled: bool,
        varlen: bool,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.dim, self.dtype, self.pooled, self.varlen = dim, dtype, pooled, varlen
        weight_name = str(DataType(str(weight_dtype).removeprefix("torch.")))
        self.kernel = _causal_conv1d_rollback_kernel(
            dim, width, dilation, state_len, self.dtype_str, weight_name, has_bias, silu, pooled,
            varlen,
        )  # fmt: skip

        self.init_config()
        # The index tensors this build leaves unread are declared one element wide.
        self._unread = torch.zeros(1, dtype=torch.int32, device=self._build_device)

    @property
    def default_config(self) -> dict:
        threads, vec = _rollback_block(self.dim)
        return {"threads": threads, "vec": vec}

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(**cfg)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        bias: torch.Tensor | None = None,
        conv_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = torch.empty_like(x)
        self.program(self._compile)(
            x,
            weight.view(-1),
            bias if bias is not None else weight.view(-1)[: self.dim],
            conv_state.view(conv_state.shape[0], -1),
            conv_state_indices if self.pooled else self._unread,
            num_accepted_tokens,
            query_start_loc if self.varlen else self._unread,
            y,
        )
        return y
