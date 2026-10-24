"""Depthwise causal short convolution of a Mamba-2 block: prefill over packed sequences and the
one-token decode step, both with fused SiLU and a per-sequence convolution state.

For a sequence occupying packed tokens ``[a, e)``, with ``h`` its ``W - 1`` preceding tokens
(``conv_state[s]`` where ``has_initial_state[s]``, zeros otherwise)::

    xp[c, :]       = concat(h[c, :], x[a:e, c])
    y[a + i, c]    = act(bias[c] + sum_k weight[c, k] * xp[c, i + k])
    conv_state[s] <- xp[c, e - a : e - a + W - 1]       (its last W - 1 tokens)

An empty sequence (``e == a``) keeps its state.
The products accumulate in float32 and the output rounds once to the input dtype.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tvm import DataType

from tileops.kernels.constants import LOG2E
from tileops.kernels.elementwise._division import approx_fdiv
from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops.kernels.mamba.call_spec import (
    CausalConv1dCall,
    CausalConv1dFwdInterface,
    CausalConv1dUpdateCall,
    CausalConv1dUpdateFwdInterface,
)

__all__ = ["CausalConv1dKernel", "CausalConv1dUpdateKernel"]


def _activate(acc, silu: bool):
    """``acc / (1 + 2^(-acc log2 e))`` in float32 when *silu*.

    The divisor is at least one, inside ``approx_fdiv``'s range; past ``2**126`` its zero
    quotient is SiLU's limit.
    """
    if not silu:
        return acc
    return approx_fdiv(acc, T.float32(1.0) + T.exp2(-acc * T.float32(LOG2E)))


def _vector(dim: int, dtype: torch.dtype) -> int:
    """Channels a thread holds: the widest of one 16-byte vector, 4, 2 or 1 that divides *dim*."""
    widest = 16 // torch.empty((), dtype=dtype).element_size()
    return next(v for v in (widest, 4, 2, 1) if v <= widest and dim % v == 0)


@functools.lru_cache(maxsize=64)
def _causal_conv1d_kernel(
    dim: int,
    width: int,
    dtype: str,
    has_bias: bool,
    has_init: bool,
    silu: bool,
):
    hist = width - 1

    # exp2 as one MUFU op, denormals flushed.
    @tilelang.jit(out_idx=[], compile_flags=["--use_fast_math"])
    def _func(vec: int, warps: int, seg: int, steps: int, stages: int):
        total = T.dynamic("total")
        batch = T.dynamic("batch")
        lanes = 32
        block_d = lanes * vec
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

        @T.prim_func
        def main(
            x: T.Tensor((total, dim), dtype),
            weight: T.Tensor((dim * width,), dtype),
            bias: T.Tensor((dim,), dtype),
            conv_state: T.Tensor((batch, dim, hist), dtype),
            cu_seqlens: T.Tensor((batch + 1,), "int32"),
            has_initial_state: T.Tensor((batch,), "uint8"),
            y: T.Tensor((total, dim), dtype),
        ):
            # A warp walks `chunk` packed tokens of block_d channels; each lane reads only the
            # ring slots it copied, so warps need no barrier.
            #
            # Only the warp whose chunk holds a sequence's first token reads or writes that
            # sequence's state: it computes the sequence's first W-1 outputs, even past its
            # chunk's end, and writes the final state last.
            with T.Kernel(
                T.ceildiv(dim, block_d),
                T.ceildiv(T.ceildiv(total, chunk), warps),
                threads=lanes * warps,
            ) as (bd, bc):
                ring = T.alloc_shared((warps, stages, seg, block_d), dtype)
                tx = T.get_thread_binding()
                lane = tx % lanes
                w = tx // lanes
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
                                        hold[k * vec + v] = conv_state[seq, c0 + v, k]
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
                                                    hold[k * vec + v] = conv_state[seq, c0 + v, k]
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
                                                seq, c0 + v, end - start + p
                                            ]
                                for p in T.unroll(hist):
                                    for v in T.unroll(vec):
                                        conv_state[seq, c0 + v, p] = hold[p * vec + v]
                            seq = seq + 1

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
        config: Optional ``{"vec", "warps", "seg", "steps", "stages"}`` override.
        device_index: The device the kernel runs on.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: CausalConv1dCall) -> Entry:
        args = (call.dim, call.width, call.dtype, call.has_bias, call.has_init, call.silu)
        index = call.device.index if call.device is not None else None
        return (*args, index), lambda: cls(*args, device_index=index)

    def __init__(
        self,
        dim: int,
        width: int,
        dtype: torch.dtype,
        has_bias: bool,
        has_init: bool,
        silu: bool,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.dim, self.width, self.dtype = dim, width, dtype
        self.kernel = _causal_conv1d_kernel(dim, width, self.dtype_str, has_bias, has_init, silu)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # The fastest measured over the manifest rows on H200.
        return {
            "vec": _vector(self.dim, self.dtype),
            "warps": 4,
            "seg": self.width * (8 // self.width),
            "steps": 8,
            "stages": 2,
        }

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        has_initial_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y = torch.empty_like(x)
        batch = cu_seqlens.shape[0] - 1
        # An absent bias or has_initial_state is never read; a view of its shape fills the slot.
        if has_initial_state is None:
            flags = cu_seqlens.view(torch.uint8)[:batch]
        else:
            flags = has_initial_state.view(torch.uint8)
        self.kernel(**self.config)(
            vector_aligned(x),
            vector_aligned(weight).view(-1),
            bias if bias is not None else x[0],
            conv_state,
            cu_seqlens,
            flags,
            y,
        )
        return y


@functools.lru_cache(maxsize=64)
def _causal_conv1d_update_kernel(
    dim: int, width: int, dtype: str, has_bias: bool, silu: bool, vec: int
):
    hist = width - 1

    @tilelang.jit(out_idx=[])
    def _func(threads: int):
        batch = T.dynamic("batch")
        block_d = threads * vec

        @T.prim_func
        def main(
            x: T.Tensor((batch, dim), dtype),
            weight: T.Tensor((dim * width,), dtype),
            bias: T.Tensor((dim,), dtype),
            conv_state: T.Tensor((batch * dim * hist,), dtype),
            y: T.Tensor((batch, dim), dtype),
        ):
            # A CTA is block_d channels of one sequence; a thread holds vec of them, whose
            # vec x W weights and vec x (W-1) state elements are each contiguous.
            with T.Kernel(T.ceildiv(dim, block_d), batch, threads=threads) as (bd, b):
                c0 = bd * block_d + T.get_thread_binding() * vec
                base = (b * dim + c0) * hist
                st = T.alloc_local((vec * hist,), dtype)
                wraw = T.alloc_local((vec * width,), dtype)
                xv = T.alloc_local((vec,), dtype)
                bv = T.alloc_local((vec,), dtype)
                acc = T.alloc_local((vec,), "float32")
                out = T.alloc_local((vec,), dtype)
                if c0 < dim:
                    for q in T.unroll(hist):
                        for v in T.vectorized(vec):
                            st[q * vec + v] = conv_state[base + q * vec + v]
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
                            conv_state[base + q * vec + v] = st[q * vec + v]

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
        config: Optional ``{"threads"}`` override.
        device_index: The device the kernel runs on.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: CausalConv1dUpdateCall) -> Entry:
        args = (call.dim, call.width, call.dtype, call.has_bias, call.silu)
        index = call.device.index if call.device is not None else None
        return (*args, index), lambda: cls(*args, device_index=index)

    def __init__(
        self,
        dim: int,
        width: int,
        dtype: torch.dtype,
        has_bias: bool,
        silu: bool,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.dim, self.width, self.dtype = dim, width, dtype
        self.vec = _vector(dim, dtype)
        self.kernel = _causal_conv1d_update_kernel(
            dim, width, self.dtype_str, has_bias, silu, self.vec
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"threads": 128}

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y = torch.empty_like(x)
        # The state is read and written in vectors; one off a 16-byte boundary is advanced
        # in an aligned copy and copied back.
        state = vector_aligned(conv_state)
        self.kernel(**self.config)(
            vector_aligned(x),
            vector_aligned(weight).view(-1),
            vector_aligned(bias) if bias is not None else x[0],
            state.view(-1),
            y,
        )
        if state is not conv_state:
            conv_state.copy_(state)
        return y
