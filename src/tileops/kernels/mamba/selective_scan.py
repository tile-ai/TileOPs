"""Mamba-1 S6 selective scan: a first-order recurrence per channel and state whose decay and
input both follow the input.

With ``dt = softplus(delta + delta_bias)`` and channel ``d`` reading group ``g`` of B and C::

    h[t, n] = exp(dt[d, t] * A[d, n]) * h[t - 1, n] + dt[d, t] * u[d, t] * B[g, n, t]
    y[d, t] = (sum_n C[g, n, t] * h[t, n] + D[d] * u[d, t]) * silu(z[d, t])

``h`` starts from ``initial_states`` or zeros and restarts from zeros where ``seq_idx``
changes. Every product and sum is float32; ``y`` rounds once to the input dtype.

Two decompositions of the time axis serve it. ``SelectiveScanChunkKernel`` splits it over the
lanes of a warp and joins the lanes' segments with a warp scan; ``SelectiveScanStepKernel``
walks it in order, one thread a group of states, for sequences too short to fill a warp.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import LOG2E, VECTOR_ACCESS_BYTES
from tileops.kernels.elementwise._division import approx_fdiv
from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops.kernels.mamba.call_spec import SelectiveScanCall, SelectiveScanFwdInterface

__all__ = ["SelectiveScanChunkKernel", "SelectiveScanStepKernel"]


def _step(x, softplus: bool):
    """``softplus(x)`` in float32 when *softplus*, else *x*.

    ``softplus(x) = max(x, 0) + log1p(v)`` with ``v = exp(-|x|)``, and ``log1p(v) = 2 atanh(s)``
    with ``s = v / (2 + v) <= 1/3``. The atanh series through ``s**15`` truncates below 2e-9
    relative, so the result stays within a few float32 units in the last place without a
    branch; past ``x = 20`` it rounds to ``x``, as torch's threshold.
    """
    if not softplus:
        return x
    v = T.exp2(-T.abs(x) * T.float32(LOG2E))
    s = approx_fdiv(v, T.float32(2) + v)
    s2 = s * s
    series = T.float32(1 / 15)
    for k in (13, 11, 9, 7, 5, 3, 1):
        series = series * s2 + T.float32(1 / k)
    return T.max(x, T.float32(0)) + T.float32(2) * s * series


def _gate(acc, z):
    """``acc * silu(z)`` in float32; the divisor is at least one, inside ``approx_fdiv``'s range."""
    return approx_fdiv(acc * z, T.float32(1) + T.exp2(-z * T.float32(LOG2E)))


@functools.lru_cache(maxsize=64)
def _selective_scan_chunk_kernel(
    dim: int,
    seqlen: int,
    dstate: int,
    groups: int,
    dtype: str,
    has_skip: bool,
    has_gate: bool,
    has_bias: bool,
    has_seq_idx: bool,
    has_init: bool,
    softplus: bool,
):
    group_dim = dim // groups
    # A row of u / delta / z / y starts on a 16-byte vector boundary.
    row_vectors = (
        seqlen * torch.empty((), dtype=getattr(torch, dtype)).element_size() % VECTOR_ACCESS_BYTES
        == 0
    )

    @tilelang.jit(out_idx=[], compile_flags=["--use_fast_math"])
    def _func(warps: int, items: int):
        batch = T.dynamic("batch")
        chunk = 32 * items
        # State n's rate and carried state live in lane n % 32's slot n // 32.
        slots = (dstate + 31) // 32
        if group_dim % warps:
            raise ValueError(f"{warps} warps do not divide a group's {group_dim} channels")
        # A lane's items are one or more 16-byte vectors.
        vectors = (
            row_vectors
            and items
            * torch.empty((), dtype=getattr(torch, dtype)).element_size()
            % VECTOR_ACCESS_BYTES
            == 0
        )

        @T.macro
        def scan_states(
            Bs, Cs, rate, carry, dt, du, av, bv, yv, bn, cn, lane, span, kept, cuts, masked: bool
        ):
            """Every state of one run: each lane scans its segment from zero, a shuffle scan joins
            the segments, and each lane rescans from its left neighbour's end. *masked* zeroes the
            decay of the steps whose bit of *cuts* is set."""
            hl = T.alloc_var("float32")
            pl = T.alloc_var("float32")
            hu = T.alloc_var("float32")
            pu = T.alloc_var("float32")
            h = T.alloc_var("float32")
            for q in T.unroll(slots):
                for nl in T.serial(T.min(32, dstate - q * 32)):
                    n = q * 32 + nl
                    r_n = T.shfl_sync(rate[q], nl)
                    c_n = T.shfl_sync(carry[q], nl)
                    for k in T.vectorized(items):
                        bn[k] = Bs[n, lane * items + k]
                    for k in T.vectorized(items):
                        cn[k] = Cs[n, lane * items + k]
                    # The lane's segment from zero: its state, and its decay as one
                    # exponential of the summed steps.
                    hl = T.float32(0)
                    for k in T.unroll(items):
                        av[k] = T.exp2(dt[k] * r_n)
                        if masked:
                            av[k] = T.if_then_else((cuts >> k) & 1 == 1, T.float32(0), av[k])
                        bv[k] = du[k] * T.cast(bn[k], "float32")
                        hl = av[k] * hl + bv[k]
                    pl = T.exp2(span * r_n) * kept
                    # Inclusive scan of (decay, state) over the lanes.
                    for s in T.unroll(5):
                        hu = T.shfl_up(hl, 1 << s)
                        pu = T.shfl_up(pl, 1 << s)
                        if lane >= (1 << s):
                            hl = pl * hu + hl
                            pl = pl * pu
                    # Each lane restarts from the state its left neighbour ends on.
                    hu = T.shfl_up(hl, 1)
                    pu = T.shfl_up(pl, 1)
                    h = T.if_then_else(lane > 0, pu * c_n + hu, c_n)
                    for k in T.unroll(items):
                        h = av[k] * h + bv[k]
                        yv[k] = yv[k] + T.cast(cn[k], "float32") * h
                    ends = T.shfl_sync(hl, 31) + T.shfl_sync(pl, 31) * c_n
                    if lane == nl:
                        carry[q] = ends

        @T.prim_func
        def main(
            u: T.Tensor((batch, dim, seqlen), dtype),
            delta: T.Tensor((batch, dim, seqlen), dtype),
            A: T.Tensor((dim, dstate), "float32"),
            B_in: T.Tensor((batch, groups, dstate, seqlen), dtype),
            C_in: T.Tensor((batch, groups, dstate, seqlen), dtype),
            D_in: T.Tensor((dim,), "float32"),
            z: T.Tensor((batch, dim, seqlen), dtype),
            delta_bias: T.Tensor((dim,), "float32"),
            seq_idx: T.Tensor((batch, seqlen), "int32"),
            initial_states: T.Tensor((batch, dim, dstate), "float32"),
            y: T.Tensor((batch, dim, seqlen), dtype),
            final_states: T.Tensor((batch, dim, dstate), "float32"),
        ):
            # A warp is one channel; lane l holds items [l * items, (l + 1) * items) of each
            # chunk-long run of the sequence. The CTA's channels share one group, so B and C
            # of the run are staged once for all its warps.
            with T.Kernel(dim // warps, batch, threads=32 * warps) as (bc, b):
                T.annotate_min_blocks_per_sm(2)
                tx = T.get_thread_binding()
                lane = tx % 32
                c = bc * warps + tx // 32
                g = (bc * warps) // group_dim
                Bs = T.alloc_shared((dstate, chunk), dtype)
                Cs = T.alloc_shared((dstate, chunk), dtype)
                rate = T.alloc_local((slots,), "float32")
                carry = T.alloc_local((slots,), "float32")
                dt = T.alloc_local((items,), "float32")
                du = T.alloc_local((items,), "float32")
                uf = T.alloc_local((items,), "float32")
                av = T.alloc_local((items,), "float32")
                bv = T.alloc_local((items,), "float32")
                yv = T.alloc_local((items,), "float32")
                raw_u = T.alloc_local((items,), dtype)
                raw_d = T.alloc_local((items,), dtype)
                raw_z = T.alloc_local((items,), dtype)
                ids = T.alloc_local((items,), "int32")
                out = T.alloc_local((items,), dtype)
                bn = T.alloc_local((items,), dtype)
                cn = T.alloc_local((items,), dtype)
                span = T.alloc_var("float32")
                kept = T.alloc_var("float32")
                # Bit k: step k of the lane's segment starts a sequence. A mask, not an array,
                # so the packed form fits the registers of two CTAs an SM.
                cuts = T.alloc_var("int32")
                for q in T.unroll(slots):
                    nq = T.min(q * 32 + lane, dstate - 1)
                    rate[q] = A[c, nq] * T.float32(LOG2E)
                    carry[q] = initial_states[b, c, nq] if has_init else T.float32(0)

                for ci in T.serial(T.ceildiv(seqlen, chunk)):
                    t0 = ci * chunk
                    base = t0 + lane * items
                    full = t0 + chunk <= seqlen
                    T.sync_threads()
                    T.copy(B_in[b, g, 0:dstate, t0 : t0 + chunk], Bs)
                    T.copy(C_in[b, g, 0:dstate, t0 : t0 + chunk], Cs)
                    T.sync_threads()
                    if full and vectors:
                        for k in T.vectorized(items):
                            raw_u[k] = u[b, c, base + k]
                        for k in T.vectorized(items):
                            raw_d[k] = delta[b, c, base + k]
                    else:
                        for k in T.unroll(items):
                            raw_u[k] = u[b, c, T.min(base + k, seqlen - 1)]
                            raw_d[k] = delta[b, c, T.min(base + k, seqlen - 1)]
                    if has_seq_idx:
                        if full and vectors:
                            for k in T.vectorized(items):
                                ids[k] = seq_idx[b, base + k]
                        else:
                            for k in T.unroll(items):
                                ids[k] = seq_idx[b, T.min(base + k, seqlen - 1)]
                    span = T.float32(0)
                    cuts = 0
                    for k in T.unroll(items):
                        t = base + k
                        # A step past the sequence decays by one and adds nothing.
                        live = t < seqlen
                        x = T.cast(raw_d[k], "float32") + (
                            delta_bias[c] if has_bias else T.float32(0)
                        )
                        dt[k] = T.if_then_else(live, _step(x, softplus), T.float32(0))
                        uf[k] = T.if_then_else(live, T.cast(raw_u[k], "float32"), T.float32(0))
                        du[k] = dt[k] * uf[k]
                        yv[k] = T.float32(0)
                        span = span + dt[k]
                        if has_seq_idx:
                            before = T.if_then_else(
                                k > 0, ids[T.max(k - 1, 0)], seq_idx[b, T.max(base - 1, 0)]
                            )
                            starts = T.And(T.And(t > 0, live), ids[k] != before)
                            cuts = cuts | (T.cast(starts, "int32") << k)
                    # A sequence's first step starts from zero: its decay is zero.
                    kept = T.if_then_else(cuts == 0, T.float32(1), T.float32(0))
                    if has_seq_idx:
                        # Only a run with a sequence start pays for the masked decay.
                        if T.any_sync(cuts != 0) != 0:
                            scan_states(
                                Bs,
                                Cs,
                                rate,
                                carry,
                                dt,
                                du,
                                av,
                                bv,
                                yv,
                                bn,
                                cn,
                                lane,
                                span,
                                kept,
                                cuts,
                                True,
                            )
                        else:
                            scan_states(
                                Bs,
                                Cs,
                                rate,
                                carry,
                                dt,
                                du,
                                av,
                                bv,
                                yv,
                                bn,
                                cn,
                                lane,
                                span,
                                kept,
                                cuts,
                                False,
                            )
                    else:
                        scan_states(
                            Bs,
                            Cs,
                            rate,
                            carry,
                            dt,
                            du,
                            av,
                            bv,
                            yv,
                            bn,
                            cn,
                            lane,
                            span,
                            kept,
                            cuts,
                            False,
                        )
                    if has_gate:
                        if full and vectors:
                            for k in T.vectorized(items):
                                raw_z[k] = z[b, c, base + k]
                        else:
                            for k in T.unroll(items):
                                raw_z[k] = z[b, c, T.min(base + k, seqlen - 1)]
                    for k in T.unroll(items):
                        acc = yv[k] + uf[k] * D_in[c] if has_skip else yv[k]
                        if has_gate:
                            acc = _gate(acc, T.cast(raw_z[k], "float32"))
                        out[k] = T.cast(acc, dtype)
                    if full and vectors:
                        for k in T.vectorized(items):
                            y[b, c, base + k] = out[k]
                    else:
                        for k in T.unroll(items):
                            if base + k < seqlen:
                                y[b, c, base + k] = out[k]
                for q in T.unroll(slots):
                    if q * 32 + lane < dstate:
                        final_states[b, c, q * 32 + lane] = carry[q]

        return main

    return _func


@functools.lru_cache(maxsize=64)
def _selective_scan_step_kernel(
    dim: int,
    seqlen: int,
    dstate: int,
    groups: int,
    dtype: str,
    has_skip: bool,
    has_gate: bool,
    has_bias: bool,
    has_seq_idx: bool,
    has_init: bool,
    softplus: bool,
):
    group_dim = dim // groups

    @tilelang.jit(out_idx=[], compile_flags=["--use_fast_math"])
    def _func(threads: int, states: int):
        batch = T.dynamic("batch")
        # A channel's lanes, each holding `states` of its states; they sum y with shfl_xor.
        lanes = dstate // states
        channels = threads // lanes
        if dstate % states or 32 % lanes:
            raise ValueError(
                f"{states} states a thread do not split {dstate} over a power of two of lanes"
            )

        @T.prim_func
        def main(
            u: T.Tensor((batch, dim, seqlen), dtype),
            delta: T.Tensor((batch, dim, seqlen), dtype),
            A: T.Tensor((dim, dstate), "float32"),
            B_in: T.Tensor((batch, groups, dstate, seqlen), dtype),
            C_in: T.Tensor((batch, groups, dstate, seqlen), dtype),
            D_in: T.Tensor((dim,), "float32"),
            z: T.Tensor((batch, dim, seqlen), dtype),
            delta_bias: T.Tensor((dim,), "float32"),
            seq_idx: T.Tensor((batch, seqlen), "int32"),
            initial_states: T.Tensor((batch, dim, dstate), "float32"),
            y: T.Tensor((batch, dim, seqlen), dtype),
            final_states: T.Tensor((batch, dim, dstate), "float32"),
        ):
            with T.Kernel(T.ceildiv(dim, channels), batch, threads=threads) as (bc, b):
                tx = T.get_thread_binding()
                c = bc * channels + tx // lanes
                n0 = (tx % lanes) * states
                rate = T.alloc_local((states,), "float32")
                h = T.alloc_local((states,), "float32")
                bv = T.alloc_local((states,), dtype)
                cv = T.alloc_local((states,), dtype)
                acc = T.alloc_var("float32")
                dt = T.alloc_var("float32")
                skip = T.alloc_var("float32")
                if c < dim:
                    g = c // group_dim
                    # Read before the arithmetic, so its load is not one more round trip after it.
                    skip = D_in[c] if has_skip else T.float32(0)
                    for n in T.vectorized(states):
                        rate[n] = A[c, n0 + n]
                    if has_init:
                        for n in T.vectorized(states):
                            h[n] = initial_states[b, c, n0 + n]
                    else:
                        for n in T.unroll(states):
                            h[n] = T.float32(0)
                    for t in T.serial(seqlen):
                        for n in T.unroll(states):
                            bv[n] = B_in[b, g, n0 + n, t]
                            cv[n] = C_in[b, g, n0 + n, t]
                        x = T.cast(delta[b, c, t], "float32") + (
                            delta_bias[c] if has_bias else T.float32(0)
                        )
                        # A variable, so the softplus is computed once, not at each use.
                        dt = _step(x, softplus)
                        uf = T.cast(u[b, c, t], "float32")
                        du = dt * uf
                        if has_seq_idx and T.And(
                            t > 0, seq_idx[b, t] != seq_idx[b, T.max(t - 1, 0)]
                        ):
                            for n in T.unroll(states):
                                h[n] = T.float32(0)
                        acc = T.float32(0)
                        for n in T.unroll(states):
                            h[n] = T.exp2(dt * rate[n] * T.float32(LOG2E)) * h[n] + du * T.cast(
                                bv[n], "float32"
                            )
                            acc = acc + T.cast(cv[n], "float32") * h[n]
                        for s in T.unroll(5):
                            if (1 << s) < lanes:
                                acc = acc + T.shfl_xor(acc, 1 << s)
                        if has_skip:
                            acc = acc + uf * skip
                        if has_gate:
                            acc = _gate(acc, T.cast(z[b, c, t], "float32"))
                        if tx % lanes == 0:
                            y[b, c, t] = T.cast(acc, dtype)
                    for n in T.vectorized(states):
                        final_states[b, c, n0 + n] = h[n]

        return main

    return _func


class _SelectiveScanKernel(Kernel, SelectiveScanFwdInterface):
    """The build and call shared by the two decompositions of the time axis."""

    supported_archs: list[int] = [80, 86, 89, 90]
    _builder = None

    @classmethod
    def entry_for(cls, call: SelectiveScanCall) -> Entry:
        args = (
            call.dim,
            call.seqlen,
            call.dstate,
            call.groups,
            call.dtype,
            call.has_skip,
            call.has_gate,
            call.has_bias,
            call.has_seq_idx,
            call.has_init,
            call.softplus,
        )
        index = call.device.index if call.device is not None else None
        return (*args, index), lambda: cls(*args, device_index=index)

    def __init__(
        self,
        dim: int,
        seqlen: int,
        dstate: int,
        groups: int,
        dtype: torch.dtype,
        has_skip: bool,
        has_gate: bool,
        has_bias: bool,
        has_seq_idx: bool,
        has_init: bool,
        softplus: bool,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.dim, self.seqlen, self.dstate, self.groups, self.dtype = (
            dim,
            seqlen,
            dstate,
            groups,
            dtype,
        )
        self.kernel = type(self)._builder(
            dim,
            seqlen,
            dstate,
            groups,
            self.dtype_str,
            has_skip,
            has_gate,
            has_bias,
            has_seq_idx,
            has_init,
            softplus,
        )
        self.init_config(config)

    def forward(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B_in: torch.Tensor,
        C_in: torch.Tensor,
        D_in: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        delta_bias: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        initial_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = u.shape[0]
        y = torch.empty_like(u)
        final_states = torch.empty(
            (batch, self.dim, self.dstate), dtype=torch.float32, device=u.device
        )
        # An absent input is never read; an uninitialized buffer of its shape fills the slot.
        per_channel = torch.empty((self.dim,), dtype=torch.float32, device=u.device)
        self.kernel(**self.config)(
            vector_aligned(u),
            vector_aligned(delta),
            vector_aligned(A),
            vector_aligned(B_in),
            vector_aligned(C_in),
            D_in if D_in is not None else per_channel,
            vector_aligned(z) if z is not None else u,
            delta_bias if delta_bias is not None else per_channel,
            vector_aligned(seq_idx)
            if seq_idx is not None
            else torch.empty((batch, self.seqlen), dtype=torch.int32, device=u.device),
            vector_aligned(initial_states) if initial_states is not None else final_states,
            y,
            final_states,
        )
        return y, final_states


class SelectiveScanChunkKernel(_SelectiveScanKernel):
    """The time axis split over a warp's lanes, the lanes' segments joined by a warp scan.

    A warp is one channel and walks the sequence in runs of ``32 * items`` steps, lane ``l``
    holding steps ``[l * items, (l + 1) * items)`` of each. For each state, every lane scans
    its segment from zero, a five-step shuffle scan combines the segments' (decay, state)
    pairs, and every lane rescans its segment from the state its left neighbour ends on. The
    CTA's ``warps`` channels share B and C, staged once a run in shared memory.

    Args:
        dim: Channels, ``D``.
        seqlen: Steps, ``S``.
        dstate: States a channel, ``N``.
        groups: Groups of channels sharing B and C, ``G``.
        dtype: float16, bfloat16 or float32.
        has_skip: Whether ``D_in * u`` is added.
        has_gate: Whether the output is gated by ``silu(z)``.
        has_bias: Whether ``delta_bias`` is added to ``delta``.
        has_seq_idx: Whether ``seq_idx`` restarts the state.
        has_init: Whether the state starts from ``initial_states``.
        softplus: Whether ``dt`` is the softplus of ``delta + delta_bias``.
        config: Optional ``{"warps", "items"}`` override.
        device_index: The device the kernel runs on.
    """

    general = True
    _builder = staticmethod(_selective_scan_chunk_kernel)

    @property
    def default_config(self) -> dict:
        # 8 channels a CTA. A run is the shortest power of two of 32-step blocks covering the
        # sequence, up to 512 steps: a run computes every step it covers, so a longer run on a
        # short sequence is padding. The fastest measured from 48 to 4096 steps on H200.
        warps = next(w for w in (8, 4, 2, 1) if (self.dim // self.groups) % w == 0)
        items = 1
        while items < 16 and 32 * items < self.seqlen:
            items *= 2
        return {"warps": warps, "items": items}


class SelectiveScanStepKernel(_SelectiveScanKernel):
    """The time axis walked in order: a thread is a group of one channel's states.

    For a sequence too short to fill a warp's lanes: decode, one step a call, is the case it
    is for. A channel's ``dstate / states`` threads each hold ``states`` of its states in
    registers and sum ``y`` with a shuffle.

    Args:
        Those of :class:`SelectiveScanChunkKernel`, with ``config`` an optional
        ``{"threads", "states"}`` override.
    """

    preferred_over = frozenset({"selective_scan_chunk"})
    _builder = staticmethod(_selective_scan_step_kernel)

    # Longest sequence it serves. On H200 at 48 steps it is 1.65x faster than the chunk kernel
    # at batch 16 and 1.19x slower at batch 1; from 64 steps the chunk kernel is faster at both.
    max_seqlen = 48

    @classmethod
    def refusal(cls, call: SelectiveScanCall) -> Optional[str]:
        if call.seqlen > cls.max_seqlen:
            return f"walks at most {cls.max_seqlen} steps in order, not {call.seqlen}"
        return None

    @property
    def default_config(self) -> dict:
        # Eight states a thread, one 16-byte load of B and of C, where they split over a power
        # of two of lanes: the fastest measured on the decode rows on H200, 128 threads a CTA.
        # Otherwise the fewest that do, or all of them.
        splits = (8, 4, 2, 1, 16, 32, 64, 128, 256)
        states = next(
            (s for s in splits if self.dstate % s == 0 and 32 % (self.dstate // s) == 0),
            self.dstate,
        )
        return {"threads": 128, "states": states}
