"""Read-only request-local Q/K rotation into private temporary pages.

An FP8 query or cache is dequantized by its scales as it is read, so the pages it writes
are 16-bit and carry no scale: a rotation commutes with a scalar scale.

Over a 16-bit cache the pages may hold only the first ``key_cols`` (rotated) key channels.

Under a left window the pages hold only the keys from the key tile where the request's first
query's window starts: the attention reads no key tile before it.
"""

import math

import tilelang
import tilelang.language as T

from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES

__all__ = []


@target_cache(maxsize=32)
@tilelang.jit(out_idx=[11, 12, 13])
def _gqa_paged_rope_prepare_kernel(
    batch,
    heads,
    heads_kv,
    dim,
    page,
    width,
    output_width,
    sm_count,
    positions,
    rotary_dim,
    layout,
    dtype,
    q_dtype,
    kv_dtype,
    per_request_kv_scale,
    key_cols,
    values,
    window_left,
    key_tile,
):
    """Rotate Q into QR and live keys into KR's ``key_cols`` leading channels; with *values*,
    copy V into VR, else VR is a one-row placeholder. With *window_left*, keys start at the
    *key_tile*-row tile holding the first query's window start.

    A thread owns one unit of a row: ``vec`` rotation pairs, read and written as two
    ``vec``-element vectors, or ``2 * vec`` copied channels past the rotated width."""
    q_fp8 = q_dtype != dtype
    kv_fp8 = kv_dtype != dtype
    group = heads // heads_kv
    kv_scale_shape = (batch, heads_kv) if per_request_kv_scale else (1,)
    total_q = T.dynamic("total_q")
    pool_rows = T.dynamic("pool_rows")
    half = rotary_dim // 2
    capacity = page * output_width
    # Rotation pairs a thread takes: at most one 16-byte vector of the 16-bit output, and a
    # divisor of each rotated half and each copied tail, so every access is a whole vector.
    vec = math.gcd(
        VECTOR_ACCESS_BYTES // 2, half, (dim - rotary_dim) // 2, (key_cols - rotary_dim) // 2
    )
    rotated_units = half // vec
    q_units = rotated_units + (dim - rotary_dim) // (2 * vec)
    k_units = rotated_units + (key_cols - rotary_dim) // (2 * vec)
    # Eight blocks of 256 threads fill the SM90 limit of 2048 resident threads per SM, which
    # hides gather latency. Workers loop over live pages, independently of padded table capacity.
    threads = 256
    q_blocks = T.ceildiv(total_q * heads * q_units, threads)
    workers_per_sm = 2048 // threads
    kv_blocks = min(
        tilelang.cdiv(capacity * heads_kv * k_units, threads),
        tilelang.cdiv(sm_count * workers_per_sm, batch),
    )

    def kv_scale(S, request, head):
        """A cache tensor's dequantization scale for *request* and *head*: one over a 16-bit
        cache."""
        if not kv_fp8:
            return 1.0
        return S[request, head] if per_request_kv_scale else S[0]

    def rotated_channels(unit):
        """The first channels of a rotation unit's two vectors: its pairs' low and high
        halves, adjacent runs when interleaved."""
        if layout == "neox":
            return unit * vec, half + unit * vec
        return unit * 2 * vec, unit * 2 * vec + vec

    def copied_channels(unit):
        """The first channels of a copy unit's two adjacent vectors, past the rotated width."""
        first = rotary_dim + (unit - rotated_units) * 2 * vec
        return first, first + vec

    @T.macro
    def load(x, raw, Src, row, head, c0, c1, scale):
        """Read two vectors of *Src* as float32 into *x*, multiplied by *scale*."""
        for e in T.vectorized(vec):
            raw[e] = Src[row, head, c0 + e]
        for e in T.vectorized(vec):
            raw[vec + e] = Src[row, head, c1 + e]
        for e in T.unroll(2 * vec):
            x[e] = T.cast(raw[e], "float32") * scale

    @T.macro
    def store(Dst, row, head, c0, c1, x, out):
        """Write *x* as two vectors of *Dst*."""
        for e in T.unroll(2 * vec):
            out[e] = T.cast(x[e], dtype)
        for e in T.vectorized(vec):
            Dst[row, head, c0 + e] = out[e]
        for e in T.vectorized(vec):
            Dst[row, head, c1 + e] = out[vec + e]

    @T.macro
    def move(Dst, drow, Src, srow, head, unit, pos, scale, raw, regs, turn):
        """Write one unit of row *srow* of *Src* to row *drow* of *Dst*; with *turn*, a
        rotation unit's vec pairs are rotated by the angles of position *pos* on the way.
        *regs* holds the thread's float32 values, output vector, angles and angle tables."""
        x, out, cos, sin, Cos, Sin = regs
        if unit < rotated_units:
            load(x, raw, Src, srow, head, *rotated_channels(unit), scale)
            if turn:
                for e in T.vectorized(vec):
                    cos[e] = Cos[pos, unit * vec + e]
                    sin[e] = Sin[pos, unit * vec + e]
                for p in T.unroll(vec):
                    lo = p if layout == "neox" else 2 * p
                    hi = vec + p if layout == "neox" else 2 * p + 1
                    c = T.cast(cos[p], "float32")
                    s = T.cast(sin[p], "float32")
                    x0 = x[lo]
                    x1 = x[hi]
                    x[lo] = x0 * c - x1 * s
                    x[hi] = x1 * c + x0 * s
            store(Dst, drow, head, *rotated_channels(unit), x, out)
        else:
            load(x, raw, Src, srow, head, *copied_channels(unit), scale)
            store(Dst, drow, head, *copied_channels(unit), x, out)

    @T.macro
    def clear(Dst, row, head, unit, regs):
        """Write zeros to one unit of row *row* of *Dst*."""
        x, out = regs[0], regs[1]
        for e in T.unroll(2 * vec):
            x[e] = 0.0
        if unit < rotated_units:
            store(Dst, row, head, *rotated_channels(unit), x, out)
        else:
            store(Dst, row, head, *copied_channels(unit), x, out)

    @T.prim_func
    def main(
        Q: T.Tensor((total_q, heads, dim), q_dtype),
        K: T.Tensor((pool_rows, heads_kv, dim), kv_dtype),
        V: T.Tensor((pool_rows, heads_kv, dim), kv_dtype),
        Table: T.Tensor((batch, width), "int32"),
        Lengths: T.Tensor((batch,), "int32"),
        CuQ: T.Tensor((batch + 1,), "int32"),
        QS: T.Tensor((batch, heads_kv), "float32"),
        KS: T.Tensor(kv_scale_shape, "float32"),
        VS: T.Tensor(kv_scale_shape, "float32"),
        Cos: T.Tensor((positions, half), dtype),
        Sin: T.Tensor((positions, half), dtype),
        QR: T.Tensor((total_q, heads, dim), dtype),
        KR: T.Tensor((batch * capacity, heads_kv, key_cols), dtype),
        VR: T.Tensor((batch * capacity if values else 1, heads_kv, dim), dtype),
    ):
        with T.Kernel(q_blocks + batch * kv_blocks, threads=threads) as bx:
            tx = T.get_thread_binding()
            raw_q = T.alloc_local([2 * vec], q_dtype)
            raw_kv = T.alloc_local([2 * vec], kv_dtype)
            x = T.alloc_local([2 * vec], "float32")
            out = T.alloc_local([2 * vec], dtype)
            cos = T.alloc_local([vec], dtype)
            sin = T.alloc_local([vec], dtype)
            regs = (x, out, cos, sin, Cos, Sin)
            if bx < q_blocks:
                idx = bx * threads + tx
                token = idx // (heads * q_units)
                head = idx // q_units % heads
                unit = idx % q_units
                if token < total_q:
                    lo = T.alloc_var("int32", init=0)
                    hi = T.alloc_var("int32", init=batch - 1)
                    for _ in T.serial(max(1, (batch - 1).bit_length())):
                        mid = (lo + hi) // 2
                        if CuQ[mid + 1] <= token:
                            lo = mid + 1
                        else:
                            hi = mid
                    pos = token - CuQ[lo] + Lengths[lo] - (CuQ[lo + 1] - CuQ[lo])
                    qscale = QS[lo, head // group] if q_fp8 else 1.0
                    move(QR, token, Q, token, head, unit, pos, qscale, raw_q, regs, True)
            else:
                request = (bx - q_blocks) // kv_blocks
                worker = (bx - q_blocks) % kv_blocks
                if CuQ[request + 1] - CuQ[request] > 0:
                    # TMA may read the last live page's padding, so initialize that
                    # page too; pages beyond it are never read and need no writes.
                    live_rows = T.ceildiv(Lengths[request], page) * page
                    # Without a window the first key is a constant zero, which the loop's index
                    # arithmetic folds away.
                    first = 0
                    if window_left >= 0:
                        q_len = CuQ[request + 1] - CuQ[request]
                        first = T.alloc_var(
                            "int32",
                            init=T.max(0, Lengths[request] - q_len - window_left)
                            // key_tile
                            * key_tile,
                        )
                    live_units = (live_rows - first) * heads_kv * k_units
                    for step in T.serial(T.ceildiv(live_units, kv_blocks * threads)):
                        slot = (step * kv_blocks + worker) * threads + tx
                        key = first + slot // (heads_kv * k_units)
                        kv_head = slot // k_units % heads_kv
                        kv_unit = slot % k_units
                        if key < live_rows:
                            dst = request * capacity + key
                            if key < Lengths[request]:
                                src = Table[request, key // page] * page + key % page
                                ks = kv_scale(KS, request, kv_head)
                                move(KR, dst, K, src, kv_head, kv_unit, key, ks, raw_kv, regs, True)
                                if values:
                                    vs = kv_scale(VS, request, kv_head)
                                    move(
                                        VR,
                                        dst,
                                        V,
                                        src,
                                        kv_head,
                                        kv_unit,
                                        key,
                                        vs,
                                        raw_kv,
                                        regs,
                                        False,
                                    )
                            else:
                                clear(KR, dst, kv_head, kv_unit, regs)
                                if values:
                                    clear(VR, dst, kv_head, kv_unit, regs)

    return main
