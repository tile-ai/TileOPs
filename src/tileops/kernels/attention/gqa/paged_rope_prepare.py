"""Read-only request-local Q/K rotation into private temporary pages.

An FP8 query or cache is dequantized by its scales as it is read, so the pages it writes
are 16-bit and carry no scale: a rotation commutes with a scalar scale.
"""

import functools

import tilelang
import tilelang.language as T

__all__ = []


@functools.lru_cache(maxsize=32)
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
    q_dtype=None,
    kv_dtype=None,
    per_request_kv_scale=False,
):
    q_dtype = q_dtype or dtype
    kv_dtype = kv_dtype or dtype
    q_fp8 = q_dtype != dtype
    kv_fp8 = kv_dtype != dtype
    group = heads // heads_kv
    kv_scale_shape = (batch, heads_kv) if per_request_kv_scale else (1,)
    total_q = T.dynamic("total_q")
    pool_rows = T.dynamic("pool_rows")
    half = rotary_dim // 2
    pairs = dim // 2
    capacity = page * output_width
    # Four scalar pairs per thread fill a coalesced 1024-pair block.
    threads, pairs_per_thread = 256, 4
    block = threads * pairs_per_thread
    q_blocks = T.ceildiv(total_q * heads * pairs, block)
    # Fill the SM90 limit of 2048 resident threads per SM to hide gather latency.
    # Workers loop over live pages, independently of padded table capacity.
    workers_per_sm = 2048 // threads
    kv_blocks = min(
        tilelang.cdiv(capacity * heads_kv * pairs, block),
        tilelang.cdiv(sm_count * workers_per_sm, batch),
    )

    def q_value(x, scale, request, head):
        """A query element as float32, dequantized when FP8."""
        x = T.cast(x, "float32")
        return x * scale[request, head // group] if q_fp8 else x

    def kv_value(x, scale, request, head):
        """A cache element as float32, dequantized when FP8."""
        x = T.cast(x, "float32")
        if not kv_fp8:
            return x
        return x * (scale[request, head] if per_request_kv_scale else scale[0])

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
        KR: T.Tensor((batch * capacity, heads_kv, dim), dtype),
        VR: T.Tensor((batch * capacity, heads_kv, dim), dtype),
    ):
        with T.Kernel(q_blocks + batch * kv_blocks, threads=threads) as bx:
            if bx < q_blocks:
                for lane, item in T.Parallel(threads, pairs_per_thread):
                    idx = bx * block + lane * pairs_per_thread + item
                    token = idx // (heads * pairs)
                    head = idx // pairs % heads
                    freq = idx % pairs
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
                        if freq < half:
                            d0 = freq if layout == "neox" else freq * 2
                            d1 = freq + half if layout == "neox" else freq * 2 + 1
                            c = T.cast(Cos[pos, freq], "float32")
                            s = T.cast(Sin[pos, freq], "float32")
                            x0 = q_value(Q[token, head, d0], QS, lo, head)
                            x1 = q_value(Q[token, head, d1], QS, lo, head)
                            QR[token, head, d0] = x0 * c - x1 * s
                            QR[token, head, d1] = x1 * c + x0 * s
                        elif q_fp8:
                            QR[token, head, freq * 2] = q_value(
                                Q[token, head, freq * 2], QS, lo, head
                            )
                            QR[token, head, freq * 2 + 1] = q_value(
                                Q[token, head, freq * 2 + 1], QS, lo, head
                            )
                        else:
                            QR[token, head, freq * 2] = Q[token, head, freq * 2]
                            QR[token, head, freq * 2 + 1] = Q[token, head, freq * 2 + 1]
            else:
                request = (bx - q_blocks) // kv_blocks
                worker = (bx - q_blocks) % kv_blocks
                if CuQ[request + 1] - CuQ[request] > 0:
                    # TMA may read the last live page's padding, so initialize that
                    # page too; pages beyond it are never read and need no writes.
                    live_rows = T.ceildiv(Lengths[request], page) * page
                    live_blocks = T.ceildiv(live_rows * heads_kv * pairs, block)
                    for step in T.serial(T.ceildiv(live_blocks, kv_blocks)):
                        tile = step * kv_blocks + worker
                        for lane, item in T.Parallel(threads, pairs_per_thread):
                            idx = tile * block + lane * pairs_per_thread + item
                            key = idx // (heads_kv * pairs)
                            head = idx // pairs % heads_kv
                            freq = idx % pairs
                            if key < live_rows:
                                dst = request * capacity + key
                                if key < Lengths[request]:
                                    src = Table[request, key // page] * page + key % page
                                    if freq < half:
                                        d0 = freq if layout == "neox" else freq * 2
                                        d1 = freq + half if layout == "neox" else freq * 2 + 1
                                        c = T.cast(Cos[key, freq], "float32")
                                        s = T.cast(Sin[key, freq], "float32")
                                        x0 = kv_value(K[src, head, d0], KS, request, head)
                                        x1 = kv_value(K[src, head, d1], KS, request, head)
                                        KR[dst, head, d0] = x0 * c - x1 * s
                                        KR[dst, head, d1] = x1 * c + x0 * s
                                    elif kv_fp8:
                                        for e in T.unroll(2):
                                            KR[dst, head, freq * 2 + e] = kv_value(
                                                K[src, head, freq * 2 + e], KS, request, head
                                            )
                                    else:
                                        KR[dst, head, freq * 2] = K[src, head, freq * 2]
                                        KR[dst, head, freq * 2 + 1] = K[src, head, freq * 2 + 1]
                                    if kv_fp8:
                                        for e in T.unroll(2):
                                            VR[dst, head, freq * 2 + e] = kv_value(
                                                V[src, head, freq * 2 + e], VS, request, head
                                            )
                                    else:
                                        VR[dst, head, freq * 2] = V[src, head, freq * 2]
                                        VR[dst, head, freq * 2 + 1] = V[src, head, freq * 2 + 1]
                                else:
                                    KR[dst, head, freq * 2] = 0
                                    KR[dst, head, freq * 2 + 1] = 0
                                    VR[dst, head, freq * 2] = 0
                                    VR[dst, head, freq * 2 + 1] = 0

    return main
