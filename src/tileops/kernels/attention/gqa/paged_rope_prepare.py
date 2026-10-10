"""Read-only request-local Q/K rotation into private temporary pages."""

import functools

import tilelang
import tilelang.language as T


@functools.lru_cache(maxsize=32)
@tilelang.jit(out_idx=[8, 9, 10])
def preprocess_kernel(
    batch,
    total_q,
    heads,
    heads_kv,
    dim,
    page,
    width,
    pool_rows,
    positions,
    rotary_dim,
    layout,
    dtype,
    request_threshold=0,
):
    half = rotary_dim // 2
    pairs = dim // 2
    capacity = page * width
    block = 1024
    q_blocks = tilelang.cdiv(total_q * heads * pairs, block)
    kv_blocks = tilelang.cdiv(capacity * heads_kv * pairs, block)

    @T.prim_func
    def main(
        Q: T.Tensor((total_q, heads, dim), dtype),
        K: T.Tensor((pool_rows, heads_kv, dim), dtype),
        V: T.Tensor((pool_rows, heads_kv, dim), dtype),
        Table: T.Tensor((batch, width), "int32"),
        Lengths: T.Tensor((batch,), "int32"),
        CuQ: T.Tensor((batch + 1,), "int32"),
        Cos: T.Tensor((positions, half), dtype),
        Sin: T.Tensor((positions, half), dtype),
        QR: T.Tensor((total_q, heads, dim), dtype),
        KR: T.Tensor((batch * capacity, heads_kv, dim), dtype),
        VR: T.Tensor((batch * capacity, heads_kv, dim), dtype),
    ):
        with T.Kernel(q_blocks + batch * kv_blocks, threads=256) as bx:
            if bx < q_blocks:
                for lane, item in T.Parallel(256, 4):
                    idx = bx * block + lane * 4 + item
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
                        if CuQ[lo + 1] - CuQ[lo] > request_threshold:
                            pos = token - CuQ[lo] + Lengths[lo] - (CuQ[lo + 1] - CuQ[lo])
                            if freq < half:
                                d0 = freq if layout == "neox" else freq * 2
                                d1 = freq + half if layout == "neox" else freq * 2 + 1
                                if Lengths[lo] > 0:
                                    c = T.cast(Cos[pos, freq], "float32")
                                    s = T.cast(Sin[pos, freq], "float32")
                                    x0 = T.cast(Q[token, head, d0], "float32")
                                    x1 = T.cast(Q[token, head, d1], "float32")
                                    QR[token, head, d0] = x0 * c - x1 * s
                                    QR[token, head, d1] = x1 * c + x0 * s
                                else:
                                    QR[token, head, d0] = 0
                                    QR[token, head, d1] = 0
                            else:
                                QR[token, head, freq * 2] = Q[token, head, freq * 2]
                                QR[token, head, freq * 2 + 1] = Q[token, head, freq * 2 + 1]
            else:
                request = (bx - q_blocks) // kv_blocks
                tile = (bx - q_blocks) % kv_blocks
                if CuQ[request + 1] - CuQ[request] > request_threshold:
                    for lane, item in T.Parallel(256, 4):
                        idx = tile * block + lane * 4 + item
                        key = idx // (heads_kv * pairs)
                        head = idx // pairs % heads_kv
                        freq = idx % pairs
                        if key < capacity:
                            dst = request * capacity + key
                            if key < Lengths[request] and CuQ[request + 1] > CuQ[request]:
                                src = Table[request, key // page] * page + key % page
                                if freq < half:
                                    d0 = freq if layout == "neox" else freq * 2
                                    d1 = freq + half if layout == "neox" else freq * 2 + 1
                                    c = T.cast(Cos[key, freq], "float32")
                                    s = T.cast(Sin[key, freq], "float32")
                                    x0 = T.cast(K[src, head, d0], "float32")
                                    x1 = T.cast(K[src, head, d1], "float32")
                                    KR[dst, head, d0] = x0 * c - x1 * s
                                    KR[dst, head, d1] = x1 * c + x0 * s
                                else:
                                    KR[dst, head, freq * 2] = K[src, head, freq * 2]
                                    KR[dst, head, freq * 2 + 1] = K[src, head, freq * 2 + 1]
                                VR[dst, head, freq * 2] = V[src, head, freq * 2]
                                VR[dst, head, freq * 2 + 1] = V[src, head, freq * 2 + 1]
                            else:
                                KR[dst, head, freq * 2] = 0
                                KR[dst, head, freq * 2 + 1] = 0
                                VR[dst, head, freq * 2] = 0
                                VR[dst, head, freq * 2 + 1] = 0

    return main
