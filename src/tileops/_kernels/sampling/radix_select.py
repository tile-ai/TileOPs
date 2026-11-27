"""The radix select of a row's k-th largest key that both top-k masks run.

A cluster of CTAs holds a row in registers and settles the k-th largest key one 8-bit digit
at a time. Here are the row's layout in registers, its load, the key transform, the select
itself, and the launch policy that decides which rows fit and how wide a row spreads. What
each mask does with the threshold stays in its own file.
"""

import operator
from types import SimpleNamespace

import tilelang.language as T
import torch

from tileops._kernels.constants import MAX_PORTABLE_CLUSTER_BLOCKS, VECTOR_ACCESS_BYTES
from tileops._kernels.warp_reduce import warp_reduce
from tileops._utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["cluster_plan", "key_select", "row_refusal"]


@T.macro
def merge_counts(hist, total, sum32, q, tx, cluster: int, slabs: int, threads: int):
    """Sum slab ``q`` of every CTA of the cluster into ``total``."""
    if cluster == 1:
        T.sync_threads()
        for i in T.serial(-(-slabs // threads)):
            if i * threads + tx < slabs:
                total[i * threads + tx] = hist[q * slabs + i * threads + tx]
    else:
        T.cluster_sync()
        for i in T.serial(-(-slabs // threads)):
            if i * threads + tx < slabs:
                sum32[0] = T.uint32(0)
                for c in T.serial(cluster):
                    sum32[0] = sum32[0] + T.call_extern(
                        "uint32",
                        "tileops::cluster_load_u32",
                        T.address_of(hist[q * slabs + i * threads + tx]),
                        c,
                    )
                total[i * threads + tx] = sum32[0]
    T.sync_threads()


@T.macro
def rank_in_bins(total, target, acc, tx, span: int):
    """Warp 0: into ``acc``, the bin of ``total`` holding rank ``target`` counted from
    the largest, that rank inside it, and the bin's count.

    Lane ``tx`` owns ``span`` bins. ``acc[0]`` stays -1 when the bins hold fewer than
    ``target`` keys.
    """
    acc[0] = -1
    acc[3] = 0
    for i in T.serial(span):
        acc[3] = acc[3] + T.cast(total[tx * span + i], "int32")
    acc[1] = acc[3]
    for stage in T.unroll(WARP_SHUFFLE_STAGES):
        up = T.shfl_down(acc[1], 1 << stage, width=WARP_LANES)
        if tx + (1 << stage) < WARP_LANES:
            acc[1] = acc[1] + up
    # This lane's bins hold the ranks in (acc[1] - acc[3], acc[1]].
    acc[2] = acc[1] - acc[3]
    if (acc[2] < target) & (target <= acc[1]):
        for i in T.serial(span):
            b = tx * span + span - 1 - i
            count = T.cast(total[b], "int32")
            if (acc[2] < target) & (target <= acc[2] + count):
                acc[0] = b
                acc[1] = target - acc[2]
                acc[3] = count
            acc[2] = acc[2] + count


def cluster_limit(arch: int) -> int:
    """CTAs one row may span: thread-block clusters start at SM90."""
    return MAX_PORTABLE_CLUSTER_BLOCKS if arch >= 90 else 1


def row_refusal(call, threads: int, max_slots: int) -> str | None:
    """Why a launch of ``threads`` a CTA cannot hold the call's rows, or ``None``.

    ``max_slots`` bounds a thread's slots where a cluster spreads a row.
    """
    if call.batch * call.vocab > 2**31 - 1:
        return f"indexes elements with int32, and B * V = {call.batch * call.vocab}"
    # Where a row stays in one CTA: the fewest slots holding a 128256-value float32 vocabulary
    # at 1024 threads; those past the registers spill to local memory.
    max_slots_one_cta = 32
    slots = max_slots if cluster_limit(call.arch) > 1 else max_slots_one_cta
    widest = (
        cluster_limit(call.arch) * threads * (VECTOR_ACCESS_BYTES // call.dtype.itemsize) * slots
    )
    if call.vocab > widest:
        return f"supports rows of at most {widest} values, and V = {call.vocab}"
    return None


def cluster_plan(call, threads: int, max_slots: int) -> dict:
    """As many CTAs per row as it takes to fill the device, each holding an equal run.

    A cluster costs a barrier across its CTAs on every pass, so a batch that already fills
    the device puts one CTA on each row and only a batch short of it spreads a row wider; a
    row too long to hold in ``max_slots`` per thread spreads wider still. The run is then
    exactly the row's share, since a slot no key lands in is a register the passes scan for
    nothing.
    """
    vec = VECTOR_ACCESS_BYTES // call.dtype.itemsize
    limit = cluster_limit(call.arch)
    cluster = 1
    while cluster < limit and 2 * cluster * call.batch <= call.sm_count:
        cluster *= 2
    slots = -(-call.vocab // (cluster * threads * vec))
    while cluster < limit and slots > max_slots:
        cluster *= 2
        slots = -(-call.vocab // (cluster * threads * vec))
    return {"threads": threads, "cluster": cluster, "slots": slots}


def key_select(vocab: int, dtype: str, threads: int, cluster: int, slots: int) -> SimpleNamespace:
    """The register layout of a row of ``vocab`` logits and the macros that select its k-th key.

    A CTA holds ``slots`` 16-byte vectors a thread, as ``words`` 32-bit words each. A value
    is held as its key: its bits with the sign bit flipped when it is non-negative and every
    bit flipped when it is negative, so keys order as the values do, and every NaN takes the
    largest key, as ``sort`` places NaN first. A row's cluster pads its last vectors with
    -inf, which no rank below ``vocab`` reaches.
    """
    # A missed sample bracket adds a radix pass without changing the selected rank.
    bracket_sigmas = 4.0
    bracket_slack = 3.0
    itemsize = torch.empty((), dtype=getattr(torch, dtype)).element_size()
    vec = VECTOR_ACCESS_BYTES // itemsize
    words = VECTOR_ACCESS_BYTES // 4
    per_word = 4 // itemsize
    packed = per_word == 2
    aligned = vocab % vec == 0
    bits = 8 * itemsize
    sign = 1 << (bits - 1)
    inf_bits = {"float16": 0x7C00, "bfloat16": 0x7F80, "float32": 0x7F800000}[dtype]
    neg_inf = sign | inf_bits
    # Each 16-bit half of a word, or the whole word.
    halves = 0x00010001 if packed else 1
    word_neg_inf = neg_inf * halves
    # One pass of the select settles one digit of the key, over this many bins.
    radix_bits = 8
    bins = 1 << radix_bits
    digits = bits // radix_bits
    # What the cluster reduces per pass: one count per digit value, plus the keys whose
    # digit is above the ones the pass counts.
    slabs = bins + 1
    chunk = slots * threads * vec
    assert cluster * chunk >= vocab
    held = slots * words
    warps = threads // WARP_LANES
    assert threads % WARP_LANES == 0 and bins % WARP_LANES == 0
    # The threads whose first element is inside the row are the ones that sample it, and
    # their count is the population a sample rank stands for.
    if aligned:
        in_row = [max(0, min(threads, -(-(vocab - c * chunk) // vec))) for c in range(cluster)]
    else:
        in_row = [max(0, min(threads, vocab - c * chunk)) for c in range(cluster)]
    sampled = sum(in_row)
    # Every digit, plus one retry of the first for a bracket that missed.
    rounds = digits + 1

    def element(j, c, h):
        """The offset in the CTA's chunk of half ``h`` of word ``c`` of slot ``j``."""
        if aligned:
            return (j * threads + T.get_thread_binding()) * vec + c * per_word + h
        return ((j * words + c) * per_word + h) * threads + T.get_thread_binding()

    def key_of(vals, i, h):
        """Key ``h`` of word ``i`` of the held keys."""
        if packed:
            return (vals[i] >> T.uint32(16 * h)) & T.uint32(0xFFFF)
        return vals[i]

    def keeps_both_zeros(key):
        """``key``, lowered to the key of -0.0 when it is the key of either zero, so that a
        bound on one zero keeps the other, as a float comparison does."""
        return T.if_then_else(
            (key == T.uint32(sign)) | (key == T.uint32(sign - 1)),
            T.uint32(sign - 1),
            key,
        )

    @T.macro
    def load_row(vals, x, base, out_of_row):
        """Read the CTA's chunk of the row at ``base`` into ``vals``, -inf past the row."""
        for j in T.unroll(slots):
            if aligned:
                if element(j, 0, 0) < out_of_row:
                    T.call_extern(
                        "handle",
                        "tileops::load16_evict_first",
                        T.address_of(vals[j * words]),
                        T.address_of(x[base + element(j, 0, 0)]),
                    )
                else:
                    for c in T.unroll(words):
                        vals[j * words + c] = T.uint32(word_neg_inf)
            else:
                for c in T.unroll(words):
                    vals[j * words + c] = T.uint32(word_neg_inf)
                    for h in T.unroll(per_word):
                        if element(j, c, h) < out_of_row:
                            if packed:
                                piece = T.cast(
                                    T.reinterpret(x[base + element(j, c, h)], "uint16"),
                                    "uint32",
                                )
                                vals[j * words + c] = (
                                    vals[j * words + c]
                                    & T.uint32(0xFFFF0000 if h == 0 else 0x0000FFFF)
                                ) | (piece << T.uint32(16 * h))
                            else:
                                vals[j * words + c] = T.reinterpret(
                                    x[base + element(j, c, h)], "uint32"
                                )

    @T.macro
    def to_keys(vals):
        """Replace each held value by its key."""
        for i in T.unroll(held):
            if packed:
                flip = ((vals[i] >> T.uint32(15)) & T.uint32(halves)) * T.uint32(0x7FFF)
                nan = T.call_extern(
                    "uint32",
                    "__vcmpgtu2",
                    vals[i] & T.uint32(0x7FFF7FFF),
                    T.uint32(inf_bits * halves),
                )
                vals[i] = (vals[i] ^ (flip | T.uint32(0x80008000))) | nan
            else:
                vals[i] = T.if_then_else(
                    (vals[i] & T.uint32(0x7FFFFFFF)) > T.uint32(inf_bits),
                    T.uint32(0xFFFFFFFF),
                    vals[i]
                    ^ T.if_then_else(
                        (vals[i] >> T.uint32(31)) != T.uint32(0),
                        T.uint32(0xFFFFFFFF),
                        T.uint32(0x80000000),
                    ),
                )

    @T.macro
    def store_slot(word, j, out_of_row, dst, at):
        """Write the values of slot ``j``, held in ``word``, to ``dst`` at chunk ``at``."""
        if aligned:
            if element(j, 0, 0) < out_of_row:
                T.call_extern(
                    "handle",
                    "tileops::store16",
                    T.address_of(dst[at + element(j, 0, 0)]),
                    T.address_of(word[0]),
                )
        else:
            for c in T.unroll(words):
                for h in T.unroll(per_word):
                    if element(j, c, h) < out_of_row:
                        if packed:
                            dst[at + element(j, c, h)] = T.reinterpret(
                                T.cast((word[c] >> T.uint32(16 * h)) & T.uint32(0xFFFF), "uint16"),
                                dtype,
                            )
                        else:
                            dst[at + element(j, c, h)] = T.reinterpret(word[c], dtype)

    @T.macro
    def select_kth(
        vals,
        kk,
        done,
        out_of_row,
        hist,
        total,
        red,
        pre,
        st,
        acc,
        sum32,
        above,
        digit_shift,
        settled,
        counted,
    ):
        """Leave in ``pre[0]`` the smallest key carrying the ``kk``-th largest key's settled
        digits; ``done`` nonzero skips the select and leaves ``pre[0]`` zero.

        ``st`` holds: 0 the rank still to find inside the settled bits, 1 whether the select
        has finished, 2 the winning bin and 3 the rank inside it, 4 whether this pass
        settled the threshold, 5 the digit being settled, 6 and 7 the lowest and highest
        digit value the pass counts, 8 whether the pass found the rank. Every CTA of a
        cluster reduces the same counts, so all of them finish on the same pass.
        ``digit_shift``, ``settled`` and ``counted`` are thread-local copies of the pass's
        shift, settled bits and counted digit values: every mention of a shared element is a
        load of it, and the loop over the held keys mentions each of them per key.
        """
        tx = T.get_thread_binding()
        # The digit values that can hold the k-th key, from one sample per thread.
        for i in T.serial(-(-slabs // threads)):
            if i * threads + tx < slabs:
                hist[i * threads + tx] = T.uint32(0)
        if tx == 0:
            st[6] = 0
            st[7] = bins - 1
        T.sync_threads()
        if element(0, 0, 0) < out_of_row:
            sample = (vals[0] & T.uint32(0xFFFF)) if packed else vals[0]
            T.atomic_add(
                hist[T.cast(sample >> T.uint32(bits - radix_bits), "int32")],
                T.uint32(1),
            )
        merge_counts(hist, total, sum32, 0, tx, cluster, slabs, threads)
        if tx < WARP_LANES:
            expected = T.cast(kk, "float32") * T.float32(sampled / vocab)
            spread = T.float32(bracket_sigmas) * T.sqrt(expected) + T.float32(bracket_slack)
            rank_in_bins(
                total,
                T.cast(T.floor(expected - spread), "int32"),
                acc,
                tx,
                bins // WARP_LANES,
            )
            if acc[0] >= 0:
                st[7] = acc[0]
            rank_in_bins(
                total,
                T.cast(T.ceil(expected + spread), "int32"),
                acc,
                tx,
                bins // WARP_LANES,
            )
            if acc[0] >= 0:
                st[6] = acc[0]
        if tx == 0:
            pre[0] = T.uint32(0)
            st[0] = kk
            st[1] = done
            st[5] = 0
        T.sync_threads()

        for q in T.serial(rounds):
            if st[1] == 0:
                # The digit this pass settles starts here; the bits below it are what a
                # pass that settles the threshold leaves unset.
                digit_shift[0] = T.uint32(digits - 1 - st[5]) * T.uint32(radix_bits)
                settled[0] = pre[0]
                counted[0] = st[6]
                counted[1] = st[7]
                # ``hist`` alternates slabs, so a peer still reading the last pass's counts never
                # sees them zeroed; the samples hold slab 0.
                slab = (q + 1) % 2
                for i in T.serial(-(-slabs // threads)):
                    if i * threads + tx < slabs:
                        hist[slab * slabs + i * threads + tx] = T.uint32(0)
                if tx == 0:
                    st[4] = 0
                    st[8] = 0
                T.sync_threads()
                above[0] = 0
                for i in T.unroll(held):
                    for h in T.unroll(per_word):
                        key = key_of(vals, i, h)
                        # Two shifts, so the first digit does not shift by the whole key
                        # width.
                        moved = key >> digit_shift[0]
                        if (moved >> T.uint32(radix_bits)) == settled[0]:
                            digit = T.cast(moved & T.uint32(bins - 1), "int32")
                            if digit > counted[1]:
                                above[0] = above[0] + 1
                            elif digit >= counted[0]:
                                T.atomic_add(hist[slab * slabs + digit], T.uint32(1))
                warp_reduce(above, operator.add)
                if tx % WARP_LANES == 0:
                    red[tx // WARP_LANES] = above[0]
                T.sync_threads()
                if tx == 0:
                    acc[0] = 0
                    for w in T.serial(warps):
                        acc[0] = acc[0] + red[w]
                    hist[slab * slabs + bins] = T.cast(acc[0], "uint32")
                merge_counts(hist, total, sum32, slab, tx, cluster, slabs, threads)
                if tx < WARP_LANES:
                    # The keys above the counted digits already outrank the target, so the
                    # counted bins hold what is left of it.
                    acc[0] = -1
                    if T.cast(total[bins], "int32") < st[0]:
                        rank_in_bins(
                            total,
                            st[0] - T.cast(total[bins], "int32"),
                            acc,
                            tx,
                            bins // WARP_LANES,
                        )
                    if acc[0] >= 0:
                        st[2] = acc[0]
                        st[3] = acc[1]
                        st[8] = 1
                        # One key carries this prefix, so its remaining bits cannot change
                        # which keys the mask keeps.
                        if acc[3] == 1:
                            st[4] = 1
                T.sync_threads()
                if tx == 0:
                    # A bracket that did not hold the rank counts every digit on the retry,
                    # which always holds it.
                    st[6] = 0
                    st[7] = bins - 1
                    if st[8] == 1:
                        # A pass that settled the threshold leaves the bits below its digit
                        # zero, the smallest key carrying it.
                        pre[0] = (
                            (pre[0] << T.uint32(radix_bits)) | T.cast(st[2], "uint32")
                        ) << T.if_then_else(st[4] == 1, digit_shift[0], T.uint32(0))
                        st[0] = st[3]
                        st[5] = st[5] + 1
                    # Set last: the parser re-reads the guard of this pass's own region
                    # after every barrier inside it.
                    st[1] = T.if_then_else((st[4] == 1) | (st[5] == digits), 1, 0)
                T.sync_threads()

    return SimpleNamespace(
        words=words,
        per_word=per_word,
        packed=packed,
        bits=bits,
        sign=sign,
        halves=halves,
        word_neg_inf=word_neg_inf,
        digits=digits,
        slabs=slabs,
        chunk=chunk,
        held=held,
        warps=warps,
        key_of=key_of,
        keeps_both_zeros=keeps_both_zeros,
        load_row=load_row,
        to_keys=to_keys,
        store_slot=store_slot,
        select_kth=select_kth,
    )
