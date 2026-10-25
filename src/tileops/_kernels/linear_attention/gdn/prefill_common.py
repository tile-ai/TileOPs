"""Gated DeltaNet (GDN) private helpers shared by the prefill stages."""

import tilelang.language as T

# What the comparator's L2 normalization adds under the square root before taking the
# reciprocal, from `fla.modules.l2norm.l2norm_fwd`. The block solve and the forward both
# form a reciprocal norm and must add the same thing.
L2NORM_EPS: float = 1e-6

# The edge of the blocks the chunk inverse works in; the solve that stages them and the
# macro that inverts them must agree on it.
INVERSE_BLOCK: int = 16


def step_size(raw, beta_sigmoid: bool, allow_neg_eigval: bool):
    """The delta-rule step size, from what the op handed the kernel.

    The sigmoid runs in float32 and the result returns to *raw*'s dtype, which is what a
    caller transforming ``beta`` itself would hand every stage. Keeping the wider value
    would give the triangular solve and the recurrence two different step sizes, since the
    solve stores it at the activation dtype and the recurrence in float32.

    Args:
        raw: The value read from ``beta``, already indexed.
        beta_sigmoid: ``beta`` carries raw logits rather than the step size.
        allow_neg_eigval: The transform is ``2 * sigmoid`` rather than ``sigmoid``.

    Returns:
        An expression in *raw*'s dtype, which is *raw* itself where no sigmoid applies.
    """
    if not beta_sigmoid:
        return raw
    scaled = T.sigmoid(T.cast(raw, "float32")) * (2.0 if allow_neg_eigval else 1.0)
    return T.Cast(raw.dtype, scaled)


def block_inverse_macro(accum_dtype: str, solve_dtype: str):
    """Four warps' inverse of a 64 x 64 unit lower-triangular matrix, in 16 x 16 blocks.

    The matrix is I + L with L strictly lower. ``lower`` holds L and ``inv`` receives the
    inverse, both as ``(4, 16, 64)`` buffers whose ``[r]`` is rows 16 r to 16 r + 15, so
    block (r, c) is ``[r, :, 16 c : 16 c + 16]``. The macro writes the inverse's blocks on
    and below the diagonal and leaves those above untouched. Each block is rounded to
    *solve_dtype* where the one-warp block solve rounds it, and its partial sums add their
    terms in that solve's order.

    Warp w inverts diagonal block w, then the blocks below the diagonal follow in three
    steps, each closed by a barrier all four warps arrive at:

    - step 1: T10, T21 and T32, and the first terms of T20, T30 and T31;
    - step 2: T20 and T31, and the second term of T30;
    - step 3: T30.

    Args:
        accum_dtype: The dtype each product accumulates in.
        solve_dtype: The dtype of ``lower``, ``inv`` and ``partial``.

    Returns:
        A ``T.macro`` taking ``(tx, base, lower, inv, partial, acc0, acc1, acc2, acc3,
        steps, parity)``, run by the 128 threads ``base`` to ``base + 127`` with ``tx`` the
        thread index: ``partial`` is a ``(7, 16, 16)`` scratch buffer, ``acc0`` to ``acc3``
        one 16 x 16 fragment for each warp, ``steps`` three barriers of 128 arrivals and
        ``parity`` their phase.
    """
    c = INVERSE_BLOCK

    def block(buf, r, col):
        return buf[r, :, col * c : (col + 1) * c]

    @T.macro
    def to_partial(a, b, acc, partial, slot):
        # partial[slot] = a b, rounded to the solve dtype.
        T.clear(acc)
        T.gemm(a, b, acc)
        for i, j in T.Parallel(c, c):
            partial[slot, i, j] = acc[i, j]
        T.sync_warp()

    @T.macro
    def add_partial(a, b, acc, partial, slot):
        # partial[slot] = partial[slot] + a b, the sum rounded to the solve dtype.
        T.clear(acc)
        T.gemm(a, b, acc)
        for i, j in T.Parallel(c, c):
            partial[slot, i, j] = partial[slot, i, j] + acc[i, j]
        T.sync_warp()

    @T.macro
    def to_inverse(a, b, acc, inv, r, col):
        # Block (r, col) of the inverse = -a b, rounded to the solve dtype.
        T.clear(acc)
        T.gemm(a, b, acc)
        for i, j in T.Parallel(c, c):
            inv[r, i, col * c + j] = -acc[i, j]
        T.sync_warp()

    @T.macro
    def invert(tx, base, lower, inv, partial, acc0, acc1, acc2, acc3, steps, parity):
        # Diagonal block w's inverse X solves X = I - L_ww X one row after another. Lane l
        # of warp w carries columns 2l and 2l + 1 in float32 registers; they share each
        # row of L_ww, and each column sums odd and even terms apart to shorten its chain.
        # The rows are stored once all are formed, so no row's loads wait behind the
        # previous row's stores.
        warp = (tx - base) // 32
        lane = (tx - base) % 32
        if lane < c // 2:
            x0 = T.alloc_local([c], accum_dtype)
            x1 = T.alloc_local([c], accum_dtype)
            # Row r of L_ww, read whole: a few wide loads instead of one per coefficient.
            arow = T.alloc_local([c], solve_dtype)
            col = lane * 2
            for r in T.unroll(c):
                for m in T.vectorized(c):
                    arow[m] = lower[warp, r, warp * c + m]
                e0 = T.alloc_var(accum_dtype)
                o0 = T.alloc_var(accum_dtype)
                e1 = T.alloc_var(accum_dtype)
                o1 = T.alloc_var(accum_dtype)
                e0 = T.if_then_else(r == col, T.float32(1.0), T.float32(0.0))
                e1 = T.if_then_else(r == col + 1, T.float32(1.0), T.float32(0.0))
                o0 = T.float32(0.0)
                o1 = T.float32(0.0)
                for m in T.unroll(r):
                    coef = -T.cast(arow[m], accum_dtype)
                    if m % 2 == 0:
                        e0 += coef * x0[m]
                        e1 += coef * x1[m]
                    else:
                        o0 += coef * x0[m]
                        o1 += coef * x1[m]
                x0[r] = e0 + o0
                x1[r] = e1 + o1
            for r in T.unroll(c):
                inv[warp, r, warp * c + col] = T.cast(x0[r], solve_dtype)
                inv[warp, r, warp * c + col + 1] = T.cast(x1[r], solve_dtype)
        T.barrier_arrive(steps[0])
        T.barrier_wait(steps[0], parity)

        # Step 1. Block (w, w - 1) = -X_w L_{w,w-1} X_{w-1} for w = 1, 2, 3; warp 0 forms
        # the first terms L20 X0, L30 X0 and L31 X1.
        if tx < base + 32:
            to_partial(block(lower, 2, 0), block(inv, 0, 0), acc0, partial, 4)
            to_partial(block(lower, 3, 0), block(inv, 0, 0), acc0, partial, 5)
            to_partial(block(lower, 3, 1), block(inv, 1, 1), acc0, partial, 6)
        elif tx < base + 64:
            to_partial(block(inv, 1, 1), block(lower, 1, 0), acc1, partial, 1)
            to_inverse(partial[1, :, :], block(inv, 0, 0), acc1, inv, 1, 0)
        elif tx < base + 96:
            to_partial(block(inv, 2, 2), block(lower, 2, 1), acc2, partial, 2)
            to_inverse(partial[2, :, :], block(inv, 1, 1), acc2, inv, 2, 1)
        else:
            to_partial(block(inv, 3, 3), block(lower, 3, 2), acc3, partial, 3)
            to_inverse(partial[3, :, :], block(inv, 2, 2), acc3, inv, 3, 2)
        T.barrier_arrive(steps[1])
        T.barrier_wait(steps[1], parity)

        # Step 2. T20 = -X2 (L20 X0 + L21 T10), T31 = -X3 (L31 X1 + L32 T21), and warp 0
        # adds L31 T10 to T30's sum.
        if tx < base + 32:
            add_partial(block(lower, 3, 1), block(inv, 1, 0), acc0, partial, 5)
        elif tx < base + 64:
            add_partial(block(lower, 2, 1), block(inv, 1, 0), acc1, partial, 4)
            to_inverse(block(inv, 2, 2), partial[4, :, :], acc1, inv, 2, 0)
        elif tx < base + 96:
            add_partial(block(lower, 3, 2), block(inv, 2, 1), acc2, partial, 6)
            to_inverse(block(inv, 3, 3), partial[6, :, :], acc2, inv, 3, 1)
        T.barrier_arrive(steps[2])
        T.barrier_wait(steps[2], parity)

        # Step 3. T30 = -X3 (L30 X0 + L31 T10 + L32 T20).
        if tx >= base + 96:
            add_partial(block(lower, 3, 2), block(inv, 2, 0), acc3, partial, 5)
            to_inverse(block(inv, 3, 3), partial[5, :, :], acc3, inv, 3, 0)

    return invert
