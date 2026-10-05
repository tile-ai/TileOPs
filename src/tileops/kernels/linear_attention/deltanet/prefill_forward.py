"""Ungated DeltaNet prefill forward: each partition's output from its starting state."""

import functools

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.linear_attention.gdn.prefill_common import L2NORM_EPS

__all__ = ["deltanet_prefill_fwd"]


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
    },
    compile_flags=["-O3", "-DENABLE_BF16"],
)
def _build_prefill_fwd_kernel(
    heads: int,
    dim: int,
    scale: float,
    dtype: str,
    state_dtype: str,
    seqlen_dtype: str,
    use_initial_state: bool,
    is_cp: bool,
    l2norm: bool,
    num_stages: int = 2,
):
    """One CTA walks one partition's chunks for one head and 64 state columns.

    Each chunk applies ``Vd = Ab (V - K S)`` with ``Ab = A diag(beta)``, then
    ``O = scale Q S + tril(scale Q K^T) Vd`` and ``S += K^T Vd``. ``W = Ab K`` and
    ``U = Ab V`` do not depend on the state, so a warp group forms them a chunk ahead. The
    state is held value-major, so the chain's two products, ``S^T W^T`` and ``Vd^T K``,
    take their left operand from registers and the chain never waits on shared memory.
    """
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    raw_batch_size = T.dynamic("raw_batch_size")
    C = 64
    # The value-major state is the left operand of a warp group's product, which takes 64
    # rows, so one CTA carries 64 state columns.
    block_dv = 64
    accum_dtype = "float32"
    # A build that does not normalize never reads this tensor, and the host hands it one
    # token so the allocation carries no cost.
    rnorm_tokens = num_tokens if l2norm else 1
    # Key columns one pass of the query's row reduction holds.
    reduce_width = 32

    @T.prim_func
    def prefill_fwd_kernel(
        q: T.Tensor((1, num_tokens, heads, dim), dtype),
        k: T.Tensor((1, num_tokens, heads, dim), dtype),
        v: T.Tensor((1, num_tokens, heads, dim), dtype),
        a: T.Tensor((1, num_tokens, heads, C), dtype),
        b: T.Tensor((1, num_tokens, heads), dtype),
        k_rnorm: T.Tensor((1, rnorm_tokens, heads), accum_dtype),
        h0: T.Tensor((raw_batch_size, heads, dim, dim), state_dtype),
        partition_h: T.Tensor((batch_size, heads, dim, dim), dtype),
        partition_m: T.Tensor((batch_size, heads, dim, dim), dtype),
        cu_seqlens: T.Tensor([batch_size + 1], seqlen_dtype),
        cp_seq_map: T.Tensor([batch_size], seqlen_dtype),
        raw_cu_seqlens: T.Tensor([raw_batch_size + 1], seqlen_dtype),
        first_partition: T.Tensor([raw_batch_size + 1], seqlen_dtype),
        o: T.Tensor((1, num_tokens, heads, dim), dtype),
        ht: T.Tensor((raw_batch_size, heads, dim, dim), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(dim, block_dv) * batch_size * heads, threads=512) as (bbhv,):
            bbh, bv = bbhv // T.ceildiv(dim, block_dv), bbhv % T.ceildiv(dim, block_dv)
            bb, bh = bbh // heads, bbh % heads
            col = bv * block_dv

            seq_start = T.alloc_var("int32")
            seq_end = T.alloc_var("int32")
            seq_start = cu_seqlens[bb]
            seq_end = cu_seqlens[bb + 1]
            raw_idx = T.alloc_var("int32")
            raw_idx = cp_seq_map[bb] if is_cp else bb
            store_state = T.alloc_var("bool")
            store_state = (raw_cu_seqlens[raw_idx + 1] == seq_end) if is_cp else True
            num_iters = T.alloc_var("int32")
            num_iters = T.ceildiv(seq_end - seq_start, C)
            # Partitions of this sequence ahead of this one, whose transitions the state
            # passes through before this partition starts.
            first = T.alloc_var("int32")
            first = first_partition[raw_idx] if is_cp else bb
            num_prior = T.alloc_var("int32")
            num_prior = bb - first

            q_shared = T.alloc_shared((num_stages, C, dim), dtype)
            k_shared = T.alloc_shared((num_stages, C, dim), dtype)
            v_shared = T.alloc_shared((num_stages, C, block_dv), dtype)
            a_shared = T.alloc_shared((num_stages, C, C), dtype)
            b_shared = T.alloc_shared((num_stages, C), accum_dtype, scope="shared")
            w_shared = T.alloc_shared((2, C, dim), dtype)
            ut_shared = T.alloc_shared((2, block_dv, C), dtype)
            st_shared = T.alloc_shared((2, block_dv, dim), dtype)
            vdt_shared = T.alloc_shared((2, block_dv, C), dtype)
            o_shared = T.alloc_shared((2, C, block_dv), dtype)
            if is_cp:
                pm_shared = T.alloc_shared((2, dim, dim), dtype)
                ph_shared = T.alloc_shared((2, dim, block_dv), dtype)
                pmh_ready = T.alloc_barrier(arrive_count=[32] * 2)
                pmh_free = T.alloc_barrier(arrive_count=[128] * 2)
            if l2norm:
                kn_shared = T.alloc_shared((num_stages, C), accum_dtype, scope="shared")
                qn_shared = T.alloc_shared((C,), accum_dtype, scope="shared")

            # Stage ``i % num_stages`` holds chunk ``i``'s inputs, buffer ``i % 2`` its products.
            data_ready = T.alloc_barrier(arrive_count=[96] * num_stages)
            data_free = T.alloc_barrier(arrive_count=[384] * num_stages)
            wu_ready = T.alloc_barrier(arrive_count=[128] * 2)
            wu_free = T.alloc_barrier(arrive_count=[128] * 2)
            s_ready = T.alloc_barrier(arrive_count=[128] * 2)
            vd_ready = T.alloc_barrier(arrive_count=[128] * 2)
            s_free = T.alloc_barrier(arrive_count=[128] * 2)
            ab_ready = T.alloc_barrier(arrive_count=128)
            o_ready = T.alloc_barrier(arrive_count=128)

            # Both sides of these buffers use the accumulator's fragment layout, eight rows a
            # warp apart; swizzling keeps those rows off one bank.
            T.annotate_layout(
                {
                    ut_shared: tilelang.layout.make_swizzled_layout(ut_shared),
                    o_shared: tilelang.layout.make_swizzled_layout(o_shared),
                }
            )

            tx = T.get_thread_binding()

            if tx < 128:
                # The state's chain, value-major: S^T -> S^T W^T -> Vd^T -> Vd^T K -> S^T. Both
                # products read their left operand from registers.
                T.set_max_nreg(168, 1)
                s_fragment = T.alloc_fragment((block_dv, dim), accum_dtype)
                s_operand = T.alloc_fragment((block_dv, dim), dtype)
                t_fragment = T.alloc_fragment((block_dv, C), accum_dtype)
                vd_operand = T.alloc_fragment((block_dv, C), dtype)
                if use_initial_state:
                    for i, j in T.Parallel(block_dv, dim):
                        s_fragment[i, j] = h0[raw_idx, bh, j, col + i]
                else:
                    T.clear(s_fragment)
                if is_cp:
                    # S^T <- S^T M_p^T + H_p^T through every earlier partition of the sequence.
                    for j_p in T.serial(num_prior):
                        slot = j_p % 2
                        T.copy(s_fragment, s_operand)
                        T.barrier_wait(pmh_ready[slot], (j_p // 2) % 2)
                        T.gemm(
                            s_operand,
                            pm_shared[slot, :, :],
                            s_fragment,
                            transpose_B=True,
                            clear_accum=True,
                        )
                        for i, j in T.Parallel(block_dv, dim):
                            s_fragment[i, j] += T.cast(ph_shared[slot, j, i], accum_dtype)
                        T.barrier_arrive(pmh_free[slot])

                for i_s in T.serial(num_iters):
                    st = i_s % 2
                    ds = i_s % num_stages
                    T.copy(s_fragment, s_operand)
                    T.barrier_wait(wu_ready[st], (i_s // 2) % 2)
                    T.gemm(
                        s_operand,
                        w_shared[st, :, :],
                        t_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.barrier_wait(s_free[st], (i_s // 2 + 1) % 2)
                    T.copy(s_operand, st_shared[st, :, :])
                    T.fence_proxy_async()
                    T.barrier_arrive(s_ready[st])
                    # A row past the sequence's end reads a row of A that is not this
                    # chunk's, so its update is zeroed here rather than trusted to beta.
                    for j_v, j_s in T.Parallel(block_dv, C):
                        t_fragment[j_v, j_s] = T.if_then_else(
                            seq_start + i_s * C + j_s < seq_end,
                            T.cast(ut_shared[st, j_v, j_s], accum_dtype) - t_fragment[j_v, j_s],
                            0.0,
                        )
                    T.copy(t_fragment, vd_operand)
                    T.gemm(vd_operand, k_shared[ds, :, :], s_fragment, clear_accum=False)
                    T.copy(vd_operand, vdt_shared[st, :, :])
                    T.fence_proxy_async()
                    T.barrier_arrive(vd_ready[st])
                    T.barrier_arrive(wu_free[st])
                    T.barrier_arrive(data_free[ds])

                if store_state:
                    for i, j in T.Parallel(block_dv, dim):
                        ht[raw_idx, bh, j, col + i] = s_fragment[i, j]

            elif tx < 256:
                # W = Ab K and U^T = V^T Ab^T, a chunk ahead of the chain.
                T.set_max_nreg(152, 1)
                w_fragment = T.alloc_fragment((C, dim), accum_dtype)
                ut_fragment = T.alloc_fragment((block_dv, C), accum_dtype)
                if l2norm:
                    square_fragment = T.alloc_fragment((C, reduce_width), accum_dtype)
                    sumsq_fragment = T.alloc_fragment((C,), accum_dtype)

                for i_s in T.serial(num_iters):
                    st = i_s % 2
                    ds = i_s % num_stages
                    dp = (i_s // num_stages) % 2
                    T.barrier_wait(data_ready[ds], dp)
                    T.barrier_wait(wu_free[st], (i_s // 2 + 1) % 2)
                    if l2norm:
                        # Every stage reads the normalized query and key, so both are
                        # normalized once in place. The key's reciprocal norm came from the
                        # block solve; the query's is reduced here.
                        T.clear(sumsq_fragment)
                        for start in T.serial(dim // reduce_width):
                            T.copy(
                                q_shared[ds, :, start * reduce_width : (start + 1) * reduce_width],
                                square_fragment,
                            )
                            for j_s, j_k in T.Parallel(C, reduce_width):
                                square_fragment[j_s, j_k] *= square_fragment[j_s, j_k]
                            T.reduce_sum(square_fragment, sumsq_fragment, dim=1, clear=False)
                        for j_s in T.Parallel(C):
                            qn_shared[j_s] = T.rsqrt(sumsq_fragment[j_s] + L2NORM_EPS)
                    for j_s, j_t in T.Parallel(C, C):
                        a_shared[ds, j_s, j_t] = T.cast(
                            T.cast(a_shared[ds, j_s, j_t], accum_dtype) * b_shared[ds, j_t], dtype
                        )
                    if l2norm:
                        for j_s, j_k in T.Parallel(C, dim):
                            k_shared[ds, j_s, j_k] = T.cast(
                                T.cast(k_shared[ds, j_s, j_k], accum_dtype) * kn_shared[ds, j_s],
                                dtype,
                            )
                            q_shared[ds, j_s, j_k] = T.cast(
                                T.cast(q_shared[ds, j_s, j_k], accum_dtype) * qn_shared[j_s], dtype
                            )
                    T.fence_proxy_async()
                    T.barrier_arrive(ab_ready)
                    T.barrier_wait(ab_ready, i_s % 2)
                    T.gemm(a_shared[ds, :, :], k_shared[ds, :, :], w_fragment, clear_accum=True)
                    T.copy(w_fragment, w_shared[st, :, :])
                    T.gemm(
                        v_shared[ds, :, :],
                        a_shared[ds, :, :],
                        ut_fragment,
                        transpose_A=True,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.copy(ut_fragment, ut_shared[st, :, :])
                    T.fence_proxy_async()
                    T.barrier_arrive(wu_ready[st])
                    T.barrier_arrive(data_free[ds])

            elif tx < 384:
                # O = scale Q S + tril(scale Q K^T) Vd, off the state's chain.
                T.set_max_nreg(152, 1)
                o_fragment = T.alloc_fragment((C, block_dv), accum_dtype)
                p_fragment = T.alloc_fragment((C, C), accum_dtype)
                p_operand = T.alloc_fragment((C, C), dtype)
                for i_s in T.serial(num_iters):
                    st = i_s % 2
                    ds = i_s % num_stages
                    T.barrier_wait(s_ready[st], (i_s // 2) % 2)
                    T.gemm(
                        q_shared[ds, :, :],
                        k_shared[ds, :, :],
                        p_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    for j_s, j_t in T.Parallel(C, C):
                        p_fragment[j_s, j_t] = T.if_then_else(
                            j_s >= j_t, p_fragment[j_s, j_t] * scale, 0.0
                        )
                    T.copy(p_fragment, p_operand)
                    T.gemm(
                        q_shared[ds, :, :],
                        st_shared[st, :, :],
                        o_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    for j_s, j_v in T.Parallel(C, block_dv):
                        o_fragment[j_s, j_v] *= scale
                    T.barrier_wait(vd_ready[st], (i_s // 2) % 2)
                    T.gemm(p_operand, vdt_shared[st, :, :], o_fragment, transpose_B=True)
                    T.barrier_arrive(s_free[st])
                    T.barrier_arrive(data_free[ds])
                    # The output is staged through shared memory so rows are written whole;
                    # a buffer is rewritten two chunks later, after every thread has passed
                    # the next chunk's barrier.
                    T.copy(o_fragment, o_shared[st, :, :])
                    T.barrier_arrive(o_ready)
                    T.barrier_wait(o_ready, i_s % 2)
                    left = seq_start + i_s * C
                    for j_s, j_v in T.Parallel(C, block_dv):
                        if left + j_s < seq_end:
                            o[0, left + j_s, bh, col + j_v] = o_shared[st, j_s, j_v]

            else:
                T.set_max_nreg(40, 0)
                if tx < 384 + 32:
                    for i_s in T.serial(num_iters):
                        ds = i_s % num_stages
                        dp = (i_s // num_stages) % 2
                        T.barrier_wait(data_free[ds], 1 - dp)
                        left = seq_start + i_s * C
                        T.tma_copy(
                            q[0, left : left + C, bh, 0:dim],
                            q_shared[ds, :, :],
                            barrier=data_ready[ds],
                        )
                        T.tma_copy(
                            k[0, left : left + C, bh, 0:dim],
                            k_shared[ds, :, :],
                            barrier=data_ready[ds],
                        )
                        T.barrier_arrive(data_ready[ds])
                elif tx < 384 + 64:
                    for i_s in T.serial(num_iters):
                        ds = i_s % num_stages
                        dp = (i_s // num_stages) % 2
                        T.barrier_wait(data_free[ds], 1 - dp)
                        left = seq_start + i_s * C
                        T.tma_copy(
                            v[0, left : left + C, bh, col : col + block_dv],
                            v_shared[ds, :, :],
                            barrier=data_ready[ds],
                        )
                        T.tma_copy(
                            a[0, left : left + C, bh, 0:C],
                            a_shared[ds, :, :],
                            barrier=data_ready[ds],
                        )
                        T.barrier_arrive(data_ready[ds])
                elif tx < 384 + 96:
                    for i_s in T.serial(num_iters):
                        ds = i_s % num_stages
                        dp = (i_s // num_stages) % 2
                        T.barrier_wait(data_free[ds], 1 - dp)
                        left = seq_start + i_s * C
                        for j_s in T.Parallel(C):
                            b_shared[ds, j_s] = T.if_then_else(
                                left + j_s < seq_end,
                                T.cast(b[0, T.min(left + j_s, seq_end - 1), bh], accum_dtype),
                                0.0,
                            )
                        if l2norm:
                            for j_s in T.Parallel(C):
                                kn_shared[ds, j_s] = k_rnorm[0, T.min(left + j_s, seq_end - 1), bh]
                        T.barrier_arrive(data_ready[ds])
                elif is_cp:
                    for j_p in T.serial(num_prior):
                        slot = j_p % 2
                        T.barrier_wait(pmh_free[slot], (j_p // 2 + 1) % 2)
                        T.tma_copy(
                            partition_m[first + j_p, bh, 0:dim, 0:dim],
                            pm_shared[slot, :, :],
                            barrier=pmh_ready[slot],
                        )
                        T.tma_copy(
                            partition_h[first + j_p, bh, 0:dim, col : col + block_dv],
                            ph_shared[slot, :, :],
                            barrier=pmh_ready[slot],
                        )
                        T.barrier_arrive(pmh_ready[slot])

    return prefill_fwd_kernel


def deltanet_prefill_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    partitions: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | None = None,
    k_rnorm: torch.Tensor | None = None,
    l2norm: bool = False,
    num_stages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed ungated DeltaNet prefill output and final state.

    Args:
        q, k, v: ``[1, num_tokens, heads, dim]`` packed activations, ``dim`` 64 or 128,
            and 64 where ``partitions`` is given.
        a: ``[1, num_tokens, heads, 64]`` per-chunk ``(I + strict_tril(diag(beta) K K^T))^-1``.
        beta: ``[1, num_tokens, heads]`` step sizes.
        scale: Query scale.
        initial_state: ``[num_sequences, heads, dim, dim]`` starting state of each sequence,
            or ``None`` for a zero start.
        cu_seqlens: Sequence offsets.
        partitions: Where sequences are split, ``(offsets, sequence_of, first_of,
            partition_h, partition_m)``: int32 partition offsets, each partition's sequence,
            each sequence's first partition followed by the partition count, and each
            partition's state from a zero start and its transition. A partition reaches its
            starting state through every earlier partition of its sequence.
        k_rnorm: ``[1, num_tokens, heads]`` reciprocal key norms from the block solve.
        l2norm: Normalize the query and the key.
        num_stages: Chunks of input the copy engine fetches ahead; by default three for a
            64-wide state and the two a 128-wide state's shared memory holds.

    Returns:
        The output, like ``v``, and the float32 final state of every sequence.
    """
    _, _, heads, dim = k.shape
    num_sequences = cu_seqlens.shape[0] - 1
    is_cp = partitions is not None
    if num_stages is None:
        num_stages = 3 if dim == 64 else 2
    if is_cp:
        rows, sequence_of, first_of, partition_h, partition_m = partitions
    else:
        rows, sequence_of, first_of = cu_seqlens, cu_seqlens[:-1], cu_seqlens
        # The unpartitioned build never reads these; their extent binds the row count.
        partition_h = partition_m = torch.empty(
            (num_sequences, heads, dim, dim), dtype=k.dtype, device=k.device
        )
    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(
            (num_sequences, heads, dim, dim), dtype=torch.float32, device=k.device
        )
    final_state = torch.empty(
        (num_sequences, heads, dim, dim), dtype=torch.float32, device=k.device
    )
    o = torch.empty_like(v)
    if k_rnorm is None:
        k_rnorm = torch.empty((1, 1, heads), dtype=torch.float32, device=k.device)
    kernel = _build_prefill_fwd_kernel(
        heads,
        dim,
        float(scale),
        str(q.dtype).removeprefix("torch."),
        str(initial_state.dtype).removeprefix("torch."),
        str(rows.dtype).removeprefix("torch."),
        use_initial_state,
        is_cp,
        l2norm,
        num_stages,
    )
    kernel(
        q,
        k,
        v,
        a,
        beta,
        k_rnorm,
        initial_state,
        partition_h,
        partition_m,
        rows,
        sequence_of,
        cu_seqlens,
        first_of,
        o,
        final_state,
    )
    return o, final_state
