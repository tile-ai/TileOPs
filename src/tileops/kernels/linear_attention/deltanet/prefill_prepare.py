"""Ungated DeltaNet prefill partition pass: each partition's local state and transition."""

import functools

import tilelang
import tilelang.language as T
import torch

__all__ = ["deltanet_partition_states"]


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
    },
    compile_flags=["-O3", "-DENABLE_BF16"],
)
def _build_partition_states_kernel(
    heads: int,
    dim: int,
    dtype: str,
    l2norm: bool,
    num_stages: int,
):
    """One CTA walks one partition's chunks for one head.

    From a zero start the partition reaches ``H``, and from a state ``S`` it reaches
    ``M S + H``. Each chunk takes ``H += K^T (U - W H)`` and ``M -= K^T (W M)``, with
    ``W = Ab K``, ``U = Ab V`` and ``Ab = A diag(beta)``. A warp group forms ``W`` and
    ``U`` a chunk ahead, and the two chains run in their own warp groups.
    """
    num_partitions = T.dynamic("num_partitions")
    num_tokens = T.dynamic("num_tokens")
    C = 64
    accum_dtype = "float32"
    rnorm_tokens = num_tokens if l2norm else 1

    @T.prim_func
    def partition_states_kernel(
        k: T.Tensor((1, num_tokens, heads, dim), dtype),
        v: T.Tensor((1, num_tokens, heads, dim), dtype),
        a: T.Tensor((1, num_tokens, heads, C), dtype),
        b: T.Tensor((1, num_tokens, heads), dtype),
        k_rnorm: T.Tensor((1, rnorm_tokens, heads), accum_dtype),
        offsets: T.Tensor([num_partitions + 1], "int32"),
        replay: T.Tensor([num_partitions], "int32"),
        ht: T.Tensor((num_partitions, heads, dim, dim), dtype),
        mt: T.Tensor((num_partitions, heads, dim, dim), dtype),
    ):
        with T.Kernel(num_partitions * heads, threads=512) as (bph,):
            bp, bh = bph // heads, bph % heads
            start = offsets[bp]
            num_iters = replay[bp]

            k_shared = T.alloc_shared((num_stages, C, dim), dtype)
            v_shared = T.alloc_shared((num_stages, C, dim), dtype)
            a_shared = T.alloc_shared((num_stages, C, C), dtype)
            b_shared = T.alloc_shared((num_stages, C), accum_dtype, scope="shared")
            w_shared = T.alloc_shared((2, C, dim), dtype)
            u_shared = T.alloc_shared((2, C, dim), dtype)
            h_shared = T.alloc_shared((dim, dim), dtype)
            vd_shared = T.alloc_shared((C, dim), dtype)
            m_shared = T.alloc_shared((dim, dim), dtype)
            z_shared = T.alloc_shared((C, dim), dtype)
            if l2norm:
                kn_shared = T.alloc_shared((num_stages, C), accum_dtype, scope="shared")

            data_ready = T.alloc_barrier(arrive_count=[96] * num_stages)
            data_free = T.alloc_barrier(arrive_count=[384] * num_stages)
            wu_ready = T.alloc_barrier(arrive_count=[128] * 2)
            wu_free = T.alloc_barrier(arrive_count=[256] * 2)
            ab_ready = T.alloc_barrier(arrive_count=128)
            h_ready = T.alloc_barrier(arrive_count=128)
            vd_ready = T.alloc_barrier(arrive_count=128)
            m_ready = T.alloc_barrier(arrive_count=128)
            z_ready = T.alloc_barrier(arrive_count=128)

            # The chains read U in the accumulator's fragment layout, eight rows a warp apart;
            # swizzling keeps those rows off one bank.
            T.annotate_layout({u_shared: tilelang.layout.make_swizzled_layout(u_shared)})

            tx = T.get_thread_binding()

            if tx < 128:
                # H += K^T (U - W H), from a zero start.
                T.set_max_nreg(152, 1)
                h_fragment = T.alloc_fragment((dim, dim), accum_dtype)
                t_fragment = T.alloc_fragment((C, dim), accum_dtype)
                T.clear(h_fragment)
                for i_s in T.serial(num_iters):
                    st = i_s % 2
                    ds = i_s % num_stages
                    dp = (i_s // num_stages) % 2
                    T.barrier_wait(wu_ready[st], (i_s // 2) % 2)
                    T.copy(h_fragment, h_shared)
                    T.fence_proxy_async()
                    T.barrier_arrive(h_ready)
                    T.barrier_wait(h_ready, i_s % 2)
                    T.gemm(w_shared[st, :, :], h_shared, t_fragment, clear_accum=True)
                    for j_s, j_v in T.Parallel(C, dim):
                        t_fragment[j_s, j_v] = (
                            T.cast(u_shared[st, j_s, j_v], accum_dtype) - t_fragment[j_s, j_v]
                        )
                    T.copy(t_fragment, vd_shared)
                    T.fence_proxy_async()
                    T.barrier_arrive(vd_ready)
                    T.barrier_wait(vd_ready, i_s % 2)
                    T.gemm(k_shared[ds, :, :], vd_shared, h_fragment, transpose_A=True)
                    T.barrier_arrive(wu_free[st])
                    T.barrier_arrive(data_free[ds])
                T.copy(h_fragment, ht[bp, bh, 0:dim, 0:dim])

            elif tx < 256:
                # M -= K^T (W M), from the identity.
                T.set_max_nreg(152, 1)
                m_fragment = T.alloc_fragment((dim, dim), accum_dtype)
                z_fragment = T.alloc_fragment((C, dim), accum_dtype)
                for i, j in T.Parallel(dim, dim):
                    m_fragment[i, j] = T.if_then_else(i == j, 1.0, 0.0)
                for i_s in T.serial(num_iters):
                    st = i_s % 2
                    ds = i_s % num_stages
                    dp = (i_s // num_stages) % 2
                    T.barrier_wait(wu_ready[st], (i_s // 2) % 2)
                    T.copy(m_fragment, m_shared)
                    T.fence_proxy_async()
                    T.barrier_arrive(m_ready)
                    T.barrier_wait(m_ready, i_s % 2)
                    T.gemm(w_shared[st, :, :], m_shared, z_fragment, clear_accum=True)
                    for j_s, j_k in T.Parallel(C, dim):
                        z_fragment[j_s, j_k] = -z_fragment[j_s, j_k]
                    T.copy(z_fragment, z_shared)
                    T.fence_proxy_async()
                    T.barrier_arrive(z_ready)
                    T.barrier_wait(z_ready, i_s % 2)
                    T.gemm(k_shared[ds, :, :], z_shared, m_fragment, transpose_A=True)
                    T.barrier_arrive(wu_free[st])
                    T.barrier_arrive(data_free[ds])
                T.copy(m_fragment, mt[bp, bh, 0:dim, 0:dim])

            elif tx < 384:
                # W = Ab K and U = Ab V, a chunk ahead of both chains.
                T.set_max_nreg(152, 1)
                w_fragment = T.alloc_fragment((C, dim), accum_dtype)
                u_fragment = T.alloc_fragment((C, dim), accum_dtype)
                for i_s in T.serial(num_iters):
                    st = i_s % 2
                    ds = i_s % num_stages
                    dp = (i_s // num_stages) % 2
                    T.barrier_wait(data_ready[ds], dp)
                    T.barrier_wait(wu_free[st], (i_s // 2 + 1) % 2)
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
                    T.fence_proxy_async()
                    T.barrier_arrive(ab_ready)
                    T.barrier_wait(ab_ready, i_s % 2)
                    T.gemm(a_shared[ds, :, :], k_shared[ds, :, :], w_fragment, clear_accum=True)
                    T.copy(w_fragment, w_shared[st, :, :])
                    T.gemm(a_shared[ds, :, :], v_shared[ds, :, :], u_fragment, clear_accum=True)
                    T.copy(u_fragment, u_shared[st, :, :])
                    T.fence_proxy_async()
                    T.barrier_arrive(wu_ready[st])
                    T.barrier_arrive(data_free[ds])

            else:
                T.set_max_nreg(48, 0)
                if tx < 384 + 32:
                    for i_s in T.serial(num_iters):
                        ds = i_s % num_stages
                        dp = (i_s // num_stages) % 2
                        T.barrier_wait(data_free[ds], 1 - dp)
                        left = start + i_s * C
                        T.tma_copy(
                            k[0, left : left + C, bh, 0:dim],
                            k_shared[ds, :, :],
                            barrier=data_ready[ds],
                        )
                        T.tma_copy(
                            a[0, left : left + C, bh, 0:C],
                            a_shared[ds, :, :],
                            barrier=data_ready[ds],
                        )
                        T.barrier_arrive(data_ready[ds])
                elif tx < 384 + 64:
                    for i_s in T.serial(num_iters):
                        ds = i_s % num_stages
                        dp = (i_s // num_stages) % 2
                        T.barrier_wait(data_free[ds], 1 - dp)
                        left = start + i_s * C
                        T.tma_copy(
                            v[0, left : left + C, bh, 0:dim],
                            v_shared[ds, :, :],
                            barrier=data_ready[ds],
                        )
                        T.barrier_arrive(data_ready[ds])
                elif tx < 384 + 96:
                    for i_s in T.serial(num_iters):
                        ds = i_s % num_stages
                        dp = (i_s // num_stages) % 2
                        T.barrier_wait(data_free[ds], 1 - dp)
                        left = start + i_s * C
                        for j_s in T.Parallel(C):
                            b_shared[ds, j_s] = T.cast(b[0, left + j_s, bh], accum_dtype)
                        if l2norm:
                            for j_s in T.Parallel(C):
                                kn_shared[ds, j_s] = k_rnorm[0, left + j_s, bh]
                        T.barrier_arrive(data_ready[ds])

    return partition_states_kernel


def deltanet_partition_states(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    beta: torch.Tensor,
    offsets: torch.Tensor,
    replay: torch.Tensor,
    k_rnorm: torch.Tensor | None = None,
    l2norm: bool = False,
    num_stages: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each partition's state from a zero start, and the transition it applies to a state.

    Args:
        k, v: ``[1, num_tokens, heads, 64]`` packed activations.
        a: ``[1, num_tokens, heads, 64]`` per-chunk ``(I + strict_tril(diag(beta) K K^T))^-1``.
        beta: ``[1, num_tokens, heads]`` step sizes.
        offsets: ``[num_partitions + 1]`` int32 partition offsets.
        replay: ``[num_partitions]`` int32 whole chunks each partition replays: all of a
            partition that is not its sequence's last, none of a last one, whose state no
            later partition reads.
        k_rnorm: ``[1, num_tokens, heads]`` reciprocal key norms from the block solve.
        l2norm: Normalize the key.
        num_stages: Chunks of input the copy engine fetches ahead.

    Returns:
        ``H`` and ``M``, each ``[num_partitions, heads, 64, 64]`` in the activation dtype.
    """
    _, num_tokens, heads, dim = k.shape
    num_partitions = offsets.shape[0] - 1
    ht = torch.empty((num_partitions, heads, dim, dim), dtype=k.dtype, device=k.device)
    mt = torch.empty_like(ht)
    if k_rnorm is None:
        k_rnorm = torch.empty((1, 1, heads), dtype=torch.float32, device=k.device)
    kernel = _build_partition_states_kernel(
        heads, dim, str(k.dtype).removeprefix("torch."), l2norm, num_stages
    )
    kernel(k, v, a, beta, k_rnorm, offsets, replay, ht, mt)
    return ht, mt
