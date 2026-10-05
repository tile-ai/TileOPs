"""Ungated partition scan: the state each prefill partition starts from."""

import functools

import tilelang
import tilelang.language as T
import torch

__all__ = ["partition_scan"]


@functools.lru_cache(maxsize=32)
# TileLang's warp specialization has every producer thread wait on the empty mbarrier while
# one thread issues the copy; the pipeline below is issued by the consumer instead.
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _partition_scan_kernel(
    heads: int,
    dim: int,
    block_dv: int,
    num_stages: int,
    has_initial_state: bool,
    state_dtype: str,
    buffer_dtype: str,
):
    num_partitions = T.dynamic("num_partitions")
    num_sequences = T.dynamic("num_sequences")
    accum_dtype = "float32"

    @T.prim_func
    def partition_scan_kernel(
        initial_state: T.Tensor([num_sequences, heads, dim, dim], state_dtype),
        partition_h: T.Tensor([num_partitions, heads, dim, dim], buffer_dtype),
        partition_m: T.Tensor([num_partitions, heads, dim, dim], buffer_dtype),
        first_partition: T.Tensor([num_sequences + 1], "int32"),
        start_state: T.Tensor([num_partitions, heads, dim, dim], state_dtype),
    ):
        with T.Kernel(dim // block_dv, heads, num_sequences, threads=128) as (bv, bh, bs):
            first = first_partition[bs]
            steps = first_partition[bs + 1] - first - 1
            col = bv * block_dv

            h_fragment = T.alloc_fragment((dim, block_dv), accum_dtype)
            h_shared = T.alloc_shared((dim, block_dv), buffer_dtype)
            ht_shared = T.alloc_shared((dim, block_dv), buffer_dtype)
            m_shared = T.alloc_shared((dim, dim), buffer_dtype)

            # A fragment-to-global copy ahead of the loop is dropped, because the fragment's
            # layout comes from how the loop consumes it; the first state is stored directly.
            if has_initial_state:
                T.copy(initial_state[bs, bh, 0:dim, col : col + block_dv], h_fragment)
                for i, j in T.Parallel(dim, block_dv):
                    start_state[first, bh, i, col + j] = initial_state[bs, bh, i, col + j]
            else:
                T.clear(h_fragment)
                for i, j in T.Parallel(dim, block_dv):
                    start_state[first, bh, i, col + j] = T.cast(0, state_dtype)

            # h_{p+1} = M_p h_p + H_p. The transition and the local state do not depend on h,
            # so the pipeline fetches them ahead and only the product waits on the chain.
            for p in T.Pipelined(steps, num_stages=num_stages):
                T.copy(partition_m[first + p, bh, 0:dim, 0:dim], m_shared)
                T.copy(partition_h[first + p, bh, 0:dim, col : col + block_dv], ht_shared)
                T.copy(h_fragment, h_shared)
                T.copy(ht_shared, h_fragment)
                T.gemm(m_shared, h_shared, h_fragment, clear_accum=False)
                T.copy(h_fragment, start_state[first + p + 1, bh, 0:dim, col : col + block_dv])

    return partition_scan_kernel


def partition_scan(
    initial_state: torch.Tensor | None,
    partition_h: torch.Tensor,
    partition_m: torch.Tensor,
    first_partition: torch.Tensor,
    block_dv: int = 16,
    num_stages: int = 2,
) -> torch.Tensor:
    """Each partition's starting state from every earlier partition's state and transition.

    Args:
        initial_state: ``[num_sequences, heads, dim, dim]`` state each sequence starts from,
            or ``None`` for a zero start.
        partition_h: ``[num_partitions, heads, dim, dim]`` state each partition reaches from
            a zero start.
        partition_m: ``[num_partitions, heads, dim, dim]`` transition each partition applies
            to the state it starts from.
        first_partition: ``[num_sequences + 1]`` int32 index of each sequence's first
            partition, and the partition count last.
        block_dv: State columns one CTA carries.
        num_stages: Partitions the pipeline fetches ahead of the product.

    Returns:
        ``[num_partitions, heads, dim, dim]`` starting state of every partition, in the
        dtype of ``initial_state`` or float32 without one.
    """
    num_partitions, heads, dim, _ = partition_h.shape
    num_sequences = first_partition.shape[0] - 1
    has_initial_state = initial_state is not None
    state_dtype = initial_state.dtype if has_initial_state else torch.float32
    if initial_state is None:
        # The build that starts from zero never reads it; one sequence's worth keeps the
        # dynamic extent bound.
        initial_state = torch.empty(
            (num_sequences, heads, dim, dim), dtype=state_dtype, device=partition_h.device
        )
    start_state = torch.empty(
        (num_partitions, heads, dim, dim), dtype=state_dtype, device=partition_h.device
    )
    kernel = _partition_scan_kernel(
        heads,
        dim,
        block_dv,
        num_stages,
        has_initial_state,
        str(state_dtype).removeprefix("torch."),
        str(partition_h.dtype).removeprefix("torch."),
    )
    kernel(initial_state, partition_h, partition_m, first_partition, start_state)
    return start_state
