"""The single-token delta-rule decode program, shared by the gated and ungated ops.

One block owns a column tile of one ``(batch, value head)`` recurrent state and advances
it by one token; a lane group within the block splits the key dimension for one column.
The build flags decide which nodes the program contains: a build without a gate reads no
gate and applies no decay, and a build without a starting state reads no state and starts
the step from zero.
"""

import math
from collections.abc import Callable

import tilelang
import tilelang.language as T

from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import LOG2E, MAX_BLOCK_THREADS
from tileops._kernels.state_pool import state_row
from tileops._utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["decode_launch", "delta_decode_tl"]


def decode_launch(
    batch: int, value_heads: int, dim: int, state_v_first: bool, sm_count: int
) -> dict:
    """The block shape the decode program runs this call in.

    One block owns ``threads // lane_group`` state columns and its lane group reads the key
    dimension of one of them, so the group is sized by whichever state axis is contiguous:
    one lane per column over a key-major state, a wide group over a value-major one. A
    key-major grid holding fewer than ``blocks_per_sm`` blocks per SM takes a second warp
    and splits the key dimension across it instead.

    Over a value-major state a lane holds four 16-byte vectors of its row, so ``dim // 16``
    lanes cover one row, and a block is two warps: the smallest block that still fills an
    SM's threads under its resident-block limit (2048 threads in 32 blocks on SM80 and SM90,
    1536 in 24 on SM89), so each block is a short burst of loads and many are in flight.

    Re-fit ``blocks_per_sm`` and either block shape by sweeping ``threads`` and
    ``lane_group`` against the decode workload rows of `benchmarks/ops/bench_deltanet.py`
    and `benchmarks/ops/bench_gdn.py`.
    """
    if state_v_first:
        return {"threads": 2 * WARP_LANES, "lane_group": dim // 16}
    blocks_per_sm = 6
    if batch * value_heads * (dim // WARP_LANES) >= blocks_per_sm * sm_count:
        return {"threads": WARP_LANES, "lane_group": 1}
    return {"threads": 2 * WARP_LANES, "lane_group": 2}


def _run_start(k_rank, run, lane_group: int, key_run: int):
    """The first key position of a lane's *run*-th run over a value-major state row: runs of
    *key_run* contiguous positions, *lane_group* runs apart."""
    return (k_rank + run * lane_group) * key_run


def _key_of(k_rank, ii, lane_group: int, key_run: int):
    """The key position of a lane's *ii*-th element of a value-major state row."""
    return _run_start(k_rank, ii // key_run, lane_group, key_run) + ii % key_run


# Seven recurrence flags, the pool and a block shape over two state widths and two dtypes:
# a process that exercises several variants at several shapes outgrows the 32 entries one
# variant needs.
@target_cache(maxsize=64)
def delta_decode_tl(
    batch: int,
    heads: int,
    value_heads: int,
    dim: int,
    scale: float,
    dtype: str,
    *,
    gated: bool,
    gate_in_kernel: bool,
    beta_sigmoid: bool,
    allow_neg_eigval: bool,
    l2norm: bool,
    has_initial_state: bool,
    state_v_first: bool,
    threads: int,
    lane_group: int,
    pooled: bool = False,
) -> Callable:
    """Build the decode program for one recurrence variant and block shape.

    Args:
        batch: Sequences in the call.
        heads: Query and key heads.
        value_heads: Value, gate, beta and state heads; a multiple of *heads*.
        dim: The square state width, which is both K and V.
        scale: Query scale, applied after the optional L2 normalization.
        dtype: TileLang name of the token dtype.
        gated: Read a per-head log decay and multiply the state slice by its exponential.
        gate_in_kernel: Read the gate raw and form ``-exp(A_log) * softplus(g + dt_bias)``.
        beta_sigmoid: Read beta as a logit and apply the sigmoid.
        allow_neg_eigval: Double the post-sigmoid beta.
        l2norm: Normalize the query and key rows before the step.
        has_initial_state: Read the caller's state; otherwise start the step from zero.
        state_v_first: Lay the state out value-major ``[B, HV, V, K]``.
        pooled: Read sequence ``b``'s state from slot ``slots[b]`` of a state pool and write
            the advanced state back there, instead of returning it; requires
            *has_initial_state*.
        threads: Threads one block runs, a whole number of warps.
        lane_group: Lanes that split the key dimension for one state column; the rest of
            the block's threads take the other columns of its tile.

    Returns:
        A TileLang jit builder returning the program.
    """
    if dim not in (64, 128):
        raise ValueError(f"delta-rule decode requires K == V in (64, 128), got {dim}")
    if value_heads % heads != 0:
        raise ValueError(f"value_heads={value_heads} must be a multiple of heads={heads}")
    if threads % WARP_LANES != 0 or threads > MAX_BLOCK_THREADS:
        raise ValueError(f"threads={threads} must be a whole number of warps, at most one block")
    if WARP_LANES % lane_group != 0:
        raise ValueError(f"lane_group={lane_group} must divide the {WARP_LANES} lanes of one warp")
    if dim % lane_group != 0:
        raise ValueError(f"dim={dim} must be divisible by lane_group={lane_group}")

    v_tile = threads // lane_group
    if dim % v_tile != 0:
        raise ValueError(f"dim={dim} must be divisible by the {v_tile} columns one block owns")
    if pooled and not has_initial_state:
        raise ValueError("a state pool is read and written in place, so it is an initial state")

    value_tiles = dim // v_tile
    total_blocks = batch * value_heads * value_tiles
    k_chunk = dim // lane_group
    # Key positions a lane moves as one 16-byte vector of a value-major state row.
    key_run = math.gcd(k_chunk, 4)

    heads_ratio = value_heads // heads
    lane_group_stages = lane_group.bit_length() - 1
    beta_gain = 2.0 if allow_neg_eigval else 1.0
    l2norm_eps = 1e-6
    # Above this `softplus(x)` is `x` to within float32, which is also what keeps `exp(x)`
    # from overflowing. torch and FLA take the same guard.
    softplus_linear_above = 20.0

    # A pooled build writes the advanced state into the pool, so only the output is returned.
    @tilelang.jit(
        out_idx=[-1] if pooled else [-2, -1],
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def _decode():
        # A parameter the flags leave unread is declared one element wide; the kernel hands
        # it a placeholder rather than a buffer of the shape it would otherwise carry.
        unread = [1]
        gate_shape = [batch, 1, value_heads] if gated else unread
        gate_param_shape = [value_heads] if gate_in_kernel else unread
        state_shape = [batch, value_heads, dim, dim] if has_initial_state else unread
        pool_shape = [T.dynamic("pool"), value_heads, dim, dim]
        token_shape = [batch, 1, heads, dim]
        value_shape = [batch, 1, value_heads, dim]
        head_shape = [batch, 1, value_heads]
        full_state_shape = [batch, value_heads, dim, dim]

        @T.macro
        def step(q, k, v, g, beta, A_log, dt_bias, state, slots, o, final_state):
            with T.Kernel(total_blocks, threads=threads) as (block,):
                tx = T.get_thread_binding()
                value_tile = block % value_tiles
                batch_head = block // value_tiles
                batch_idx = batch_head // value_heads
                head_idx = batch_head - batch_idx * value_heads
                qk_head_idx = head_idx // heads_ratio
                value_lane = tx // lane_group
                k_rank = tx - value_lane * lane_group
                value_idx = value_tile * v_tile + value_lane
                # Consecutive lanes land on consecutive elements of whichever state axis is
                # contiguous: the value columns under the key-major layout, the key positions
                # under the value-major one, where a lane holds runs of key_run contiguous
                # positions, lane_group runs apart, each moved as one vector.
                k_begin = k_rank * k_chunk

                k_shared = T.alloc_shared([dim], "float32")
                q_shared = T.alloc_shared([dim], "float32")
                state_local = T.alloc_local([k_chunk], "float32")
                old_partial = T.alloc_var("float32", init=0.0)
                out_partial = T.alloc_var("float32", init=0.0)
                q_square = T.alloc_var("float32", init=0.0)
                k_square = T.alloc_var("float32", init=0.0)

                for i in T.serial(T.ceildiv(dim, threads)):
                    kk = tx + i * threads
                    if kk < dim:
                        k_shared[kk] = T.cast(k[batch_idx, 0, qk_head_idx, kk], "float32")
                        q_value = T.cast(q[batch_idx, 0, qk_head_idx, kk], "float32")
                        q_shared[kk] = q_value if l2norm else q_value * scale
                T.sync_threads()

                if l2norm:
                    # Every warp folds the whole row, so the butterfly leaves both norms in
                    # every thread of a multi-warp block with no second pass over shared.
                    lane = tx - tx // WARP_LANES * WARP_LANES
                    for i in T.serial(T.ceildiv(dim, WARP_LANES)):
                        kk = lane + i * WARP_LANES
                        if kk < dim:
                            q_square += q_shared[kk] * q_shared[kk]
                            k_square += k_shared[kk] * k_shared[kk]
                    for stage in T.unroll(WARP_SHUFFLE_STAGES):
                        q_square += T.shfl_xor(q_square, 1 << stage, width=WARP_LANES)
                        k_square += T.shfl_xor(k_square, 1 << stage, width=WARP_LANES)
                    q_gain = T.rsqrt(q_square + l2norm_eps) * scale
                    k_gain = T.rsqrt(k_square + l2norm_eps)
                    for i in T.serial(T.ceildiv(dim, threads)):
                        kk = tx + i * threads
                        if kk < dim:
                            q_shared[kk] *= q_gain
                            k_shared[kk] *= k_gain
                    T.sync_threads()

                beta_value = T.cast(beta[batch_idx, 0, head_idx], "float32")
                if beta_sigmoid:
                    beta_value = T.sigmoid(beta_value) * beta_gain

                if gated:
                    log_decay = T.cast(g[batch_idx, 0, head_idx], "float32")
                    if gate_in_kernel:
                        shifted = log_decay + dt_bias[head_idx]
                        log_decay = -T.exp2(A_log[head_idx] * LOG2E) * T.if_then_else(
                            shifted > softplus_linear_above, shifted, T.log1p(T.exp(shifted))
                        )
                    decay = T.exp2(log_decay * LOG2E)

                if has_initial_state and state_v_first:
                    for run in T.serial(k_chunk // key_run):
                        for e in T.vectorized(key_run):
                            state_local[run * key_run + e] = state[
                                state_row(slots, batch_idx),
                                head_idx,
                                value_idx,
                                _run_start(k_rank, run, lane_group, key_run) + e,
                            ]
                    for ii in T.serial(k_chunk):
                        decayed_state = state_local[ii] * decay if gated else state_local[ii]
                        state_local[ii] = decayed_state
                        old_partial += (
                            decayed_state * k_shared[_key_of(k_rank, ii, lane_group, key_run)]
                        )
                elif has_initial_state:
                    for ii in T.serial(k_chunk):
                        kk = k_begin + ii
                        previous = state[state_row(slots, batch_idx), head_idx, kk, value_idx]
                        decayed_state = previous * decay if gated else previous
                        state_local[ii] = decayed_state
                        old_partial += decayed_state * k_shared[kk]

                if has_initial_state:
                    for stage in T.unroll(lane_group_stages):
                        old_partial += T.shfl_xor(old_partial, 1 << stage, width=lane_group)

                value_new = beta_value * (
                    T.cast(v[batch_idx, 0, head_idx, value_idx], "float32") - old_partial
                )

                if state_v_first:
                    for ii in T.serial(k_chunk):
                        kk = _key_of(k_rank, ii, lane_group, key_run)
                        if has_initial_state:
                            updated = state_local[ii] + k_shared[kk] * value_new
                        else:
                            updated = k_shared[kk] * value_new
                        state_local[ii] = updated
                        out_partial += updated * q_shared[kk]
                    for run in T.serial(k_chunk // key_run):
                        for e in T.vectorized(key_run):
                            final_state[
                                state_row(slots, batch_idx),
                                head_idx,
                                value_idx,
                                _run_start(k_rank, run, lane_group, key_run) + e,
                            ] = state_local[run * key_run + e]
                else:
                    for ii in T.serial(k_chunk):
                        kk = k_begin + ii
                        if has_initial_state:
                            updated = state_local[ii] + k_shared[kk] * value_new
                        else:
                            updated = k_shared[kk] * value_new
                        final_state[state_row(slots, batch_idx), head_idx, kk, value_idx] = updated
                        out_partial += updated * q_shared[kk]

                for stage in T.unroll(lane_group_stages):
                    out_partial += T.shfl_down(out_partial, 1 << stage, width=lane_group)
                if k_rank == 0:
                    o[batch_idx, 0, head_idx, value_idx] = T.cast(out_partial, dtype)

        if pooled:

            @T.prim_func
            def delta_decode(
                q: T.Tensor(token_shape, dtype),
                k: T.Tensor(token_shape, dtype),
                v: T.Tensor(value_shape, dtype),
                g: T.Tensor(gate_shape, dtype),
                beta: T.Tensor(head_shape, dtype),
                A_log: T.Tensor(gate_param_shape, "float32"),
                dt_bias: T.Tensor(gate_param_shape, "float32"),
                state: T.Tensor(pool_shape, "float32"),
                slots: T.Tensor([batch], "int32"),
                o: T.Tensor(value_shape, dtype),
            ):
                step(q, k, v, g, beta, A_log, dt_bias, state, slots, o, state)

        else:

            @T.prim_func
            def delta_decode(
                q: T.Tensor(token_shape, dtype),
                k: T.Tensor(token_shape, dtype),
                v: T.Tensor(value_shape, dtype),
                g: T.Tensor(gate_shape, dtype),
                beta: T.Tensor(head_shape, dtype),
                A_log: T.Tensor(gate_param_shape, "float32"),
                dt_bias: T.Tensor(gate_param_shape, "float32"),
                state: T.Tensor(state_shape, "float32"),
                o: T.Tensor(value_shape, dtype),
                final_state: T.Tensor(full_state_shape, "float32"),
            ):
                step(q, k, v, g, beta, A_log, dt_bias, state, None, o, final_state)

        return delta_decode

    return _decode
