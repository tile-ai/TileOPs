"""The SM90 single-token delta-rule decode program, shared by the gated and ungated ops.

One block owns a column tile of one ``(batch, value head)`` recurrent state and advances
it by one token; a lane group within the block splits the key dimension for one column.
The build flags decide which nodes the program contains: a build without a gate reads no
gate and applies no decay, and a build without a starting state reads no state and starts
the step from zero.
"""

import functools
from typing import Callable

import tilelang
import tilelang.language as T

from tileops.kernels.constants import LOG2E, MAX_BLOCK_THREADS
from tileops.utils import WARP_LANES, WARP_SHUFFLE_STAGES, get_sm_count

__all__ = ["decode_launch", "delta_decode_sm90_tl"]


def decode_launch(
    batch: int, value_heads: int, dim: int, state_v_first: bool, device_index: int | None
) -> dict:
    """The block shape the decode program runs this call in.

    One block owns ``threads // lane_group`` state columns and its lane group reads the key
    dimension of one of them, so the group is sized by whichever state axis is contiguous:
    one lane per column over a key-major state, a wide group over a value-major one. A
    key-major grid holding fewer than ``blocks_per_sm`` blocks per SM takes a second warp
    and splits the key dimension across it instead.

    Re-fit ``blocks_per_sm`` and either block shape by sweeping ``threads`` and
    ``lane_group`` against the decode workload rows of `benchmarks/ops/bench_deltanet.py`
    and `benchmarks/ops/bench_gdn.py`.
    """
    if state_v_first:
        return {"threads": 8 * WARP_LANES, "lane_group": WARP_LANES // 2}
    blocks_per_sm = 6
    if batch * value_heads * (dim // WARP_LANES) >= blocks_per_sm * get_sm_count(device_index):
        return {"threads": WARP_LANES, "lane_group": 1}
    return {"threads": 2 * WARP_LANES, "lane_group": 2}


# Seven recurrence flags and a block shape over two state widths and two dtypes: a process
# that exercises several variants at several shapes outgrows the 32 entries one variant needs.
@functools.lru_cache(maxsize=64)
def delta_decode_sm90_tl(
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
        threads: Threads one block runs, a whole number of warps.
        lane_group: Lanes that split the key dimension for one state column; the rest of
            the block's threads take the other columns of its tile.

    Returns:
        A TileLang jit builder returning the program.
    """
    if dim not in (64, 128):
        raise ValueError(f"Hopper delta-rule decode requires K == V in (64, 128), got {dim}")
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

    value_tiles = dim // v_tile
    total_blocks = batch * value_heads * value_tiles
    k_chunk = dim // lane_group
    heads_ratio = value_heads // heads
    lane_group_stages = lane_group.bit_length() - 1
    beta_gain = 2.0 if allow_neg_eigval else 1.0
    l2norm_eps = 1e-6
    # Above this `softplus(x)` is `x` to within float32, which is also what keeps `exp(x)`
    # from overflowing. torch and FLA take the same guard.
    softplus_linear_above = 20.0

    @tilelang.jit(out_idx=[-2, -1], compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"])
    def _decode():
        # A parameter the flags leave unread is declared one element wide; the kernel hands
        # it a placeholder rather than a buffer of the shape it would otherwise carry.
        unread = [1]
        gate_shape = [batch, 1, value_heads] if gated else unread
        gate_param_shape = [value_heads] if gate_in_kernel else unread
        state_shape = [batch, value_heads, dim, dim] if has_initial_state else unread
        token_shape = [batch, 1, heads, dim]
        value_shape = [batch, 1, value_heads, dim]
        head_shape = [batch, 1, value_heads]
        full_state_shape = [batch, value_heads, dim, dim]

        @T.prim_func
        def delta_decode_sm90(
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
                # Consecutive lanes must land on consecutive elements of whichever state
                # axis is contiguous: the value columns under the key-major layout, the key
                # positions under the value-major one.
                k_begin = k_rank if state_v_first else k_rank * k_chunk
                k_step = lane_group if state_v_first else 1

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

                if has_initial_state:
                    for ii in T.serial(k_chunk):
                        kk = k_begin + ii * k_step
                        if state_v_first:
                            previous = state[batch_idx, head_idx, value_idx, kk]
                        else:
                            previous = state[batch_idx, head_idx, kk, value_idx]
                        decayed_state = previous * decay if gated else previous
                        state_local[ii] = decayed_state
                        old_partial += decayed_state * k_shared[kk]

                    for stage in T.unroll(lane_group_stages):
                        old_partial += T.shfl_xor(old_partial, 1 << stage, width=lane_group)

                value_new = beta_value * (
                    T.cast(v[batch_idx, 0, head_idx, value_idx], "float32") - old_partial
                )

                for ii in T.serial(k_chunk):
                    kk = k_begin + ii * k_step
                    if has_initial_state:
                        updated = state_local[ii] + k_shared[kk] * value_new
                    else:
                        updated = k_shared[kk] * value_new
                    if state_v_first:
                        final_state[batch_idx, head_idx, value_idx, kk] = updated
                    else:
                        final_state[batch_idx, head_idx, kk, value_idx] = updated
                    out_partial += updated * q_shared[kk]

                for stage in T.unroll(lane_group_stages):
                    out_partial += T.shfl_down(out_partial, 1 << stage, width=lane_group)
                if k_rank == 0:
                    o[batch_idx, 0, head_idx, value_idx] = T.cast(out_partial, dtype)

        return delta_decode_sm90

    return _decode
