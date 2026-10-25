"""Gated DeltaNet (GDN) prefill in one kernel: the chunk inverse, the state walk and the output."""

import functools

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.constants import (
    LOG2E,
    NAMED_BARRIERS,
    SETMAXNREG_GRANULARITY,
    SM90_REGISTERS,
    WARPGROUP_THREADS,
    WGMMA_ROWS,
)
from tileops._kernels.linear_attention.gdn.prefill_common import step_size

__all__ = ["FUSED_PREFILL_DIM", "fused_gdr_prefill", "fused_gdr_prefill_program"]

# Key and value width the kernel and its helpers are written for: the 128-byte swizzle holds
# 64 of the 128 columns an atom, and two state warp groups hold 64 value rows each. The
# dispatcher reads it to route a call here.
FUSED_PREFILL_DIM = 128


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
    compile_flags=[
        "-O3",
        "-DENABLE_BF16",
        *csrc_include("gdn_inverse_helper.h", "gdn_scan_helper.h"),
    ],
)
def _build_fused_prefill_kernel(
    heads: int,
    key_heads: int,
    scale: float,
    dtype: str,
    gate_dtype: str,
    beta_dtype: str,
    seqlen_dtype: str,
    use_initial_state: bool,
    is_cp: bool,
    state_v_first: bool,
    beta_sigmoid: bool,
    allow_neg_eigval: bool,
):
    """One CTA walks one (sequence, value head), its 128 x 128 state value-major in registers.

    Per 64-token chunk, with gc the within-chunk cumulative log gate, gl its last entry and
    G[s, t] = e^(gc_s - gc_t):

        T   = (I + strict_lower(diag(beta) (G * K K^T)))^-1 diag(beta)
        P   = scale lower(G * Q K^T)
        X^T = V^T - (S^T K^T) diag(e^gc)                     Vn^T = X^T T^T
        O^T = scale (S^T Q^T) diag(e^gc) + Vn^T P^T
        S^T = e^gl S^T + Vn^T diag(e^(gl - gc)) K

    Four warp groups share each chunk:

    - two state groups hold S^T, 64 value rows each, so every product that reads the state
      takes it from registers, and each writes its 64 output columns;
    - the auxiliary group forms T and P a chunk ahead (gdn_inverse_helper.h);
    - the producer group copies the tiles in, prepares the gate vectors and copies the output
      out.
    """
    num_tokens = T.dynamic("num_tokens")
    batch_size = T.dynamic("batch_size")
    raw_batch_size = T.dynamic("raw_batch_size")
    dim = FUSED_PREFILL_DIM
    # One WGMMA M extent per chunk, the size gdn_inverse_helper.h inverts.
    chunk = WGMMA_ROWS
    accum_dtype = "float32"
    inv_dtype = "float16"
    group = heads // key_heads
    state_shape = (batch_size, heads, dim, dim)
    final_shape = (raw_batch_size, heads, dim, dim)
    suffix = "bf16" if dtype == "bfloat16" else "fp16"
    # Value rows of S^T one state group holds.
    rows = dim // 2
    stages = 3
    threads = 4 * WARPGROUP_THREADS
    state_threads = 2 * WARPGROUP_THREADS

    # Registers a thread of each group may hold. A state group keeps S^T (64 floats), its
    # bfloat16 copy and three 64 x 64 accumulators live at once; the auxiliary group's inverse
    # fits in 104 without spilling; the producer only issues copies and scalar work.
    state_regs, aux_regs, producer_regs = 192, 104, 24
    for regs in (state_regs, aux_regs, producer_regs):
        assert regs % SETMAXNREG_GRANULARITY == 0, regs
    assert (2 * state_regs + aux_regs + producer_regs) * WARPGROUP_THREADS <= SM90_REGISTERS

    # Named barriers: two that order the state groups' products, one for the inverse's
    # exchange. TileLang's own synchronization takes lower ids.
    order_barrier = 8
    inverse_barrier = 13
    assert order_barrier + 1 < NAMED_BARRIERS and inverse_barrier < NAMED_BARRIERS
    assert inverse_barrier not in (order_barrier, order_barrier + 1)

    @T.macro
    def produce(
        i_s,
        seq_start,
        seq_end,
        bh,
        bhg,
        q,
        k,
        v,
        g,
        b,
        q_shared,
        k_shared,
        v_shared,
        g_shared,
        b_shared,
        eg_shared,
        grev_shared,
        glast_shared,
        data_ready,
        data_free,
        ptx,
    ):
        # Chunk i_s's tiles and gate vectors into stage i_s % stages once every consumer has
        # left the chunk it held, one token a thread. Rows past the sequence's end take a zero
        # step size and repeat its last gate, so they move neither the state nor a valid output.
        left = seq_start + i_s * chunk
        ds = i_s % stages
        T.barrier_wait(data_free[ds], ((i_s // stages) % 2) ^ 1)
        if ptx < 32:
            T.tma_copy(
                q[0, left : left + chunk, bhg, 0:dim],
                q_shared[ds, :, :],
                barrier=data_ready[ds],
                leader_scope_threads=32,
            )
            T.tma_copy(
                k[0, left : left + chunk, bhg, 0:dim],
                k_shared[ds, :, :],
                barrier=data_ready[ds],
                leader_scope_threads=32,
            )
            T.tma_copy(
                v[0, left : left + chunk, bh, 0:dim],
                v_shared[ds, :, :],
                barrier=data_ready[ds],
                leader_scope_threads=32,
            )
        valid = left + ptx < seq_end
        gate = T.cast(g[0, T.min(left + ptx, seq_end - 1), bh], accum_dtype)
        last = T.cast(g[0, T.min(left + chunk, seq_end) - 1, bh], accum_dtype)
        # The auxiliary group takes the gates in log2 units.
        g_shared[ds, ptx] = gate * LOG2E
        b_shared[ds, ptx] = T.if_then_else(
            valid,
            T.cast(
                step_size(b[0, T.min(left + ptx, seq_end - 1), bh], beta_sigmoid, allow_neg_eigval),
                accum_dtype,
            ),
            0.0,
        )
        # e^gc, e^(gl - gc) (zero past the sequence's end) and e^gl.
        eg_shared[ds, ptx] = T.exp2(gate * LOG2E)
        grev_shared[ds, ptx] = T.if_then_else(valid, T.exp2((last - gate) * LOG2E), 0.0)
        if ptx == 0:
            glast_shared[ds, 0] = T.exp2(last * LOG2E)
        T.barrier_arrive(data_ready[ds])

    @T.macro
    def store_output(i_s, seq_start, seq_end, bh, o, o_shared, o_full, o_empty, half, lane):
        # One state group's 64 output columns of chunk i_s, by one warp: whole chunks by the
        # copy engine, the sequence's last rows by the warp, so no row of the next sequence is
        # written.
        left = seq_start + i_s * chunk
        T.barrier_wait(o_full[half], i_s % 2)
        if left + chunk <= seq_end:
            T.tma_copy(
                o_shared[:, half * rows : half * rows + rows],
                o[0, left : left + chunk, bh, half * rows : half * rows + rows],
                leader_scope_threads=32,
            )
            T.tma_store_wait()
        else:
            T.call_extern(
                "handle",
                "tileops::gdn_copy_rows_" + suffix,
                T.address_of(o_shared[0, 0]),
                half,
                T.address_of(o[0, left, bh, half * rows]),
                heads * dim,
                seq_end - left,
                lane,
            )
        T.sync_warp()
        if lane == 0:
            T.barrier_arrive(o_empty[half])

    @T.macro
    def state_walk(
        row,
        bb,
        bh,
        raw_idx,
        num_iters,
        q_shared,
        k_shared,
        v_shared,
        t_shared,
        p_shared,
        eg_shared,
        grev_shared,
        glast_shared,
        o_shared,
        h0,
        ht,
        data_ready,
        data_free,
        aux_ready,
        aux_free,
        o_full,
        o_empty,
        store_state,
    ):
        # One state group walks value rows row to row + 63 of S^T.
        group = row // rows
        tx = T.get_thread_binding()
        s_fragment = T.alloc_fragment((rows, dim), accum_dtype)
        s_operand = T.alloc_fragment((rows, dim), dtype)
        sk_fragment = T.alloc_fragment((rows, chunk), accum_dtype)
        # X^T's A operand, in the registers gdn_scan_helper.h fills: 64 x 64 over 128 threads.
        x_operand = T.alloc_local((rows * chunk // WARPGROUP_THREADS,), dtype)
        vn_fragment = T.alloc_fragment((rows, chunk), accum_dtype)
        vn_operand = T.alloc_fragment((rows, chunk), dtype)
        vd_operand = T.alloc_fragment((rows, chunk), dtype)
        o_fragment = T.alloc_fragment((rows, chunk), accum_dtype)
        glast = T.alloc_var(accum_dtype)

        if use_initial_state:
            for i, j in T.Parallel(rows, dim):
                s_fragment[i, j] = (
                    h0[bb, bh, row + i, j] if state_v_first else h0[bb, bh, j, row + i]
                )
        else:
            T.clear(s_fragment)

        # The state groups take the tensor cores in turn, the first group first: each waits on
        # its own barrier before issuing a product and arrives at the other's after.
        own_turn = order_barrier + group
        other_turn = order_barrier + 1 - group
        if group == 1:
            T.call_extern("handle", "tileops::gdn_bar_arrive", order_barrier, state_threads)
        for i_s in T.serial(num_iters):
            st = i_s % 2
            phase = (i_s // 2) % 2
            ds = i_s % stages
            T.barrier_wait(data_ready[ds], (i_s // stages) % 2)
            T.copy(s_fragment, s_operand)
            # S^T K^T and S^T Q^T, both done before X^T forms so the state's operand copy is
            # not live beside X^T.
            T.sync_threads(own_turn, state_threads)
            T.wgmma_gemm(
                s_operand, k_shared[ds, :, :], sk_fragment, transpose_B=True, clear_accum=True
            )
            T.wait_wgmma(0)
            T.wgmma_gemm(
                s_operand, q_shared[ds, :, :], o_fragment, transpose_B=True, clear_accum=True
            )
            T.call_extern("handle", "tileops::gdn_bar_arrive", other_turn, state_threads)
            T.wait_wgmma(0)
            T.barrier_wait(aux_ready[st], phase)
            T.call_extern(
                "handle",
                "tileops::gdn_form_x_" + suffix,
                x_operand.data,
                sk_fragment.data,
                T.address_of(v_shared[ds, 0, 0]),
                row,
                T.address_of(eg_shared[ds, 0]),
                tx,
            )
            T.sync_threads(own_turn, state_threads)
            T.call_extern(
                "handle",
                "tileops::gdn_issue_xt_" + suffix,
                vn_fragment.data,
                x_operand.data,
                T.address_of(t_shared[st, 0, 0]),
            )
            T.call_extern("handle", "tileops::gdn_bar_arrive", other_turn, state_threads)
            T.wait_wgmma(0)
            glast = glast_shared[ds, 0]
            for i_v, j_s in T.Parallel(rows, chunk):
                vd_operand[i_v, j_s] = T.cast(vn_fragment[i_v, j_s] * grev_shared[ds, j_s], dtype)
            for i_v, j_s in T.Parallel(rows, chunk):
                vn_operand[i_v, j_s] = T.cast(vn_fragment[i_v, j_s], dtype)
            for i_v, j_s in T.Parallel(rows, chunk):
                o_fragment[i_v, j_s] *= scale * eg_shared[ds, j_s]
            for i_v, i_k in T.Parallel(rows, dim):
                s_fragment[i_v, i_k] *= glast
            T.sync_threads(own_turn, state_threads)
            T.wgmma_gemm(vd_operand, k_shared[ds, :, :], s_fragment, clear_accum=False)
            T.wait_wgmma(0)
            T.wgmma_gemm(
                vn_operand, p_shared[st, :, :], o_fragment, transpose_B=True, clear_accum=False
            )
            T.call_extern("handle", "tileops::gdn_bar_arrive", other_turn, state_threads)
            T.wait_wgmma(0)
            T.barrier_arrive(data_free[ds])
            T.barrier_arrive(aux_free[st])
            # This group's 64 output columns, transposed into the tile once the producer has
            # copied out the previous chunk's.
            T.barrier_wait(o_empty[group], (i_s % 2) ^ 1)
            T.call_extern(
                "handle",
                "tileops::gdn_store_ot_" + suffix,
                o_fragment.data,
                T.address_of(o_shared[0, 0]),
                row,
                tx,
            )
            T.fence_proxy_async()
            T.barrier_arrive(o_full[group])

        if store_state:
            for i, j in T.Parallel(rows, dim):
                if state_v_first:
                    ht[raw_idx, bh, row + i, j] = s_fragment[i, j]
                else:
                    ht[raw_idx, bh, j, row + i] = s_fragment[i, j]

    @T.prim_func
    def fused_gdr_prefill_kernel(
        q: T.Tensor((1, num_tokens, key_heads, dim), dtype),
        k: T.Tensor((1, num_tokens, key_heads, dim), dtype),
        v: T.Tensor((1, num_tokens, heads, dim), dtype),
        g: T.Tensor((1, num_tokens, heads), gate_dtype),
        b: T.Tensor((1, num_tokens, heads), beta_dtype),
        h0: T.Tensor(state_shape, accum_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], seqlen_dtype),
        cp_seq_map: T.Tensor([batch_size], seqlen_dtype),
        raw_cu_seqlens: T.Tensor([raw_batch_size + 1], seqlen_dtype),
        o: T.Tensor((1, num_tokens, heads, dim), dtype),
        ht: T.Tensor(final_shape, accum_dtype),
    ):
        with T.Kernel(batch_size * heads, threads=threads) as (bbh,):
            bb, bh = bbh // heads, bbh % heads
            bhg = bh // group

            seq_start = T.alloc_var("int32")
            seq_end = T.alloc_var("int32")
            seq_start = cu_seqlens[bb]
            seq_end = cu_seqlens[bb + 1]
            raw_idx = T.alloc_var("int32")
            raw_idx = cp_seq_map[bb] if is_cp else bb
            store_state = T.alloc_var("bool")
            store_state = (raw_cu_seqlens[raw_idx + 1] == seq_end) if is_cp else True
            num_iters = T.alloc_var("int32")
            num_iters = T.ceildiv(seq_end - seq_start, chunk)

            q_shared = T.alloc_shared((stages, chunk, dim), dtype)
            k_shared = T.alloc_shared((stages, chunk, dim), dtype)
            v_shared = T.alloc_shared((stages, chunk, dim), dtype)
            g_shared = T.alloc_shared((stages, chunk), accum_dtype, scope="shared")
            b_shared = T.alloc_shared((stages, chunk), accum_dtype, scope="shared")
            eg_shared = T.alloc_shared((stages, chunk), accum_dtype, scope="shared")
            grev_shared = T.alloc_shared((stages, chunk), accum_dtype, scope="shared")
            glast_shared = T.alloc_shared((stages, 1), accum_dtype, scope="shared")
            t_shared = T.alloc_shared((2, chunk, chunk), dtype)
            p_shared = T.alloc_shared((2, chunk, chunk), dtype)
            o_shared = T.alloc_shared((chunk, dim), dtype)
            # The inverse's exchange tiles, float16 in the swizzle gdn_inverse_helper.h reads.
            n_shared = T.alloc_shared((chunk, chunk), inv_dtype)
            xd_shared = T.alloc_shared((chunk, chunk), inv_dtype)

            # Stage i % stages holds chunk i's inputs and buffer i % 2 its T and P.
            data_ready = T.alloc_barrier(arrive_count=[2 * 32] * stages)
            data_free = T.alloc_barrier(arrive_count=[state_threads + WARPGROUP_THREADS] * stages)
            aux_ready = T.alloc_barrier(arrive_count=[WARPGROUP_THREADS] * 2)
            aux_free = T.alloc_barrier(arrive_count=[state_threads] * 2)
            # State group h's half of the output tile is written, and copied out.
            o_full = T.alloc_barrier(arrive_count=[WARPGROUP_THREADS] * 2)
            o_empty = T.alloc_barrier(arrive_count=[1] * 2)

            # Tiles the hand-written code addresses: the 128-byte swizzle, in 64-column atoms.
            T.annotate_layout(
                {
                    q_shared: tilelang.layout.make_wgmma_swizzled_layout(q_shared, continuity=64),
                    k_shared: tilelang.layout.make_wgmma_swizzled_layout(k_shared, continuity=64),
                    v_shared: tilelang.layout.make_wgmma_swizzled_layout(v_shared, continuity=64),
                    o_shared: tilelang.layout.make_wgmma_swizzled_layout(o_shared, continuity=64),
                    t_shared: tilelang.layout.make_swizzled_layout(t_shared),
                    p_shared: tilelang.layout.make_swizzled_layout(p_shared),
                    n_shared: tilelang.layout.make_swizzled_layout(n_shared),
                    xd_shared: tilelang.layout.make_swizzled_layout(xd_shared),
                }
            )

            tx = T.get_thread_binding()

            if tx < WARPGROUP_THREADS:
                T.set_max_nreg(state_regs, 1)
                state_walk(
                    0,
                    bb,
                    bh,
                    raw_idx,
                    num_iters,
                    q_shared,
                    k_shared,
                    v_shared,
                    t_shared,
                    p_shared,
                    eg_shared,
                    grev_shared,
                    glast_shared,
                    o_shared,
                    h0,
                    ht,
                    data_ready,
                    data_free,
                    aux_ready,
                    aux_free,
                    o_full,
                    o_empty,
                    store_state,
                )
            elif tx < state_threads:
                T.set_max_nreg(state_regs, 1)
                state_walk(
                    rows,
                    bb,
                    bh,
                    raw_idx,
                    num_iters,
                    q_shared,
                    k_shared,
                    v_shared,
                    t_shared,
                    p_shared,
                    eg_shared,
                    grev_shared,
                    glast_shared,
                    o_shared,
                    h0,
                    ht,
                    data_ready,
                    data_free,
                    aux_ready,
                    aux_free,
                    o_full,
                    o_empty,
                    store_state,
                )
            elif tx < state_threads + WARPGROUP_THREADS:
                # T and P a chunk ahead of the state.
                T.set_max_nreg(aux_regs, 0)
                aux_tx = tx - state_threads
                for i_s in T.serial(num_iters):
                    st = i_s % 2
                    phase = (i_s // 2) % 2
                    ds = i_s % stages
                    T.barrier_wait(aux_free[st], phase ^ 1)
                    T.barrier_wait(data_ready[ds], (i_s // stages) % 2)
                    T.call_extern(
                        "handle",
                        "tileops::gdn_chunk_tp_" + suffix,
                        T.address_of(q_shared[ds, 0, 0]),
                        T.address_of(k_shared[ds, 0, 0]),
                        T.address_of(g_shared[ds, 0]),
                        T.address_of(b_shared[ds, 0]),
                        T.float32(scale),
                        T.address_of(n_shared[0, 0]),
                        T.address_of(xd_shared[0, 0]),
                        T.address_of(t_shared[st, 0, 0]),
                        T.address_of(p_shared[st, 0, 0]),
                        aux_tx,
                        inverse_barrier,
                    )
                    T.barrier_arrive(aux_ready[st])
                    T.barrier_arrive(data_free[ds])
            else:
                # Copies in by the first two warps, the output's copies out by the third.
                T.set_max_nreg(producer_regs, 0)
                ptx = tx - state_threads - WARPGROUP_THREADS
                if ptx < chunk:
                    for i_s in T.serial(num_iters):
                        produce(
                            i_s,
                            seq_start,
                            seq_end,
                            bh,
                            bhg,
                            q,
                            k,
                            v,
                            g,
                            b,
                            q_shared,
                            k_shared,
                            v_shared,
                            g_shared,
                            b_shared,
                            eg_shared,
                            grev_shared,
                            glast_shared,
                            data_ready,
                            data_free,
                            ptx,
                        )
                elif ptx < chunk + 32:
                    for i_s in T.serial(num_iters):
                        store_output(
                            i_s,
                            seq_start,
                            seq_end,
                            bh,
                            o,
                            o_shared,
                            o_full,
                            o_empty,
                            0,
                            ptx - chunk,
                        )
                        store_output(
                            i_s,
                            seq_start,
                            seq_end,
                            bh,
                            o,
                            o_shared,
                            o_full,
                            o_empty,
                            1,
                            ptx - chunk,
                        )

    return fused_gdr_prefill_kernel


def _check_inputs(q, k, v, g, b, cu_seqlens, initial_state, cp_seq_map, raw_cu_seqlens):
    """Raise ValueError for a call the kernel does not serve."""
    if q.dim() != 4 or k.shape != q.shape or v.dim() != 4 or q.shape[0] != 1:
        raise ValueError(
            "q and k must be [1, num_tokens, key_heads, D] and v [1, num_tokens, "
            f"heads, D]; got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    _, num_tokens, key_heads, dim = q.shape
    heads = v.shape[2]
    if dim != FUSED_PREFILL_DIM or v.shape[3] != FUSED_PREFILL_DIM:
        raise ValueError(f"the fused prefill serves D = {FUSED_PREFILL_DIM}, got {dim}")
    if v.shape[:2] != q.shape[:2] or key_heads < 1 or heads % key_heads != 0:
        raise ValueError(f"value heads {heads} must be a multiple of key heads {key_heads}")
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"q, k and v must share bfloat16 or float16, got {q.dtype}")
    if g.shape != (1, num_tokens, heads) or b.shape != (1, num_tokens, heads):
        raise ValueError("g and b must be [1, num_tokens, heads]")
    tensors = [q, k, v, g, b, cu_seqlens]
    for tensor in tensors:
        if not tensor.is_cuda or tensor.device != q.device:
            raise ValueError("every tensor must be on q's CUDA device")
    if cu_seqlens.dtype not in (torch.int32, torch.int64) or cu_seqlens.dim() != 1:
        raise ValueError("cu_seqlens must be a 1-D int32 or int64 tensor")
    if (cp_seq_map is None) != (raw_cu_seqlens is None):
        raise ValueError("cp_seq_map and raw_cu_seqlens come together")
    if cp_seq_map is not None:
        for name, tensor in (("cp_seq_map", cp_seq_map), ("raw_cu_seqlens", raw_cu_seqlens)):
            if tensor.dtype != cu_seqlens.dtype or tensor.device != q.device:
                raise ValueError(f"{name} must match cu_seqlens's dtype and device")
        if cp_seq_map.shape != (cu_seqlens.shape[0] - 1,):
            raise ValueError("cp_seq_map holds one entry a row of cu_seqlens")
    if initial_state is not None:
        rows = cu_seqlens.shape[0] - 1
        expected = (rows, heads, FUSED_PREFILL_DIM, FUSED_PREFILL_DIM)
        if tuple(initial_state.shape) != expected or initial_state.dtype != torch.float32:
            raise ValueError(f"initial_state must be float32 {expected}")
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on q's device")


def fused_gdr_prefill_program(
    heads: int,
    key_heads: int,
    scale: float,
    dtype: torch.dtype,
    gate_dtype: torch.dtype,
    beta_dtype: torch.dtype,
    seqlen_dtype: torch.dtype,
    use_initial_state: bool,
    is_cp: bool,
    state_v_first: bool = False,
    beta_sigmoid: bool = False,
    allow_neg_eigval: bool = False,
):
    """The program :func:`fused_gdr_prefill` launches for these heads, dtypes and flags."""
    return _build_fused_prefill_kernel(
        heads,
        key_heads,
        float(scale),
        str(dtype).removeprefix("torch."),
        str(gate_dtype).removeprefix("torch."),
        str(beta_dtype).removeprefix("torch."),
        str(seqlen_dtype).removeprefix("torch."),
        use_initial_state,
        is_cp,
        state_v_first,
        beta_sigmoid,
        allow_neg_eigval,
    )


def fused_gdr_prefill(
    program,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    cp_seq_map: torch.Tensor | None = None,
    raw_cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed GDN prefill output and final state in one kernel, one CTA a (sequence, value head).

    Args:
        program: What :func:`fused_gdr_prefill_program` built for these tensors.
        q, k: ``[1, num_tokens, key_heads, 128]`` packed queries and keys.
        v: ``[1, num_tokens, heads, 128]`` packed values.
        g: ``[1, num_tokens, heads]`` gate, accumulated within each 64-token chunk.
        b: ``[1, num_tokens, heads]`` step sizes, or their logits where *program* applies
            the sigmoid.
        cu_seqlens: Offsets of the rows each CTA walks: from 0, non-decreasing, ending at
            ``num_tokens``. Their values stay on the device and are not checked.
        initial_state: ``[rows, heads, 128, 128]`` float32 starting state, or ``None``.
        cp_seq_map: The sequence each row belongs to where sequences are partitioned.
        raw_cu_seqlens: The sequences' offsets where they are partitioned.

    Returns:
        The output, like ``v``, and the float32 final state of every sequence.

    Raises:
        ValueError: A shape, dtype, layout or device the kernel does not serve.
    """
    _check_inputs(q, k, v, g, b, cu_seqlens, initial_state, cp_seq_map, raw_cu_seqlens)
    # The copy engine and the kernel's indexing read dense rows.
    q, k, v, g, b = (tensor.contiguous() for tensor in (q, k, v, g, b))
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    dim = k.shape[3]
    heads = v.shape[2]
    rows = cu_seqlens.shape[0] - 1
    if cp_seq_map is None:
        cp_seq_map = cu_seqlens[:-1]
        raw_cu_seqlens = cu_seqlens
    if initial_state is None:
        initial_state = torch.empty((rows, heads, dim, dim), dtype=torch.float32, device=k.device)
    final_state = torch.empty(
        (raw_cu_seqlens.shape[0] - 1, heads, dim, dim), dtype=torch.float32, device=k.device
    )
    o = torch.empty_like(v)
    program(q, k, v, g, b, initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens, o, final_state)
    return o, final_state
