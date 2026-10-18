import torch

from tileops.kernels.linear_attention.chunk_plan import (
    CHUNK_SIZE,
    _plan,
    packed_offsets,
    scan_value_tile,
)

# The GDN and KDA packages export the kernels that run this driver, so the program
# builders it takes from them are imported where they are called.


def chunk_plan(call):
    return _plan(call.dim_k, call.dtype.itemsize, call.num_sequences, call.smem_budget, call.arch)


def chunk_refusal(call):
    if chunk_plan(call) is None:
        return f"needs more than {call.smem_budget} bytes of shared memory for {call.num_sequences} sequences"
    return None


def cumulative_gate(call, gate, offsets, *, a_log, dt_bias):
    from tileops.kernels.linear_attention.gdn.prefill_prepare import (
        prefill_chunk_local_cumsum_bthd_tl,
    )

    program = prefill_chunk_local_cumsum_bthd_tl(
        call.batch * call.seq_len,
        call.num_sequences,
        call.value_heads,
        CHUNK_SIZE,
        str(call.dtype).removeprefix("torch."),
        "int64",
        call.gate_in_kernel,
    )
    if call.gate_in_kernel:
        return program(gate, offsets, a_log, dt_bias)
    return program(gate, offsets)


def prepare_gate(call, inputs, offsets, *, gate_mode):
    _, _, _, g, beta, _, _, a_log, dt_bias = inputs
    total = call.batch * call.seq_len
    if gate_mode == "channel":
        return g.reshape(1, total, call.value_heads, call.dim_k)
    if gate_mode == "none":
        return torch.empty((1, total, call.value_heads), dtype=torch.float32, device=beta.device)
    return cumulative_gate(
        call, g.reshape(1, total, call.value_heads), offsets, a_log=a_log, dt_bias=dt_bias
    )


def build_programs(call, num_sequences, *, gate_mode):
    from tileops.kernels.linear_attention.kda.chunk_programs import (
        chunk_prepare_program,
        chunk_scan_program,
    )

    total = call.batch * call.seq_len
    plan = _plan(call.dim_k, call.dtype.itemsize, num_sequences, call.smem_budget, call.arch)
    if plan is None:
        raise ValueError(f"{num_sequences} sequences need more shared memory than the device has")
    lean, value_tile = plan
    value_tile = scan_value_tile(
        value_tile,
        call.arch,
        total=total,
        num_seqs=num_sequences,
        value_heads=call.value_heads,
        dim=call.dim_k,
    )
    dtype = str(call.dtype).removeprefix("torch.")
    prepare = chunk_prepare_program(
        call.heads,
        call.value_heads,
        call.dim_k,
        call.dim_v,
        CHUNK_SIZE,
        dtype,
        call.scale,
        call.l2norm,
        total,
        num_sequences,
        lean=lean,
        gate_mode=gate_mode,
        beta_sigmoid=call.beta_sigmoid,
        allow_neg_eigval=call.allow_neg_eigval,
    )
    if gate_mode == "channel":
        return prepare, _split_scan(call, num_sequences, total, dtype)
    scan = chunk_scan_program(
        call.value_heads,
        call.dim_k,
        call.dim_v,
        CHUNK_SIZE,
        dtype,
        total,
        num_sequences,
        value_tile=value_tile,
        state_v_first=call.state_v_first,
    )
    return prepare, scan


def _split_scan(call, num_sequences, total, dtype):
    """The scan as two launches: the state walk over the chunks, then every chunk's output.

    The state walk takes the narrowest value tile whose CTAs fit one wave: a narrower tile
    shortens each CTA's chain of matmuls, a second wave doubles it. On H200, 65536 tokens of
    32 value heads walked in 1793 us at 32 columns (128 CTAs) against 3043 at 16 (256); 16
    sequences of 8192 tokens and 4 heads in 334 us at 64 columns (128 CTAs) against 472 at 32.
    """
    from tileops.kernels.linear_attention.kda.chunk_programs import (
        chunk_output_program,
        chunk_state_program,
    )

    value_tile = next(
        (
            width
            for width in (16, 32, 64)
            if width <= call.dim_v
            and num_sequences * call.value_heads * (call.dim_v // width) <= call.sm_count
        ),
        call.dim_v,
    )
    state_scan = chunk_state_program(
        call.value_heads,
        call.dim_k,
        call.dim_v,
        CHUNK_SIZE,
        dtype,
        total,
        num_sequences,
        value_tile,
    )
    output = chunk_output_program(
        call.value_heads, call.dim_k, call.dim_v, CHUNK_SIZE, dtype, total, num_sequences
    )

    def scan(w, u, qg, kg, aqk, dec, h0, offsets):
        starts, values, final_state = state_scan(w, u, kg, dec, h0, offsets)
        return output(qg, aqk, starts, values, offsets), final_state

    return scan


def prepare_inputs(call, tensors, *, program, gate_mode):
    q, k, v, gate, beta, offsets = tensors
    if gate_mode == "channel":
        return program(q, k, v, gate, beta, offsets)
    from tileops.kernels.linear_attention.gdn.prefill_prepare import prefill_blocksolve_A_bthd

    inverse, _ = prefill_blocksolve_A_bthd(
        k,
        gate,
        beta,
        offsets,
        CHUNK_SIZE,
        use_gate=gate_mode == "head",
        l2norm=call.l2norm,
        beta_sigmoid=call.beta_sigmoid,
        allow_neg_eigval=call.allow_neg_eigval,
    )
    return program(q, k, v, gate, beta, offsets, inverse)


def run_chunk_prefill(call, inputs, *, gate_mode):
    q, k, v, _, beta, state, offsets, _, _ = inputs
    total = call.batch * call.seq_len
    offsets = packed_offsets(call.batch, call.seq_len, offsets, q.device)
    num_sequences = offsets.numel() - 1
    q, k, v, beta = (tensor.reshape(1, total, *tensor.shape[2:]) for tensor in (q, k, v, beta))
    gate = prepare_gate(call, inputs, offsets, gate_mode=gate_mode)
    if state is None:
        state = torch.zeros(
            num_sequences,
            call.value_heads,
            call.dim_k,
            call.dim_v,
            dtype=torch.float32,
            device=q.device,
        )
    prepare, scan = build_programs(call, num_sequences, gate_mode=gate_mode)
    prepared = prepare_inputs(
        call, (q, k, v, gate, beta, offsets), program=prepare, gate_mode=gate_mode
    )
    output, final_state = scan(*prepared, state, offsets)
    return output.reshape(call.batch, call.seq_len, call.value_heads, call.dim_v), final_state
