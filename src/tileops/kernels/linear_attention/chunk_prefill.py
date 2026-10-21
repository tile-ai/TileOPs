import torch

from tileops.kernels.linear_attention.chunk_plan import (
    CHUNK_SIZE,
    _plan,
    packed_offsets,
    scan_value_tile,
)

# The GDN and KDA packages export the kernels that run this driver, so the program
# builders it takes from them are imported where they are called.


def split_prepare_enabled(call):
    """Whether the channel-gate call prepares its chunks with the split SM90 launches.

    They serve every sequence length, head width, normalization, scale and gate bound: on
    H200 the joint preparation was slower for raw gates at every length and head count
    measured, dense and packed. A given log gate takes the same launches after its own
    ordered FP32 prefix. The joint preparation stays finite for strong decays only by
    forming each Gram entry from per-pair decay differences, which costs more than these
    launches on H200.
    """
    return call.arch == 90


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


def build_programs(call, num_sequences, *, gate_mode, initial_state=True, dense=False):
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
    prepare_factory = chunk_prepare_program
    split_options = {}
    if gate_mode == "channel" and split_prepare_enabled(call):
        from tileops.kernels.linear_attention.kda.prepare_split import split_prepare_program

        prepare_factory = split_prepare_program
        if dense:
            # Equal-length sequences have a known chunk count; packed ones bound it.
            split_options["grid_chunks"] = call.batch * -(-call.seq_len // CHUNK_SIZE)
    prepare = prepare_factory(
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
        **(
            {
                "gate_in_kernel": call.gate_in_kernel,
                "has_dt_bias": call.has_dt_bias,
                "lower_bound": call.lower_bound,
            }
            if gate_mode == "channel"
            else {}
        ),
        **split_options,
    )
    if gate_mode == "channel":
        return prepare, _split_scan(call, num_sequences, total, dtype, initial_state)
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


def _split_scan(call, num_sequences, total, dtype, initial_state=True):
    """The scan as the state walk over the chunks, forming the outputs in the walk or after it.

    The state walk takes the narrowest value tile whose CTAs fit one wave: a narrower tile
    shortens each CTA's chain of matmuls, a second wave doubles it. On H200, 65536 tokens of
    32 value heads walked in 1793 us at 32 columns (128 CTAs) against 3043 at 16 (256); 16
    sequences of 8192 tokens and 4 heads in 334 us at 64 columns (128 CTAs) against 472 at 32.

    The walk forms each chunk's output itself unless its CTAs are too few for the length of
    its sequences: then the extra products lengthen a chain that the chunk-parallel output
    launch would run beside idle SMs. The CTA-count thresholds below were tuned on H200.
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
    ctas = num_sequences * call.value_heads * (call.dim_v // value_tile)
    chunks = total // (num_sequences * CHUNK_SIZE)
    if call.arch != 90:
        # The output-in-walk thresholds were tuned on H200 only.
        with_output = False
    elif chunks <= 16:
        with_output = True
    elif chunks <= 256:
        with_output = ctas * 4 > call.sm_count
    else:
        with_output = ctas * 2 >= call.sm_count
    threads = 128
    if value_tile == 128 and (with_output or (call.gate_in_kernel and split_prepare_enabled(call))):
        # No narrower tile fits one wave: walk the whole state with two warpgroups.
        threads = 256
    elif with_output and value_tile == 64:
        # Two warpgroups also share the output's products.
        threads = 256
    state_scan = chunk_state_program(
        call.value_heads,
        call.dim_k,
        call.dim_v,
        CHUNK_SIZE,
        dtype,
        total,
        num_sequences,
        value_tile,
        threads,
        arch=call.arch,
        with_output=with_output,
        initial_state=initial_state,
    )
    if with_output:
        return state_scan
    output = chunk_output_program(
        call.value_heads, call.dim_k, call.dim_v, CHUNK_SIZE, dtype, total, num_sequences
    )

    def scan(w, u, qg, kg, aqk, dec, h0, offsets):
        starts, values, final_state = state_scan(w, u, kg, dec, h0, offsets)
        return output(qg, aqk, starts, values, offsets), final_state

    return scan


def prepare_inputs(call, tensors, *, program, gate_mode, gate_params=()):
    q, k, v, gate, beta, offsets = tensors
    if gate_mode == "channel":
        return program(q, k, v, gate, beta, offsets, *gate_params)
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
    dense = not call.varlen
    offsets = packed_offsets(call.batch, call.seq_len, offsets, q.device)
    num_sequences = offsets.numel() - 1
    q, k, v, beta = (tensor.reshape(1, total, *tensor.shape[2:]) for tensor in (q, k, v, beta))
    gate = prepare_gate(call, inputs, offsets, gate_mode=gate_mode)
    # The channel-gate walk starts from zero itself when there is no initial state, so the
    # state buffer it is handed need not be filled.
    initial_state = state is not None or gate_mode != "channel"
    if state is None:
        state = (torch.zeros if initial_state else torch.empty)(
            num_sequences,
            call.value_heads,
            call.dim_k,
            call.dim_v,
            dtype=torch.float32,
            device=q.device,
        )
    prepare, scan = build_programs(
        call, num_sequences, gate_mode=gate_mode, initial_state=initial_state, dense=dense
    )
    prepared = prepare_inputs(
        call,
        (q, k, v, gate, beta, offsets),
        program=prepare,
        gate_mode=gate_mode,
        gate_params=(inputs[-2], inputs[-1] if inputs[-1] is not None else inputs[-2])
        if gate_mode == "channel" and call.gate_in_kernel
        else (),
    )
    output, final_state = scan(*prepared, state, offsets)
    return output.reshape(call.batch, call.seq_len, call.value_heads, call.dim_v), final_state
