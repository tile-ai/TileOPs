import torch

from tileops._kernels.kernel_base import CompiledProgram
from tileops._kernels.linear_attention.chunk_plan import (
    CHUNK_SIZE,
    launch_plan,
    packed_offsets,
    scan_value_tile,
)

# The GDN and KDA packages export the kernels that run this driver, so the program
# builders it takes from them are imported where they are called.


def split_prepare_enabled(call):
    """Whether a channel-gate chunk call prepares its chunks with the split SM90 launches.

    They stay finite for strong decays of either gate form; their launch shapes are chosen
    for SM90, so elsewhere the joint chunk program serves the call.
    """
    return call.arch == 90


def chunk_plan(call):
    return launch_plan(
        call.dim_k, call.dtype.itemsize, call.num_sequences, call.smem_budget, call.arch
    )


def chunk_refusal(call):
    if chunk_plan(call) is None:
        return f"needs more than {call.smem_budget} bytes of shared memory for {call.num_sequences} sequences"
    return None


def cumulative_gate_program(call):
    """The program that accumulates a head gate within each chunk."""
    from tileops._kernels.linear_attention.gdn.prefill_prepare import (
        prefill_chunk_local_cumsum_bthd_tl,
    )

    return prefill_chunk_local_cumsum_bthd_tl(
        call.batch * call.seq_len,
        call.num_sequences,
        call.value_heads,
        CHUNK_SIZE,
        str(call.dtype).removeprefix("torch."),
        "int64",
        call.gate_in_kernel,
    )


def prepare_gate(call, inputs, offsets, *, gate_mode, program):
    _, _, _, g, beta, _, _, a_log, dt_bias = inputs
    total = call.batch * call.seq_len
    if gate_mode == "channel":
        return g.reshape(1, total, call.value_heads, call.dim_k)
    if gate_mode == "none":
        return torch.empty((1, total, call.value_heads), dtype=torch.float32, device=beta.device)
    gate = g.reshape(1, total, call.value_heads)
    if call.gate_in_kernel:
        return program(gate, offsets, a_log, dt_bias)
    return program(gate, offsets)


def solve_program(call, num_sequences, offsets_dtype, *, gate_mode):
    """The block solve that forms each chunk's inverse ahead of a head-gate or ungated scan."""
    from tileops._kernels.linear_attention.gdn.prefill_prepare import prefill_blocksolve_A_bthd_tl

    return prefill_blocksolve_A_bthd_tl(
        call.batch * call.seq_len,
        num_sequences,
        call.value_heads,
        call.heads,
        CHUNK_SIZE,
        call.dim_k,
        str(call.dtype).removeprefix("torch."),
        str(offsets_dtype).removeprefix("torch."),
        use_gate=gate_mode == "head",
        l2norm=call.l2norm,
        beta_sigmoid=call.beta_sigmoid,
        allow_neg_eigval=call.allow_neg_eigval,
    )


def chunk_programs(
    config, call, gate_mode, num_sequences, initial_state, offsets_dtype, split_output=False
):
    """The programs one chunked prefill launch runs: the gate cumsum, the block solve, the
    chunk preparation and the scan, ``None`` for a pass the gate mode skips.

    A ``Kernel.program`` build: *config* is unread, and the rest are the launch's facts.
    """
    gate = solve = None
    if gate_mode == "head":
        gate = CompiledProgram(cumulative_gate_program(call))
    if gate_mode != "channel":
        solve = CompiledProgram(
            solve_program(call, num_sequences, offsets_dtype, gate_mode=gate_mode)
        )
    prepare, scan = build_programs(
        call,
        num_sequences,
        gate_mode=gate_mode,
        initial_state=initial_state,
        dense=not call.varlen,
        split_output=split_output,
    )
    return gate, solve, prepare, scan


def build_programs(
    call, num_sequences, *, gate_mode, initial_state=True, dense=False, split_output=False
):
    from tileops._kernels.linear_attention.kda.chunk_programs import (
        chunk_prepare_program,
        chunk_scan_program,
    )

    total = call.batch * call.seq_len
    plan = launch_plan(call.dim_k, call.dtype.itemsize, num_sequences, call.smem_budget, call.arch)
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
        from tileops._kernels.linear_attention.kda.prepare_split import split_prepare_program

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
    prepare = CompiledProgram(prepare)
    if gate_mode == "channel" or split_output:
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
    return prepare, CompiledProgram(scan)


def _split_scan(call, num_sequences, total, dtype, initial_state=True):
    """The scan as the state walk over the chunks, forming the outputs in the walk or after it.

    The state walk takes the narrowest value tile whose CTAs fit one wave: a narrower tile
    shortens each CTA's chain of matmuls, a second wave doubles it.

    Forming each chunk's output in the walk saves the output launch and the round trip of
    the chunk states and new values through global memory, but adds the output's products
    to the walk's serial chain, which the chunk-parallel output launch would run beside
    otherwise idle SMs. The longer the sequences, the longer that chain, so the walk takes
    the outputs only when its CTAs already cover a larger share of the SMs.
    """
    from tileops._kernels.linear_attention.kda.chunk_programs import (
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
        # The crossovers below are set for the SM90 walk only; other architectures keep
        # the chunk-parallel output launch.
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
    state_scan = CompiledProgram(state_scan)
    if with_output:
        return state_scan
    output = CompiledProgram(
        chunk_output_program(
            call.value_heads, call.dim_k, call.dim_v, CHUNK_SIZE, dtype, total, num_sequences
        )
    )

    def scan(w, u, qg, kg, aqk, dec, h0, offsets):
        starts, values, final_state = state_scan(w, u, kg, dec, h0, offsets)
        return output(qg, aqk, starts, values, offsets), final_state

    return scan


def prepare_inputs(call, tensors, *, programs, gate_mode, gate_params=()):
    q, k, v, gate, beta, offsets = tensors
    _, solve, prepare, _ = programs
    if gate_mode == "channel":
        return prepare(q, k, v, gate, beta, offsets, *gate_params)
    from tileops._kernels.linear_attention.gdn.prefill_prepare import prefill_blocksolve_A_bthd

    inverse, _ = prefill_blocksolve_A_bthd(
        solve, k, gate, beta, offsets, CHUNK_SIZE, l2norm=call.l2norm
    )
    return prepare(q, k, v, gate, beta, offsets, inverse)


def run_chunk_prefill(kernel, call, inputs, *, gate_mode, split_output=False):
    """Run one chunked prefill launch on the programs *kernel* holds for its facts."""
    q, k, v, _, beta, state, offsets, _, _ = inputs
    total = call.batch * call.seq_len
    offsets = packed_offsets(call.batch, call.seq_len, offsets, q.device)
    num_sequences = offsets.numel() - 1
    q, k, v, beta = (tensor.reshape(1, total, *tensor.shape[2:]) for tensor in (q, k, v, beta))
    # The channel-gate walk starts from zero itself when there is no initial state, so the
    # state buffer it is handed need not be filled.
    initial_state = state is not None or gate_mode != "channel"
    programs = kernel.program(
        chunk_programs, call, gate_mode, num_sequences, initial_state, offsets.dtype, split_output
    )
    gate = prepare_gate(call, inputs, offsets, gate_mode=gate_mode, program=programs[0])
    if state is None:
        state = (torch.zeros if initial_state else torch.empty)(
            num_sequences,
            call.value_heads,
            call.dim_k,
            call.dim_v,
            dtype=torch.float32,
            device=q.device,
        )
    gate_params = ()
    if gate_mode == "channel":
        from tileops._kernels.linear_attention.kda.gate import raw_gate_arguments

        gate_params = raw_gate_arguments(call.gate_in_kernel, *inputs[-2:])
    prepared = prepare_inputs(
        call,
        (q, k, v, gate, beta, offsets),
        programs=programs,
        gate_mode=gate_mode,
        gate_params=gate_params,
    )
    output, final_state = programs[3](*prepared, state, offsets)
    return output.reshape(call.batch, call.seq_len, call.value_heads, call.dim_v), final_state
