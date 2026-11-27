"""Mamba-2 benchmarks, one case per manifest call. The SSD stages and the full forward
(SSDChunkCumsum, SSDChunkCoupling, SSDChunkState, SSDStatePassing, SSDChunkScan) run against
vLLM's Triton kernels, which carry mamba_ssm's over packed varlen tokens, and
mamba_chunk_scan_combined_varlen; the causal convolution ahead of them against vLLM's too.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    baseline_module,
    compiled_reference,
    flashinfer_op,
    private_inputs,
    vllm_op,
    vllm_state_lines,
)
from tileops._ops.mamba.causal_conv1d import CausalConv1dFwdOp, CausalConv1dUpdateFwdOp
from tileops._ops.mamba.mamba2_fwd import Mamba2FwdOp
from tileops._ops.mamba.selective_scan import SelectiveScanFwdOp
from tileops._ops.mamba.ssd_chunk_coupling import SSDChunkCouplingFwdOp
from tileops._ops.mamba.ssd_chunk_cumsum import SSDChunkCumsumFwdOp
from tileops._ops.mamba.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops._ops.mamba.ssd_chunk_state import SSDChunkStateFwdOp
from tileops._ops.mamba.ssd_recurrent import SSDRecurrentFwdOp
from tileops._ops.mamba.ssd_state_passing import SSDStatePassingFwdOp
from workloads.mamba import ssd_decode_result

VLLM_CONV1D = "model_executor.layers.mamba.ops.causal_conv1d"

# vLLM's Mamba-2 kernels, which carry mamba_ssm's Triton SSD stages over packed varlen tokens.
_VLLM_SSD = "vllm.model_executor.layers.mamba.ops."


def _vllm_ssd(module: str, name: str):
    return getattr(baseline_module(_VLLM_SSD + module, VLLM_TAG), name)


def _vllm_chunks(batch: int, seqlen: int, chunk_size: int, device: torch.device):
    """vLLM's chunking of *batch* packed sequences of *seqlen* tokens: the chunk boundaries,
    each sequence's last chunk, and each chunk's sequence."""
    chunks = -(-seqlen // chunk_size)
    within = torch.arange(chunks, device=device) * chunk_size
    starts = (torch.arange(batch, device=device)[:, None] * seqlen + within).flatten()
    bounds = torch.cat([starts, starts.new_tensor([batch * seqlen])]).to(torch.int32)
    last = (torch.arange(1, batch + 1, device=device) * chunks - 1).to(torch.int32)
    owner = torch.arange(batch, device=device).repeat_interleave(chunks).to(torch.int32)
    return bounds, last, owner


def _to_vllm_chunks(t: torch.Tensor) -> torch.Tensor:
    """(b, h, c, L) as vLLM's (h, b * c, L)."""
    return t.transpose(0, 1).flatten(1, 2).contiguous()


def _from_vllm_chunks(t: torch.Tensor, batch: int) -> torch.Tensor:
    """vLLM's (h, b * c, L) as (b, h, c, L)."""
    return t.unflatten(1, (batch, -1)).transpose(0, 1)


def _end_states(entering: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """States entering each chunk, (b, c, ...), as vLLM's states leaving each chunk, packed,
    and the states entering each sequence. A sequence's last state is never read."""
    leaving = torch.cat([entering[:, 1:], torch.zeros_like(entering[:, :1])], dim=1)
    return leaving.flatten(0, 1).contiguous(), entering[:, 0].contiguous()


def _torch_baselines(implementations: dict, ref_program) -> None:
    implementations["torch-ref"] = ref_program
    implementations[TORCH_COMPILE_TAG] = compiled_reference(ref_program)


@pytest.mark.parametrize("case", bench.cases(SSDChunkCouplingFwdOp))
def test_ssd_chunk_coupling_fwd_bench(case) -> None:
    """The CB stage on its own, over the shapes the Mamba-2 configs give it."""
    chunk_len = case.params["chunk_size"]
    op = SSDChunkCouplingFwdOp(**case.arguments)
    bmm_chunk_fwd = _vllm_ssd("ssd_bmm", "_bmm_chunk_fwd")
    c_in = case.inputs[0]
    batch, seqlen = c_in.shape[:2]
    bounds, _, _ = _vllm_chunks(batch, seqlen, chunk_len, c_in.device)

    def vllm_fn(c, b):
        cb = bmm_chunk_fwd(
            c.flatten(0, 1), b.flatten(0, 1), chunk_len, bounds, causal=True, output_dtype=c.dtype
        )
        return cb.unflatten(0, (batch, -1)).tril()

    bench.Runner(op, case).compare({VLLM_TAG: vllm_fn, "torch": case.reference})


@pytest.mark.parametrize("case", bench.cases(SSDChunkCumsumFwdOp))
def test_ssd_chunk_cumsum_fwd_bench(case) -> None:
    dt, A, dt_bias = case.inputs
    op = SSDChunkCumsumFwdOp(**case.arguments)
    chunk_cumsum_fwd = _vllm_ssd("ssd_chunk_state", "_chunk_cumsum_fwd")
    batch, seqlen = dt.shape[:2]
    bounds, _, _ = _vllm_chunks(batch, seqlen, op.chunk_size, dt.device)

    # vLLM returns (dA_cumsum, dt_out) with a float32 dt_out; return TileOPs' order and dtype.
    def vllm_fwd():
        dA_cumsum, dt_out = chunk_cumsum_fwd(
            dt.flatten(0, 1),
            A,
            op.chunk_size,
            bounds,
            dt_bias=dt_bias,
            dt_softplus=op.dt_softplus,
            dt_limit=(op.dt_min, op.dt_max),
        )
        return (
            _from_vllm_chunks(dt_out, batch).to(op.out_dtype),
            _from_vllm_chunks(dA_cumsum, batch),
        )

    implementations = {VLLM_TAG: bench.Implementation(run=vllm_fwd, args=())}
    _torch_baselines(implementations, case.reference)
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(SSDChunkScanFwdOp))
def test_ssd_chunk_scan_fwd_bench(case) -> None:
    x, cb, dA_cumsum, C, prev_states, dt = case.inputs
    op = SSDChunkScanFwdOp(**case.arguments)
    chunk_scan_fwd = _vllm_ssd("ssd_chunk_scan", "_chunk_scan_fwd")
    batch, seqlen = x.shape[:2]
    bounds, _, owner = _vllm_chunks(batch, seqlen, dt.shape[-1], x.device)
    leaving, initial = _end_states(prev_states)
    cb_v, dt_v, dA_v = cb.flatten(0, 1), _to_vllm_chunks(dt), _to_vllm_chunks(dA_cumsum)
    out = torch.empty_like(x.flatten(0, 1))

    def vllm_fwd():
        chunk_scan_fwd(
            cb_v,
            x.flatten(0, 1),
            dt_v,
            dA_v,
            C.flatten(0, 1),
            leaving,
            bounds,
            out,
            owner,
            initial_states=initial,
        )
        # Match the op's FP32 output; the cast is part of this baseline's timing.
        return out.unflatten(0, (batch, seqlen)).float()

    implementations = {VLLM_TAG: bench.Implementation(run=vllm_fwd, args=())}
    _torch_baselines(implementations, case.reference)
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(SSDChunkStateFwdOp))
def test_ssd_chunk_state_fwd_bench(case) -> None:
    x, Bmat, dt, dA_cumsum, seq_idx = case.inputs
    op = SSDChunkStateFwdOp(**case.arguments)
    implementations = {}

    # vLLM restarts only at chunk boundaries, so a seq_idx inside a chunk has no equivalent.
    if seq_idx is None:
        chunk_state_fwd = _vllm_ssd("ssd_chunk_state", "_chunk_state_fwd")
        batch, seqlen = x.shape[:2]
        bounds, _, _ = _vllm_chunks(batch, seqlen, dt.shape[-1], x.device)
        dt_v, dA_v = _to_vllm_chunks(dt), _to_vllm_chunks(dA_cumsum)

        def vllm_fwd():
            states = chunk_state_fwd(Bmat.flatten(0, 1), x.flatten(0, 1), dt_v, dA_v, bounds)
            return states.unflatten(0, (batch, -1))

        implementations[VLLM_TAG] = bench.Implementation(run=vllm_fwd, args=())

    _torch_baselines(implementations, case.reference)
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(SSDStatePassingFwdOp))
def test_ssd_state_passing_fwd_bench(case) -> None:
    states, dA_chunk_cumsum, initial_states = case.inputs
    op = SSDStatePassingFwdOp(**case.arguments)
    state_passing_fwd = _vllm_ssd("ssd_state_passing", "_state_passing_fwd")
    batch, chunks = states.shape[:2]
    last = (torch.arange(1, batch + 1, device=states.device) * chunks - 1).to(torch.int32)
    # vLLM reads each chunk's total decay from the last slot of a full cumsum.
    dA_v = dA_chunk_cumsum.transpose(0, 1).flatten(1).unsqueeze(-1).contiguous()
    packed = states.flatten(0, 1)
    first = initial_states if initial_states is not None else torch.zeros_like(states[:, 0])

    # vLLM returns each chunk's leaving state; TileOPs, each entering state and the final one.
    def vllm_fwd():
        leaving = state_passing_fwd(
            packed, dA_v, last, initial_states=initial_states, out_dtype=torch.float32
        ).unflatten(0, (batch, chunks))
        entering = torch.cat([first.float()[:, None], leaving[:, :-1]], dim=1)
        return entering, leaving[:, -1]

    implementations = {VLLM_TAG: bench.Implementation(run=vllm_fwd, args=())}
    _torch_baselines(implementations, case.reference)
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(SSDRecurrentFwdOp))
def test_ssd_decode_bench(case) -> None:
    x = case.inputs[2]
    op = SSDRecurrentFwdOp(**case.arguments)
    selective_state_update = _vllm_ssd("mamba_ssm", "selective_state_update")

    skip = torch.zeros(x.shape[1:], dtype=torch.float32, device=x.device)
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)

    def vllm_fn(A, dt, x, B, C, state):
        # FP32 x preserves the contract's output.
        selective_state_update(state, x.float(), dt, A, B, C, skip, skip, out=out)
        return out

    flashinfer_update = flashinfer_op("mamba.selective_state_update")

    skip_tied = torch.zeros(x.shape[1], 1, dtype=torch.float32, device=x.device).expand(x.shape[1:])

    def flashinfer_fn(A, dt, x, B, C, state):
        rates = A[:, :1, :1].contiguous().expand_as(A)
        steps = dt[:, :, :1].contiguous().expand_as(dt)
        return flashinfer_update(state, x.float(), steps, rates, B.float(), C.float(), skip_tied)

    def private_state(run):
        # Every implementation updates the recurrent state in place.
        return private_inputs(run, case.inputs, 5)

    bench.Runner(op, case).compare(
        {
            VLLM_TAG: private_state(lambda *args: ssd_decode_result(vllm_fn, *args)),
            FLASHINFER_TAG: private_state(lambda *args: ssd_decode_result(flashinfer_fn, *args)),
            "torch-ref": private_state(case.reference),
            TORCH_COMPILE_TAG: private_state(compiled_reference(case.reference)),
        }
    )


@pytest.mark.parametrize("case", bench.cases(Mamba2FwdOp))
def test_mamba2_fwd_bench(case):
    x, dt, A, B, C, dt_bias, initial_states = case.inputs
    op = Mamba2FwdOp(**case.arguments)
    reference = case.reference
    combined = _vllm_ssd("ssd_combined", "mamba_chunk_scan_combined_varlen")
    batch, seqlen = x.shape[:2]
    bounds, last, owner = _vllm_chunks(batch, seqlen, op.chunk_size, x.device)
    seqlens = torch.arange(batch + 1, device=x.device, dtype=torch.int32) * seqlen
    out = torch.empty_like(x.flatten(0, 1))

    # Only the five leading tensors are positional, so every path gets identical clone
    # treatment; the optional tensors are captured.
    reference_args = (x, dt, A, B, C)

    def _vllm_wrapper(x, dt, A, B, C):
        final_states = combined(
            x.flatten(0, 1),
            dt.flatten(0, 1),
            A,
            B.flatten(0, 1),
            C.flatten(0, 1),
            op.chunk_size,
            seqlens,
            bounds,
            last,
            owner,
            out,
            dt_bias=dt_bias,
            initial_states=initial_states,
            dt_softplus=op.dt_softplus,
            # The intermediate states must share the float32 initial states' dtype.
            state_dtype=torch.float32,
        )
        return out.unflatten(0, (batch, seqlen)).float(), final_states.float()

    def _torch_wrapper(x, dt, A, B, C):
        return reference(x, dt, A, B, C, dt_bias, initial_states)

    implementations = {
        VLLM_TAG: bench.Implementation(run=_vllm_wrapper, args=reference_args),
        "torch-ref": bench.Implementation(run=_torch_wrapper, args=reference_args),
        TORCH_COMPILE_TAG: bench.Implementation(
            run=compiled_reference(_torch_wrapper), args=reference_args
        ),
    }
    bench.Runner(op, case).compare(implementations)


def _vllm_conv_lines(conv_state: torch.Tensor, slots: torch.Tensor | None, count: int):
    """vLLM's cache lines for *conv_state*, the line of each of the *count* sequences, and the
    reset that loads the state into them; without *slots*, sequence i holds state i."""
    if slots is None:
        slots = torch.arange(count, dtype=torch.int32, device=conv_state.device)
    return vllm_state_lines(conv_state, slots)


def _vllm_taps_differ(case) -> str | None:
    """Why vLLM's taps are not *case*'s, or None."""
    if case.arguments.get("dilation", 1) != 1:
        return "vLLM convolves adjacent taps only"
    return None


def _vllm_conv_differs(case) -> str | None:
    """Why vLLM's convolution computes something else than *case*, or None."""
    differs = _vllm_taps_differ(case)
    if differs is not None:
        return differs
    if case.inputs[1].dtype != case.inputs[0].dtype:
        return "vLLM takes the weight in the activation dtype"
    return None


@pytest.mark.parametrize("case", bench.cases(CausalConv1dFwdOp))
def test_causal_conv1d_bench(case) -> None:
    """Prefill against vLLM, which takes the (D, Tt) transposed view of the same storage."""
    x, weight, conv_state, cu_seqlens, bias, has_initial_state, cache_indices = case.inputs
    op = CausalConv1dFwdOp(**case.arguments)
    causal_conv1d_fn = vllm_op("causal_conv1d_fn", module=VLLM_CONV1D)
    lines, indices, reset = _vllm_conv_lines(conv_state, cache_indices, cu_seqlens.numel() - 1)
    # vLLM reads the flags unconditionally; absent, no sequence has a state.
    flags = (
        has_initial_state
        if has_initial_state is not None
        else torch.zeros(cu_seqlens.numel() - 1, dtype=torch.bool, device=x.device)
    )

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        out = causal_conv1d_fn(
            x.t(),
            weight,
            bias,
            lines,
            cu_seqlens,
            cache_indices=indices,
            has_initial_state=flags,
            activation=op.activation,
        )
        return out.t(), lines[1:]

    vllm = bench.Implementation(
        run=run, args=(), reset=reset, noncomparable_reason=_vllm_conv_differs(case)
    )
    bench.Runner(op, case).compare({VLLM_TAG: vllm})


@pytest.mark.parametrize("case", bench.cases(CausalConv1dUpdateFwdOp))
def test_causal_conv1d_update_bench(case) -> None:
    x, weight, conv_state, bias, slots, accepted, query_start_loc = case.inputs
    op = CausalConv1dUpdateFwdOp(**case.arguments)
    causal_conv1d_update = vllm_op("causal_conv1d_update", module=VLLM_CONV1D)
    count = x.shape[0] if query_start_loc is None else query_start_loc.numel() - 1
    lines, indices, reset = _vllm_conv_lines(conv_state, slots, count)
    out = torch.empty_like(x)
    varlen = {}
    if query_start_loc is not None:
        varlen = {"query_start_loc": query_start_loc, "max_query_len": op.max_query_len}

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        causal_conv1d_update(
            x, lines, weight, bias, activation=op.activation, conv_state_indices=indices,
            num_accepted_tokens=accepted, out=out, **varlen,
        )  # fmt: skip
        return out, lines[1:]

    # vLLM's causal_conv1d_update multiplies a float32 weight in float32, as the reference does.
    implementations = {
        VLLM_TAG: bench.Implementation(
            run=run, args=(), reset=reset, noncomparable_reason=_vllm_taps_differ(case)
        )
    }
    if accepted is not None:
        # The reference advances its state argument in place.
        implementations["torch-ref"] = private_inputs(case.torch, case.inputs, 2)
    bench.Runner(op, case).compare(implementations)


def _vllm_selective_scan(case) -> bench.Implementation:
    """vLLM's selective scan, which writes ``y`` over ``z`` (``delta`` without a gate) and the
    final states into its cache lines.

    A batch without ``seq_idx`` runs as ``B`` sequences of the ``(B, D, S)`` layout; with it,
    every sequence of every row runs in the ``(D, total)`` varlen layout, one line each.
    """
    u, delta, A, B_in, C_in, D_in, z, delta_bias, seq_idx, initial_states = case.inputs
    batch, dim, seqlen = u.shape
    scan = vllm_op("selective_scan_fn", module="model_executor.layers.mamba.ops.mamba_ssm")
    if seq_idx is None:
        count = batch

        def layout(t):
            return t

        starts = None
        last = torch.arange(1, batch + 1, device=u.device)
    else:
        cuts = [
            row * seqlen + step
            for row in range(batch)
            for step in [
                0,
                *(torch.nonzero(seq_idx[row, 1:] != seq_idx[row, :-1]) + 1).flatten().tolist(),
            ]
        ]
        count = len(cuts)
        starts = torch.tensor([*cuts, batch * seqlen], dtype=torch.int32, device=u.device)
        # The state after row b is the state of its last sequence.
        last = torch.tensor(
            [
                i + 1
                for i, (a, b) in enumerate(zip(cuts, cuts[1:] + [batch * seqlen], strict=True))
                if b % seqlen == 0
            ],
            device=u.device,
        )

        def layout(t):
            return None if t is None else t.movedim(0, -2).flatten(-2)

    lines = torch.empty((count + 1, dim, A.shape[1]), dtype=torch.float32, device=u.device)
    indices = torch.arange(1, count + 1, dtype=torch.int32, device=u.device)
    flags = torch.full((count,), initial_states is not None, dtype=torch.bool, device=u.device)
    shared = [layout(t) for t in (u, delta, B_in, C_in, z)]
    args = [None if t is None else t.clone() for t in shared]

    def reset() -> None:
        for private, source in zip(args, shared, strict=True):
            if private is not None:
                private.copy_(source)
        if initial_states is not None:
            lines[1:].copy_(initial_states)

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        x, step, b, c, gate = args
        y = scan(
            x,
            lines,
            step,
            A,
            b,
            c,
            D_in,
            gate,
            delta_bias,
            case.arguments["delta_softplus"],
            query_start_loc=starts,
            cache_indices=indices,
            has_initial_state=flags,
        )
        if seq_idx is not None:
            y = y.unflatten(-1, (batch, seqlen)).movedim(-2, 0)
        return y, lines[last]

    return bench.Implementation(run=run, args=(), reset=reset)


@pytest.mark.parametrize("case", bench.cases(SelectiveScanFwdOp))
def test_selective_scan_bench(case) -> None:
    """The Mamba-1 scan against vLLM's CUDA scan, and for one decode step vLLM's Triton
    state update."""
    u, delta, A, B_in, C_in, D_in, z, delta_bias, seq_idx, initial_states = case.inputs
    op = SelectiveScanFwdOp(**case.arguments)
    softplus = case.arguments["delta_softplus"]
    implementations = {VLLM_TAG: _vllm_selective_scan(case)}

    # The state update takes one step with one group of B and C, and requires a state.
    if u.shape[2] == 1 and B_in.shape[1] == 1 and initial_states is not None:
        selective_state_update = _vllm_ssd("mamba_ssm", "selective_state_update")
        # vLLM takes D and the delta bias as tensors and writes into a given output.
        zeros = torch.zeros(u.shape[1], dtype=torch.float32, device=u.device)
        out = torch.empty(u.shape[:2], dtype=u.dtype, device=u.device)

        def update_fn(u, delta, A, B_in, C_in, D_in, z, delta_bias, seq_idx, state):
            selective_state_update(
                state,
                u[..., 0],
                delta[..., 0],
                A,
                B_in[:, 0, :, 0],
                C_in[:, 0, :, 0],
                zeros if D_in is None else D_in,
                zeros if delta_bias is None else delta_bias,
                None if z is None else z[..., 0],
                dt_softplus=softplus,
                out=out,
            )
            return out[..., None], state

        implementations["vllm-state-update"] = private_inputs(update_fn, case.inputs, 9)

    bench.Runner(op, case).compare(implementations)
