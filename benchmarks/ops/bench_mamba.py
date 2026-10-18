"""Mamba-2 benchmarks, one case per manifest call. The SSD stages and the full forward
(SSDChunkCumsum, SSDChunkCoupling, SSDChunkState, SSDStatePassing, SSDChunkScan) run against the
mamba_ssm Triton kernels and mamba_chunk_scan_combined; the causal convolution ahead of them
against vLLM's Triton kernels and Dao-AILab's causal-conv1d.
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
    optional_baseline,
    private_inputs,
    vllm_op,
)
from tileops.ops.mamba.causal_conv1d import CausalConv1dFwdOp, CausalConv1dUpdateFwdOp
from tileops.ops.mamba.mamba2_fwd import Mamba2FwdOp
from tileops.ops.mamba.ssd_chunk_coupling import SSDChunkCouplingFwdOp
from tileops.ops.mamba.ssd_chunk_cumsum import SSDChunkCumsumFwdOp
from tileops.ops.mamba.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops.ops.mamba.ssd_chunk_state import SSDChunkStateFwdOp
from tileops.ops.mamba.ssd_recurrent import SSDRecurrentFwdOp
from tileops.ops.mamba.ssd_state_passing import SSDStatePassingFwdOp
from workloads.mamba import ssd_decode_result

VLLM_CONV1D = "model_executor.layers.mamba.ops.causal_conv1d"
DAO_CONV1D_TAG = "causal-conv1d"

# Optional mamba_ssm Triton baselines
_mamba_chunk_cumsum_fwd = optional_baseline(
    "mamba_ssm.ops.triton.ssd_chunk_state", "mamba-ssm", "_chunk_cumsum_fwd"
)
_mamba_chunk_scan_fwd = optional_baseline(
    "mamba_ssm.ops.triton.ssd_chunk_scan", "mamba-ssm", "_chunk_scan_fwd"
)
_mamba_chunk_state_fwd = optional_baseline(
    "mamba_ssm.ops.triton.ssd_chunk_state", "mamba-ssm", "_chunk_state_fwd"
)
_mamba_state_passing_fwd = optional_baseline(
    "mamba_ssm.ops.triton.ssd_state_passing", "mamba-ssm", "_state_passing_fwd"
)
_mamba_chunk_scan_combined = optional_baseline(
    "mamba_ssm.ops.triton.ssd_combined", "mamba-ssm", "mamba_chunk_scan_combined"
)


def _torch_baselines(implementations: dict, ref_program) -> None:
    implementations["torch-ref"] = ref_program
    implementations[TORCH_COMPILE_TAG] = compiled_reference(ref_program)


@pytest.mark.parametrize("case", bench.cases(SSDChunkCouplingFwdOp), ids=lambda case: case.id)
def test_ssd_chunk_coupling_fwd_bench(case) -> None:
    """The CB stage on its own, over the shapes the Mamba-2 configs give it."""
    chunk_len = case.params["chunk_len"]
    op = SSDChunkCouplingFwdOp(**case.arguments)
    _bmm_chunk_fwd = baseline_module("mamba_ssm.ops.triton.ssd_bmm", "mamba-ssm")._bmm_chunk_fwd

    def mamba_fn(c, b):
        return _bmm_chunk_fwd(c, b, chunk_len, causal=True, output_dtype=c.dtype).tril()

    bench.Runner(op, case).compare({"mamba": mamba_fn, "torch": case.reference})


@pytest.mark.parametrize("case", bench.cases(SSDChunkCumsumFwdOp), ids=lambda case: case.id)
def test_ssd_chunk_cumsum_fwd_bench(case) -> None:
    dt, A, dt_bias = case.inputs
    op = SSDChunkCumsumFwdOp(**case.arguments)
    implementations = {}

    # _chunk_cumsum_fwd returns (dA_cumsum, dt_out) with a float32 dt_out; the baseline
    # returns them in TileOPs' order and dtype.
    if _mamba_chunk_cumsum_fwd is not None:

        def mamba_fwd():
            dA_cumsum, dt_out = _mamba_chunk_cumsum_fwd(
                dt, A, op.chunk_len, dt_bias=dt_bias, dt_softplus=op.dt_softplus
            )
            return dt_out.to(op.out_dtype), dA_cumsum

        implementations["mamba"] = bench.Implementation(run=mamba_fwd, args=())

    _torch_baselines(implementations, case.reference)
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(SSDChunkScanFwdOp), ids=lambda case: case.id)
def test_ssd_chunk_scan_fwd_bench(case) -> None:
    x, cb, dA_cumsum, C, prev_states, dt = case.inputs
    op = SSDChunkScanFwdOp(**case.arguments)
    implementations = {}

    if _mamba_chunk_scan_fwd is not None:
        # mamba signature: _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, ...)
        def mamba_fwd():
            # Match the op's FP32 output; the cast is part of this baseline's timing.
            out, _ = _mamba_chunk_scan_fwd(cb, x, dt, dA_cumsum, C, prev_states)
            return out.float()

        implementations["mamba"] = bench.Implementation(run=mamba_fwd, args=())

    _torch_baselines(implementations, case.reference)
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(SSDChunkStateFwdOp), ids=lambda case: case.id)
def test_ssd_chunk_state_fwd_bench(case) -> None:
    x, Bmat, dt, dA_cumsum, seq_idx = case.inputs
    op = SSDChunkStateFwdOp(**case.arguments)
    implementations = {}

    if _mamba_chunk_state_fwd is not None:
        # dt and dA_cumsum share TileOPs' (b, h, c, L) layout.
        def mamba_fwd():
            return _mamba_chunk_state_fwd(Bmat, x, dt, dA_cumsum, seq_idx=seq_idx)

        implementations["mamba"] = bench.Implementation(run=mamba_fwd, args=())

    _torch_baselines(implementations, case.reference)
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(SSDStatePassingFwdOp), ids=lambda case: case.id)
def test_ssd_state_passing_fwd_bench(case) -> None:
    states, dA_chunk_cumsum, initial_states = case.inputs
    op = SSDStatePassingFwdOp(**case.arguments)
    implementations = {}

    if _mamba_state_passing_fwd is not None:
        # dA_chunk_cumsum shares TileOPs' (b, h, c) layout; float32 output as TileOPs.
        def mamba_fwd():
            return _mamba_state_passing_fwd(
                states, dA_chunk_cumsum, initial_states=initial_states, out_dtype=torch.float32
            )

        implementations["mamba"] = bench.Implementation(run=mamba_fwd, args=())

    _torch_baselines(implementations, case.reference)
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(SSDRecurrentFwdOp), ids=lambda case: case.id)
def test_ssd_decode_bench(case) -> None:
    x = case.inputs[2]
    op = SSDRecurrentFwdOp(**case.arguments)

    selective_state_update = baseline_module(
        "mamba_ssm.ops.triton.selective_state_update", "mamba-ssm"
    ).selective_state_update

    skip = torch.zeros(x.shape[1:], dtype=torch.float32, device=x.device)

    def mamba_fn(A, dt, x, B, C, state):
        # FP32 x preserves the contract's output; explicit zero bias avoids an upstream None-stride bug.
        return selective_state_update(state, x.float(), dt, A, B, C, D=skip, dt_bias=skip)

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
            "mamba": private_state(lambda *args: ssd_decode_result(mamba_fn, *args)),
            FLASHINFER_TAG: private_state(lambda *args: ssd_decode_result(flashinfer_fn, *args)),
            "torch-ref": private_state(case.reference),
            TORCH_COMPILE_TAG: private_state(compiled_reference(case.reference)),
        }
    )


@pytest.mark.parametrize("case", bench.cases(Mamba2FwdOp), ids=lambda case: case.id)
def test_mamba2_fwd_bench(case):
    x, dt, A, B, C, dt_bias, initial_states = case.inputs
    op = Mamba2FwdOp(**case.arguments)
    implementations = {}
    reference = case.reference

    # Only the five leading tensors are positional, so every path gets identical clone
    # treatment; the optional tensors are captured.
    reference_args = (x, dt, A, B, C)
    if _mamba_chunk_scan_combined is not None:

        def _mamba_wrapper(x, dt, A, B, C):
            out, final_states = _mamba_chunk_scan_combined(
                x,
                dt,
                A,
                B,
                C,
                op.chunk_size,
                dt_bias=dt_bias,
                dt_softplus=op.dt_softplus,
                initial_states=initial_states,
                return_final_states=True,
            )
            return out.float(), final_states.float()

        implementations["mamba"] = bench.Implementation(run=_mamba_wrapper, args=reference_args)

    def _torch_wrapper(x, dt, A, B, C):
        return reference(x, dt, A, B, C, dt_bias, initial_states)

    implementations["torch-ref"] = bench.Implementation(run=_torch_wrapper, args=reference_args)
    implementations[TORCH_COMPILE_TAG] = bench.Implementation(
        run=compiled_reference(_torch_wrapper), args=reference_args
    )

    bench.Runner(op, case).compare(implementations)


def _vllm_state_lines(conv_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's cache lines for *conv_state*, and the indices of the sequences' lines.

    vLLM skips cache line 0, its null block, so the sequences use lines 1 to B.
    """
    lines = torch.empty(
        (conv_state.shape[0] + 1, *conv_state.shape[1:]),
        dtype=conv_state.dtype,
        device=conv_state.device,
    )
    indices = torch.arange(1, conv_state.shape[0] + 1, dtype=torch.int32, device=conv_state.device)
    return lines, indices


@pytest.mark.parametrize("case", bench.cases(CausalConv1dFwdOp), ids=lambda case: case.id)
def test_causal_conv1d_bench(case) -> None:
    """Prefill against vLLM, which takes the (D, Tt) transposed view of the same storage.

    Dao-AILab's causal_conv1d_fn takes one initial state a batch row and returns no final
    state with packed sequences (seq_idx), so it cannot express this op's per-sequence state.
    """
    x, weight, conv_state, cu_seqlens, bias, has_initial_state = case.inputs
    op = CausalConv1dFwdOp(**case.arguments)
    causal_conv1d_fn = vllm_op("causal_conv1d_fn", module=VLLM_CONV1D)
    lines, indices = _vllm_state_lines(conv_state)
    # vLLM reads the flags unconditionally; absent, no sequence has a state.
    flags = (
        has_initial_state
        if has_initial_state is not None
        else torch.zeros(cu_seqlens.numel() - 1, dtype=torch.bool, device=x.device)
    )

    def reset() -> None:
        lines[1:].copy_(conv_state)

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

    bench.Runner(op, case).compare({VLLM_TAG: bench.Implementation(run=run, args=(), reset=reset)})


@pytest.mark.parametrize("case", bench.cases(CausalConv1dUpdateFwdOp), ids=lambda case: case.id)
def test_causal_conv1d_update_bench(case) -> None:
    x, weight, conv_state, bias = case.inputs
    op = CausalConv1dUpdateFwdOp(**case.arguments)
    causal_conv1d_update = vllm_op("causal_conv1d_update", module=VLLM_CONV1D)
    lines, indices = _vllm_state_lines(conv_state)
    out = torch.empty_like(x)

    def reset() -> None:
        lines[1:].copy_(conv_state)

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        causal_conv1d_update(
            x, lines, weight, bias, activation=op.activation, conv_state_indices=indices, out=out
        )
        return out, lines[1:]

    implementations = {VLLM_TAG: bench.Implementation(run=run, args=(), reset=reset)}
    dao_update = optional_baseline("causal_conv1d", DAO_CONV1D_TAG, "causal_conv1d_update")
    if dao_update is not None:

        def dao(x, weight, conv_state, bias):
            return dao_update(x, conv_state, weight, bias, op.activation), conv_state

        implementations[DAO_CONV1D_TAG] = private_inputs(dao, case.inputs, 2)
    bench.Runner(op, case).compare(implementations)
