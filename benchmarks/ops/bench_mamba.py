"""Mamba-2 SSD benchmarks, one case per manifest call, against the mamba_ssm Triton kernels: each stage,
and the full forward (DaCumsum, CBProducer, SSDChunkState, SSDStatePassing, SSDChunkScan) against
mamba_chunk_scan_combined.
"""

import pytest
import torch

from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    compiled_reference,
    flashinfer_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.mamba.mamba2_fwd import Mamba2FwdOp
from tileops.ops.mamba.ssd_chunk_coupling import SSDChunkCouplingFwdOp
from tileops.ops.mamba.ssd_chunk_cumsum import SSDChunkCumsumFwdOp
from tileops.ops.mamba.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops.ops.mamba.ssd_chunk_state import SSDChunkStateFwdOp
from tileops.ops.mamba.ssd_recurrent import SSDRecurrentFwdOp
from tileops.ops.mamba.ssd_state_passing import SSDStatePassingFwdOp
from workloads.mamba import (
    CBProducerFwdCall,
    DaCumsumFwdCall,
    Mamba2FwdCall,
    SSDChunkScanFwdCall,
    SSDChunkStateFwdCall,
    SSDDecodeFwdCall,
    SSDStatePassingFwdCall,
)

# Optional mamba_ssm Triton baselines
try:
    from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd as _mamba_chunk_cumsum_fwd
except ImportError:
    _mamba_chunk_cumsum_fwd = None

try:
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd as _mamba_chunk_scan_fwd
except ImportError:
    _mamba_chunk_scan_fwd = None

try:
    from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd as _mamba_chunk_state_fwd
except ImportError:
    _mamba_chunk_state_fwd = None

try:
    from mamba_ssm.ops.triton.ssd_state_passing import (
        _state_passing_fwd as _mamba_state_passing_fwd,
    )
except ImportError:
    _mamba_state_passing_fwd = None

try:
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined as _mamba_chunk_scan_combined,
    )
except ImportError:
    _mamba_chunk_scan_combined = None


def _torch_baselines(functors: dict, ref_program) -> None:
    functors["torch-ref"] = ref_program
    functors[TORCH_COMPILE_TAG] = compiled_reference(ref_program)


@pytest.mark.parametrize("call", manifest_calls(SSDChunkCouplingFwdOp))
def test_cb_producer_fwd_bench(call) -> None:
    """The CB stage on its own, over the shapes the Mamba-2 configs give it."""
    workload = CBProducerFwdCall(call)
    inputs = workload.gen_inputs()
    op = SSDChunkCouplingFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)
    from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd

    def mamba_fn(c, b):
        return _bmm_chunk_fwd(c, b, call.ix["chunk_len"], causal=True, output_dtype=c.dtype).tril()

    bm.compare({"tileops": op, "mamba": mamba_fn, "torch": workload.ref_program}, *inputs)


@pytest.mark.parametrize("call", manifest_calls(SSDChunkCumsumFwdOp))
def test_da_cumsum_fwd_bench(call) -> None:
    workload = DaCumsumFwdCall(call)
    dt, A, dt_bias = inputs = workload.gen_inputs()
    op = SSDChunkCumsumFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)
    functors = {"tileops": op}

    # _chunk_cumsum_fwd returns (dA_cumsum, dt_out) with a float32 dt_out; the baseline
    # returns them in TileOPs' order and dtype.
    if _mamba_chunk_cumsum_fwd is not None:

        def mamba_fwd():
            dA_cumsum, dt_out = _mamba_chunk_cumsum_fwd(
                dt, A, op.chunk_len, dt_bias=dt_bias, dt_softplus=op.dt_softplus
            )
            return dt_out.to(op.out_dtype), dA_cumsum

        functors["mamba"] = (mamba_fwd, ())

    _torch_baselines(functors, workload.ref_program)
    bm.compare(functors, *inputs)


@pytest.mark.parametrize("call", manifest_calls(SSDChunkScanFwdOp))
def test_ssd_chunk_scan_fwd_bench(call) -> None:
    workload = SSDChunkScanFwdCall(call)
    x, cb, dA_cumsum, C, prev_states, dt = inputs = workload.gen_inputs()
    op = SSDChunkScanFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)
    functors = {"tileops": op}

    if _mamba_chunk_scan_fwd is not None:
        # mamba signature: _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, ...)
        def mamba_fwd():
            # Match the op's FP32 output; the cast is part of this baseline's timing.
            out, _ = _mamba_chunk_scan_fwd(cb, x, dt, dA_cumsum, C, prev_states)
            return out.float()

        functors["mamba"] = (mamba_fwd, ())

    _torch_baselines(functors, workload.ref_program)
    bm.compare(
        functors,
        *inputs,
    )


@pytest.mark.parametrize("call", manifest_calls(SSDChunkStateFwdOp))
def test_ssd_chunk_state_fwd_bench(call) -> None:
    workload = SSDChunkStateFwdCall(call)
    x, Bmat, dt, dA_cumsum, seq_idx = inputs = workload.gen_inputs()
    op = SSDChunkStateFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)
    functors = {"tileops": op}

    if _mamba_chunk_state_fwd is not None:
        # dt and dA_cumsum share TileOPs' (b, h, c, L) layout.
        def mamba_fwd():
            return _mamba_chunk_state_fwd(Bmat, x, dt, dA_cumsum, seq_idx=seq_idx)

        functors["mamba"] = (mamba_fwd, ())

    _torch_baselines(functors, workload.ref_program)
    bm.compare(
        functors,
        *inputs,
    )


@pytest.mark.parametrize("call", manifest_calls(SSDStatePassingFwdOp))
def test_ssd_state_passing_fwd_bench(call) -> None:
    workload = SSDStatePassingFwdCall(call)
    states, dA_chunk_cumsum, initial_states = inputs = workload.gen_inputs()
    op = SSDStatePassingFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)
    functors = {"tileops": op}

    if _mamba_state_passing_fwd is not None:
        # dA_chunk_cumsum shares TileOPs' (b, h, c) layout; float32 output as TileOPs.
        def mamba_fwd():
            return _mamba_state_passing_fwd(
                states, dA_chunk_cumsum, initial_states=initial_states, out_dtype=torch.float32
            )

        functors["mamba"] = (mamba_fwd, ())

    _torch_baselines(functors, workload.ref_program)
    bm.compare(functors, *inputs)


@pytest.mark.parametrize("call", manifest_calls(SSDRecurrentFwdOp))
def test_ssd_decode_bench(call) -> None:
    workload = SSDDecodeFwdCall(call)
    A, dt, x, B_in, C_in, state = workload.gen_inputs()
    op = SSDRecurrentFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)

    from mamba_ssm.ops.triton.selective_state_update import selective_state_update

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

    def reset_state(fn):
        private = state.clone()

        def run(A, dt, x, B, C, source_state):
            private.copy_(source_state)
            return fn(A, dt, x, B, C, private)

        return run

    functors = {
        "tileops": reset_state(op),
        "mamba": reset_state(mamba_fn),
        FLASHINFER_TAG: reset_state(flashinfer_fn),
        "torch-ref": reset_state(workload.ref_program),
        TORCH_COMPILE_TAG: reset_state(compiled_reference(workload.ref_program)),
    }
    bm.compare(functors, A, dt, x, B_in, C_in, state)


@pytest.mark.parametrize("call", manifest_calls(Mamba2FwdOp))
def test_mamba2_fwd_bench(call):
    workload = Mamba2FwdCall(call)
    inputs = workload.gen_inputs()
    x, dt, A, B, C, dt_bias, initial_states = inputs
    op = Mamba2FwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)
    functors = {"tileops": op}

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

        functors["mamba"] = (_mamba_wrapper, reference_args)

    def _torch_wrapper(x, dt, A, B, C):
        return workload.ref_program(x, dt, A, B, C, dt_bias, initial_states)

    functors["torch-ref"] = (_torch_wrapper, reference_args)
    functors[TORCH_COMPILE_TAG] = (compiled_reference(_torch_wrapper), reference_args)

    bm.compare(
        functors,
        *inputs,
    )
