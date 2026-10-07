"""Mamba-2 SSD benchmarks, one case per manifest call, against the mamba_ssm Triton kernels: each stage,
and the full forward (SSDChunkCumsum, SSDChunkCoupling, SSDChunkState, SSDStatePassing, SSDChunkScan) against
mamba_chunk_scan_combined.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    compiled_reference,
    flashinfer_op,
    private_inputs,
)
from tileops.ops.mamba.mamba2_fwd import Mamba2FwdOp
from tileops.ops.mamba.ssd_chunk_coupling import SSDChunkCouplingFwdOp
from tileops.ops.mamba.ssd_chunk_cumsum import SSDChunkCumsumFwdOp
from tileops.ops.mamba.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops.ops.mamba.ssd_chunk_state import SSDChunkStateFwdOp
from tileops.ops.mamba.ssd_recurrent import SSDRecurrentFwdOp
from tileops.ops.mamba.ssd_state_passing import SSDStatePassingFwdOp
from workloads.mamba import ssd_decode_result

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


@pytest.mark.parametrize("case", bench.cases(SSDChunkCouplingFwdOp), ids=lambda case: case.id)
def test_ssd_chunk_coupling_fwd_bench(case) -> None:
    """The CB stage on its own, over the shapes the Mamba-2 configs give it."""
    chunk_len = case.params["chunk_len"]
    op = SSDChunkCouplingFwdOp(**case.arguments)
    from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd

    def mamba_fn(c, b):
        return _bmm_chunk_fwd(c, b, chunk_len, causal=True, output_dtype=c.dtype).tril()

    bench.Runner(op, case).compare({"mamba": mamba_fn, "torch": case.reference})


@pytest.mark.parametrize("case", bench.cases(SSDChunkCumsumFwdOp), ids=lambda case: case.id)
def test_ssd_chunk_cumsum_fwd_bench(case) -> None:
    dt, A, dt_bias = case.inputs
    op = SSDChunkCumsumFwdOp(**case.arguments)
    functors = {}

    # _chunk_cumsum_fwd returns (dA_cumsum, dt_out) with a float32 dt_out; the baseline
    # returns them in TileOPs' order and dtype.
    if _mamba_chunk_cumsum_fwd is not None:

        def mamba_fwd():
            dA_cumsum, dt_out = _mamba_chunk_cumsum_fwd(
                dt, A, op.chunk_len, dt_bias=dt_bias, dt_softplus=op.dt_softplus
            )
            return dt_out.to(op.out_dtype), dA_cumsum

        functors["mamba"] = bench.Implementation(run=mamba_fwd, args=())

    _torch_baselines(functors, case.reference)
    bench.Runner(op, case).compare(functors)


@pytest.mark.parametrize("case", bench.cases(SSDChunkScanFwdOp), ids=lambda case: case.id)
def test_ssd_chunk_scan_fwd_bench(case) -> None:
    x, cb, dA_cumsum, C, prev_states, dt = case.inputs
    op = SSDChunkScanFwdOp(**case.arguments)
    functors = {}

    if _mamba_chunk_scan_fwd is not None:
        # mamba signature: _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, ...)
        def mamba_fwd():
            # Match the op's FP32 output; the cast is part of this baseline's timing.
            out, _ = _mamba_chunk_scan_fwd(cb, x, dt, dA_cumsum, C, prev_states)
            return out.float()

        functors["mamba"] = bench.Implementation(run=mamba_fwd, args=())

    _torch_baselines(functors, case.reference)
    bench.Runner(op, case).compare(functors)


@pytest.mark.parametrize("case", bench.cases(SSDChunkStateFwdOp), ids=lambda case: case.id)
def test_ssd_chunk_state_fwd_bench(case) -> None:
    x, Bmat, dt, dA_cumsum, seq_idx = case.inputs
    op = SSDChunkStateFwdOp(**case.arguments)
    functors = {}

    if _mamba_chunk_state_fwd is not None:
        # dt and dA_cumsum share TileOPs' (b, h, c, L) layout.
        def mamba_fwd():
            return _mamba_chunk_state_fwd(Bmat, x, dt, dA_cumsum, seq_idx=seq_idx)

        functors["mamba"] = bench.Implementation(run=mamba_fwd, args=())

    _torch_baselines(functors, case.reference)
    bench.Runner(op, case).compare(functors)


@pytest.mark.parametrize("case", bench.cases(SSDStatePassingFwdOp), ids=lambda case: case.id)
def test_ssd_state_passing_fwd_bench(case) -> None:
    states, dA_chunk_cumsum, initial_states = case.inputs
    op = SSDStatePassingFwdOp(**case.arguments)
    functors = {}

    if _mamba_state_passing_fwd is not None:
        # dA_chunk_cumsum shares TileOPs' (b, h, c) layout; float32 output as TileOPs.
        def mamba_fwd():
            return _mamba_state_passing_fwd(
                states, dA_chunk_cumsum, initial_states=initial_states, out_dtype=torch.float32
            )

        functors["mamba"] = bench.Implementation(run=mamba_fwd, args=())

    _torch_baselines(functors, case.reference)
    bench.Runner(op, case).compare(functors)


@pytest.mark.parametrize("case", bench.cases(SSDRecurrentFwdOp), ids=lambda case: case.id)
def test_ssd_decode_bench(case) -> None:
    x = case.inputs[2]
    op = SSDRecurrentFwdOp(**case.arguments)

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
    functors = {}
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

        functors["mamba"] = bench.Implementation(run=_mamba_wrapper, args=reference_args)

    def _torch_wrapper(x, dt, A, B, C):
        return reference(x, dt, A, B, C, dt_bias, initial_states)

    functors["torch-ref"] = bench.Implementation(run=_torch_wrapper, args=reference_args)
    functors[TORCH_COMPILE_TAG] = bench.Implementation(
        run=compiled_reference(_torch_wrapper), args=reference_args
    )

    bench.Runner(op, case).compare(functors)
