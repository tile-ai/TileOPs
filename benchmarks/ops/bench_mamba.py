"""Mamba-2 SSD stage benchmarks, one case per manifest call, against the mamba_ssm Triton stages."""

import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.mamba.cb_producer import CBProducerFwdOp
from tileops.ops.mamba.da_cumsum import DaCumsumFwdOp
from tileops.ops.mamba.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops.ops.mamba.ssd_chunk_state import SSDChunkStateFwdOp
from tileops.ops.mamba.ssd_decode import SSDDecodeFwdOp
from tileops.ops.mamba.ssd_state_passing import SSDStatePassingFwdOp
from workloads.mamba import (
    CBProducerFwdCall,
    DaCumsumFwdCall,
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


def _torch_baselines(functors: dict, ref_program) -> None:
    functors["torch-ref"] = ref_program
    functors[TORCH_COMPILE_TAG] = compiled_reference(ref_program)


@pytest.mark.parametrize("call", manifest_calls(CBProducerFwdOp))
def test_cb_producer_fwd_bench(call) -> None:
    """The CB stage on its own, over the shapes the Mamba-2 configs give it."""
    test = CBProducerFwdCall(call)
    inputs = test.gen_inputs()
    op = CBProducerFwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)
    bm.compare({"tileops": op, "torch": (test.ref_program, inputs)}, *inputs)


@pytest.mark.parametrize("call", manifest_calls(DaCumsumFwdOp))
def test_da_cumsum_fwd_bench(call) -> None:
    test = DaCumsumFwdCall(call)
    dt, A, dt_bias = inputs = test.gen_inputs()
    op = DaCumsumFwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)
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

    _torch_baselines(functors, test.ref_program)
    bm.compare(functors, *inputs)


@pytest.mark.parametrize("call", manifest_calls(SSDChunkScanFwdOp))
def test_ssd_chunk_scan_fwd_bench(call) -> None:
    test = SSDChunkScanFwdCall(call)
    x, cb, dA_cumsum, C, prev_states, dt = inputs = test.gen_inputs()
    op = SSDChunkScanFwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)
    functors = {"tileops": op}

    if _mamba_chunk_scan_fwd is not None:
        # mamba signature: _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, ...)
        def mamba_fwd():
            return _mamba_chunk_scan_fwd(cb, x, dt, dA_cumsum, C, prev_states)

        functors["mamba"] = (mamba_fwd, ())

    _torch_baselines(functors, test.ref_program)
    bm.compare(functors, *inputs)


@pytest.mark.parametrize("call", manifest_calls(SSDChunkStateFwdOp))
def test_ssd_chunk_state_fwd_bench(call) -> None:
    test = SSDChunkStateFwdCall(call)
    x, Bmat, dt, dA_cumsum, seq_idx = inputs = test.gen_inputs()
    op = SSDChunkStateFwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)
    functors = {"tileops": op}

    if _mamba_chunk_state_fwd is not None:
        # dt and dA_cumsum share TileOPs' (b, h, c, L) layout.
        def mamba_fwd():
            return _mamba_chunk_state_fwd(Bmat, x, dt, dA_cumsum, seq_idx=seq_idx)

        functors["mamba"] = (mamba_fwd, ())

    _torch_baselines(functors, test.ref_program)
    bm.compare(functors, *inputs)


@pytest.mark.parametrize("call", manifest_calls(SSDStatePassingFwdOp))
def test_ssd_state_passing_fwd_bench(call) -> None:
    test = SSDStatePassingFwdCall(call)
    states, dA_chunk_cumsum, initial_states = inputs = test.gen_inputs()
    op = SSDStatePassingFwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)
    functors = {"tileops": op}

    if _mamba_state_passing_fwd is not None:
        # dA_chunk_cumsum shares TileOPs' (b, h, c) layout; float32 output as TileOPs.
        def mamba_fwd():
            return _mamba_state_passing_fwd(
                states, dA_chunk_cumsum, initial_states=initial_states, out_dtype=torch.float32
            )

        # Pre-warm: run once outside bm.profile so the Triton autotuner
        # selects its best config before the CUPTI window opens.
        mamba_fwd()
        torch.cuda.synchronize()

        functors["mamba"] = (mamba_fwd, ())

    _torch_baselines(functors, test.ref_program)
    bm.compare(functors, *inputs)


@pytest.mark.parametrize("call", manifest_calls(SSDDecodeFwdOp))
def test_ssd_decode_bench(call) -> None:
    test = SSDDecodeFwdCall(call)
    A, dt, x, B_in, C_in, state = test.gen_inputs()
    op = SSDDecodeFwdOp(**test.arguments())
    bm = ManifestBenchmark(op, test)

    # Each implementation updates its own copy of the state in place.
    functors = {
        "tileops": op,
        "torch-ref": (test.ref_program, (A, dt, x, B_in, C_in, state.clone())),
        TORCH_COMPILE_TAG: (
            compiled_reference(test.ref_program),
            (A, dt, x, B_in, C_in, state.clone()),
        ),
    }
    bm.compare(functors, A, dt, x, B_in, C_in, state.clone())
