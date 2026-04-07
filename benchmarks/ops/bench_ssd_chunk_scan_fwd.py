from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_ssd_chunk_scan_fwd import (
    SsdChunkScanFwdTest,
    ssd_chunk_scan_fwd_ref,
)
from tileops.ops.ssd_chunk_scan_fwd import SsdChunkScanFwdOp

try:
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd
except ImportError:
    _chunk_scan_fwd = None


def _to_mamba_inputs(x, cb, dA_cumsum, C, prev_states, dt):
    b, c, L, h, p = x.shape
    n = C.shape[-1]

    x_m = x.reshape(b, c * L, h, p).contiguous()
    cb_m = cb.contiguous()                          # (b, c, h, L, L) -- already correct
    dt_m = dA_cumsum.contiguous()                   # mamba's dt_m == dA_cumsum here
    dA_m = dA_cumsum.contiguous()
    C_m = C.reshape(b, c * L, h, n).contiguous()   # ngroups == nheads
    # TileOPs prev_states: (b, c, h, n, p) -> mamba states: (b, c, h, p, n)
    states_m = prev_states.permute(0, 1, 2, 4, 3).contiguous()

    return x_m, cb_m, dt_m, dA_m, C_m, states_m


class SsdChunkScanFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        b, c, L, h, p, n = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state,
        )
        # History path (Step 4): C @ prev_states per token
        #   b * c * L * h matmuls of shape (1, n) x (n, p) -> 2*n*p FLOPs each
        history_flops = b * c * L * h * 2 * n * p
        # Intra-chunk path (Step 1): lower-triangular lcb @ x
        #   b * c * h causal GEMMs of size (L, L) x (L, p) -> L*(L+1)/2 * 2*p FLOPs each
        diag_flops = b * c * h * (L * (L + 1) // 2) * 2 * p
        return float(history_flops + diag_flops)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        b, c, L, h, p, n = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state,
        )
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads (input dtype): x + cb + C + prev_states + dt
        reads = (
            b * c * L * h * p          # x
            + b * c * h * L * L        # cb
            + b * c * L * h * n        # C
            + b * c * h * n * p        # prev_states
            + b * c * L * h            # dt
        ) * elem
        # Reads (float32): dA_cumsum
        reads += b * h * c * L * 4
        # Writes (float32): out
        writes = b * c * L * h * p * 4
        return float(reads + writes)


# SSD chunk scan forward benchmark parameters.
#
# The forward kernel is the Mamba2 prefill primitive. Cases cover both
# small unit-scale configs and realistic Mamba2 model workloads.
#
# Model-to-shape mapping (Mamba2 defaults):
#   n_heads = d_model / 32,  head_dim = 64,  d_state = 128,  chunk_len = 256
#   num_chunks = seq_len // chunk_len  (chunk_len=256: 2k->8, 4k->16, 32k->128)
#
#   130M -> n_heads=24   370M -> n_heads=32   780M -> n_heads=48
#   1.3B -> n_heads=64   2.7B -> n_heads=80
#
# Schema: (batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune)
_SSD_CHUNK_SCAN_FWD_BENCH_PARAMS = [
    # ── unit-scale ──
    pytest.param(1, 2,  64, 4,  64,  32, torch.float16,  False, id="b1-c2-L64-h4-p64-n32-fp16"),
    pytest.param(2, 4,  64, 8,  64,  64, torch.float16,  False, id="b2-c4-L64-h8-p64-n64-fp16"),
    pytest.param(1, 2, 128, 4, 128,  32, torch.bfloat16, False, id="b1-c2-L128-h4-p128-n32-bf16"),
    pytest.param(2, 2,  64, 4,  64,  32, torch.bfloat16, False, id="b2-c2-L64-h4-p64-n32-bf16"),
    # ── 130M (n_heads=24) ──
    pytest.param(1,  16, 256, 24, 64, 128, torch.float16, True, id="latency-130m-4k"),
    pytest.param(8,  16, 256, 24, 64, 128, torch.float16, True, id="serving-130m-4k"),
    pytest.param(4, 128, 256, 24, 64, 128, torch.float16, True, id="longctx-130m-32k"),
    # ── 370M (n_heads=32) ──
    pytest.param(1,  16, 256, 32, 64, 128, torch.float16, True, id="latency-370m-4k"),
    pytest.param(8,  16, 256, 32, 64, 128, torch.float16, True, id="serving-370m-4k"),
    pytest.param(4, 128, 256, 32, 64, 128, torch.float16, True, id="longctx-370m-32k"),
    pytest.param(32,  8, 256, 32, 64, 128, torch.float16, True, id="throughput-370m-2k"),
    # ── 780M (n_heads=48) ──
    pytest.param(1,  16, 256, 48, 64, 128, torch.float16, True, id="latency-780m-4k"),
    pytest.param(8,  16, 256, 48, 64, 128, torch.float16, True, id="serving-780m-4k"),
    pytest.param(4, 128, 256, 48, 64, 128, torch.float16, True, id="longctx-780m-32k"),
    pytest.param(16,  8, 256, 48, 64, 128, torch.float16, True, id="throughput-780m-2k"),
    # ── 1.3B (n_heads=64) ──
    pytest.param(1,  16, 256, 64, 64, 128, torch.float16, True, id="latency-1p3b-4k"),
    pytest.param(8,  16, 256, 64, 64, 128, torch.float16, True, id="serving-1p3b-4k"),
    pytest.param(2, 128, 256, 64, 64, 128, torch.float16, True, id="longctx-1p3b-32k"),
    pytest.param(8,   8, 256, 64, 64, 128, torch.float16, True, id="throughput-1p3b-2k"),
    # ── 2.7B (n_heads=80) ──
    pytest.param(1,  16, 256, 80, 64, 128, torch.float16, True, id="latency-2p7b-4k"),
    pytest.param(4,  16, 256, 80, 64, 128, torch.float16, True, id="serving-2p7b-4k"),
    pytest.param(2, 128, 256, 80, 64, 128, torch.float16, True, id="longctx-2p7b-32k"),
    pytest.param(4,   8, 256, 80, 64, 128, torch.float16, True, id="throughput-2p7b-2k"),
]


@pytest.mark.parametrize(
    "batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune",
    _SSD_CHUNK_SCAN_FWD_BENCH_PARAMS,
)
def test_ssd_chunk_scan_fwd_bench(
    batch: int, num_chunks: int, chunk_len: int, n_heads: int,
    d_head: int, d_state: int, dtype: torch.dtype, tune: bool,
) -> None:
    test = SsdChunkScanFwdTest(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype)
    bm = SsdChunkScanFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = SsdChunkScanFwdOp(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    if _chunk_scan_fwd is not None:
        x, cb, dA_cumsum, C, prev_states, dt = inputs
        x_m, cb_m, dt_m, dA_m, C_m, states_m = _to_mamba_inputs(
            x, cb, dA_cumsum, C, prev_states, dt,
        )

        def mamba_fwd():
            return _chunk_scan_fwd(cb_m, x_m, dt_m, dA_m, C_m, states_m)

        result_mamba = bm.profile(mamba_fwd)
        BenchmarkReport.record(op, locals(), result_mamba, tag="mamba")
    else:
        def baseline(*args):
            return ssd_chunk_scan_fwd_ref(*args)
        result_bl = bm.profile(baseline, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
