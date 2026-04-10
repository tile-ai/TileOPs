from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.ssd_state_passing_fwd import SsdStatePassingFwdOp
from workloads.ops.ssd_state_passing_fwd import SsdStatePassingFwdTest


def ssd_state_passing_fwd_ref(
    states: torch.Tensor,
    dA_chunk_cumsum: torch.Tensor,
    initial_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for ssd_state_passing_fwd (benchmark-local copy)."""
    b, c, h, d = states.shape
    out = []
    s = initial_states.float()

    for ci in range(c):
        scale = torch.exp(dA_chunk_cumsum[:, :, ci]).unsqueeze(-1)
        u = states[:, ci, :, :].float()
        s = scale * s + u
        out.append(s.clone())

    return torch.stack(out, dim=1), s

try:
    from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd
except ImportError:
    _state_passing_fwd = None


class SsdStatePassingFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        b, c, h, d = t.batch, t.num_chunks, t.n_heads, t.d_state
        # Per chunk: scale multiply + add for each (b, h, d) element
        # 2 FLOPs (mul + add) per element per chunk
        flops = b * c * h * d * 2
        return float(flops)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        b, c, h, d = t.batch, t.num_chunks, t.n_heads, t.d_state
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads (input dtype): states
        reads = b * c * h * d * elem
        # Reads (float32): dA_chunk_cumsum + initial_states
        reads += b * h * c * 4 + b * h * d * 4
        # Writes (float32): out + final_states
        writes = (b * c * h * d + b * h * d) * 4
        return float(reads + writes)


# State passing benchmark parameters.
#
# Model-to-shape mapping (Mamba-2 defaults):
#   n_heads = d_model / 32,  d_state = 128
#   num_chunks = seq_len // chunk_len  (chunk_len=256: 2k->8, 4k->16, 32k->128)
#
#   130M -> n_heads=24   370M -> n_heads=32   780M -> n_heads=48
#   1.3B -> n_heads=64   2.7B -> n_heads=80
#
# Schema: (batch, num_chunks, n_heads, d_state, dtype, tune)
_SSD_STATE_PASSING_FWD_BENCH_PARAMS = [
    # ── unit-scale ──
    pytest.param(1, 2, 4,  32, torch.float16,  False, id="b1-c2-h4-d32-fp16"),
    pytest.param(2, 4, 8,  64, torch.float16,  False, id="b2-c4-h8-d64-fp16"),
    pytest.param(1, 2, 4,  32, torch.bfloat16, False, id="b1-c2-h4-d32-bf16"),
    pytest.param(2, 4, 8,  64, torch.bfloat16, False, id="b2-c4-h8-d64-bf16"),
    # ── 130M (n_heads=24) ──
    pytest.param(1,  16, 24, 128, torch.float16, True, id="latency-130m-4k"),
    pytest.param(8,  16, 24, 128, torch.float16, True, id="serving-130m-4k"),
    pytest.param(4, 128, 24, 128, torch.float16, True, id="longctx-130m-32k"),
    # ── 370M (n_heads=32) ──
    pytest.param(1,  16, 32, 128, torch.float16, True, id="latency-370m-4k"),
    pytest.param(8,  16, 32, 128, torch.float16, True, id="serving-370m-4k"),
    pytest.param(4, 128, 32, 128, torch.float16, True, id="longctx-370m-32k"),
    pytest.param(32,  8, 32, 128, torch.float16, True, id="throughput-370m-2k"),
    # ── 780M (n_heads=48) ──
    pytest.param(1,  16, 48, 128, torch.float16, True, id="latency-780m-4k"),
    pytest.param(8,  16, 48, 128, torch.float16, True, id="serving-780m-4k"),
    pytest.param(4, 128, 48, 128, torch.float16, True, id="longctx-780m-32k"),
    # ── 1.3B (n_heads=64) ──
    pytest.param(1,  16, 64, 128, torch.float16, True, id="latency-1p3b-4k"),
    pytest.param(8,  16, 64, 128, torch.float16, True, id="serving-1p3b-4k"),
    pytest.param(2, 128, 64, 128, torch.float16, True, id="longctx-1p3b-32k"),
    # ── 2.7B (n_heads=80) ──
    pytest.param(1,  16, 80, 128, torch.float16, True, id="latency-2p7b-4k"),
    pytest.param(4,  16, 80, 128, torch.float16, True, id="serving-2p7b-4k"),
    pytest.param(2, 128, 80, 128, torch.float16, True, id="longctx-2p7b-32k"),
]


@pytest.mark.parametrize(
    "batch, num_chunks, n_heads, d_state, dtype, tune",
    _SSD_STATE_PASSING_FWD_BENCH_PARAMS,
)
def test_ssd_state_passing_fwd_bench(
    batch: int, num_chunks: int, n_heads: int, d_state: int, dtype: torch.dtype, tune: bool,
) -> None:
    test = SsdStatePassingFwdTest(batch, num_chunks, n_heads, d_state, dtype)
    bm = SsdStatePassingFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = SsdStatePassingFwdOp(batch, num_chunks, n_heads, d_state, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    if _state_passing_fwd is not None:
        states, dA_chunk_cumsum, initial_states = inputs

        def mamba_fwd():
            # mamba_ssm _state_passing_fwd expects (b, h, c) for dA_chunk_cumsum,
            # matching TileOPs layout — no permutation needed.
            return _state_passing_fwd(
                states.contiguous(),
                dA_chunk_cumsum.contiguous(),
                initial_states=initial_states.contiguous(),
            )

        result_mamba = bm.profile(mamba_fwd)
        BenchmarkReport.record(op, locals(), result_mamba, tag="mamba")
    else:
        def baseline(states, dA_chunk_cumsum, initial_states):
            return ssd_state_passing_fwd_ref(states, dA_chunk_cumsum, initial_states)
        result_bl = bm.profile(baseline, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
