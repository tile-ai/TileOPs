from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.da_cumsum import DaCumsumFwdOp
from tileops.ops.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops.ops.ssd_chunk_state import SSDChunkStateFwdOp
from tileops.ops.ssd_decode import SSDDecodeOp
from tileops.ops.ssd_state_passing import SSDStatePassingFwdOp
from workloads.mamba import (
    DaCumsumFwdFixture,
    DaCumsumFwdTest,
    SSDChunkScanFwdFixture,
    SSDChunkScanFwdTest,
    SSDChunkStateFwdFixture,
    SSDChunkStateFwdTest,
    SSDDecodeFixture,
    SSDDecodeTest,
    SSDStatePassingFwdFixture,
    SSDStatePassingFwdTest,
)

# ---------------------------------------------------------------------------
# Optional mamba_ssm Triton baselines
# ---------------------------------------------------------------------------
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


def da_cumsum_fwd_ref(
    dt: torch.Tensor,
    A: torch.Tensor,
    num_chunks: int,
    chunk_len: int,
) -> torch.Tensor:
    """PyTorch reference for da_cumsum_fwd (benchmark-local copy)."""
    b, S, h = dt.shape
    Q = chunk_len
    C = num_chunks
    dt_chunked = dt.float().reshape(b, C, Q, h)
    dA = dt_chunked * A.float()
    dA_cumsum = dA.cumsum(dim=2)
    return dA_cumsum.permute(0, 3, 1, 2).contiguous()


class DaCumsumFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        b, c, L, h = t.batch, t.num_chunks, t.chunk_len, t.n_heads
        # One multiply (dt * A) and one add per element for the inclusive scan
        # Total: 2 * b * c * L * h
        return float(2 * b * c * L * h)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        b, c, L, h = t.batch, t.num_chunks, t.chunk_len, t.n_heads
        # float32 throughout
        elem = 4
        # Reads: dt (b, c*L, h) + A (h,)
        reads = (b * c * L * h + h) * elem
        # Writes: dA_cumsum (b, h, c, L)
        writes = b * h * c * L * elem
        return float(reads + writes)


@DaCumsumFwdFixture
def test_da_cumsum_fwd_bench(batch, num_chunks, chunk_len, n_heads, tune):
    test = DaCumsumFwdTest(batch, num_chunks, chunk_len, n_heads)
    bm = DaCumsumFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = DaCumsumFwdOp(batch, num_chunks, chunk_len, n_heads, seq_len=num_chunks * chunk_len, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(dt, A):
        return da_cumsum_fwd_ref(dt, A, num_chunks, chunk_len)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record("da_cumsum_fwd", locals(), result_bl, tag="baseline")


def ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt, n_groups):
    """Official-aligned PyTorch reference for chunk scan.

    Inputs (official layouts):
      x:           [B, S, H, P]        dtype
      cb:          [B, C, G, L, L]     dtype    group-owned
      dA_cumsum:   [B, H, C, L]        float32
      C:           [B, S, G, N]        dtype    group-owned
      prev_states: [B, C, H, P, N]     dtype    P before N
      dt:          [B, H, C, L]        dtype

    Output: [B, S, H, P]  float32
    """
    b, S, h, p = x.shape
    _, _, c, L = dA_cumsum.shape
    n = C.shape[-1]
    g = n_groups
    heads_per_group = h // g

    x_chunked = x.float().reshape(b, c, L, h, p)             # [B, C, L, H, P]
    C_chunked = C.float().reshape(b, c, L, g, n)              # [B, C, L, G, N]
    # broadcast C from groups to heads: [B, C, L, H, N]
    C_heads = C_chunked[:, :, :, torch.arange(h, device=x.device) // heads_per_group, :]

    # dA_cumsum: [B, H, C, L] -> [B, C, L, H] for broadcast
    dA = dA_cumsum.float().permute(0, 2, 3, 1)  # [B, C, L, H]

    # --- History path: exp(dA_l) * C[l] @ prev_states[p, n] ---
    # prev_states: [B, C, H, P, N]
    # C_heads:     [B, C, L, H, N] -> einsum over n: [B, C, L, H, P]
    y_off = torch.einsum("bclhn,bchpn->bclhp", C_heads, prev_states.float())
    y_off = y_off * torch.exp(dA).unsqueeze(-1)  # scale by exp(dA_l)

    # --- Intra-chunk path: sum_{s<=l} cb[l,s] * exp(dA_l - dA_s) * dt[s] * x[s] ---
    # cb: [B, C, G, L, L]; broadcast to heads [B, C, H, L, L]
    cb_chunked = cb.float()  # [B, C, G, L, L]
    cb_heads = cb_chunked[:, :, torch.arange(h, device=x.device) // heads_per_group, :, :]

    # decay[b,c,h,l,s] = exp(dA_cumsum[l] - dA_cumsum[s])
    dA_l = dA_cumsum.float().unsqueeze(-1)  # [B, H, C, L, 1]
    dA_s = dA_cumsum.float().unsqueeze(-2)  # [B, H, C, 1, L]
    decay = torch.exp(dA_l - dA_s)         # [B, H, C, L, L]

    # causal mask
    mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
    decay = decay.masked_fill(~mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), 0.0)
    decay = decay.permute(0, 2, 1, 3, 4)   # [B, C, H, L, L]

    # dt: [B, H, C, L] -> [B, C, H, 1, L]
    dt_s = dt.float().permute(0, 2, 1, 3).unsqueeze(-2)  # [B, C, H, 1, L]

    # lcb[b,c,h,l,s] = cb[l,s] * decay[l,s] * dt[s]
    lcb = cb_heads * decay * dt_s  # [B, C, H, L, L]

    # y_diag[b,c,l,h,p] = sum_s lcb[b,c,h,l,s] * x[b,c,s,h,p]
    y_diag = torch.einsum("bchls,bcshp->bclhp", lcb, x_chunked)

    # combine and reshape to [B, S, H, P]
    out = (y_off + y_diag).reshape(b, S, h, p)
    return out


class SSDChunkScanFwdBenchmark(BenchmarkBase):
    """FLOPs and memory bandwidth for ssd_chunk_scan_fwd."""

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        b, c, L, h, p, n = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state,
        )
        # History path: C @ prev_states per token
        #   b * c * L * h matmuls of shape (1, n) x (n, p) -> 2*n*p FLOPs each
        history_flops = b * c * L * h * 2 * n * p
        # Intra-chunk path: lower-triangular lcb @ x
        #   b * c * h causal GEMMs of size (L, L) x (L, p) -> L*(L+1)/2 * 2*p FLOPs each
        diag_flops = b * c * h * (L * (L + 1) // 2) * 2 * p
        return float(history_flops + diag_flops)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        b, c, L, h, p, n, g = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state, t.n_groups,
        )
        S = c * L
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads (input dtype): x + cb + C + prev_states + dt
        reads = (
            b * S * h * p           # x          [B, S, H, P]
            + b * c * g * L * L     # cb         [B, C, G, L, L]
            + b * S * g * n         # C          [B, S, G, N]
            + b * c * h * p * n     # prev_states [B, C, H, P, N]
            + b * h * c * L         # dt         [B, H, C, L]
        ) * elem
        # Reads (float32): dA_cumsum [B, H, C, L]
        reads += b * h * c * L * 4
        # Writes (float32): out [B, S, H, P]
        writes = b * S * h * p * 4
        return float(reads + writes)


# ---------------------------------------------------------------------------
# Benchmark parameters
#
# Model-to-shape mapping (Mamba-2 defaults):
#   n_heads = d_model / 32,  head_dim = 64,  d_state = 128,  chunk_len = 256
#   num_chunks = seq_len // chunk_len  (chunk_len=256: 2k->8, 4k->16, 32k->128)
#   n_groups = 1 (Mamba-2 standard)
#
#   130M -> n_heads=24   370M -> n_heads=32   780M -> n_heads=48
#   1.3B -> n_heads=64   2.7B -> n_heads=80
#
# Schema: (batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune)
# ---------------------------------------------------------------------------
_SSD_CHUNK_SCAN_FWD_BENCH_PARAMS = [
    # ── unit-scale ──
    pytest.param(1, 2,  64, 4,  64,  32, 1, torch.float16,  False, id="b1-c2-L64-h4-p64-n32-fp16"),
    pytest.param(2, 4,  64, 8,  64,  64, 2, torch.float16,  False, id="b2-c4-L64-h8-p64-n64-fp16"),
    pytest.param(1, 2, 128, 4, 128,  32, 1, torch.bfloat16, False, id="b1-c2-L128-h4-p128-n32-bf16"),
    pytest.param(2, 2,  64, 4,  64,  32, 2, torch.bfloat16, False, id="b2-c2-L64-h4-p64-n32-bf16"),
    # ── 130M (n_heads=24) ──
    pytest.param(1,  16, 256, 24, 64, 128, 1, torch.float16, True, id="latency-130m-4k"),
    pytest.param(8,  16, 256, 24, 64, 128, 1, torch.float16, True, id="serving-130m-4k"),
    pytest.param(4, 128, 256, 24, 64, 128, 1, torch.float16, True, id="longctx-130m-32k"),
    # ── 370M (n_heads=32) ──
    pytest.param(1,  16, 256, 32, 64, 128, 1, torch.float16, True, id="latency-370m-4k"),
    pytest.param(8,  16, 256, 32, 64, 128, 1, torch.float16, True, id="serving-370m-4k"),
    pytest.param(4, 128, 256, 32, 64, 128, 1, torch.float16, True, id="longctx-370m-32k"),
    pytest.param(32,  8, 256, 32, 64, 128, 1, torch.float16, True, id="throughput-370m-2k"),
    # ── 780M (n_heads=48) ──
    pytest.param(1,  16, 256, 48, 64, 128, 1, torch.float16, True, id="latency-780m-4k"),
    pytest.param(8,  16, 256, 48, 64, 128, 1, torch.float16, True, id="serving-780m-4k"),
    pytest.param(4, 128, 256, 48, 64, 128, 1, torch.float16, True, id="longctx-780m-32k"),
    pytest.param(16,  8, 256, 48, 64, 128, 1, torch.float16, True, id="throughput-780m-2k"),
    # ── 1.3B (n_heads=64) ──
    pytest.param(1,  16, 256, 64, 64, 128, 1, torch.float16, True, id="latency-1p3b-4k"),
    pytest.param(8,  16, 256, 64, 64, 128, 1, torch.float16, True, id="serving-1p3b-4k"),
    pytest.param(2, 128, 256, 64, 64, 128, 1, torch.float16, True, id="longctx-1p3b-32k"),
    pytest.param(8,   8, 256, 64, 64, 128, 1, torch.float16, True, id="throughput-1p3b-2k"),
    # ── 2.7B (n_heads=80) ──
    pytest.param(1,  16, 256, 80, 64, 128, 1, torch.float16, True, id="latency-2p7b-4k"),
    pytest.param(4,  16, 256, 80, 64, 128, 1, torch.float16, True, id="serving-2p7b-4k"),
    pytest.param(2, 128, 256, 80, 64, 128, 1, torch.float16, True, id="longctx-2p7b-32k"),
    pytest.param(4,   8, 256, 80, 64, 128, 1, torch.float16, True, id="throughput-2p7b-2k"),
]


@pytest.mark.parametrize(
    "batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune",
    _SSD_CHUNK_SCAN_FWD_BENCH_PARAMS,
)
def test_ssd_chunk_scan_fwd_bench(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = SSDChunkScanFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
    )
    bm = SSDChunkScanFwdBenchmark(test)
    inputs = test.gen_inputs()  # x, cb, dA_cumsum, C, prev_states, dt

    # ── TileOPs kernel ──
    op = SSDChunkScanFwdOp(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # ── Mamba-2 Triton baseline ──
    if _mamba_chunk_scan_fwd is not None:
        x, cb, dA_cumsum, C, prev_states, dt = inputs
        # All tensors are already in official mamba_ssm layout
        # mamba signature: _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, ...)
        def mamba_fwd():
            return _mamba_chunk_scan_fwd(cb, x, dt, dA_cumsum, C, prev_states)

        result_mamba = bm.profile(mamba_fwd)
        BenchmarkReport.record(op, locals(), result_mamba, tag="mamba")
    else:
        def torch_ref(x, cb, dA_cumsum, C, prev_states, dt):
            return ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt, n_groups)

        result_bl = bm.profile(torch_ref, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


def ssd_chunk_state_fwd_ref(
    x: torch.Tensor,
    Bmat: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    n_groups: int,
    seq_idx=None,
) -> torch.Tensor:
    """PyTorch reference for ssd_chunk_state_fwd (benchmark-local copy)."""
    b, seq_len, h, p = x.shape
    _, _, c, Q = dt.shape
    n = Bmat.shape[-1]
    heads_per_group = h // n_groups

    x_chunked = x.float().reshape(b, c, Q, h, p)
    B_chunked = Bmat.float().reshape(b, c, Q, n_groups, n)
    B_heads = B_chunked[:, :, :, torch.arange(h) // heads_per_group, :]

    dA = dA_cumsum.float().permute(0, 2, 1, 3)
    dA_end = dA[:, :, :, -1:]
    decay = torch.exp(torch.clamp(dA_end - dA, max=0.0))

    dt_chunked = dt.float().permute(0, 2, 1, 3)
    weight = decay * dt_chunked

    if seq_idx is not None:
        seq_chunked = seq_idx.reshape(b, c, Q)
        seq_end = seq_chunked[..., -1:]
        same = (seq_chunked == seq_end).unsqueeze(3)
        weight = weight * same.permute(0, 1, 3, 2)

    w = weight.permute(0, 1, 3, 2).unsqueeze(-1).unsqueeze(-1)
    contrib = w * B_heads.unsqueeze(-1) * x_chunked.unsqueeze(-2)
    out = contrib.sum(dim=2)
    return out.permute(0, 1, 2, 4, 3)


class SSDChunkStateFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        b, c, Q, h, p, n = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state,
        )
        # For each (b, c, h) block we do a rank-1 outer-product accumulation
        # over Q positions: Q * (p + n) multiply-adds, giving Q * p * n * 2 FLOPs
        # (treating the outer product as n*p MACs per position).
        flops = b * c * h * Q * n * p * 2
        return float(flops)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        b, c, Q, h, p, n, g = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state, t.n_groups,
        )
        seq_len = c * Q
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads (input dtype): x + Bmat
        reads = (
            b * seq_len * h * p      # x
            + b * seq_len * g * n    # Bmat
        ) * elem
        # Reads (float32): dt + dA_cumsum
        reads += b * h * c * Q * 4 * 2
        # Writes (float32): out
        writes = b * c * h * n * p * 4
        return float(reads + writes)


_SSD_CHUNK_STATE_FWD_BENCH_PARAMS = [
    pytest.param(1, 2,  64, 4,  64,  32, 1, torch.float16,  False, False, id="b1-c2-L64-h4-p64-n32-g1-fp16"),
    pytest.param(2, 4,  64, 8,  64,  64, 2, torch.float16,  False, False, id="b2-c4-L64-h8-p64-n64-g2-fp16"),
    pytest.param(1, 2, 128, 4, 128,  32, 1, torch.bfloat16, False, False, id="b1-c2-L128-h4-p128-n32-g1-bf16"),
    pytest.param(2, 2,  64, 4,  64,  32, 2, torch.bfloat16, False, False, id="b2-c2-L64-h4-p64-n32-g2-bf16"),
    pytest.param(2, 4,  64, 8,  64,  64, 2, torch.float16,  False, True,  id="b2-c4-L64-h8-p64-n64-g2-seqidx-fp16"),
]


@pytest.mark.parametrize(
    "batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune, has_seq_idx",
    _SSD_CHUNK_STATE_FWD_BENCH_PARAMS,
)
def test_ssd_chunk_state_fwd_bench(
    batch: int, num_chunks: int, chunk_len: int, n_heads: int, d_head: int,
    d_state: int, n_groups: int, dtype: torch.dtype, tune: bool, has_seq_idx: bool,
) -> None:
    test = SSDChunkStateFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, has_seq_idx,
    )
    bm = SSDChunkStateFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = SSDChunkStateFwdOp(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
        has_seq_idx=has_seq_idx, tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    if _mamba_chunk_state_fwd is not None:
        x, Bmat, dt, dA_cumsum, seq_idx = inputs

        def mamba_fwd():
            # mamba_ssm _chunk_state_fwd expects (b, h, c, L) for dt/dA_cumsum,
            # matching TileOPs layout — no permutation needed.
            return _mamba_chunk_state_fwd(
                Bmat.contiguous(),
                x.contiguous(),
                dt.contiguous(),
                dA_cumsum.contiguous(),
                seq_idx=seq_idx,
            )

        result_mamba = bm.profile(mamba_fwd)
        BenchmarkReport.record(op, locals(), result_mamba, tag="mamba")
    else:
        def baseline(x, Bmat, dt, dA_cumsum, seq_idx):
            return ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, n_groups=n_groups, seq_idx=seq_idx)
        result_bl = bm.profile(baseline, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


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


class SSDStatePassingFwdBenchmark(BenchmarkBase):

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
    test = SSDStatePassingFwdTest(batch, num_chunks, n_heads, d_state, dtype)
    bm = SSDStatePassingFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = SSDStatePassingFwdOp(batch, num_chunks, n_heads, d_state, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    if _mamba_state_passing_fwd is not None:
        states, dA_chunk_cumsum, initial_states = inputs

        def mamba_fwd():
            # mamba_ssm _state_passing_fwd expects (b, h, c) for dA_chunk_cumsum,
            # matching TileOPs layout — no permutation needed.
            return _mamba_state_passing_fwd(
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


def ssd_decode_ref(
    A: torch.Tensor,      # (H,)          float32
    dt: torch.Tensor,     # (B, H)        float32
    x: torch.Tensor,      # (B, H, P)     any dtype
    B_in: torch.Tensor,   # (B, G, N)     any dtype
    C_in: torch.Tensor,   # (B, G, N)     any dtype
    state: torch.Tensor,  # (B, H, P, N)  float32  -- updated in-place
) -> torch.Tensor:
    """PyTorch reference for ssd_decode (benchmark-local copy)."""
    B, H = dt.shape
    G = B_in.shape[1]
    heads_per_group = H // G

    dA = torch.exp(dt.float() * A.float())

    head_idx = torch.arange(H, device=B_in.device) // heads_per_group
    B_heads = B_in.float()[:, head_idx, :]
    C_heads = C_in.float()[:, head_idx, :]

    dBx = (
        dt.float()[:, :, None, None]
        * x.float()[:, :, :, None]
        * B_heads[:, :, None, :]
    )

    new_state = dA[:, :, None, None] * state.float() + dBx
    state.copy_(new_state)

    y_out = torch.einsum("bhpn,bhn->bhp", state.float(), C_heads)
    return y_out


class SSDDecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        b, h, p, n = t.batch, t.n_heads, t.d_head, t.d_state
        # State update: dA * old_s + dt * x * B  -> 3 muls + 1 add per (b,h,p,n)
        # Output accum: new_s * C                 -> 1 mul + 1 add per (b,h,p,n)
        # Total: 6 * b * h * p * n
        return float(6 * b * h * p * n)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        b, h, p, n, g = t.batch, t.n_heads, t.d_head, t.d_state, t.n_groups
        f32 = torch.float32.itemsize
        dtype_bytes = self.workload.dtype.itemsize
        # Reads: A(h) + dt(b,h) + x(b,h,p) + B_in(b,g,n) + C_in(b,g,n) + state(b,h,p,n)
        reads = (
            h * f32
            + b * h * f32
            + b * h * p * dtype_bytes
            + 2 * b * g * n * dtype_bytes
            + b * h * p * n * f32
        )
        # Writes: state(b,h,p,n) + y_out(b,h,p)
        writes = (b * h * p * n + b * h * p) * f32
        return float(reads + writes)


# Mamba2 (SSD) decode benchmark parameters.
#
# Model-to-shape mapping (Mamba2 defaults):
#   n_heads = d_model * expand / headdim = d_model * 2 / 64
#   headdim = 64,  d_state = 128,  n_groups = 1 (official default: all heads share B/C)
#
#   130M (d_model=768)  -> n_heads=24   370M (d_model=1024) -> n_heads=32
#   780M (d_model=1536) -> n_heads=48   1.3B (d_model=2048) -> n_heads=64
#   2.7B (d_model=2560) -> n_heads=80
#
# Schema: (batch, n_heads, d_head, d_state, n_groups, dtype, tune)
_SSD_DECODE_BENCH_PARAMS = [
    # ── smoke / unit-scale ──
    pytest.param(1,  4,  64,  16, 1, torch.float16,  False, id="b1-h4-p64-n16-g1-fp16"),
    pytest.param(2,  8,  64,  32, 2, torch.float16,  False, id="b2-h8-p64-n32-g2-fp16"),
    pytest.param(1,  4,  64,  16, 1, torch.bfloat16, False, id="b1-h4-p64-n16-g1-bf16"),
    pytest.param(2,  8, 128,  64, 4, torch.bfloat16, False, id="b2-h8-p128-n64-g4-bf16"),
    # ── 130M (n_heads=24) ──
    pytest.param(1,  24, 64, 128, 1, torch.float16, True, id="latency-130m"),
    pytest.param(8,  24, 64, 128, 1, torch.float16, True, id="serving-130m"),
    pytest.param(64, 24, 64, 128, 1, torch.float16, True, id="throughput-130m"),
    # ── 370M (n_heads=32) ──
    pytest.param(1,  32, 64, 128, 1, torch.float16, True, id="latency-370m"),
    pytest.param(8,  32, 64, 128, 1, torch.float16, True, id="serving-370m"),
    pytest.param(64, 32, 64, 128, 1, torch.float16, True, id="throughput-370m"),
    # ── 780M (n_heads=48) ──
    pytest.param(1,  48, 64, 128, 1, torch.float16, True, id="latency-780m"),
    pytest.param(8,  48, 64, 128, 1, torch.float16, True, id="serving-780m"),
    pytest.param(32, 48, 64, 128, 1, torch.float16, True, id="throughput-780m"),
    # ── 1.3B (n_heads=64) ──
    pytest.param(1,  64, 64, 128, 1, torch.float16, True, id="latency-1p3b"),
    pytest.param(8,  64, 64, 128, 1, torch.float16, True, id="serving-1p3b"),
    pytest.param(16, 64, 64, 128, 1, torch.float16, True, id="throughput-1p3b"),
    # ── 2.7B (n_heads=80) ──
    pytest.param(1,  80, 64, 128, 1, torch.float16, True, id="latency-2p7b"),
    pytest.param(4,  80, 64, 128, 1, torch.float16, True, id="serving-2p7b"),
    pytest.param(8,  80, 64, 128, 1, torch.float16, True, id="throughput-2p7b"),
]


@pytest.mark.parametrize(
    "batch, n_heads, d_head, d_state, n_groups, dtype, tune",
    _SSD_DECODE_BENCH_PARAMS,
)
def test_ssd_decode_bench(
    batch: int, n_heads: int, d_head: int, d_state: int,
    n_groups: int, dtype: torch.dtype, tune: bool,
) -> None:
    test = SSDDecodeTest(batch, n_heads, d_head, d_state, n_groups, dtype)
    bm = SSDDecodeBenchmark(test)
    A, dt, x, B_in, C_in, state = test.gen_inputs()

    # Clone state before each profile run so both start from identical initial
    # conditions (op mutates state in-place across iterations).
    state_for_op = state.clone()
    state_bl = state.clone()

    op = SSDDecodeOp(batch, n_heads, d_head, d_state, n_groups, dtype, tune=tune)
    result = bm.profile(op, A, dt, x, B_in, C_in, state_for_op)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(A, dt, x, B_in, C_in, state):
        return ssd_decode_ref(A, dt, x, B_in, C_in, state)

    result_bl = bm.profile(baseline, A, dt, x, B_in, C_in, state_bl)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")
