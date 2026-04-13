from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.da_cumsum import DaCumsumFwdOp
from tileops.ops.ssd_chunk_scan import SsdChunkScanFwdOp
from tileops.ops.ssd_chunk_state import SsdChunkStateFwdOp
from tileops.ops.ssd_decode import SsdDecodeOp
from tileops.ops.ssd_state_passing import SsdStatePassingFwdOp
from workloads.da_cumsum import DaCumsumFwdFixture, DaCumsumFwdTest
from workloads.ssd_chunk_scan import SsdChunkScanFwdFixture, SsdChunkScanFwdTest
from workloads.ssd_chunk_state import SsdChunkStateFwdFixture, SsdChunkStateFwdTest
from workloads.ssd_decode import SsdDecodeFixture, SsdDecodeTest
from workloads.ssd_state_passing import SsdStatePassingFwdFixture, SsdStatePassingFwdTest


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


def ssd_chunk_scan_fwd_torch(x, cb, dA_cumsum, C, prev_states, dt):
    """Triton-aligned PyTorch reference for chunk scan."""
    b, c, L, h, p = x.shape

    y_off = torch.einsum("bclhn,bchnp->bclhp", C.float(), prev_states.float())
    a_l = dA_cumsum.permute(0, 2, 3, 1).unsqueeze(-1)
    y_off = y_off * torch.exp(a_l)

    a_lhs = dA_cumsum.unsqueeze(-1)
    a_rhs = dA_cumsum.unsqueeze(-2)
    decay = torch.exp(a_lhs - a_rhs)
    mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
    decay = decay.masked_fill(~mask, 0)
    decay = decay.permute(0, 2, 1, 3, 4)
    dt_s = dt.float().permute(0, 1, 3, 2).unsqueeze(-2)
    lcb = cb.float() * decay * dt_s
    y_diag = torch.einsum("bchls,bcshp->bclhp", lcb, x.float())

    return y_off + y_diag


class SsdChunkScanFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
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
        t = self.workload
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


@SsdChunkScanFwdFixture
def test_ssd_chunk_scan_fwd_bench(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune):
    test = SsdChunkScanFwdTest(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype)
    bm = SsdChunkScanFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = SsdChunkScanFwdOp(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(*args):
        return ssd_chunk_scan_fwd_torch(*args)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record("ssd_chunk_scan_fwd", locals(), result_bl, tag="baseline")


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


class SsdChunkStateFwdBenchmark(BenchmarkBase):

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


@SsdChunkStateFwdFixture
def test_ssd_chunk_state_fwd_bench(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune, has_seq_idx,
):
    test = SsdChunkStateFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, has_seq_idx,
    )
    bm = SsdChunkStateFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = SsdChunkStateFwdOp(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
        has_seq_idx=has_seq_idx, tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(x, Bmat, dt, dA_cumsum, seq_idx):
        return ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, n_groups=n_groups, seq_idx=seq_idx)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record("ssd_chunk_state_fwd", locals(), result_bl, tag="baseline")


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


@SsdStatePassingFwdFixture
def test_ssd_state_passing_fwd_bench(batch, num_chunks, n_heads, d_state, dtype, tune):
    test = SsdStatePassingFwdTest(batch, num_chunks, n_heads, d_state, dtype)
    bm = SsdStatePassingFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = SsdStatePassingFwdOp(batch, num_chunks, n_heads, d_state, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(states, dA_chunk_cumsum, initial_states):
        return ssd_state_passing_fwd_ref(states, dA_chunk_cumsum, initial_states)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record("ssd_state_passing_fwd", locals(), result_bl, tag="baseline")


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


class SsdDecodeBenchmark(BenchmarkBase):

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


@SsdDecodeFixture
def test_ssd_decode_bench(batch, n_heads, d_head, d_state, n_groups, dtype, tune):
    test = SsdDecodeTest(batch, n_heads, d_head, d_state, n_groups, dtype)
    bm = SsdDecodeBenchmark(test)
    A, dt, x, B_in, C_in, state = test.gen_inputs()

    # Clone state before each profile run so both start from identical initial
    # conditions (op mutates state in-place across iterations).
    state_for_op = state.clone()
    state_bl = state.clone()

    op = SsdDecodeOp(batch, n_heads, d_head, d_state, n_groups, dtype, tune=tune)
    result = bm.profile(op, A, dt, x, B_in, C_in, state_for_op)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(A, dt, x, B_in, C_in, state):
        return ssd_decode_ref(A, dt, x, B_in, C_in, state)

    result_bl = bm.profile(baseline, A, dt, x, B_in, C_in, state_bl)
    BenchmarkReport.record("ssd_decode", locals(), result_bl, tag="baseline")
