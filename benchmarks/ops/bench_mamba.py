
import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import ManifestBenchmark
from benchmarks.ops.attention.manifest_params import manifest_params
from tileops.manifest import load_workloads
from tileops.ops.da_cumsum import DaCumsumFwdOp
from tileops.ops.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops.ops.ssd_chunk_state import SSDChunkStateFwdOp
from tileops.ops.ssd_decode import SSDDecodeFwdOp
from tileops.ops.ssd_state_passing import SSDStatePassingFwdOp
from workloads.mamba import (
    DaCumsumFwdWorkload,
    SSDChunkScanFwdWorkload,
    SSDChunkStateFwdWorkload,
    SSDDecodeWorkload,
    SSDStatePassingFwdWorkload,
    ssd_state_passing_fwd_ref,
)

_DA_CUMSUM_OP_NAME = "DaCumsumFwdOp"
_CHUNK_STATE_OP_NAME = "SSDChunkStateFwdOp"
_CHUNK_SCAN_OP_NAME = "SSDChunkScanFwdOp"
_STATE_PASSING_OP_NAME = "SSDStatePassingFwdOp"
_DECODE_OP_NAME = "SSDDecodeFwdOp"


def _da_cumsum_args(w: dict) -> tuple:
    """Constructor arguments for one manifest workload row."""
    batch, seq_len, n_heads = w["dt_shape"]
    chunk_len = w["chunk_len"]
    return (batch, seq_len // chunk_len, chunk_len, n_heads,
            "dt_bias_shape" in w, bool(w.get("dt_softplus", True)))


def _chunk_state_args(w: dict) -> tuple:
    batch, _, n_heads, d_head = w["x_shape"]
    _, _, n_groups, d_state = w["Bmat_shape"]
    return (batch, w["dt_shape"][2], w["dt_shape"][3], n_heads, d_head, d_state,
            n_groups, "seq_idx_shape" in w)


def _chunk_scan_args(w: dict) -> tuple:
    batch, _, n_heads, d_head = w["x_shape"]
    _, num_chunks, n_groups, chunk_len, _ = w["cb_shape"]
    return batch, num_chunks, chunk_len, n_heads, d_head, w["C_shape"][3], n_groups


def _state_passing_args(w: dict) -> tuple:
    batch, num_chunks, n_heads, d_state = w["states_shape"]
    return batch, num_chunks, n_heads, d_state, "initial_states_shape" in w


def _decode_args(w: dict) -> tuple:
    n_heads, d_head, d_state = w["A_shape"]
    return w["x_shape"][0], n_heads, d_head, d_state, w["B_in_shape"][1]


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


def da_cumsum_fwd_ref(
    dt: torch.Tensor,
    A: torch.Tensor,
    num_chunks: int,
    chunk_len: int,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    dt_min: float = 0.0,
    dt_max: float = float("inf"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for da_cumsum_fwd (benchmark-local copy).

    Returns:
        dt_out:    (batch, n_heads, num_chunks, chunk_len) float32
        dA_cumsum: (batch, n_heads, num_chunks, chunk_len) float32
    """
    b, S, h = dt.shape
    Q = chunk_len
    C = num_chunks
    dt_val = dt.float()
    if dt_bias is not None:
        dt_val = dt_val + dt_bias.float()
    if dt_softplus:
        dt_val = F.softplus(dt_val)
    dt_val = torch.clamp(dt_val, min=dt_min, max=dt_max)
    dt_chunked = dt_val.reshape(b, C, Q, h)
    dt_out = dt_chunked.permute(0, 3, 1, 2).contiguous()          # (b, h, C, Q)
    dA = dt_chunked * A.float()
    dA_cumsum = dA.cumsum(dim=2).permute(0, 3, 1, 2).contiguous() # (b, h, C, Q)
    return dt_out, dA_cumsum


@pytest.mark.parametrize(
    "batch, num_chunks, chunk_len, n_heads, has_dt_bias, dt_softplus, dtype, tune",
    manifest_params(load_workloads(_DA_CUMSUM_OP_NAME), _da_cumsum_args, tune=False),
)
def test_da_cumsum_fwd_bench(batch, num_chunks, chunk_len, n_heads, has_dt_bias, dt_softplus, dtype, tune):
    test = DaCumsumFwdWorkload(
        batch, num_chunks, chunk_len, n_heads,
        has_dt_bias=has_dt_bias, dt_softplus=dt_softplus, dtype=dtype,
    )
    inputs = test.gen_inputs()  # (dt_raw, A, dt_bias)

    op = DaCumsumFwdOp(
        chunk_len=chunk_len,
        dt_softplus=dt_softplus,
        dtype=dtype,
        tune=tune,
    )
    bm = ManifestBenchmark(_DA_CUMSUM_OP_NAME, op, test)
    functors = {"tileops": op}

    # ── Mamba-2 Triton baseline ──
    # _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=...)
    # returns (dA_cumsum, dt_out) — note reversed order vs TileOPs (dt_out, dA_cumsum)
    if _mamba_chunk_cumsum_fwd is not None:
        mamba_dt_bias = inputs[2] if has_dt_bias else None

        def mamba_fwd():
            return _mamba_chunk_cumsum_fwd(
                inputs[0].contiguous(),
                inputs[1].contiguous(),
                chunk_len,
                dt_bias=mamba_dt_bias.contiguous() if mamba_dt_bias is not None else None,
                dt_softplus=dt_softplus,
            )

        functors["mamba"] = (mamba_fwd, ())
    else:
        def baseline(dt_raw, A, dt_bias):
            return da_cumsum_fwd_ref(
                dt_raw, A, num_chunks, chunk_len,
                dt_bias=dt_bias if has_dt_bias else None,
                dt_softplus=dt_softplus,
            )
        functors["torch-ref"] = baseline

    bm.compare(functors, *inputs, record_as=op, params=locals())




def ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt, n_groups):
    """PyTorch reference for ssd_chunk_scan_fwd (benchmark-local copy).

    Inputs (official layouts):
      x:           [B, S, H, P]        dtype
      cb:          [B, C, G, L, L]     dtype    group-owned
      dA_cumsum:   [B, H, C, L]        float32
      C:           [B, S, G, N]        dtype    group-owned
      prev_states: [B, C, H, P, N]     float32  P before N
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
@pytest.mark.parametrize(
    "batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune",
    manifest_params(load_workloads(_CHUNK_SCAN_OP_NAME), _chunk_scan_args, tune=False),
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
    test = SSDChunkScanFwdWorkload(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
    )
    inputs = test.gen_inputs()  # x, cb, dA_cumsum, C, prev_states, dt

    # ── TileOPs kernel ──
    op = SSDChunkScanFwdOp(tune=tune)
    bm = ManifestBenchmark(_CHUNK_SCAN_OP_NAME, op, test)
    functors = {"tileops": op}

    # ── Mamba-2 Triton baseline ──
    if _mamba_chunk_scan_fwd is not None:
        x, cb, dA_cumsum, C, prev_states, dt = inputs
        # All tensors are already in official mamba_ssm layout
        # mamba signature: _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, ...)
        def mamba_fwd():
            return _mamba_chunk_scan_fwd(cb, x, dt, dA_cumsum, C, prev_states)

        functors["mamba"] = (mamba_fwd, ())
    else:
        def torch_ref(x, cb, dA_cumsum, C, prev_states, dt):
            return ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt, n_groups)

        functors["torch-ref"] = torch_ref

    bm.compare(functors, *inputs, record_as=op, params=locals())


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
        same = ((seq_end >= 0) & (seq_chunked == seq_end)).unsqueeze(3)
        weight = weight * same.permute(0, 1, 3, 2)

    w = weight.permute(0, 1, 3, 2).unsqueeze(-1).unsqueeze(-1)
    contrib = w * B_heads.unsqueeze(-1) * x_chunked.unsqueeze(-2)
    out = contrib.sum(dim=2)
    return out.permute(0, 1, 2, 4, 3)


@pytest.mark.parametrize(
    "batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, has_seq_idx,"
    " dtype, tune",
    manifest_params(load_workloads(_CHUNK_STATE_OP_NAME), _chunk_state_args, tune=False),
)
def test_ssd_chunk_state_fwd_bench(
    batch: int, num_chunks: int, chunk_len: int, n_heads: int, d_head: int,
    d_state: int, n_groups: int, has_seq_idx: bool, dtype: torch.dtype, tune: bool,
) -> None:
    test = SSDChunkStateFwdWorkload(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, has_seq_idx,
    )
    inputs = test.gen_inputs()

    op = SSDChunkStateFwdOp(tune=tune)
    bm = ManifestBenchmark(_CHUNK_STATE_OP_NAME, op, test)
    functors = {"tileops": op}

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

        functors["mamba"] = (mamba_fwd, ())
    else:
        def baseline(x, Bmat, dt, dA_cumsum, seq_idx):
            return ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, n_groups=n_groups, seq_idx=seq_idx)
        functors["torch-ref"] = baseline

    bm.compare(functors, *inputs, record_as=op, params=locals())


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
@pytest.mark.parametrize(
    "batch, num_chunks, n_heads, d_state, has_initial_states, dtype, tune",
    manifest_params(load_workloads(_STATE_PASSING_OP_NAME), _state_passing_args, tune=False),
)
def test_ssd_state_passing_fwd_bench(
    batch: int, num_chunks: int, n_heads: int, d_state: int,
    has_initial_states: bool, dtype: torch.dtype, tune: bool,
) -> None:
    test = SSDStatePassingFwdWorkload(
        batch, num_chunks, n_heads, d_state, dtype, has_initial_states
    )
    inputs = test.gen_inputs()
    states, dA_chunk_cumsum, initial_states = inputs

    op = SSDStatePassingFwdOp(tune=tune)
    bm = ManifestBenchmark(_STATE_PASSING_OP_NAME, op, test)
    functors = {"tileops": op}

    if _mamba_state_passing_fwd is not None:
        def mamba_fwd():
            # mamba_ssm _state_passing_fwd: dA_chunk_cumsum layout (b, h, c) matches
            # TileOPs — no permutation needed.  out_dtype=float32 matches TileOPs output
            # dtype for an apples-to-apples comparison.
            return _mamba_state_passing_fwd(
                states.contiguous(),
                dA_chunk_cumsum.contiguous(),
                initial_states=(
                    None if initial_states is None else initial_states.contiguous()
                ),
                out_dtype=torch.float32,
            )

        # Pre-warm: run once outside bm.profile so the Triton autotuner
        # selects its best config before the CUPTI window opens.
        mamba_fwd()
        torch.cuda.synchronize()

        functors["mamba"] = (mamba_fwd, ())
    else:
        def baseline(states, dA_chunk_cumsum, initial_states):
            return ssd_state_passing_fwd_ref(states, dA_chunk_cumsum, initial_states)
        functors["torch-ref"] = baseline

    bm.compare(functors, *inputs, record_as=op, params=locals())


def ssd_decode_ref(
    A: torch.Tensor,      # (H, P, N)     float32
    dt: torch.Tensor,     # (B, H, P)     float32
    x: torch.Tensor,      # (B, H, P)     any dtype
    B_in: torch.Tensor,   # (B, G, N)     any dtype
    C_in: torch.Tensor,   # (B, G, N)     any dtype
    state: torch.Tensor,  # (B, H, P, N)  float32  -- updated in-place
) -> torch.Tensor:
    """PyTorch reference for ssd_decode (benchmark-local copy)."""
    B, H, P = dt.shape
    G = B_in.shape[1]
    heads_per_group = H // G

    # dA[b, h, p, n] = exp(dt[b, h, p] * A[h, p, n])
    dA = torch.exp(dt.float()[:, :, :, None] * A.float()[None, :, :, :])

    head_idx = torch.arange(H, device=B_in.device) // heads_per_group
    B_heads = B_in.float()[:, head_idx, :]   # (B, H, N)
    C_heads = C_in.float()[:, head_idx, :]   # (B, H, N)

    # dBx[b, h, p, n] = dt[b, h, p] * x[b, h, p] * B[b, h, n]
    dBx = (
        dt.float()[:, :, :, None]
        * x.float()[:, :, :, None]
        * B_heads[:, :, None, :]
    )

    new_state = dA * state.float() + dBx
    state.copy_(new_state)

    y_out = torch.einsum("bhpn,bhn->bhp", state.float(), C_heads)
    return y_out


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
@pytest.mark.parametrize(
    "batch, n_heads, d_head, d_state, n_groups, dtype, tune",
    manifest_params(load_workloads(_DECODE_OP_NAME), _decode_args, tune=False),
)
def test_ssd_decode_bench(
    batch: int, n_heads: int, d_head: int, d_state: int,
    n_groups: int, dtype: torch.dtype, tune: bool,
) -> None:
    test = SSDDecodeWorkload(batch, n_heads, d_head, d_state, n_groups, dtype)
    A, dt, x, B_in, C_in, state = test.gen_inputs()

    # Clone state before each profile run so both start from identical initial
    # conditions (op mutates state in-place across iterations).
    state_for_op = state.clone()
    state_bl = state.clone()

    op = SSDDecodeFwdOp(tune=tune)
    bm = ManifestBenchmark(_DECODE_OP_NAME, op, test)
    functors = {"tileops": op}

    def baseline(A, dt, x, B_in, C_in, state):
        return ssd_decode_ref(A, dt, x, B_in, C_in, state)

    functors["torch-ref"] = (baseline, (A, dt, x, B_in, C_in, state_bl, ))
    bm.compare(functors, A, dt, x, B_in, C_in, state_for_op, record_as=op, params=locals())
