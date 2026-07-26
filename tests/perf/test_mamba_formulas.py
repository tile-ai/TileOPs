"""Unit tests for the Mamba-2 / State-Space Dual (SSD) roofline helpers.

These exercise the (flops, bytes) accounting for the mamba family manifest
entries, which use ``roofline.func``. Each helper is driven through a
lightweight attribute stub (no CUDA build required). Conditional tensor
presence (dt_bias / seq_idx / initial_states) is hard-wired per variant
function, so every public variant helper is exercised explicitly and the
composite ``mamba2_*_roofline`` FLOP totals are locked to the sum of the
matching standalone stage helpers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tileops.perf import formulas

pytestmark = pytest.mark.smoke

# Small representative Mamba-2 geometry: S = NC * Q, G divides H.
B, NC, Q, H, P, N, G = 2, 4, 128, 4, 16, 32, 1
S = NC * Q
TOKENS = B * S * H


def _da_cumsum_op(dt_softplus: bool) -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, seq_len=S, n_heads=H, dt_softplus=dt_softplus,
        dtype=torch.float16)


def _da_cumsum_expected(has_dt_bias: bool, dt_softplus: bool) -> tuple[int, int]:
    flops = (3 + (1 if has_dt_bias else 0) + (4 if dt_softplus else 0)) * TOKENS
    nbytes = (
        TOKENS * 4                              # dt read (fp32)
        + H * 4                                 # A read
        + (H * 4 if has_dt_bias else 0)         # dt_bias read
        + TOKENS * 2                            # dt_out write (fp16)
        + TOKENS * 4                            # dA_cumsum write
    )
    return flops, nbytes


@pytest.mark.parametrize("dt_softplus", [False, True])
def test_da_cumsum_fwd_roofline(dt_softplus: bool):
    assert formulas.da_cumsum_fwd_roofline(
        _da_cumsum_op(dt_softplus)) == _da_cumsum_expected(False, dt_softplus)


@pytest.mark.parametrize("dt_softplus", [False, True])
def test_da_cumsum_bias_fwd_roofline(dt_softplus: bool):
    assert formulas.da_cumsum_bias_fwd_roofline(
        _da_cumsum_op(dt_softplus)) == _da_cumsum_expected(True, dt_softplus)


def test_cb_producer_roofline():
    op = SimpleNamespace(
        batch=B, num_chunks=NC, n_groups=G, chunk_len=Q, d_state=N,
        dtype=torch.float16)
    flops, nbytes = formulas.cb_producer_roofline(op)
    # Causal masking halves the 2*Q*Q*N GEMM work per (batch, chunk, group).
    assert flops == B * NC * G * Q * Q * N
    assert nbytes == (2 * B * S * G * N * 2 + B * NC * G * Q * Q * 2)


def _chunk_state_op() -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, num_chunks=NC, chunk_len=Q, n_heads=H, d_head=P, d_state=N,
        n_groups=G, dtype=torch.float16)


def _chunk_state_expected(has_seq_idx: bool) -> tuple[int, int]:
    flops = 2 * B * NC * H * P * N * Q + 4 * TOKENS + TOKENS * P
    nbytes = (
        TOKENS * P * 2                  # x
        + B * S * G * N * 2             # Bmat
        + TOKENS * 2                    # dt
        + TOKENS * 4                    # dA_cumsum
        + (B * S * 4 if has_seq_idx else 0)  # seq_idx
        + B * NC * H * P * N * 4        # states out
    )
    return flops, nbytes


def test_ssd_chunk_state_fwd_roofline():
    assert formulas.ssd_chunk_state_fwd_roofline(
        _chunk_state_op()) == _chunk_state_expected(False)


def test_ssd_chunk_state_seq_idx_fwd_roofline():
    assert formulas.ssd_chunk_state_seq_idx_fwd_roofline(
        _chunk_state_op()) == _chunk_state_expected(True)


def _state_passing_op(d_state: int) -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, num_chunks=NC, n_heads=H, d_state=d_state,
        dtype=torch.float32)


def _state_passing_expected(has_initial_states: bool,
                            d_state: int) -> tuple[int, int]:
    state_elems = B * NC * H * d_state
    # One multiply-add per state element; the exp(dA_chunk_cumsum) decay
    # scalar is shared across the state dim -> B*H*NC cardinality.
    flops = 2 * state_elems + B * H * NC
    nbytes = (
        state_elems * 4                 # states read (fp32 workload)
        + B * H * NC * 4                # dA_chunk_cumsum
        + (B * H * d_state * 4 if has_initial_states else 0)  # initial_states
        + state_elems * 4               # out
        + B * H * d_state * 4           # final_states
    )
    return flops, nbytes


def test_ssd_state_passing_fwd_roofline():
    assert formulas.ssd_state_passing_fwd_roofline(
        _state_passing_op(N)) == _state_passing_expected(False, N)


def test_ssd_state_passing_init_states_fwd_roofline():
    assert formulas.ssd_state_passing_init_states_fwd_roofline(
        _state_passing_op(N)) == _state_passing_expected(True, N)


def test_ssd_chunk_scan_fwd_roofline():
    op = SimpleNamespace(
        batch=B, num_chunks=NC, chunk_len=Q, n_heads=H, d_head=P, d_state=N,
        n_groups=G, dtype=torch.float16)
    flops, nbytes = formulas.ssd_chunk_scan_fwd_roofline(op)
    assert flops == (2 * TOKENS * N * P + B * NC * H * Q * Q * P)
    expected_nbytes = (
        TOKENS * P * 2                  # x
        + B * NC * G * Q * Q * 2        # cb
        + TOKENS * 4                    # dA_cumsum
        + B * S * G * N * 2             # C
        + B * NC * H * P * N * 4        # prev_states
        + TOKENS * 2                    # dt
        + TOKENS * P * 4                # y out
    )
    assert nbytes == expected_nbytes


def test_ssd_decode_roofline():
    op = SimpleNamespace(
        batch=B, n_heads=H, d_head=P, d_state=N, n_groups=G,
        dtype=torch.float16)
    flops, nbytes = formulas.ssd_decode_roofline(op)
    state_elems = B * H * P * N
    # dt*A, exp, two products for dt*x*B, decay multiply, state add, and
    # the output multiply-add: eight ops per state element.
    assert flops == 8 * state_elems
    expected_nbytes = (
        H * P * N * 4                   # A
        + B * H * P * 4                 # dt
        + B * H * P * 2                 # x
        + 2 * B * G * N * 2             # B_in, C_in
        + 2 * state_elems * 4           # state read + write
        + B * H * P * 4                 # y_out
    )
    assert nbytes == expected_nbytes


def _mamba2_op() -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, seqlen=S, num_chunks=NC, chunk_size=Q, n_heads=H, d_head=P,
        d_state=N, n_groups=G, dtype=torch.float16, dt_softplus=True)


# (composite helper, has_dt_bias, has_initial_states) — one public roofline
# function per Mamba2 manifest variant.
_MAMBA2_VARIANTS = [
    (formulas.mamba2_fwd_roofline, False, False),
    (formulas.mamba2_bias_fwd_roofline, True, False),
    (formulas.mamba2_init_states_fwd_roofline, False, True),
    (formulas.mamba2_bias_init_states_fwd_roofline, True, True),
]


@pytest.mark.parametrize(("helper", "has_dt_bias", "has_initial_states"),
                         _MAMBA2_VARIANTS)
def test_mamba2_fwd_roofline_flops_equal_stage_sum(helper, has_dt_bias: bool,
                                                   has_initial_states: bool):
    """Composite FLOPs must equal the sum of the five standalone stages."""
    composite_flops, _ = helper(_mamba2_op())

    da_cumsum = (formulas.da_cumsum_bias_fwd_roofline if has_dt_bias
                 else formulas.da_cumsum_fwd_roofline)
    state_passing = (formulas.ssd_state_passing_init_states_fwd_roofline
                     if has_initial_states
                     else formulas.ssd_state_passing_fwd_roofline)

    stage_flops = 0
    stage_flops += da_cumsum(_da_cumsum_op(dt_softplus=True))[0]
    stage_flops += formulas.cb_producer_roofline(SimpleNamespace(
        batch=B, num_chunks=NC, n_groups=G, chunk_len=Q, d_state=N,
        dtype=torch.float16))[0]
    stage_flops += formulas.ssd_chunk_state_fwd_roofline(_chunk_state_op())[0]
    # State passing runs over the flattened d_head * d_state dimension.
    stage_flops += state_passing(_state_passing_op(P * N))[0]
    stage_flops += formulas.ssd_chunk_scan_fwd_roofline(SimpleNamespace(
        batch=B, num_chunks=NC, chunk_len=Q, n_heads=H, d_head=P, d_state=N,
        n_groups=G, dtype=torch.float16))[0]

    assert composite_flops == stage_flops


@pytest.mark.parametrize(("helper", "has_dt_bias", "has_initial_states"),
                         _MAMBA2_VARIANTS)
def test_mamba2_fwd_roofline_nbytes(helper, has_dt_bias: bool,
                                    has_initial_states: bool):
    _, nbytes = helper(_mamba2_op())
    state_elems = B * NC * H * P * N
    expected = (
        TOKENS * P * 2                          # x
        + TOKENS * 4                            # dt
        + 2 * B * S * G * N * 2                 # B, C
        + H * 4                                 # A
        + (H * 4 if has_dt_bias else 0)         # dt_bias
        + (B * H * P * N * 4 if has_initial_states else 0)  # initial_states
        + B * NC * G * Q * Q * 2                # cb intermediate
        + 2 * state_elems * 4                   # chunk states read + write
        + TOKENS * 2                            # dt_out
        + TOKENS * 4                            # dA_cumsum
        + TOKENS * P * 4                        # y out
    )
    assert nbytes == expected
