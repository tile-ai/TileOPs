"""Unit tests for the Mamba-2 / State-Space Dual (SSD) roofline helpers.

These exercise the (flops, bytes) accounting for the mamba family manifest
entries, which use ``roofline.func``. Each helper is driven through a
lightweight attribute stub (no CUDA build required), covering the
flag-dependent cases (``has_dt_bias`` / ``dt_softplus`` / ``has_seq_idx`` /
``has_initial_states``) and asserting exact FLOP and byte totals. The
composite ``mamba2_fwd_roofline`` FLOP total is locked to the sum of the
five standalone stage formulas.
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


def _da_cumsum_op(has_dt_bias: bool, dt_softplus: bool) -> SimpleNamespace:
    return SimpleNamespace(
        batch=B,
        seq_len=S,
        n_heads=H,
        has_dt_bias=has_dt_bias,
        dt_softplus=dt_softplus,
        dtype=torch.float16,
    )


@pytest.mark.parametrize(
    "has_dt_bias, dt_softplus",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_da_cumsum_fwd_roofline_flag_cases(has_dt_bias: bool, dt_softplus: bool):
    flops, nbytes = formulas.da_cumsum_fwd_roofline(
        _da_cumsum_op(has_dt_bias, dt_softplus))
    expected_flops = (3 + (1 if has_dt_bias else 0) +
                      (4 if dt_softplus else 0)) * TOKENS
    expected_nbytes = (
        TOKENS * 4                              # dt read (fp32)
        + H * 4                                 # A read
        + (H * 4 if has_dt_bias else 0)         # dt_bias read
        + TOKENS * 2                            # dt_out write (fp16)
        + TOKENS * 4                            # dA_cumsum write
    )
    assert flops == expected_flops
    assert nbytes == expected_nbytes


def test_cb_producer_roofline():
    op = SimpleNamespace(
        batch=B, num_chunks=NC, n_groups=G, chunk_len=Q, d_state=N,
        dtype=torch.float16)
    flops, nbytes = formulas.cb_producer_roofline(op)
    # Causal masking halves the 2*Q*Q*N GEMM work per (batch, chunk, group).
    assert flops == B * NC * G * Q * Q * N
    assert nbytes == (2 * B * S * G * N * 2 + B * NC * G * Q * Q * 2)


def _chunk_state_op(has_seq_idx: bool) -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, num_chunks=NC, chunk_len=Q, n_heads=H, d_head=P, d_state=N,
        n_groups=G, has_seq_idx=has_seq_idx, dtype=torch.float16)


@pytest.mark.parametrize("has_seq_idx", [False, True])
def test_ssd_chunk_state_fwd_roofline(has_seq_idx: bool):
    flops, nbytes = formulas.ssd_chunk_state_fwd_roofline(
        _chunk_state_op(has_seq_idx))
    assert flops == (2 * B * NC * H * P * N * Q + 4 * TOKENS + TOKENS * P)
    expected_nbytes = (
        TOKENS * P * 2                  # x
        + B * S * G * N * 2             # Bmat
        + TOKENS * 2                    # dt
        + TOKENS * 4                    # dA_cumsum
        + (B * S * 4 if has_seq_idx else 0)  # seq_idx
        + B * NC * H * P * N * 4        # states out
    )
    assert nbytes == expected_nbytes


def _state_passing_op(has_initial_states: bool, d_state: int) -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, num_chunks=NC, n_heads=H, d_state=d_state,
        has_initial_states=has_initial_states, dtype=torch.float32)


@pytest.mark.parametrize("has_initial_states", [False, True])
def test_ssd_state_passing_fwd_roofline(has_initial_states: bool):
    flops, nbytes = formulas.ssd_state_passing_fwd_roofline(
        _state_passing_op(has_initial_states, N))
    state_elems = B * NC * H * N
    # One multiply-add per state element; the exp(dA_chunk_cumsum) decay
    # scalar is shared across the state dim -> B*H*NC cardinality.
    assert flops == 2 * state_elems + B * H * NC
    expected_nbytes = (
        state_elems * 4                 # states read (fp32 workload)
        + B * H * NC * 4                # dA_chunk_cumsum
        + (B * H * N * 4 if has_initial_states else 0)  # initial_states
        + state_elems * 4               # out
        + B * H * N * 4                 # final_states
    )
    assert nbytes == expected_nbytes


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


def _mamba2_op(has_dt_bias: bool, has_initial_states: bool) -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, seqlen=S, num_chunks=NC, chunk_size=Q, n_heads=H, d_head=P,
        d_state=N, n_groups=G, dtype=torch.float16, dt_softplus=True,
        has_dt_bias=has_dt_bias, has_initial_states=has_initial_states)


@pytest.mark.parametrize(
    "has_dt_bias, has_initial_states",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_mamba2_fwd_roofline_flops_equal_stage_sum(has_dt_bias: bool,
                                                   has_initial_states: bool):
    """Composite FLOPs must equal the sum of the five standalone stages."""
    composite_flops, _ = formulas.mamba2_fwd_roofline(
        _mamba2_op(has_dt_bias, has_initial_states))

    stage_flops = 0
    stage_flops += formulas.da_cumsum_fwd_roofline(
        _da_cumsum_op(has_dt_bias, dt_softplus=True))[0]
    stage_flops += formulas.cb_producer_roofline(SimpleNamespace(
        batch=B, num_chunks=NC, n_groups=G, chunk_len=Q, d_state=N,
        dtype=torch.float16))[0]
    stage_flops += formulas.ssd_chunk_state_fwd_roofline(
        _chunk_state_op(has_seq_idx=False))[0]
    # State passing runs over the flattened d_head * d_state dimension.
    stage_flops += formulas.ssd_state_passing_fwd_roofline(
        _state_passing_op(has_initial_states, P * N))[0]
    stage_flops += formulas.ssd_chunk_scan_fwd_roofline(SimpleNamespace(
        batch=B, num_chunks=NC, chunk_len=Q, n_heads=H, d_head=P, d_state=N,
        n_groups=G, dtype=torch.float16))[0]

    assert composite_flops == stage_flops


@pytest.mark.parametrize(
    "has_dt_bias, has_initial_states",
    [(False, False), (True, True)],
)
def test_mamba2_fwd_roofline_nbytes(has_dt_bias: bool,
                                    has_initial_states: bool):
    _, nbytes = formulas.mamba2_fwd_roofline(
        _mamba2_op(has_dt_bias, has_initial_states))
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
