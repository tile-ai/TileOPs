"""Composite-vs-stage contract for the Mamba-2 / State-Space Dual (SSD) rooflines.

``mamba2_fwd_roofline`` re-inlines its stage cost terms instead of calling the
standalone stage helpers, so its FLOP total is locked here against the sum of the
five stages through the independent code path, in every configuration of the two
optional inputs.
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


def _da_cumsum_op(dt_softplus: bool, has_dt_bias: bool) -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, seq_len=S, n_heads=H, dt_softplus=dt_softplus,
        dt_bias_shape=(H,) if has_dt_bias else None,
        dtype=torch.float16)


def _chunk_state_op() -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, num_chunks=NC, chunk_len=Q, n_heads=H, d_head=P, d_state=N,
        n_groups=G, dtype=torch.float16)


def _state_passing_op(d_state: int, has_initial_states: bool) -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, num_chunks=NC, n_heads=H, d_state=d_state,
        initial_states_shape=(B, H, d_state) if has_initial_states else None,
        dtype=torch.float32)


def _mamba2_op(has_dt_bias: bool, has_initial_states: bool) -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, seqlen=S, num_chunks=NC, chunk_size=Q, n_heads=H, d_head=P,
        d_state=N, n_groups=G, dtype=torch.float16, dt_softplus=True,
        dt_bias_shape=(H,) if has_dt_bias else None,
        initial_states_shape=(B, H, P, N) if has_initial_states else None)


# One public roofline function reads which optional inputs the call passed, so
# the four presence configurations exercise the same helper.
@pytest.mark.parametrize(("has_dt_bias", "has_initial_states"),
                         [(False, False), (True, False),
                          (False, True), (True, True)])
def test_mamba2_fwd_roofline_flops_equal_stage_sum(has_dt_bias: bool,
                                                   has_initial_states: bool):
    """Composite FLOPs must equal the sum of the five standalone stages."""
    composite_flops, _ = formulas.mamba2_fwd_roofline(
        _mamba2_op(has_dt_bias, has_initial_states))

    stage_flops = 0
    stage_flops += formulas.da_cumsum_fwd_roofline(
        _da_cumsum_op(dt_softplus=True, has_dt_bias=has_dt_bias))[0]
    stage_flops += formulas.cb_producer_roofline(SimpleNamespace(
        batch=B, num_chunks=NC, n_groups=G, chunk_len=Q, d_state=N,
        dtype=torch.float16))[0]
    stage_flops += formulas.ssd_chunk_state_fwd_roofline(_chunk_state_op())[0]
    # State passing runs over the flattened d_head * d_state dimension.
    stage_flops += formulas.ssd_state_passing_fwd_roofline(
        _state_passing_op(P * N, has_initial_states))[0]
    stage_flops += formulas.ssd_chunk_scan_fwd_roofline(SimpleNamespace(
        batch=B, num_chunks=NC, chunk_len=Q, n_heads=H, d_head=P, d_state=N,
        n_groups=G, dtype=torch.float16))[0]

    assert composite_flops == stage_flops
