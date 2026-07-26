"""Composite-vs-stage contract for the Mamba-2 / State-Space Dual (SSD) rooflines.

The composite ``mamba2_*_roofline`` helpers re-inline their stage cost
terms instead of calling the standalone stage helpers, so their FLOP
totals are locked here against the sum of the five stages through the
independent code path.
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


def _chunk_state_op() -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, num_chunks=NC, chunk_len=Q, n_heads=H, d_head=P, d_state=N,
        n_groups=G, dtype=torch.float16)


def _state_passing_op(d_state: int) -> SimpleNamespace:
    return SimpleNamespace(
        batch=B, num_chunks=NC, n_heads=H, d_state=d_state,
        dtype=torch.float32)


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
