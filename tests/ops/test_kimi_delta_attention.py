"""KimiDeltaAttentionFwdOp against FLA's per-token KDA recurrence."""

import pytest
import torch
import torch.nn.functional as F

from tileops.kernels.linear_attention import (
    KimiDeltaAttentionCall,
    KimiDeltaAttentionChunkPrefillFwdKernel,
    KimiDeltaAttentionFusedPrefillFwdKernel,
)
from tileops.ops import KimiDeltaAttentionFwdOp
from workloads.device import run_device

pytestmark = pytest.mark.smoke

TOLERANCE = {torch.float16: (2e-3, 2e-3), torch.bfloat16: (1.6e-2, 1.6e-2)}


def _inputs(batch, seq_len, heads, value_heads, dim, dtype, lengths=None, state=True):
    device = run_device()
    q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=dtype) * 0.1
    k = torch.randn_like(q)
    v = torch.randn(batch, seq_len, value_heads, dim, device=device, dtype=dtype) * 0.1
    g = F.logsigmoid(
        torch.rand(batch, seq_len, value_heads, dim, device=device, dtype=torch.float32)
    ).to(dtype)
    beta = torch.rand(batch, seq_len, value_heads, device=device, dtype=dtype) * 0.5
    cu_seqlens = None
    sequences = batch
    if lengths is not None:
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int64)
        sequences = len(lengths)
    initial_state = (
        torch.randn(sequences, value_heads, dim, dim, device=device, dtype=torch.float32) * 0.01
        if state
        else None
    )
    return q, k, v, g, beta, initial_state, cu_seqlens


def _reference(q, k, v, g, beta, initial_state, cu_seqlens):
    from fla.ops.kda import fused_recurrent_kda

    return fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )


def _check(dtype, *args):
    q, k, v, g, beta, initial_state, cu_seqlens = args
    op = KimiDeltaAttentionFwdOp(use_qk_l2norm_in_kernel=True)
    got_o, got_state = op(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        None if cu_seqlens is None else cu_seqlens.cpu(),
    )
    want_o, want_state = _reference(q, k, v, g, beta, initial_state, cu_seqlens)
    atol, rtol = TOLERANCE[dtype]
    torch.testing.assert_close(got_o, want_o, atol=atol, rtol=rtol)
    torch.testing.assert_close(got_state, want_state, atol=atol, rtol=rtol)


@pytest.mark.sm90
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_kimi_delta_attention_prefill_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    _check(dtype, *_inputs(1, 256, 4, 4, 128, dtype))


@pytest.mark.sm90
def test_kimi_delta_attention_prefill_packs_ragged_sequences() -> None:
    torch.manual_seed(42)
    _check(torch.bfloat16, *_inputs(1, 300, 4, 4, 128, torch.bfloat16, lengths=[100, 70, 130]))


@pytest.mark.sm90
def test_kimi_delta_attention_prefill_packs_a_batch_holding_an_empty_sequence() -> None:
    """A sequence with no token owns no chunk, and the ones around it keep their own.

    Such a sequence applies no update, so it ends on the state it started from.
    FLA leaves that slot at zero instead, so the reference is only asked about the
    sequences that carry a token.
    """
    torch.manual_seed(42)
    lengths = [100, 0, 70, 130]
    q, k, v, g, beta, state, cu_seqlens = _inputs(
        1, 300, 4, 4, 128, torch.bfloat16, lengths=lengths
    )
    op = KimiDeltaAttentionFwdOp(use_qk_l2norm_in_kernel=True)
    got_o, got_state = op(q, k, v, g, beta, state, cu_seqlens, cu_seqlens.cpu())
    want_o, want_state = _reference(q, k, v, g, beta, state, cu_seqlens)

    atol, rtol = TOLERANCE[torch.bfloat16]
    torch.testing.assert_close(got_o, want_o, atol=atol, rtol=rtol)
    for seq, length in enumerate(lengths):
        if length:
            torch.testing.assert_close(got_state[seq], want_state[seq], atol=atol, rtol=rtol)
        else:
            torch.testing.assert_close(got_state[seq], state[seq], atol=0.0, rtol=0.0)


@pytest.mark.sm90
def test_kimi_delta_attention_prefill_packs_sequences_that_fill_their_chunks() -> None:
    """The launch bound is loosest when every sequence ends on a chunk boundary.

    Four 64-token sequences need four chunks and the bound gives eight, so half the
    launch is the chunks past the last real one. They must leave the output alone.
    """
    torch.manual_seed(42)
    _check(torch.bfloat16, *_inputs(1, 256, 4, 4, 128, torch.bfloat16, lengths=[64] * 4))


@pytest.mark.sm90
def test_kimi_delta_attention_prefill_reads_offsets_a_freed_buffer_left_behind() -> None:
    """Boundaries come from the offsets this call was handed, not the last ones.

    The caching allocator hands a freed block to the next tensor of the same size,
    so a second call can carry offsets at the address and version the first one had.
    """
    torch.manual_seed(42)
    first = _inputs(1, 300, 4, 4, 128, torch.bfloat16, lengths=[100, 100, 100])
    _check(torch.bfloat16, *first)
    del first
    torch.cuda.empty_cache()
    _check(torch.bfloat16, *_inputs(1, 300, 4, 4, 128, torch.bfloat16, lengths=[50, 120, 130]))


@pytest.mark.sm90
def test_kimi_delta_attention_prefill_serves_more_value_heads_than_query_heads() -> None:
    torch.manual_seed(42)
    _check(torch.bfloat16, *_inputs(1, 256, 2, 4, 128, torch.bfloat16))


@pytest.mark.sm90
def test_kimi_delta_attention_prefill_serves_a_64_wide_state() -> None:
    torch.manual_seed(42)
    _check(torch.bfloat16, *_inputs(1, 256, 4, 4, 64, torch.bfloat16))


@pytest.mark.sm90
@pytest.mark.parametrize("batch", [1, 8], ids=["b1", "b8"])
def test_kimi_delta_attention_decode_matches_reference(batch: int) -> None:
    torch.manual_seed(42)
    _check(torch.bfloat16, *_inputs(batch, 1, 4, 4, 128, torch.bfloat16))


@pytest.mark.sm90
def test_kimi_delta_attention_decode_runs_from_a_zero_state() -> None:
    torch.manual_seed(42)
    _check(torch.bfloat16, *_inputs(4, 1, 4, 4, 128, torch.bfloat16, state=False))


@pytest.mark.sm90
def test_kimi_delta_attention_prefill_continues_across_calls() -> None:
    """The state a prefill ends on is the one the next call starts from."""
    torch.manual_seed(42)
    q, k, v, g, beta, state, _ = _inputs(1, 128, 4, 4, 128, torch.bfloat16)
    op = KimiDeltaAttentionFwdOp(use_qk_l2norm_in_kernel=True)
    first_o, carried = op(q[:, :64], k[:, :64], v[:, :64], g[:, :64], beta[:, :64], state)
    second_o, final = op(q[:, 64:], k[:, 64:], v[:, 64:], g[:, 64:], beta[:, 64:], carried)
    whole_o, whole_final = op(q, k, v, g, beta, state)
    torch.testing.assert_close(
        torch.cat([first_o, second_o], dim=1), whole_o, atol=1.6e-2, rtol=1.6e-2
    )
    torch.testing.assert_close(final, whole_final, atol=1.6e-2, rtol=1.6e-2)


def test_kimi_delta_attention_fused_prefill_takes_the_wide_launches() -> None:
    """The fused shape serves a call only when the launch covers the device."""
    wide = KimiDeltaAttentionCall(
        batch=1,
        seq_len=2048,
        sequences=8,
        heads=32,
        value_heads=32,
        dim_k=128,
        dim_v=128,
        dtype=torch.bfloat16,
        scale=128**-0.5,
        l2norm=True,
        varlen=True,
        sm_count=132,
        arch=90,
        calibration=None,
        smem_budget=0,
    )
    narrow = KimiDeltaAttentionCall(
        batch=1,
        seq_len=2048,
        sequences=1,
        heads=32,
        value_heads=32,
        dim_k=128,
        dim_v=128,
        dtype=torch.bfloat16,
        scale=128**-0.5,
        l2norm=True,
        varlen=True,
        sm_count=132,
        arch=90,
        calibration=None,
        smem_budget=0,
    )
    assert KimiDeltaAttentionFusedPrefillFwdKernel.applies(wide)
    assert not KimiDeltaAttentionFusedPrefillFwdKernel.applies(narrow)
    assert KimiDeltaAttentionChunkPrefillFwdKernel.applies(wide)
    assert KimiDeltaAttentionChunkPrefillFwdKernel.applies(narrow)


def test_kimi_delta_attention_refuses_the_variants_it_does_not_serve() -> None:
    base = dict(
        batch=1,
        seq_len=256,
        sequences=1,
        heads=4,
        value_heads=4,
        dim_k=128,
        dim_v=128,
        dtype=torch.bfloat16,
        scale=128**-0.5,
        sm_count=132,
        arch=90,
        calibration=None,
        smem_budget=0,
    )
    assert "state_v_first" in KimiDeltaAttentionCall(**base, state_v_first=True).chunk_refusal
    assert "use_gate_in_kernel" in KimiDeltaAttentionCall(**base, gate_in_kernel=True).chunk_refusal
    assert "lower_bound" in KimiDeltaAttentionCall(**base, bounded_gate=True).chunk_refusal
    assert KimiDeltaAttentionCall(**base).chunk_refusal is None


def test_kimi_delta_attention_refuses_a_value_width_its_key_buffers_do_not_hold() -> None:
    """K and V are separate dims in the contract, and this pair serves them equal.

    The chunk-local half stages the value tile in the buffers it sized for a key
    tile, so a wider V would run past them. The call is declined rather than served
    from the wrong rows.
    """
    base = dict(
        batch=1,
        seq_len=256,
        sequences=1,
        heads=4,
        value_heads=4,
        dtype=torch.bfloat16,
        scale=128**-0.5,
        sm_count=132,
        arch=90,
        calibration=None,
        smem_budget=0,
    )
    mixed = KimiDeltaAttentionCall(**base, dim_k=64, dim_v=128)
    assert "K equal to V" in mixed.chunk_refusal
    assert not KimiDeltaAttentionChunkPrefillFwdKernel.applies(mixed)
    assert not KimiDeltaAttentionFusedPrefillFwdKernel.applies(mixed)
    assert KimiDeltaAttentionCall(**base, dim_k=64, dim_v=64).chunk_refusal is None
    assert KimiDeltaAttentionCall(**base, dim_k=128, dim_v=128).chunk_refusal is None
