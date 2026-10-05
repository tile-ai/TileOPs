"""KDAFwdOp against the FP64 KDA recurrence in ``workloads/linear_attention/kda.py``."""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.kernels.linear_attention import (
    KDACall,
    KDAChunkPrefillFwdKernel,
    KDAFusedPrefillFwdKernel,
)
from tileops.ops import KDAFwdOp
from workloads.linear_attention.kda import KDAFwdWorkload
from workloads.numerics import compare_outputs

pytestmark = pytest.mark.smoke


class KDAFwdTest(KDAFwdWorkload, TestBase):
    pass


def _check(*args, **kwargs):
    test = KDAFwdTest(*args, **kwargs)
    test.check(KDAFwdOp(use_qk_l2norm_in_kernel=True), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_kda_prefill_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    _check(1, 256, 4, 128, dtype)


@pytest.mark.sm90
def test_kda_prefill_packs_ragged_sequences() -> None:
    torch.manual_seed(42)
    _check(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(100, 70, 130))


@pytest.mark.sm90
def test_kda_prefill_packs_a_batch_holding_an_empty_sequence() -> None:
    """A sequence with no token owns no chunk, and the ones around it keep their own.

    Such a sequence applies no update, so it ends on the state it started from.
    """
    torch.manual_seed(42)
    _check(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(100, 0, 70, 130))


@pytest.mark.sm90
def test_kda_prefill_packs_sequences_that_fill_their_chunks() -> None:
    """The launch bound is loosest when every sequence ends on a chunk boundary.

    Four 64-token sequences need four chunks and the bound gives eight, so half the
    launch is the chunks past the last real one. They must leave the output alone.
    """
    torch.manual_seed(42)
    _check(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(64,) * 4)


@pytest.mark.sm90
def test_kda_prefill_reads_offsets_a_freed_buffer_left_behind() -> None:
    """Boundaries come from the offsets this call was handed, not the last ones.

    The caching allocator hands a freed block to the next tensor of the same size,
    so a second call can carry offsets at the address and version the first one had.
    """
    torch.manual_seed(42)
    _check(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(100, 100, 100))
    torch.cuda.empty_cache()
    _check(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(50, 120, 130))


@pytest.mark.sm90
def test_kda_prefill_serves_more_value_heads_than_query_heads() -> None:
    torch.manual_seed(42)
    _check(1, 256, 2, 128, torch.bfloat16, value_heads=4)


@pytest.mark.sm90
def test_kda_prefill_serves_a_64_wide_state() -> None:
    torch.manual_seed(42)
    _check(1, 256, 4, 64, torch.bfloat16)


@pytest.mark.sm90
@pytest.mark.parametrize("batch", [1, 8], ids=["b1", "b8"])
def test_kda_decode_matches_reference(batch: int) -> None:
    torch.manual_seed(42)
    _check(batch, 1, 4, 128, torch.bfloat16)


@pytest.mark.sm90
def test_kda_decode_runs_from_a_zero_state() -> None:
    torch.manual_seed(42)
    _check(4, 1, 4, 128, torch.bfloat16, has_initial_state=False)


@pytest.mark.sm90
def test_kda_prefill_continues_across_calls() -> None:
    """The state a prefill ends on is the one the next call starts from."""
    torch.manual_seed(42)
    test = KDAFwdTest(1, 128, 4, 128, torch.bfloat16)
    inputs = test.gen_inputs()
    q, k, v, g, beta, state = inputs
    op = KDAFwdOp(use_qk_l2norm_in_kernel=True)
    first_o, carried = op(q[:, :64], k[:, :64], v[:, :64], g[:, :64], beta[:, :64], state)
    second_o, final = op(q[:, 64:], k[:, 64:], v[:, 64:], g[:, 64:], beta[:, 64:], carried)
    compare_outputs(
        (torch.cat([first_o, second_o], dim=1), final),
        test.ref_program(*inputs),
        test.verification(*inputs),
    )


def test_kda_refuses_the_variants_it_does_not_serve() -> None:
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
    assert "state_v_first" in KDACall(**base, state_v_first=True).chunk_refusal
    assert "use_gate_in_kernel" in KDACall(**base, gate_in_kernel=True).chunk_refusal
    assert "lower_bound" in KDACall(**base, bounded_gate=True).chunk_refusal
    assert KDACall(**base).chunk_refusal is None


def test_kda_refuses_a_value_width_its_key_buffers_do_not_hold() -> None:
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
    mixed = KDACall(**base, dim_k=64, dim_v=128)
    assert "K equal to V" in mixed.chunk_refusal
    assert not KDAChunkPrefillFwdKernel.applies(mixed)
    assert not KDAFusedPrefillFwdKernel.applies(mixed)
    assert KDACall(**base, dim_k=64, dim_v=64).chunk_refusal is None
    assert KDACall(**base, dim_k=128, dim_v=128).chunk_refusal is None
