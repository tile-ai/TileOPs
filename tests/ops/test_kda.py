"""KDAFwdOp against the FP64 KDA recurrence in ``workloads/linear_attention/kda.py``."""

import math

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.kernels.linear_attention import (
    KDACall,
)
from tileops.kernels.linear_attention.kda.prefill_fused import TWO_LAUNCH_TOKENS_PER_SM
from tileops.ops import KDAFwdOp
from tileops.utils.utils import get_sm_count
from workloads.linear_attention.kda import KDAFwdCall, KDAFwdWorkload
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call


class KDAFwdTest(KDAFwdWorkload, TestBase):
    pass


def _check(*args, **kwargs):
    test = KDAFwdTest(*args, **kwargs)
    test.check(KDAFwdOp(use_qk_l2norm_in_kernel=True), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_kda_prefill_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    _check(1, 256, 4, 128, dtype)


@pytest.mark.smoke
def test_kda_prefill_packs_ragged_sequences() -> None:
    torch.manual_seed(42)
    _check(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(100, 70, 130))


@pytest.mark.smoke
def test_kda_prefill_packs_a_batch_holding_an_empty_sequence() -> None:
    """A sequence with no token owns no chunk, and the ones around it keep their own.

    Such a sequence applies no update, so it ends on the state it started from.
    """
    torch.manual_seed(42)
    _check(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(100, 0, 70, 130))


@pytest.mark.smoke
def test_kda_prefill_packs_sequences_that_fill_their_chunks() -> None:
    """The launch bound is loosest when every sequence ends on a chunk boundary.

    Four 64-token sequences need four chunks and the bound gives eight, so half the
    launch is the chunks past the last real one. They must leave the output alone.
    """
    torch.manual_seed(42)
    _check(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(64,) * 4)


@pytest.mark.smoke
def test_kda_prefill_reads_offsets_a_freed_buffer_left_behind() -> None:
    """Boundaries come from the offsets this call was handed, not the last ones.

    The caching allocator hands a freed block to the next tensor of the same size,
    so a second call can carry offsets at the address and version the first one had.
    """
    torch.manual_seed(42)
    _check(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(100, 100, 100))
    torch.cuda.empty_cache()
    _check(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(50, 120, 130))


@pytest.mark.smoke
def test_kda_prefill_serves_more_value_heads_than_query_heads() -> None:
    torch.manual_seed(42)
    _check(1, 256, 2, 128, torch.bfloat16, value_heads=4)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "sequence_lengths",
    [pytest.param((100, 130), id="two"), pytest.param((100, 130, 70, 64), id="four")],
)
def test_kda_prefill_packs_sequences_of_16_heads(sequence_lengths) -> None:
    """Packed sequences of 16 value heads, most ending in a partial chunk."""
    torch.manual_seed(42)
    _check(1, 0, 16, 128, torch.bfloat16, sequence_lengths=sequence_lengths)


@pytest.mark.smoke
def test_kda_prefill_serves_a_64_wide_state() -> None:
    torch.manual_seed(42)
    _check(1, 256, 4, 64, torch.bfloat16)


@pytest.mark.smoke
@pytest.mark.parametrize("batch", [1, 8], ids=["b1", "b8"])
def test_kda_decode_matches_reference(batch: int) -> None:
    torch.manual_seed(42)
    _check(batch, 1, 4, 128, torch.bfloat16)


@pytest.mark.smoke
def test_kda_decode_runs_from_a_zero_state() -> None:
    torch.manual_seed(42)
    _check(4, 1, 4, 128, torch.bfloat16, has_initial_state=False)


@pytest.mark.smoke
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


@pytest.mark.smoke
@pytest.mark.sm89
def test_kda_prefill_in_99_kb_of_shared_memory() -> None:
    """SM89 gives a block 99 KB of opt-in shared memory, so a packed head dim 128 prefill
    runs on smaller programs than SM80 and SM90 build."""
    torch.manual_seed(7)
    _check(1, 0, 4, 128, torch.float16, sequence_lengths=(300, 1, 211))


@pytest.mark.smoke
@pytest.mark.sm89
def test_kda_prefill_scans_independent_value_tiles() -> None:
    """SM89 splits a long single-sequence scan with eight value heads into 16-column
    tiles, so every tile must read and write its own state and output columns.
    """
    torch.manual_seed(42)
    _check(1, 4096, 2, 128, torch.bfloat16, value_heads=8)


@pytest.mark.smoke
@pytest.mark.sm89
def test_kda_fused_prefill_reads_many_offsets_from_global_memory() -> None:
    """2048 sequences of four tokens: more boundaries than a block can stage."""
    torch.manual_seed(42)
    test = KDAFwdTest(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(4,) * 2048)
    test.check(KDAFwdOp(use_qk_l2norm_in_kernel=True), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("facts", "reason"),
    [
        pytest.param({"state_v_first": True}, "state_v_first", id="value-major-state"),
        # The chunk-local half stages a value tile in buffers sized for a key tile.
        pytest.param({"dim_k": 64, "dim_v": 128}, "K equal to V", id="value-wider-than-key"),
    ],
)
def test_kda_prefill_refuses_what_no_kernel_serves(facts: dict, reason: str) -> None:
    call = KDACall(
        **{
            "batch": 1,
            "seq_len": 256,
            "sequences": 1,
            "heads": 4,
            "value_heads": 4,
            "dim_k": 128,
            "dim_v": 128,
            "dtype": torch.bfloat16,
            "scale": 128**-0.5,
            "sm_count": 132,
            "arch": 90,
            "calibration": None,
            "smem_budget": 0,
            **facts,
        }
    )
    with pytest.raises(ValueError, match=reason):
        KDAFwdOp().key_for("kda", call)


@pytest.mark.smoke
@pytest.mark.sm89
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("dim", [64, 128])
def test_kda_fused_prefill_in_sm89_shared_memory(dtype, dim):
    """256 sequences over 16 value heads: enough blocks to cover an SM89 device, with
    output and carried state checked at each width."""
    torch.manual_seed(42)
    test = KDAFwdTest(16, 256, 16, dim, dtype)
    test.check(KDAFwdOp(use_qk_l2norm_in_kernel=True), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "batch,heads",
    [
        pytest.param(1, 4, id="chunk"),
        pytest.param(16, 16, marks=pytest.mark.sm89, id="fused-sm89"),
        pytest.param(16, 16, marks=pytest.mark.sm90, id="fused-sm90"),
    ],
)
def test_kda_prefill_correlated_keys(dtype, batch, heads):
    """Repeated keys must not destabilize the inverse, including a partial chunk."""
    torch.manual_seed(42)
    workload = KDAFwdTest(batch, 65, heads, 128, dtype)
    q, k, v, g, beta, state = workload.gen_inputs()
    inputs = (
        q,
        k[:, :1].expand_as(k).contiguous(),
        v,
        torch.full_like(g, -0.01),
        torch.full_like(beta, 0.95),
        state,
    )
    workload.check(KDAFwdOp(use_qk_l2norm_in_kernel=True), *inputs)


class KDARawGateTest(KDAFwdCall, TestBase):
    pass


def _raw_gate_test(
    batch,
    seqlen,
    dim,
    lower_bound,
    some=(),
    dtype="bfloat16",
    heads=4,
    value_heads=None,
    l2norm=True,
    **shape,
):
    """A raw-gate workload test and the op it checks."""
    call = manifest_call(
        "KDAFwdOp",
        {"T": dtype},
        B=batch,
        S=seqlen,
        H=heads,
        HV=value_heads or heads,
        K=dim,
        V=dim,
        use_qk_l2norm_in_kernel=l2norm,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
        some=list(some),
        **shape,
    )
    return KDARawGateTest(call), KDAFwdOp(**call.arguments({}))


@pytest.mark.parametrize(
    "batch,seqlen,heads,value_heads,dim,dtype,lower_bound,packed",
    [
        pytest.param(
            1, 65, 2, 4, 64, "float16", None, False, marks=pytest.mark.smoke, id="grouped-fp16"
        ),
        pytest.param(
            1,
            65,
            4,
            4,
            128,
            "bfloat16",
            -5.0,
            True,
            marks=pytest.mark.smoke,
            id="bounded-packed-tail",
        ),
        pytest.param(
            1, 1100, 1, 2, 128, "bfloat16", -5.0, False, marks=pytest.mark.full, id="long-grouped"
        ),
        pytest.param(
            16,
            65,
            16,
            16,
            64,
            "float16",
            -0.25,
            False,
            marks=pytest.mark.full,
            id="bounded-many-sequences",
        ),
    ],
)
def test_kda_raw_gate_matches_reference(
    batch, seqlen, heads, value_heads, dim, dtype, lower_bound, packed
):
    """Raw gates preserve the recurrence across grouped heads, packed tails and saturation."""
    some = ["initial_state", "dt_bias"] + (["cu_seqlens", "cu_seqlens_cpu"] if packed else [])
    test, op = _raw_gate_test(
        batch,
        seqlen,
        dim,
        lower_bound,
        some,
        dtype=dtype,
        heads=heads,
        value_heads=value_heads,
        **({"seq_lens": [0, 1, 64]} if packed else {}),
    )
    inputs = list(test.gen_inputs())
    inputs[3] = (
        torch.linspace(-100, 100, dim, device=inputs[3].device)
        .to(inputs[3])
        .expand_as(inputs[3])
        .contiguous()
    )
    test.check(op, *inputs)


@pytest.mark.smoke
def test_kda_raw_gate_takes_large_softplus_arguments_as_identity() -> None:
    """A shifted gate past exp's FP32 range keeps the finite decay -exp(A_log) * shifted.

    softplus(100) is 100 within rounding, while exp(100) overflows FP32, so evaluating it as
    log1p(exp(100)) would give an infinite decay that zeroes the carried state.
    """
    torch.manual_seed(42)
    test, op = _raw_gate_test(1, 65, 64, None, ["initial_state"])
    inputs = list(test.gen_inputs())
    inputs[3].fill_(100.0)
    # A decay of -0.25 per token keeps the carried state visible across the call.
    inputs[8].fill_(math.log(0.0025))
    test.check(op, *inputs)


@pytest.mark.smoke
def test_kda_raw_gate_decode_reads_dt_bias() -> None:
    """Decode adds each channel's dt_bias to the raw gate before the softplus."""
    torch.manual_seed(42)
    test, op = _raw_gate_test(4, 1, 64, None, ["initial_state", "dt_bias"])
    inputs = list(test.gen_inputs())
    inputs[3].zero_()
    inputs[8].zero_()
    # Channel decays from -softplus(-3) to -softplus(3) instead of -log(2) for every channel.
    inputs[9].copy_(torch.linspace(-3.0, 3.0, 64, device=inputs[9].device).repeat(4))
    test.check(op, *inputs)


@pytest.mark.smoke
def test_kda_raw_gate_decode_keeps_state_when_factors_cross_fp32_range() -> None:
    """A finite decay retains the carried state when its raw factors cross FP32's range.

    exp(88) is near FP32's largest value and exp(-88) is subnormal, yet their product,
    the decay -1, is finite.
    """
    torch.manual_seed(42)
    test, op = _raw_gate_test(8, 1, 128, None, ["initial_state"])
    inputs = list(test.gen_inputs())
    inputs[3].fill_(-88.0)
    inputs[8].fill_(88.0)
    test.check(op, *inputs)


@pytest.mark.smoke
def test_kda_bounded_gate_zero_gate_survives_overflowing_rate() -> None:
    """A zero raw gate has decay lower_bound / 2 even when exp(A_log) overflows FP32."""
    torch.manual_seed(42)
    test, op = _raw_gate_test(
        1,
        65,
        128,
        -5.0,
        ["initial_state", "dt_bias", "cu_seqlens", "cu_seqlens_cpu"],
        seq_lens=[0, 1, 64],
    )
    inputs = list(test.gen_inputs())
    inputs[3].zero_()
    inputs[8].fill_(100.0)
    inputs[9].zero_()
    test.check(op, *inputs)


@pytest.mark.parametrize(
    "batch,seqlen,lower_bound,seq_lens",
    [
        pytest.param(2, 192, None, None, marks=pytest.mark.smoke, id="dense"),
        pytest.param(1, 300, -5.0, [1, 63, 100, 136], marks=pytest.mark.full, id="bounded-packed"),
        pytest.param(128, 100, -5.0, None, marks=pytest.mark.full, id="many-sequences"),
    ],
)
def test_kda_raw_gate_serves_many_chunk_heads(batch, seqlen, lower_bound, seq_lens) -> None:
    """Calls of many chunk-heads match the reference, dense and packed with partial chunks.

    64 chunk-heads dense and packed, and 2048 chunk-heads of 128 sequences that each end in
    a partial chunk.
    """
    some = ["initial_state", "dt_bias"] + (["cu_seqlens", "cu_seqlens_cpu"] if seq_lens else [])
    shape = {"seq_lens": seq_lens} if seq_lens else {}
    test, op = _raw_gate_test(batch, seqlen, 128, lower_bound, some, heads=8, **shape)
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "batch,seqlen,heads",
    [
        pytest.param(1, 65, 4, marks=pytest.mark.smoke),
        pytest.param(16, 2, 16, marks=pytest.mark.full),
        pytest.param(4, 1, 4, marks=pytest.mark.full),
    ],
)
def test_kda_gate_modes_and_operands_remain_call_local(batch, seqlen, heads):
    """Gate mode, bound, bias presence and operand values belong to the current call."""
    for raw, lower_bound, bias in [
        (True, -0.25, True),
        (True, -3.0, True),
        (True, None, True),
        (False, None, False),
    ]:
        call = manifest_call(
            "KDAFwdOp",
            {"T": "bfloat16"},
            B=batch,
            S=seqlen,
            H=heads,
            HV=heads,
            K=64,
            V=64,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=raw,
            lower_bound=lower_bound,
            some=["initial_state"] + (["dt_bias"] if bias else []),
        )
        test = KDARawGateTest(call)
        inputs = list(test.gen_inputs())
        op = KDAFwdOp(**call.arguments({}))
        test.check(op, *inputs)
        if raw:
            inputs[8].add_(0.5)
            if bias:
                inputs[9].add_(0.75)
            test.check(op, *inputs)
            if bias:
                inputs[9] = None
                test.check(op, *inputs)


@pytest.mark.smoke
def test_kda_raw_gate_decode_continues_from_prefill() -> None:
    """Raw-gate decode consumes the FP32 state returned by raw-gate prefill."""
    test, op = _raw_gate_test(1, 65, 64, None, ["initial_state", "dt_bias"])
    inputs = test.gen_inputs()
    op = op
    first = list(inputs)
    first[:5] = [tensor[:, :-1] for tensor in inputs[:5]]
    first_o, carried = op(*first)
    second = list(inputs)
    second[:5] = [tensor[:, -1:] for tensor in inputs[:5]]
    second[5] = carried
    second_o, final = op(*second)
    compare_outputs(
        (torch.cat((first_o, second_o), dim=1), final),
        test.ref_program(*inputs),
        test.verification(*inputs),
    )


@pytest.mark.parametrize(
    "batch,heads",
    [
        pytest.param(1, 1, marks=pytest.mark.smoke, id="single-sequence"),
        pytest.param(32, 4, marks=pytest.mark.full, id="batched-sequences"),
    ],
)
def test_kda_raw_gate_preserves_each_channel_decay(batch, heads) -> None:
    """Zero beta and diagonal state expose each channel's accumulated decay in every output."""
    test, op = _raw_gate_test(
        batch, 64, 128, None, ["initial_state"], heads=heads, l2norm=False, scale=1.0
    )
    inputs = list(test.gen_inputs())
    inputs[0].fill_(1.0)
    inputs[1].zero_()
    inputs[2].zero_()
    inputs[3].uniform_(-5.0, -3.0)
    inputs[4].zero_()
    inputs[5].copy_(torch.eye(128, device=inputs[5].device))
    inputs[8].zero_()
    test.check(op, *inputs)


@pytest.mark.parametrize(
    "batch,heads",
    [
        pytest.param(1, 1, marks=pytest.mark.smoke, id="single-sequence"),
        pytest.param(32, 4, marks=pytest.mark.full, id="batched-sequences"),
    ],
)
def test_kda_bounded_gate_preserves_tiny_query_contributions(batch, heads) -> None:
    """A tiny normalized query component retains a finite contribution from a large value."""
    test, op = _raw_gate_test(batch, 64, 128, -5.0, heads=heads, scale=1.0)
    inputs = list(test.gen_inputs())
    for index in (0, 1, 2, 4, 8):
        inputs[index].zero_()
    inputs[0][:, 15, :, 0] = 1e-30
    inputs[0][:, 15, :, 1] = 1.0
    inputs[1][:, 14, :, 0] = 1.0
    inputs[2][:, 14, :, 0] = 1e35
    inputs[3].fill_(100.0)
    inputs[4][:, 14, :] = 1.0
    test.check(op, *inputs)


@pytest.mark.parametrize(
    "batch,heads",
    [
        pytest.param(1, 1, marks=pytest.mark.smoke, id="single-sequence"),
        pytest.param(32, 4, marks=pytest.mark.full, id="batched-sequences"),
    ],
)
def test_kda_bounded_gate_preserves_near_cancelling_contributions(batch, heads) -> None:
    """Small channelwise gate differences preserve the finite residual of cancelling projections."""
    test, op = _raw_gate_test(batch, 64, 128, -5.0, ["dt_bias"], heads=heads, scale=1.0)
    inputs = list(test.gen_inputs())
    for index in (0, 1, 2, 3, 4, 8, 9):
        inputs[index].zero_()
    inputs[0][:, 16, :, :2] = 1.0
    inputs[1][:, 15, :, 0] = 1.0
    inputs[1][:, 15, :, 1] = -1.0
    inputs[2][:, 15, :, 0] = 1e6
    inputs[4][:, 15, :] = 1.0
    inputs[9].view(heads, -1)[:, 1] = 1e-3
    test.check(op, *inputs)


@pytest.mark.smoke
def test_kda_raw_gate_preserves_finite_unnormalized_cancellation() -> None:
    """Large finite unnormalized operands retain a finite cancelling projection."""
    test, op = _raw_gate_test(1, 64, 128, None, heads=1, l2norm=False, scale=1.0)
    inputs = list(test.gen_inputs())
    for index in (0, 1, 2, 4, 8):
        inputs[index].zero_()
    inputs[0][0, 0, 0, 0] = 2.0**-63
    inputs[1][0, 0, 0, :] = 2.0**63
    inputs[2][0, 0, 0, 0] = 1.0
    inputs[4][0, 0, 0] = 1.0
    inputs[0][0, 1, 0, 0::2] = 2.0**63
    inputs[0][0, 1, 0, 1::2] = -(2.0**63)
    inputs[3].fill_(-100.0)
    test.check(op, *inputs)


@pytest.mark.smoke
def test_kda_raw_gate_preserves_finite_state_cancellation() -> None:
    """Large finite state contributions retain a finite cancelling output."""
    test, op = _raw_gate_test(32, 64, 128, None, scale=1.0)
    inputs = list(test.gen_inputs())
    for index in (0, 1, 2, 4, 8):
        inputs[index].zero_()
    for token in range(16):
        inputs[1][:, token, :, token] = 2.0**32
        inputs[2][:, token, :, 0] = (-1.0 if token % 2 else 1.0) * (1.5 * 2.0**127)
        inputs[4][:, token, :] = 1.0
    inputs[0][:, 0, :, 0] = 2.0**32
    inputs[0][:, 16, :, :16] = 2.0**32
    inputs[3].fill_(-100.0)
    test.check(op, *inputs)


@pytest.mark.smoke
def test_kda_raw_gate_preserves_finite_beta_correction() -> None:
    """A fractional beta retains a finite correction between large opposite values."""
    test, op = _raw_gate_test(32, 64, 128, None, scale=1.0)
    inputs = list(test.gen_inputs())
    for index in (0, 1, 2, 4, 8):
        inputs[index].zero_()
    inputs[0][:, :2, :, 0] = 2.0**32
    inputs[1][:, :2, :, 0] = 2.0**32
    inputs[2][:, 0, :, 0] = -(1.5 * 2.0**127)
    inputs[2][:, 1, :, 0] = 1.5 * 2.0**127
    inputs[4][:, 0, :] = 1.0
    inputs[4][:, 1, :] = 0.5
    inputs[3].fill_(-100.0)
    test.check(op, *inputs)


@pytest.mark.smoke
def test_kda_bounded_gate_preserves_single_step_cancellation() -> None:
    """A single small gate difference retains the residual of two opposing channels."""
    test, op = _raw_gate_test(1, 64, 128, -5.0, ["dt_bias"], heads=1, scale=1.0)
    inputs = list(test.gen_inputs())
    for index in (0, 1, 2, 4, 8, 9):
        inputs[index].zero_()
    inputs[0][0, 15, 0, 0] = 1.0
    inputs[0][0, 16, 0, :2] = 1.0
    inputs[1][0, 15, 0, 0] = 1.0
    inputs[1][0, 15, 0, 1] = -1.0
    inputs[2][0, 15, 0, 0] = 1e6
    inputs[4][0, 15, 0] = 1.0
    inputs[3].fill_(-100.0)
    inputs[3][0, 16, 0, 0] = 0.0
    inputs[3][0, 16, 0, 1] = 2.0**-19
    test.check(op, *inputs)


def _strong_decays(dtype):
    """Uniform per-token log decays whose sum over a 16-row block overflows *dtype*."""
    return (-20.0, -12.0) if dtype == torch.bfloat16 else (-4.0, -2.0)


@pytest.mark.parametrize(
    "dtype,batch,heads,value_heads,dim,l2norm,boundary_decay",
    [
        pytest.param(
            torch.bfloat16,
            1,
            4,
            4,
            128,
            True,
            9.0,
            marks=pytest.mark.smoke,
            id="normalized-128-bf16",
        ),
        pytest.param(
            torch.float16,
            1,
            2,
            4,
            64,
            False,
            1.5,
            marks=pytest.mark.smoke,
            id="unnormalized-64-fp16",
        ),
        pytest.param(
            torch.bfloat16, 16, 16, 16, 128, False, 9.0, marks=pytest.mark.smoke, id="fused-bf16"
        ),
        pytest.param(
            torch.float16, 16, 16, 16, 128, True, 1.5, marks=pytest.mark.full, id="fused-fp16"
        ),
    ],
)
def test_kda_prefill_given_gate_mixes_ordinary_and_strong_chunks(
    dtype, batch, heads, value_heads, dim, l2norm, boundary_decay
):
    """Ordinary, strong and boundary-concentrated decays in one call all stay accurate.

    The 64-token chunks rotate between the workload's decays, strong decays (also in the
    one-row tail), and *boundary_decay* (log2) per token on every row of each 16-row block
    but its first and ninth, which puts the blocks' reference rows far apart. One sequence
    reaches the chunk program, 256 (sequence, head) pairs the fused program.
    """
    torch.manual_seed(42)
    workload = KDAFwdTest(
        batch, 4 * 64 + 1, heads, dim, dtype, l2norm=l2norm, value_heads=value_heads
    )
    q, k, v, g, beta, state = workload.gen_inputs()
    chunk = torch.arange(g.shape[1], device=g.device) // 64 % 3
    row = torch.arange(g.shape[1], device=g.device) % 16
    concentrated = torch.where((row == 0) | (row == 8), 0.0, -boundary_decay * math.log(2))
    g = g.float()
    g[:, chunk == 1] = torch.empty_like(g[:, chunk == 1]).uniform_(*_strong_decays(dtype))
    g[:, chunk == 2] = concentrated[chunk == 2][None, :, None, None].expand_as(g[:, chunk == 2])
    g = g.to(dtype)
    if not l2norm:
        # Unit keys keep the delta rule contractive without the in-kernel normalization.
        q, k = (x / x.norm(dim=-1, keepdim=True) for x in (q, k))
    workload.check(KDAFwdOp(use_qk_l2norm_in_kernel=l2norm), q, k, v, g, beta, state)


@pytest.mark.sm90
@pytest.mark.parametrize(
    "packed",
    [
        pytest.param(False, marks=pytest.mark.smoke, id="dense"),
        pytest.param(True, marks=pytest.mark.full, id="varlen"),
    ],
)
def test_kda_prefill_strong_decay_sequence_among_ordinary_sequences(packed):
    """One sequence's strong decays leave the other sequences of the call untouched.

    The call holds a (sequence, value head) pair per SM and twice the tokens times value
    heads per SM from which an SM90 device takes two launches, so only the strong
    sequence's pairs walk again. Packed lengths shift its chunks off the dense grid.
    """
    heads = 32
    sm_count = get_sm_count()
    sequences = max(4, -(-sm_count // heads))
    length = -(-2 * TWO_LAUNCH_TOKENS_PER_SM * sm_count // (sequences * heads * 64)) * 64
    torch.manual_seed(42)
    if packed:
        lengths = tuple(length + (-200, 100, 200, -100)[i % 4] for i in range(sequences))
        workload = KDAFwdTest(1, 0, heads, 128, torch.bfloat16, sequence_lengths=lengths)
        start = lengths[0]
    else:
        workload = KDAFwdTest(sequences, length, heads, 128, torch.bfloat16)
        start = length
    inputs = list(workload.gen_inputs())
    g = inputs[3].clone()
    flat = g.view(-1, *g.shape[2:])
    strong = flat[start + 100 : start + 228]
    strong.copy_(torch.empty(strong.shape, device=g.device).uniform_(*_strong_decays(g.dtype)))
    inputs[3] = g
    workload.check(KDAFwdOp(use_qk_l2norm_in_kernel=True), *inputs)
