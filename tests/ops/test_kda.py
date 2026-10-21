"""KDAFwdOp against the FP64 KDA recurrence in ``workloads/linear_attention/kda.py``."""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.kernels.linear_attention import (
    KDACall,
)
from tileops.ops import KDAFwdOp
from workloads.linear_attention.kda import KDAFwdCall, KDAFwdWorkload
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call

pytestmark = pytest.mark.smoke


class KDAFwdTest(KDAFwdWorkload, TestBase):
    pass


def _check(*args, **kwargs):
    test = KDAFwdTest(*args, **kwargs)
    test.check(KDAFwdOp(use_qk_l2norm_in_kernel=True), *test.gen_inputs())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_kda_prefill_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    _check(1, 256, 4, 128, dtype)


def test_kda_prefill_packs_ragged_sequences() -> None:
    torch.manual_seed(42)
    _check(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(100, 70, 130))


def test_kda_prefill_packs_a_batch_holding_an_empty_sequence() -> None:
    """A sequence with no token owns no chunk, and the ones around it keep their own.

    Such a sequence applies no update, so it ends on the state it started from.
    """
    torch.manual_seed(42)
    _check(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(100, 0, 70, 130))


def test_kda_prefill_packs_sequences_that_fill_their_chunks() -> None:
    """The launch bound is loosest when every sequence ends on a chunk boundary.

    Four 64-token sequences need four chunks and the bound gives eight, so half the
    launch is the chunks past the last real one. They must leave the output alone.
    """
    torch.manual_seed(42)
    _check(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(64,) * 4)


def test_kda_prefill_reads_offsets_a_freed_buffer_left_behind() -> None:
    """Boundaries come from the offsets this call was handed, not the last ones.

    The caching allocator hands a freed block to the next tensor of the same size,
    so a second call can carry offsets at the address and version the first one had.
    """
    torch.manual_seed(42)
    _check(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(100, 100, 100))
    torch.cuda.empty_cache()
    _check(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(50, 120, 130))


def test_kda_prefill_serves_more_value_heads_than_query_heads() -> None:
    torch.manual_seed(42)
    _check(1, 256, 2, 128, torch.bfloat16, value_heads=4)


@pytest.mark.parametrize(
    "sequence_lengths",
    [pytest.param((100, 130), id="two"), pytest.param((100, 130, 70, 64), id="four")],
)
def test_kda_prefill_packs_sequences_of_16_heads(sequence_lengths) -> None:
    """Packed sequences of 16 value heads, most ending in a partial chunk."""
    torch.manual_seed(42)
    _check(1, 0, 16, 128, torch.bfloat16, sequence_lengths=sequence_lengths)


def test_kda_prefill_serves_a_64_wide_state() -> None:
    torch.manual_seed(42)
    _check(1, 256, 4, 64, torch.bfloat16)


@pytest.mark.parametrize("batch", [1, 8], ids=["b1", "b8"])
def test_kda_decode_matches_reference(batch: int) -> None:
    torch.manual_seed(42)
    _check(batch, 1, 4, 128, torch.bfloat16)


def test_kda_decode_runs_from_a_zero_state() -> None:
    torch.manual_seed(42)
    _check(4, 1, 4, 128, torch.bfloat16, has_initial_state=False)


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


@pytest.mark.sm89
def test_kda_prefill_in_99_kb_of_shared_memory() -> None:
    """SM89 gives a block 99 KB of opt-in shared memory, so a packed head dim 128 prefill
    runs on smaller programs than SM80 and SM90 build."""
    torch.manual_seed(7)
    _check(1, 0, 4, 128, torch.float16, sequence_lengths=(300, 1, 211))


@pytest.mark.sm89
def test_kda_prefill_scans_independent_value_tiles() -> None:
    """SM89 splits a long single-sequence scan with eight value heads into 16-column
    tiles, so every tile must read and write its own state and output columns.
    """
    torch.manual_seed(42)
    _check(1, 4096, 2, 128, torch.bfloat16, value_heads=8)


@pytest.mark.sm89
def test_kda_fused_prefill_reads_many_offsets_from_global_memory() -> None:
    """2048 sequences of four tokens: more boundaries than a block can stage."""
    torch.manual_seed(42)
    test = KDAFwdTest(1, 0, 4, 128, torch.bfloat16, sequence_lengths=(4,) * 2048)
    test.check(KDAFwdOp(use_qk_l2norm_in_kernel=True), *test.gen_inputs())


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


@pytest.mark.sm89
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("dim", [64, 128])
def test_kda_fused_prefill_in_sm89_shared_memory(dtype, dim):
    """256 sequences over 16 value heads: enough blocks to cover an SM89 device, with
    output and carried state checked at each width."""
    torch.manual_seed(42)
    test = KDAFwdTest(16, 256, 16, dim, dtype)
    test.check(KDAFwdOp(use_qk_l2norm_in_kernel=True), *test.gen_inputs())


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


@pytest.mark.parametrize(
    "dtype,batch,heads,value_heads,dim,l2norm",
    [
        pytest.param(torch.bfloat16, 1, 4, 4, 128, True, id="normalized-128-bf16"),
        pytest.param(torch.float16, 1, 2, 4, 64, False, id="unnormalized-grouped-64-fp16"),
        pytest.param(torch.bfloat16, 16, 16, 16, 128, False, id="unnormalized-many-heads-bf16"),
    ],
)
def test_kda_prefill_strong_given_gate_stays_finite(dtype, batch, heads, value_heads, dim, l2norm):
    """Strong per-token decays keep the output and final state finite and accurate.

    Each dtype takes decays whose sum over a chunk overflows its own range: normalized keys
    128 wide, unnormalized keys 64 wide shared by grouped value heads, and unnormalized keys
    over 256 (sequence, head) pairs.
    """
    decay = (-20.0, -12.0) if dtype == torch.bfloat16 else (-4.0, -2.0)
    torch.manual_seed(42)
    workload = KDAFwdTest(batch, 65, heads, dim, dtype, l2norm=l2norm, value_heads=value_heads)
    q, k, v, g, beta, state = workload.gen_inputs()
    g = torch.empty(g.shape, device=g.device).uniform_(*decay).to(dtype)
    if not l2norm:
        # Unit keys keep the delta rule contractive without the in-kernel normalization.
        q, k = (x / x.norm(dim=-1, keepdim=True) for x in (q, k))
    workload.check(KDAFwdOp(use_qk_l2norm_in_kernel=l2norm), q, k, v, g, beta, state)


class KDARawGateTest(KDAFwdCall, TestBase):
    pass


@pytest.mark.parametrize(
    "batch,seqlen,heads,value_heads,dim,dtype,lower_bound,packed,bias",
    [
        pytest.param(1, 65, 2, 4, 64, "float16", None, False, True, id="grouped-fp16"),
        pytest.param(1, 65, 4, 4, 128, "bfloat16", -5.0, True, True, id="bounded-packed-tail"),
        pytest.param(1, 1100, 1, 2, 128, "bfloat16", -5.0, False, True, id="long-grouped"),
        pytest.param(
            16, 65, 16, 16, 64, "float16", -0.25, False, True, id="bounded-many-sequences"
        ),
        pytest.param(8, 1, 4, 4, 128, "bfloat16", None, False, False, id="decode-no-bias"),
    ],
)
def test_kda_raw_gate_matches_reference(
    batch, seqlen, heads, value_heads, dim, dtype, lower_bound, packed, bias
):
    """Raw gates preserve the recurrence across grouped heads, packed tails and saturation."""
    some = ["initial_state"] + (["dt_bias"] if bias else [])
    if packed:
        some += ["cu_seqlens", "cu_seqlens_cpu"]
    call = manifest_call(
        "KDAFwdOp",
        {"T": dtype},
        B=batch,
        S=seqlen,
        H=heads,
        HV=value_heads,
        K=dim,
        V=dim,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
        some=some,
        **({"seq_lens": [0, 1, 64]} if packed else {}),
    )
    test = KDARawGateTest(call)
    inputs = list(test.gen_inputs())
    inputs[3] = (
        torch.linspace(-100, 100, dim, device=inputs[3].device)
        .to(inputs[3])
        .expand_as(inputs[3])
        .contiguous()
    )
    op = KDAFwdOp(**call.arguments({}))
    test.check(op, *inputs)
    if seqlen == 1 and not bias:
        # A finite decay must retain the carried state when raw factors cross FP32's range.
        inputs[3].fill_(-88.0)
        inputs[8].fill_(88.0)
        test.check(op, *inputs)
    if packed:
        # A zero raw gate has decay lower_bound / 2 even when exp(A_log) overflows FP32.
        inputs[3].zero_()
        inputs[8].fill_(100.0)
        inputs[9].zero_()
        test.check(op, *inputs)


@pytest.mark.parametrize(
    "batch,seqlen,lower_bound,seq_lens",
    [
        pytest.param(2, 192, None, None, id="dense"),
        pytest.param(1, 300, -5.0, [1, 63, 100, 136], id="bounded-packed"),
        pytest.param(128, 100, -5.0, None, id="many-sequences"),
    ],
)
def test_kda_raw_gate_serves_many_chunk_heads(batch, seqlen, lower_bound, seq_lens) -> None:
    """Calls of many chunk-heads match the reference, dense and packed with partial chunks.

    64 chunk-heads dense and packed, and 2048 chunk-heads of 128 sequences that each end in
    a partial chunk.
    """
    some = ["initial_state", "dt_bias"]
    if seq_lens is not None:
        some += ["cu_seqlens", "cu_seqlens_cpu"]
    call = manifest_call(
        "KDAFwdOp",
        {"T": "bfloat16"},
        B=batch,
        S=seqlen,
        H=8,
        HV=8,
        K=128,
        V=128,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
        some=some,
        **({"seq_lens": seq_lens} if seq_lens is not None else {}),
    )
    test = KDARawGateTest(call)
    test.check(KDAFwdOp(**call.arguments({})), *test.gen_inputs())


@pytest.mark.parametrize("batch,seqlen,heads", [(1, 65, 4), (16, 2, 16), (4, 1, 4)])
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


def test_kda_raw_gate_decode_continues_from_prefill() -> None:
    """Raw-gate decode consumes the FP32 state returned by raw-gate prefill."""
    call = manifest_call(
        "KDAFwdOp",
        {"T": "bfloat16"},
        B=1,
        S=65,
        H=4,
        HV=4,
        K=64,
        V=64,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        some=["initial_state", "dt_bias"],
    )
    test = KDARawGateTest(call)
    inputs = test.gen_inputs()
    op = KDAFwdOp(**call.arguments({}))
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
    [pytest.param(1, 1, id="single-sequence"), pytest.param(32, 4, id="batched-sequences")],
)
def test_kda_raw_gate_preserves_each_channel_decay(batch, heads) -> None:
    """Zero beta and diagonal state expose each channel's accumulated decay in every output."""
    call = manifest_call(
        "KDAFwdOp",
        {"T": "bfloat16"},
        B=batch,
        S=64,
        H=heads,
        HV=heads,
        K=128,
        V=128,
        scale=1.0,
        use_gate_in_kernel=True,
        some=["initial_state"],
    )
    test = KDARawGateTest(call)
    inputs = list(test.gen_inputs())
    inputs[0].fill_(1.0)
    inputs[1].zero_()
    inputs[2].zero_()
    inputs[3].uniform_(-5.0, -3.0)
    inputs[4].zero_()
    inputs[5].copy_(torch.eye(128, device=inputs[5].device))
    inputs[8].zero_()
    test.check(KDAFwdOp(**call.arguments({})), *inputs)


@pytest.mark.parametrize(
    "batch,heads",
    [pytest.param(1, 1, id="single-sequence"), pytest.param(32, 4, id="batched-sequences")],
)
def test_kda_bounded_gate_preserves_tiny_query_contributions(batch, heads) -> None:
    """A tiny normalized query component retains a finite contribution from a large value."""
    call = manifest_call(
        "KDAFwdOp",
        {"T": "bfloat16"},
        B=batch,
        S=64,
        H=heads,
        HV=heads,
        K=128,
        V=128,
        scale=1.0,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
    )
    test = KDARawGateTest(call)
    inputs = list(test.gen_inputs())
    for index in (0, 1, 2, 4, 8):
        inputs[index].zero_()
    inputs[0][:, 15, :, 0] = 1e-30
    inputs[0][:, 15, :, 1] = 1.0
    inputs[1][:, 14, :, 0] = 1.0
    inputs[2][:, 14, :, 0] = 1e35
    inputs[3].fill_(100.0)
    inputs[4][:, 14, :] = 1.0
    test.check(KDAFwdOp(**call.arguments({})), *inputs)


@pytest.mark.parametrize(
    "batch,heads",
    [pytest.param(1, 1, id="single-sequence"), pytest.param(32, 4, id="batched-sequences")],
)
def test_kda_bounded_gate_preserves_near_cancelling_contributions(batch, heads) -> None:
    """Small channelwise gate differences preserve the finite residual of cancelling projections."""
    call = manifest_call(
        "KDAFwdOp",
        {"T": "bfloat16"},
        B=batch,
        S=64,
        H=heads,
        HV=heads,
        K=128,
        V=128,
        scale=1.0,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
        some=["dt_bias"],
    )
    test = KDARawGateTest(call)
    inputs = list(test.gen_inputs())
    for index in (0, 1, 2, 3, 4, 8, 9):
        inputs[index].zero_()
    inputs[0][:, 16, :, :2] = 1.0
    inputs[1][:, 15, :, 0] = 1.0
    inputs[1][:, 15, :, 1] = -1.0
    inputs[2][:, 15, :, 0] = 1e6
    inputs[4][:, 15, :] = 1.0
    inputs[9].view(heads, -1)[:, 1] = 1e-3
    test.check(KDAFwdOp(**call.arguments({})), *inputs)


def test_kda_raw_gate_preserves_finite_unnormalized_cancellation() -> None:
    """Large finite unnormalized operands retain a finite cancelling projection."""
    call = manifest_call(
        "KDAFwdOp",
        {"T": "bfloat16"},
        B=1,
        S=64,
        H=1,
        HV=1,
        K=128,
        V=128,
        scale=1.0,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=False,
    )
    test = KDARawGateTest(call)
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
    test.check(KDAFwdOp(**call.arguments({})), *inputs)


def test_kda_raw_gate_preserves_finite_state_cancellation() -> None:
    """Large finite state contributions retain a finite cancelling output."""
    call = manifest_call(
        "KDAFwdOp",
        {"T": "bfloat16"},
        B=32,
        S=64,
        H=4,
        HV=4,
        K=128,
        V=128,
        scale=1.0,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
    )
    test = KDARawGateTest(call)
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
    test.check(KDAFwdOp(**call.arguments({})), *inputs)


def test_kda_raw_gate_preserves_finite_beta_correction() -> None:
    """A fractional beta retains a finite correction between large opposite values."""
    call = manifest_call(
        "KDAFwdOp",
        {"T": "bfloat16"},
        B=32,
        S=64,
        H=4,
        HV=4,
        K=128,
        V=128,
        scale=1.0,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
    )
    test = KDARawGateTest(call)
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
    test.check(KDAFwdOp(**call.arguments({})), *inputs)


def test_kda_bounded_gate_preserves_single_step_cancellation() -> None:
    """A single small gate difference retains the residual of two opposing channels."""
    call = manifest_call(
        "KDAFwdOp",
        {"T": "bfloat16"},
        B=1,
        S=64,
        H=1,
        HV=1,
        K=128,
        V=128,
        scale=1.0,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
        some=["dt_bias"],
    )
    test = KDARawGateTest(call)
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
    test.check(KDAFwdOp(**call.arguments({})), *inputs)
