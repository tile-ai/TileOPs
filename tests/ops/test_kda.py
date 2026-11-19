"""KDAFwdOp against the FP64 KDA recurrence in ``workloads/linear_attention/kda.py``."""

import math

import pytest
import torch

from tests.state_pool_guard import check_padded_slots_skipped
from tests.workload_test_base import TestBase
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention import (
    KDACall,
)
from tileops._kernels.linear_attention.chunk_prefill import run_chunk_prefill
from tileops._kernels.linear_attention.kda import prefill as kda_prefill
from tileops._kernels.linear_attention.kda import prepare_split
from tileops._kernels.linear_attention.kda.prepare_split import split_prepare_program
from tileops._utils.utils import get_sm_count
from tileops.backend import OpNotAvailableError
from tileops.linear_attention import KDAFwdOp
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
def test_kda_prefill_packs_a_batch_holding_an_empty_sequence() -> None:
    """A sequence with no token owns no chunk, applies no update and ends on its initial
    state; the ones around it keep their own chunks."""
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
        # Every in-tree kernel keeps one state a sequence, not one a token.
        pytest.param({"pooled": True, "rollback": True}, "num_accepted_tokens", id="rollback"),
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


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("raw", "dtype", "split"),
    [
        pytest.param(False, torch.bfloat16, False, marks=pytest.mark.smoke, id="given-gate-bf16"),
        pytest.param(False, torch.float16, True, marks=pytest.mark.smoke, id="given-gate-fp16"),
        pytest.param(True, torch.bfloat16, True, marks=pytest.mark.full, id="raw-gate"),
    ],
)
def test_kda_sm90_chunk_prefill_splits_the_preparation_of_raw_and_fp16_gates(
    raw: bool, dtype: torch.dtype, split: bool, monkeypatch
) -> None:
    """On SM90 a bf16 given gate's chunks are prepared in the one centred launch; a raw gate's
    and an fp16 given gate's in the split FP32 launches. All match the recurrence."""
    torch.manual_seed(42)
    built = []

    def counted(*args, **kwargs):
        built.append(1)
        return split_prepare_program(*args, **kwargs)

    monkeypatch.setattr(prepare_split, "split_prepare_program", counted)
    flags = {"use_qk_l2norm_in_kernel": True, "use_gate_in_kernel": raw}
    name = "bfloat16" if dtype == torch.bfloat16 else "float16"
    call = manifest_call(
        "KDAFwdOp", {"T": name}, some=["A_log", "dt_bias"] if raw else [], B=1, S=256, H=4,
        HV=4, K=128, V=128, **flags,
    )  # fmt: skip
    test = KDAFwdCallTest(call)
    test.check(KDAFwdOp(**flags), *test.gen_inputs())
    assert bool(built) == split


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("arch,budget", [(80, 166912), (89, 101376), (90, 232448)])
@pytest.mark.parametrize("dim", [64, 128])
def test_kda_raw_gate_prefill_uses_chunk_path(arch: int, budget: int, dim: int) -> None:
    # The device type the in-tree kernels run on; the call names it without placing a tensor.
    (in_tree,) = Kernel.devices
    call = KDACall(
        batch=16,
        seq_len=256,
        sequences=16,
        heads=16,
        value_heads=16,
        dim_k=dim,
        dim_v=dim,
        dtype=torch.bfloat16,
        scale=dim**-0.5,
        gate_in_kernel=True,
        sm_count=132,
        arch=arch,
        smem_budget=budget,
        device=torch.device(in_tree),
    )
    assert KDAFwdOp().key_for("kda", call) == "kda_chunk_prefill"


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


class KDAFwdCallTest(KDAFwdCall, TestBase):
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
        some=[*some, "A_log"],
        **shape,
    )
    return KDAFwdCallTest(call), KDAFwdOp(**call.arguments())


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


@pytest.mark.sm90
@pytest.mark.cuda_graph(op="KDAFwdOp")
@pytest.mark.parametrize(
    "seqlen,lower_bound",
    [
        pytest.param(1, None, marks=pytest.mark.smoke, id="unbounded-decode"),
        pytest.param(65, -5.0, marks=pytest.mark.smoke, id="bounded-prefill"),
        pytest.param(65, None, marks=pytest.mark.full, id="unbounded-prefill"),
        pytest.param(1, -5.0, marks=pytest.mark.full, id="bounded-decode"),
    ],
)
def test_kda_raw_gate_graph_replay(seqlen, lower_bound):
    """Replay reads changed gates and initial state and overwrites both returned tensors."""
    test, op = _raw_gate_test(1, seqlen, 64, lower_bound, ["initial_state", "dt_bias"])
    inputs = test.gen_inputs()
    # The keyword-only state slot inputs are absent; the rest go by position.
    positional = inputs[:-2]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*positional)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, state = op(*positional)
    for _ in range(2):
        inputs[0].neg_()
        inputs[2].mul_(0.75)
        inputs[3].add_(0.25)
        inputs[5].mul_(0.5)
        inputs[8].add_(0.125)
        inputs[9].sub_(0.125)
        output.fill_(float("nan"))
        state.fill_(float("nan"))
        graph.replay()
        test.check(op, *inputs, runs=lambda *args, **keywords: (output, state))


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
            some=["initial_state"] + (["dt_bias"] if bias else []) + (["A_log"] if raw else []),
        )
        test = KDAFwdCallTest(call)
        inputs = list(test.gen_inputs())
        op = KDAFwdOp(**call.arguments())
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
    first = list(inputs[:-2])
    first[:5] = [tensor[:, :-1] for tensor in inputs[:5]]
    first_o, carried = op(*first)
    second = list(inputs[:-2])
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
    "dim", [pytest.param(128, marks=pytest.mark.smoke), pytest.param(64, marks=pytest.mark.full)]
)
def test_kda_prefill_given_gate_drops_inside_its_blocks(dim):
    """A bf16 chunk whose decay drops 100 nats on the fifth row of every 16-row block, and
    nowhere else, lies past the bounds while the entries inside each block stay of order one,
    so its per-pair diagonal blocks decide the output at both supported widths."""
    torch.manual_seed(42)
    workload = KDAFwdTest(1, 128, 4, dim, torch.bfloat16)
    q, k, v, g, beta, state = workload.gen_inputs()
    row = torch.arange(g.shape[1], device=g.device) % 16
    g = torch.where(row == 4, -100.0, 0.0)[None, :, None, None].expand_as(g).to(g.dtype)
    workload.check(KDAFwdOp(use_qk_l2norm_in_kernel=True), q, k, v, g, beta, state)


@pytest.mark.sm90
@pytest.mark.parametrize(
    "spread",
    [
        pytest.param(None, marks=pytest.mark.smoke, id="dense"),
        # Unequal lengths of one chunk count each, most ending in a partial chunk: the one-CTA
        # walk of each sequence, its checking launch and its rerun.
        pytest.param((-60, -10, 0, -30), marks=pytest.mark.full, id="varlen-even-chunks"),
        # Lengths of different chunk counts: the chunk-parallel prefill.
        pytest.param((-200, 100, 200, -100), marks=pytest.mark.full, id="varlen-uneven-chunks"),
    ],
)
def test_kda_prefill_strong_decay_sequence_among_ordinary_sequences(spread):
    """One strongly decayed sequence leaves ordinary sequences unchanged."""
    heads = 32
    sm_count = get_sm_count()
    sequences = max(4, -(-sm_count // heads))
    length = 6144  # Keeps every supported SM90 device in the long-prefill dispatch region.
    torch.manual_seed(42)
    if spread is not None:
        lengths = tuple(length + spread[i % 4] for i in range(sequences))
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


def _pooled(dtype: torch.dtype, *, some: tuple[str, ...] = (), **row) -> KDAFwdCallTest:
    """A manifest call whose states live in a pool: sequence ``n`` starts from slot
    ``state_indices[n]`` and leaves its final state there, every other slot untouched."""
    name = "bfloat16" if dtype == torch.bfloat16 else "float16"
    call = manifest_call(
        "KDAFwdOp", {"T": name}, some=["initial_state", "state_indices", *some], **row
    )
    return KDAFwdCallTest(call)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("dtype", "dim", "flags"),
    [
        (torch.float16, 128, {}),
        (torch.bfloat16, 128, {"state_v_first": True, "use_qk_l2norm_in_kernel": True}),
        (torch.bfloat16, 64, {"state_v_first": True}),
    ],
    ids=["fp16-key-major", "bf16-value-major", "bf16-value-major-64"],
)
def test_kda_decode_updates_its_slots_of_a_state_pool(
    dtype: torch.dtype, dim: int, flags: dict
) -> None:
    """A decode step reads each sequence's state from its slot and writes it back in place,
    in either state layout; a 64-wide value-major row takes half the lanes of a 128-wide one."""
    torch.manual_seed(42)
    test = _pooled(dtype, B=5, S=1, H=2, HV=4, K=dim, V=dim, P=16, **flags)
    test.check(KDAFwdOp(**flags), *test.gen_inputs())


@pytest.mark.smoke
def test_kda_raw_gate_decode_updates_its_slots_of_a_state_pool() -> None:
    """A pooled step forms each channel's decay from the raw gate, its rate and its bias."""
    torch.manual_seed(42)
    flags = {"state_v_first": True, "use_gate_in_kernel": True, "lower_bound": -5.0}
    test = _pooled(
        torch.bfloat16, some=("A_log", "dt_bias"), B=3, S=1, H=2, HV=4, K=128, V=128, P=8,
        **flags,
    )  # fmt: skip
    test.check(KDAFwdOp(**flags), *test.gen_inputs())


@pytest.mark.smoke
def test_kda_decode_steps_a_value_major_state() -> None:
    """A value-major step in batch order."""
    torch.manual_seed(42)
    call = manifest_call(
        "KDAFwdOp", {"T": "bfloat16"}, some=["initial_state"], B=3, S=1, H=2, HV=4, K=128,
        V=128, state_v_first=True,
    )  # fmt: skip
    test = KDAFwdCallTest(call)
    test.check(KDAFwdOp(state_v_first=True), *test.gen_inputs())


# Enough (sequence, value head) pairs for the one-kernel prefill on any SM89 or SM90 device.
_FUSED_HEADS = 64
# SM89 builds the one-kernel prefill's compact program, SM90 its full one.
_FUSED_ARCHS = [
    pytest.param("sm89", marks=pytest.mark.sm89, id="sm89"),
    pytest.param("sm90", marks=pytest.mark.sm90, id="sm90"),
]


# Sequence lengths whose chunk counts differ (one is empty) and lengths of two chunks each.
_UNEVEN = [100, 0, 37, 64]
_EVEN = [100, 70, 120, 66]


_HOST = ("cu_seqlens", "cu_seqlens_cpu")


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("state_v_first", [False, True], ids=["key-major", "value-major"])
@pytest.mark.parametrize(
    ("offsets", "seq_lens", "dtype", "chunked", "arch"),
    [
        pytest.param(
            _HOST, _UNEVEN, torch.bfloat16, True, "sm89",
            marks=[pytest.mark.sm89, pytest.mark.smoke], id="uneven-chunks-sm89",
        ),
        pytest.param(
            _HOST, _UNEVEN, torch.bfloat16, True, "sm90",
            marks=[pytest.mark.sm90, pytest.mark.smoke], id="uneven-chunks-sm90",
        ),
        # fp16 chunks take the split preparation on SM90, which the one-CTA walk outruns.
        pytest.param(
            _HOST, _UNEVEN, torch.float16, False, "sm90",
            marks=[pytest.mark.sm90, pytest.mark.smoke], id="uneven-chunks-fp16-sm90",
        ),
        # Only SM90 reads the host lengths; SM89 walks every packed call in one CTA a sequence.
        pytest.param(
            ("cu_seqlens",), _EVEN, torch.bfloat16, True, "sm90",
            marks=[pytest.mark.sm90, pytest.mark.full], id="device-offsets-sm90",
        ),
        pytest.param(
            _HOST, _EVEN, torch.bfloat16, False, "sm90",
            marks=[pytest.mark.sm90, pytest.mark.full], id="even-chunks-sm90",
        ),
    ],
)  # fmt: skip
def test_kda_prefill_updates_its_slots_of_a_state_pool(
    state_v_first: bool,
    offsets: tuple,
    seq_lens: list,
    dtype: torch.dtype,
    chunked: bool,
    arch,
    monkeypatch,
) -> None:
    """Packed sequences start from and end in their slots, in either state layout; an empty
    sequence leaves its slot as it was. Enough (sequence, value head) pairs for the one-CTA
    walk of each sequence, which serves the call only when the host shows every sequence
    spanning one chunk count; on SM90 the chunk-parallel prefill serves the rest of the bf16
    calls, and SM89 keeps the one-CTA walk."""
    torch.manual_seed(42)
    runs = []

    def counted(*args, **kwargs):
        runs.append(1)
        return run_chunk_prefill(*args, **kwargs)

    monkeypatch.setattr(kda_prefill, "run_chunk_prefill", counted)
    flags = {"state_v_first": state_v_first, "use_qk_l2norm_in_kernel": True}
    test = _pooled(
        dtype, some=offsets, B=1, S=sum(seq_lens), seq_lens=seq_lens, H=4, HV=_FUSED_HEADS,
        K=128, V=128, P=12, **flags,
    )  # fmt: skip
    test.check(KDAFwdOp(**flags), *test.gen_inputs())
    assert bool(runs) == (chunked and arch == "sm90")


@pytest.mark.smoke
def test_kda_raw_gate_prefill_updates_its_slots_of_a_value_major_pool() -> None:
    """Packed raw-gate sequences, which only the chunk-parallel prefill serves, start from and
    end in their slots of a value-major pool; an empty sequence leaves its slot as it was."""
    torch.manual_seed(42)
    flags = {"state_v_first": True, "use_gate_in_kernel": True, "lower_bound": -5.0}
    test = _pooled(
        torch.bfloat16, some=("cu_seqlens", "cu_seqlens_cpu", "A_log", "dt_bias"), B=1,
        S=sum(_UNEVEN), seq_lens=_UNEVEN, H=4, HV=_FUSED_HEADS, K=128, V=128, P=12, **flags,
    )  # fmt: skip
    test.check(KDAFwdOp(**flags), *test.gen_inputs())


@pytest.mark.parametrize("pooled", [False, True], ids=["batch-order", "pool"])
@pytest.mark.parametrize(
    ("length", "heads"),
    [
        # Only the SM90 walk forms the outputs itself.
        pytest.param(65, 4, marks=[pytest.mark.sm90, pytest.mark.smoke], id="outputs-in-walk"),
        pytest.param(1200, 2, marks=pytest.mark.full, id="output-launch"),
    ],
)
def test_kda_chunk_prefill_walks_a_value_major_state(pooled: bool, length: int, heads: int) -> None:
    """A few value heads are too few for the one-CTA walk, so the chunk-parallel prefill
    starts from and returns value-major states. SM90 forms a short call's outputs in the
    walk and a longer call's in a launch of its own after it, as every call does elsewhere."""
    torch.manual_seed(42)
    flags = {"state_v_first": True, "use_qk_l2norm_in_kernel": True}
    shape = {"B": 1, "S": length, "H": heads, "HV": heads, "K": 128, "V": 128}
    if pooled:
        test = _pooled(torch.bfloat16, P=6, **shape, **flags)
    else:
        call = manifest_call(
            "KDAFwdOp", {"T": "bfloat16"}, some=["initial_state"], **shape, **flags
        )
        test = KDAFwdCallTest(call)
    test.check(KDAFwdOp(**flags), *test.gen_inputs())


@pytest.mark.parametrize(
    ("flags", "row", "padded"),
    [
        pytest.param({}, {"B": 5, "S": 1}, [1, 3], marks=pytest.mark.smoke),
        pytest.param({"state_v_first": True}, {"B": 5, "S": 1}, [1, 3], marks=pytest.mark.full),
        pytest.param(
            {},
            {
                "B": 1,
                "S": sum(_UNEVEN),
                "seq_lens": _UNEVEN,
                "HV": _FUSED_HEADS,
                "some": ("cu_seqlens",),
            },
            [0, 2],
            marks=[pytest.mark.sm80, pytest.mark.full],
        ),
        pytest.param(
            {},
            {
                "B": 1,
                "S": sum(_UNEVEN),
                "seq_lens": _UNEVEN,
                "HV": _FUSED_HEADS,
                "some": ("cu_seqlens",),
            },
            [0, 2],
            marks=[pytest.mark.sm90, pytest.mark.full],
        ),
        pytest.param(
            {"state_v_first": True},
            {
                "B": 1,
                "S": sum(_UNEVEN),
                "seq_lens": _UNEVEN,
                "HV": _FUSED_HEADS,
                "some": ("cu_seqlens",),
            },
            [0, 2],
            marks=[pytest.mark.sm80, pytest.mark.full],
        ),
        pytest.param(
            {"state_v_first": True},
            {
                "B": 1,
                "S": sum(_UNEVEN),
                "seq_lens": _UNEVEN,
                "HV": _FUSED_HEADS,
                "some": ("cu_seqlens",),
            },
            [0, 2],
            marks=[pytest.mark.sm90, pytest.mark.full],
        ),
        pytest.param(
            {},
            {
                "B": 1,
                "S": sum(_EVEN),
                "seq_lens": _EVEN,
                "HV": _FUSED_HEADS,
                "some": ("cu_seqlens", "cu_seqlens_cpu"),
            },
            [0, 2],
            marks=[pytest.mark.sm89, pytest.mark.full],
        ),
        pytest.param(
            {},
            {
                "B": 1,
                "S": sum(_EVEN),
                "seq_lens": _EVEN,
                "HV": _FUSED_HEADS,
                "some": ("cu_seqlens", "cu_seqlens_cpu"),
            },
            [0, 2],
            marks=[pytest.mark.sm90, pytest.mark.full],
        ),
        pytest.param(
            {"state_v_first": True},
            {
                "B": 1,
                "S": sum(_EVEN),
                "seq_lens": _EVEN,
                "HV": _FUSED_HEADS,
                "some": ("cu_seqlens", "cu_seqlens_cpu"),
            },
            [0, 2],
            marks=[pytest.mark.sm89, pytest.mark.full],
        ),
        pytest.param(
            {"state_v_first": True},
            {
                "B": 1,
                "S": sum(_EVEN),
                "seq_lens": _EVEN,
                "HV": _FUSED_HEADS,
                "some": ("cu_seqlens", "cu_seqlens_cpu"),
            },
            [0, 2],
            marks=[pytest.mark.sm90, pytest.mark.full],
        ),
    ],
    ids=[
        "decode-key-major",
        "decode-value-major",
        "chunk-prefill-key-major-sm80",
        "chunk-prefill-key-major-sm90",
        "chunk-prefill-value-major-sm80",
        "chunk-prefill-value-major-sm90",
        "fused-prefill-key-major-sm89",
        "fused-prefill-key-major-sm90",
        "fused-prefill-value-major-sm89",
        "fused-prefill-value-major-sm90",
    ],
)
def test_kda_skips_the_states_of_negative_slots(flags: dict, row: dict, padded: list) -> None:
    """The decode and both packed prefills write no state for a slot of -1, in either
    layout: the pool's neighbours and the slots of the padded sequences keep their states, every
    other slot ends as in the unpadded call."""
    torch.manual_seed(42)
    row = {"H": 4, "HV": 4, "K": 128, "V": 128, "P": 12, **row}
    some = row.pop("some", ())
    inputs = _pooled(torch.bfloat16, some=some, **row, **flags).gen_inputs()
    check_padded_slots_skipped(KDAFwdOp(**flags, validate_inputs=True), inputs, 5, 10, padded)


@pytest.mark.smoke
@pytest.mark.parametrize("arch", _FUSED_ARCHS)
def test_kda_fused_prefill_walks_a_value_major_state(arch) -> None:
    """Equal-length sequences start from value-major states in batch order and return them."""
    del arch
    torch.manual_seed(42)
    call = manifest_call(
        "KDAFwdOp", {"T": "float16"}, some=["initial_state"], B=4, S=100, H=4,
        HV=_FUSED_HEADS, K=128, V=128, state_v_first=True,
    )  # fmt: skip
    test = KDAFwdCallTest(call)
    test.check(KDAFwdOp(state_v_first=True), *test.gen_inputs())


@pytest.mark.smoke
def test_kda_decode_updates_a_strided_pool_in_place() -> None:
    """A pool that is a strided view is advanced in place, the call returning that view."""
    torch.manual_seed(42)
    test = _pooled(torch.bfloat16, B=4, S=1, H=2, HV=4, K=128, V=128, P=8, state_v_first=True)
    q, k, v, g, beta, pool, *rest = test.gen_inputs()
    strided = torch.empty((2 * pool.shape[0], *pool.shape[1:]), device=pool.device)[::2]
    strided.copy_(pool)
    op = KDAFwdOp(state_v_first=True)

    def run(*args, **keywords):
        # The compared state is the caller's view, so a missed copy-back fails the check.
        o, final_state = op(*args, **keywords)
        assert final_state is args[5]
        return o, final_state

    test.check(op, q, k, v, g, beta, strided, *rest, runs=run)


@pytest.mark.sm90
@pytest.mark.parametrize(
    "pooled",
    [
        pytest.param(True, marks=pytest.mark.smoke, id="value-major-pool"),
        pytest.param(False, marks=pytest.mark.full, id="value-major"),
    ],
)
def test_kda_long_prefill_walks_a_value_major_state(pooled: bool) -> None:
    """A value-major prefill with enough tokens a SM for the two-launch program and one strongly
    decayed chunk. In batch order the checking launch flags that pair and the rerun walks it
    again from its initial state; on a pool the one-launch program serves the call, since the
    first launch overwrites the initial state a rerun would read."""
    torch.manual_seed(42)
    heads = 32
    sequences = max(4, -(-get_sm_count() // heads))
    length = 6144  # Keeps every supported SM90 device in the long-prefill dispatch region.
    flags = {"state_v_first": True, "use_qk_l2norm_in_kernel": True}
    some = ["initial_state", *(["state_indices"] if pooled else [])]
    call = manifest_call(
        "KDAFwdOp", {"T": "bfloat16"}, some=some, B=sequences, S=length, H=heads, HV=heads,
        K=128, V=128, **({"P": 2 * sequences} if pooled else {}), **flags,
    )  # fmt: skip
    test = KDAFwdCallTest(call)
    inputs = list(test.gen_inputs())
    g = inputs[3].clone()
    strong = g[1, 100:228]
    strong.copy_(torch.empty(strong.shape, device=g.device).uniform_(*_strong_decays(g.dtype)))
    inputs[3] = g
    test.check(KDAFwdOp(**flags), *inputs)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_kda_refuses_a_packed_token_over_an_empty_sequence() -> None:
    """One packed token over two sequences leaves one empty: the decode reads one token and one
    state a row, so the op refuses the call instead of pairing a slot with the wrong token."""
    torch.manual_seed(42)
    test = _pooled(
        torch.bfloat16, some=("cu_seqlens",), B=1, S=1, seq_lens=[0, 1], H=2, HV=4, K=128,
        V=128, P=8, state_v_first=True,
    )  # fmt: skip
    inputs = test.gen_inputs()
    with pytest.raises(OpNotAvailableError, match="one token a sequence"):
        KDAFwdOp(state_v_first=True)(*inputs[:10], state_indices=inputs[10])
