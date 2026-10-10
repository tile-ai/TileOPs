"""Per-head attention sinks across packed request boundaries."""

import pytest
import torch

from tileops.ops import GQAVarlenFwdOp
from workloads.attention.gqa.varlen import GQAVarlenScaledWorkload
from workloads.numerics import compare_outputs


@pytest.mark.sm90
@pytest.mark.parametrize(
    "q_lens,kv_lens,heads,dim,dtype,options",
    [
        pytest.param(
            [129, 0, 17],
            [257, 0, 513],
            8,
            64,
            torch.float16,
            {},
            id="ragged-fp16",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            [65, 17],
            [129, 257],
            8,
            128,
            torch.bfloat16,
            {"is_causal": False, "softcap": 2.0},
            id="noncausal-softcap",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            [129, 3],
            [17, 0],
            8,
            64,
            torch.float16,
            {},
            id="fully-masked-rows",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            [17, 1], [0, 0], 8, 64, torch.bfloat16, {}, id="empty-kv", marks=pytest.mark.smoke
        ),
        pytest.param(
            [0, 0], [17, 0], 8, 64, torch.float16, {}, id="empty-q", marks=pytest.mark.smoke
        ),
        pytest.param(
            [65, 17],
            [129, 257],
            8,
            256,
            torch.float16,
            {},
            id="general-wide",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            [129, 17],
            [257, 513],
            8,
            64,
            torch.bfloat16,
            {"wl": 127},
            id="window-causal",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            [65, 17],
            [129, 257],
            8,
            128,
            torch.float16,
            {"wl": 32, "wr": 16, "is_causal": False, "rotary_dim": 64},
            id="window-rope",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            [65, 17],
            [129, 257],
            8,
            128,
            torch.bfloat16,
            {"rotary_dim": 64, "softcap": 2.0},
            id="persistent-rope",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            [17, 65],
            [225, 449],
            16,
            128,
            torch.float8_e4m3fn,
            {},
            id="fp8-group-eight",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            [17, 65],
            [225, 449],
            6,
            128,
            torch.float8_e4m3fn,
            {},
            id="fp8-general",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            [17, 65],
            [225, 449],
            8,
            128,
            torch.float8_e4m3fn,
            {"rotary_dim": 64, "softcap": 2.0},
            id="fp8-rope-softcap",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            [17, 65],
            [225, 449],
            8,
            128,
            torch.float8_e4m3fn,
            {"wl": 127},
            id="fp8-window",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            [129],
            [6944],
            8,
            128,
            torch.float8_e4m3fn,
            {"is_causal": False},
            id="fp8-long",
            marks=pytest.mark.full,
        ),
    ],
)
def test_gqa_varlen_sinks(q_lens, kv_lens, heads, dim, dtype, options):
    workload = GQAVarlenScaledWorkload(
        len(q_lens),
        q_lens,
        kv_lens,
        heads,
        2,
        dim,
        options.get("is_causal", True),
        options.get("wl", -1),
        options.get("wr", -1),
        dtype,
        softcap=options.get("softcap"),
        rotary_dim=options.get("rotary_dim"),
        out_dtype=torch.bfloat16 if dtype == torch.float8_e4m3fn else None,
        has_sinks=True,
    )
    op = GQAVarlenFwdOp(
        is_causal=workload.is_causal,
        window_size_left=workload.wl,
        window_size_right=workload.wr,
        softcap=workload.softcap,
        out_dtype=workload.out_dtype,
        pos_encoding_mode="rope" if workload.rotary_dim is not None else "none",
        rotary_dim=workload.rotary_dim,
    )
    # The long case also checks the first promoted extent and reuse across
    # the short/long factory boundary, without multiplying pytest nodes.
    lengths = ([512], [3361], kv_lens) if max(kv_lens) >= 6944 else (kv_lens,)
    for current_lengths in lengths:
        workload.seqlens_k = current_lengths
        inputs = workload.gen_inputs()
        if max(kv_lens) >= 6944:
            # Uniform weights make the accumulation loss observable. The
            # dequantization scale prevents the absolute tolerance hiding it.
            inputs[0].zero_()
            inputs[1].zero_()
            inputs[2].fill_(1.5)
            inputs[7].fill_(1.25)
        inputs[-1][0] = -float("inf")
        inputs[-1][-1] = 1000.0
        calls = (inputs, (*inputs[:-1], None)) if max(kv_lens) >= 6944 else (inputs,)
        for call in calls:
            compare_outputs(op(*call), workload.ref_program(*call), workload.verification(*call))


@pytest.mark.sm90
@pytest.mark.smoke
def test_gqa_varlen_sinks_reuse():
    """The cached Op tolerates changing sink presence, values, strides and packing."""
    workload = GQAVarlenScaledWorkload(
        2,
        [65, 17],
        [129, 257],
        8,
        2,
        64,
        True,
        -1,
        -1,
        torch.float16,
        has_sinks=True,
    )
    op = GQAVarlenFwdOp()
    for lengths in ([65, 17], [17, 65]):
        workload.seqlens_q = lengths
        inputs = workload.gen_inputs()
        sinks = inputs[-1].repeat_interleave(2)[::2]
        for value in (None, sinks, -sinks, None):
            call = (*inputs[:-1], value)
            compare_outputs(op(*call), workload.ref_program(*call), workload.verification(*call))


@pytest.mark.smoke
@pytest.mark.parametrize("invalid", ["shape", "dtype"])
def test_gqa_varlen_sinks_invalid(invalid):
    workload = GQAVarlenScaledWorkload(
        1,
        [1],
        [17],
        8,
        2,
        64,
        True,
        -1,
        -1,
        torch.float16,
        has_sinks=True,
    )
    inputs = workload.gen_inputs()
    sinks = inputs[-1][:2] if invalid == "shape" else inputs[-1].half()
    with pytest.raises(ValueError, match="sinks"):
        GQAVarlenFwdOp()(*inputs[:-1], sinks)
