"""Attention sinks across dense prefill and decode decompositions."""

import pytest
import torch

from tileops.ops import GQADenseFwdOp
from workloads.attention.gqa.dense import GQADensePrefillWorkload
from workloads.device import run_device
from workloads.numerics import compare_outputs


@pytest.mark.sm90
@pytest.mark.parametrize(
    "shape,dtype,options",
    [
        # Two prefill consumers, rectangular masking, and both 16-bit types.
        pytest.param(
            (2, 129, 257, 8, 2, 64), torch.float16, {}, id="prefill-fp16", marks=pytest.mark.smoke
        ),
        pytest.param(
            (1, 129, 257, 8, 2, 128),
            torch.bfloat16,
            {"is_causal": False, "softcap": 2.0},
            id="prefill-bf16-softcap",
            marks=pytest.mark.smoke,
        ),
        # Left-only and two-sided windows, with partial fused rotation.
        pytest.param(
            (1, 257, 257, 8, 2, 64),
            torch.float16,
            {"window_size_left": 127},
            id="window-causal",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (1, 257, 257, 8, 2, 128),
            torch.bfloat16,
            {
                "is_causal": False,
                "window_size_left": 32,
                "window_size_right": 16,
                "rotary_dim": 64,
                "softcap": 2.0,
            },
            id="window-rope",
            marks=pytest.mark.smoke,
        ),
        # Batched unsplit/split, padded query-head rows, and batch-one decode.
        pytest.param(
            (2, 1, 17, 8, 2, 64), torch.float16, {}, id="decode-unsplit", marks=pytest.mark.smoke
        ),
        pytest.param(
            (2, 1, 257, 6, 2, 64),
            torch.bfloat16,
            {},
            id="decode-split-tail",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (1, 1, 641, 8, 2, 128), torch.float16, {}, id="decode-bs1", marks=pytest.mark.smoke
        ),
        pytest.param(
            (1, 1, 1025, 8, 2, 128),
            torch.float16,
            {"rotary_dim": 64},
            id="decode-rope-split",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (1, 1, 1025, 32, 4, 128), torch.float16, {}, id="decode-long", marks=pytest.mark.smoke
        ),
        # FP8 lane-local row sums (including the group-eight synchronization),
        # capped logits, and context-split decode.
        pytest.param(
            (1, 129, 225, 16, 2, 128),
            torch.float8_e4m3fn,
            {},
            id="fp8-prefill",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (1, 129, 225, 8, 2, 128),
            torch.float8_e4m3fn,
            {"softcap": 2.0, "rotary_dim": 64},
            id="fp8-softcap-rope",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (1, 1, 2049, 8, 2, 128),
            torch.float8_e4m3fn,
            {},
            id="fp8-decode",
            marks=pytest.mark.smoke,
        ),
        # Long FP8 PV accumulation needs FP32 promotion for biased values.
        # Exercise both softmax schedules and the first promoted extent.
        pytest.param(
            (1, 129, 3361, 8, 1, 128),
            torch.float8_e4m3fn,
            {},
            id="fp8-causal-long",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (1, 6944, 6944, 4, 1, 128),
            torch.float8_e4m3fn,
            {"is_causal": False},
            id="fp8-noncausal-promotion",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (1, 6945, 6945, 4, 1, 128),
            torch.float8_e4m3fn,
            {"is_causal": False},
            id="fp8-noncausal-long",
            marks=pytest.mark.full,
        ),
    ],
)
def test_gqa_dense_sinks(shape, dtype, options):
    """SM90 supplies the WGMMA prefill and warp-specialized decode paths."""
    workload = GQADensePrefillWorkload(
        *shape,
        dtype,
        out_dtype=torch.bfloat16 if dtype == torch.float8_e4m3fn else None,
        sm_scale=0.125,
        has_sinks=True,
        **options,
    )
    op_options = dict(options)
    if "rotary_dim" in options:
        op_options["pos_encoding_mode"] = "rope"
    op = GQADenseFwdOp(
        **op_options,
        sm_scale=workload.sm_scale,
        out_dtype=workload.out_dtype if dtype == torch.float8_e4m3fn else None,
    )
    inputs = workload.gen_inputs()
    if dtype == torch.float8_e4m3fn and shape[2] >= 3361:
        # Constant values expose accumulation loss without cancellation. The
        # value scale prevents the absolute tolerance from hiding that loss.
        inputs[0].zero_()
        inputs[1].zero_()
        inputs[2].fill_(1.5)
        inputs[5].fill_(1.25)
    # Large finite sinks must suppress output without overflow; -inf disables
    # only that head's sink. Other heads keep distinct, finite logits.
    inputs[-1][0] = -float("inf")
    inputs[-1][-1] = 1000.0
    sink_values = (inputs[-1],)
    if dtype == torch.float8_e4m3fn and not options.get("is_causal", True):
        sink_values += (None,)
    for sinks in sink_values:
        call = (*inputs[:-1], sinks)
        compare_outputs(op(*call), workload.ref_program(*call), workload.verification(*call))


@pytest.mark.sm90
@pytest.mark.smoke
def test_gqa_dense_sinks_optional_input_reuse():
    """Sink presence and values can change between calls to the same op."""
    workload = GQADensePrefillWorkload(1, 129, 257, 8, 2, 64, torch.float16, has_sinks=True)
    inputs = workload.gen_inputs()
    op = GQADenseFwdOp()
    # A strided sink checks the public contiguous-input adaptation as well.
    sinks = torch.linspace(-4, 12, 16, device=run_device(), dtype=torch.float32)[::2]
    for value in (None, sinks, -sinks, None):
        call = (*inputs[:-1], value)
        compare_outputs(op(*call), workload.ref_program(*call), workload.verification(*call))


@pytest.mark.smoke
@pytest.mark.parametrize("shape,dtype", [((2,), torch.float32), ((8,), torch.float16)])
def test_gqa_dense_sinks_rejects_invalid_input(shape, dtype):
    """The manifest requires one FP32 logit per query head, not per KV head."""
    q = torch.empty(1, 1, 8, 64, device=run_device(), dtype=torch.float16)
    k = torch.empty(1, 17, 2, 64, device=run_device(), dtype=torch.float16)
    sinks = torch.empty(shape, device=run_device(), dtype=dtype)
    with pytest.raises(ValueError, match="sinks"):
        GQADenseFwdOp()(q, k, k, sinks=sinks)
