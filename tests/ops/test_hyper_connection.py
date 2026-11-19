"""Public semantics of the manifold hyper-connection ops. Pre: each token count's path,
leading shapes, a broadcast residual, one kernel serving every token count, and the calls no
kernel serves. Post: each launch program, ragged columns, vector widths, stream counts and the
mix dtypes."""

import pytest
import torch

from tests.workload_test_base import TestBase, smoke_first
from tileops.sequence_modeling import HyperConnectionPostFwdOp, HyperConnectionPreFwdOp
from workloads.device import run_device
from workloads.sequence_modeling import HyperConnectionPostWorkload, HyperConnectionPreCall
from workloads.workload_base import manifest_call


class HyperConnectionPreTest(HyperConnectionPreCall, TestBase):
    pass


class HyperConnectionPostTest(HyperConnectionPostWorkload, TestBase):
    pass


_SMOKE = pytest.mark.smoke
_FULL = pytest.mark.full
_DSV4 = {
    "rms_eps": 1e-6,
    "hc_pre_eps": 1e-6,
    "hc_sinkhorn_eps": 1e-6,
    "hc_post_mult_value": 2.0,
    "sinkhorn_repeat": 20,
}


def _case(leading, mark, case_id, c=4096, **params):
    return pytest.param(leading, c, {**_DSV4, **params}, marks=mark, id=case_id)


def _call(leading, params, c=4096, n=4):
    return manifest_call("HyperConnectionPreFwdOp", S=leading, n=n, C=c, **params)


_CASES = (
    # One token: a 64-token tile, the reduction split 128 ways.
    _case([1], _SMOKE, "decode"),
    # A leading shape of two axes over a partial 64-token tile.
    _case([2, 31], _FULL, "leading-dims"),
    # 128-token tiles, the last one partial, fn split in the projection.
    _case([300], _FULL, "128-token-tiles"),
    # Past 512 tokens, behind the launch that splits fn.
    _case([2500], _FULL, "split-launch"),
    # Tiles enough to fill the SMs unsplit, mixed from registers past the staging bound.
    _case([8500], _FULL, "one-split"),
    # The softmax and one column normalization.
    _case([5], _FULL, "one-sinkhorn-iteration", sinkhorn_repeat=1),
    # Streams staged 128 channels at a time.
    _case([5], _FULL, "640-channels", c=640),
)


@pytest.mark.parametrize("leading, c, params", smoke_first(list(_CASES)))
def test_hyper_connection_pre(leading, c, params) -> None:
    call = _call(leading, params, c)
    test = HyperConnectionPreTest(call)
    test.check(HyperConnectionPreFwdOp(**call.arguments()), *test.gen_inputs())


@pytest.mark.parametrize(
    "leading", [pytest.param([1], marks=_SMOKE), pytest.param([8500], marks=_FULL)]
)
def test_hyper_connection_pre_reads_a_broadcast_residual(leading) -> None:
    """The first block's residual, one stream expanded to all four along a stride-0 axis,
    gives what its materialized streams give."""
    call = _call(leading, _DSV4)
    test = HyperConnectionPreTest(call)
    residual, fn, hc_scale, hc_base = test.gen_inputs()
    residual = residual[..., :1, :].expand_as(residual)
    test.check(HyperConnectionPreFwdOp(**call.arguments()), residual, fn, hc_scale, hc_base)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "token_counts, program_sets",
    [
        pytest.param((1, 9), 1, marks=_SMOKE, id="one-launch"),
        pytest.param((3, 70), 2, marks=_FULL, id="two-launches"),
    ],
)
def test_hyper_connection_pre_serves_new_token_counts_from_one_kernel(
    token_counts, program_sets
) -> None:
    """Token counts of one launch share its programs, and one of 128-token tiles adds its own
    to the same kernel."""
    op = HyperConnectionPreFwdOp(**_DSV4)
    for tokens in token_counts:
        test = HyperConnectionPreTest(_call([tokens], _DSV4))
        test.check(op, *test.gen_inputs())
    (kernel,) = op.iter_kernels()
    assert len(kernel._generation.programs) == program_sets


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "n, c, params, reason",
    [
        (3, 4096, _DSV4, "serves four streams"),
        (4, 4000, _DSV4, "multiple of 128"),
        (4, 0, _DSV4, "multiple of 128"),
        (4, 4096, {**_DSV4, "sinkhorn_repeat": 0}, "at least one Sinkhorn iteration"),
    ],
    ids=["three-streams", "4000-channels", "no-channels", "no-sinkhorn-iteration"],
)
def test_hyper_connection_pre_refuses_what_no_kernel_serves(n, c, params, reason) -> None:
    from tileops.backend import OpNotAvailableError

    call = _call([2], params, c, n)
    op = HyperConnectionPreFwdOp(**call.arguments())
    with pytest.raises(OpNotAvailableError, match=reason):
        op(*HyperConnectionPreTest(call).gen_inputs())
    assert not op._entries_built("hyper_connection_pre")


@pytest.mark.parametrize(
    "shape, n, c, mix_dtype",
    [
        # DeepSeek-V4-Flash decoding one token, and GLM-5.3-Flash's bfloat16 mixes.
        pytest.param((1,), 4, 4096, torch.float32, marks=pytest.mark.smoke),
        pytest.param((37,), 4, 4096, torch.bfloat16, marks=pytest.mark.smoke),
        # Enough tokens for the program giving a thread four runs.
        pytest.param((600,), 4, 4096, torch.float32, marks=pytest.mark.full),
        # A last CTA of a row past its columns: 7168 columns in 4-run CTAs of 4096.
        pytest.param((500,), 4, 7168, torch.float32, marks=pytest.mark.full),
        # Columns read 2 and 1 a vector.
        pytest.param((4,), 4, 998, torch.float32, marks=pytest.mark.full),
        pytest.param((4,), 4, 999, torch.bfloat16, marks=pytest.mark.full),
        # Folded leading axes.
        pytest.param((2, 5), 4, 512, torch.float32, marks=pytest.mark.full),
        # Other stream counts: one, a mix row not a power of two, and the largest served.
        pytest.param((3,), 1, 256, torch.float32, marks=pytest.mark.full),
        pytest.param((3,), 3, 384, torch.bfloat16, marks=pytest.mark.full),
        pytest.param((9,), 8, 4096, torch.float32, marks=pytest.mark.full),
    ],
)
def test_hyper_connection_post_op(shape, n, c, mix_dtype) -> None:
    test = HyperConnectionPostTest(shape, n, c, mix_dtype)
    test.check(HyperConnectionPostFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
def test_hyper_connection_post_reads_strided_inputs() -> None:
    """Views that are not contiguous: every other token of x and of the residual streams, and
    mixes stored transposed."""
    test = HyperConnectionPostTest((6,), 4, 512)
    x, residual, post, comb = test.gen_inputs()
    device = run_device()
    x_wide = torch.randn(12, 512, dtype=x.dtype, device=device)
    x_wide[::2] = x
    residual_wide = torch.randn(12, 4, 512, dtype=residual.dtype, device=device)
    residual_wide[::2] = residual
    comb_t = comb.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert not (x_wide[::2].is_contiguous() or comb_t.is_contiguous())
    test.check(HyperConnectionPostFwdOp(), x_wide[::2], residual_wide[::2], post, comb_t)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "x_dtype, post_dtype, comb_dtype, reason",
    [
        (torch.float16, torch.float32, torch.float32, "x dtype is not bfloat16"),
        (torch.bfloat16, torch.float16, torch.float16, "post_layer_mix dtype is outside"),
        (torch.bfloat16, torch.float32, torch.bfloat16, "comb_res_mix dtype differs"),
    ],
    ids=["float16-streams", "float16-mixes", "mixed-mixes"],
)
def test_hyper_connection_post_rejects_dtypes_outside_the_signature(
    x_dtype, post_dtype, comb_dtype, reason
) -> None:
    device = run_device()
    x = torch.randn(2, 64, dtype=x_dtype, device=device)
    residual = torch.randn(2, 4, 64, dtype=x_dtype, device=device)
    post = torch.rand(2, 4, 1, device=device).to(post_dtype)
    comb = torch.rand(2, 4, 4, device=device).to(comb_dtype)
    with pytest.raises(ValueError, match=reason):
        HyperConnectionPostFwdOp()(x, residual, post, comb)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_hyper_connection_post_serves_new_token_counts_from_one_kernel() -> None:
    op = HyperConnectionPostFwdOp()
    for tokens in (3, 70):
        test = HyperConnectionPostTest((tokens,), 4, 1024)
        test.check(op, *test.gen_inputs())
    assert len(list(op.iter_kernels())) == 1


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_hyper_connection_post_refuses_streams_past_the_register_budget() -> None:
    """Nine streams give a thread 81 comb_res_mix entries to hold."""
    from tileops.backend import OpNotAvailableError

    test = HyperConnectionPostTest((2,), 9, 64)
    with pytest.raises(OpNotAvailableError, match="register budget"):
        HyperConnectionPostFwdOp()(*test.gen_inputs())
