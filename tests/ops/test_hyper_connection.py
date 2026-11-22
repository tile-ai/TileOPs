"""Public semantics of the manifold hyper-connection ops. Pre: each token count's path,
leading shapes, a broadcast residual, one kernel serving every token count, capture and
replay, and the calls no kernel serves. Post: each launch program, ragged columns, vector
widths, stream counts and the mix dtypes. Head: each program, reached from the device's SM
count, ragged columns, vector widths and stream counts."""

import pytest
import torch

from tests.workload_test_base import TestBase, smoke_first
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.sequence_modeling import HyperConnectionPreCall as PreCallSpec
from tileops.backend._dispatch import device_facts_of
from tileops.sequence_modeling import (
    HyperConnectionHeadFwdOp,
    HyperConnectionPostFwdOp,
    HyperConnectionPreFwdOp,
)
from workloads.device import run_device
from workloads.sequence_modeling import (
    HyperConnectionHeadWorkload,
    HyperConnectionPostWorkload,
    HyperConnectionPreCall,
)
from workloads.workload_base import manifest_call


class HyperConnectionPreTest(HyperConnectionPreCall, TestBase):
    pass


class HyperConnectionPostTest(HyperConnectionPostWorkload, TestBase):
    pass


class HyperConnectionHeadTest(HyperConnectionHeadWorkload, TestBase):
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
    # 128-token tiles, the last one partial.
    _case([300], _FULL, "128-token-tiles"),
    # Splits that do not divide the 256 steps, six on 132 SMs.
    _case([2500], _FULL, "uneven-splits"),
    # Tiles enough to fill the SMs unsplit.
    _case([8500], _FULL, "one-split"),
    # The softmax and one column normalization.
    _case([5], _FULL, "one-sinkhorn-iteration", sinkhorn_repeat=1),
    # Streams staged 128 channels at a time, and one step a split.
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


@pytest.mark.sm90
@pytest.mark.cuda_graph(op="HyperConnectionPreFwdOp")
@pytest.mark.smoke
def test_hyper_connection_pre_can_be_captured_and_replayed() -> None:
    """Two decode calls captured back to back consume changed inputs at their captured
    addresses: each mix launch, which may start while its projection drains, reads that
    replay's splits, and the second projection does not overwrite the first's early."""
    call = _call([1], _DSV4)
    test = HyperConnectionPreTest(call)
    op = HyperConnectionPreFwdOp(**call.arguments())
    first = test.gen_inputs()
    second = tuple(t.clone() for t in first)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(*first)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = op(*first), op(*second)
    first[0].add_(0.5)
    second[1].mul_(0.5)
    second[3].add_(0.25)
    for output in outputs:
        for tensor in output:
            tensor.fill_(float("nan"))
    graph.replay()
    test.check(op, *first, runs=lambda *args: outputs[0])
    test.check(op, *second, runs=lambda *args: outputs[1])


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "n, c, params, reason",
    [
        (3, 4096, _DSV4, "serves four streams"),
        (4, 4000, _DSV4, "multiple of 128"),
        (4, 4096, {**_DSV4, "sinkhorn_repeat": 0}, "at least one Sinkhorn iteration"),
    ],
    ids=["three-streams", "4000-channels", "no-sinkhorn-iteration"],
)
def test_hyper_connection_pre_refuses_what_no_kernel_serves(n, c, params, reason) -> None:
    from tileops.backend import OpNotAvailableError

    call = _call([2], params, c, n)
    op = HyperConnectionPreFwdOp(**call.arguments())
    with pytest.raises(OpNotAvailableError, match=reason):
        op(*HyperConnectionPreTest(call).gen_inputs())
    assert not op._entries_built("hyper_connection_pre")


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_hyper_connection_pre_fits_its_projection_ring_to_shared_memory() -> None:
    """A block that takes less shared memory than six 128-token projection slots need is
    still served, and one that cannot hold two is refused before a build."""
    from tileops.backend import OpNotAvailableError

    (in_tree,) = Kernel.devices

    def call(smem_budget):
        return PreCallSpec(
            arch=90, sm_count=132, smem_budget=smem_budget, n=4, c=4096, **_DSV4,
            device=torch.device(in_tree),
        )  # fmt: skip

    op = HyperConnectionPreFwdOp(**_DSV4)
    op.key_for("hyper_connection_pre", call(160 * 1024))
    with pytest.raises(OpNotAvailableError, match="past the 49152"):
        op.key_for("hyper_connection_pre", call(48 * 1024))


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


def _head_tokens(program: str) -> int:
    """A token count reaching *program* on the run device, whose SM count picks it: tokens a
    CTA held in registers, or streamed. Above one token, the last CTA holds one token."""
    sms = device_facts_of(torch.device(run_device())).sm_count
    return {
        "resident-1": 1,
        "resident-2": 2 * sms - 1,
        "streamed-2": 4 * sms + 1,
        "streamed-4": 4 * (5 * sms // 4 + 1) + 1,
    }[program]


@pytest.mark.parametrize(
    "program, n, c",
    [
        # DeepSeek-V4-Flash decoding one token.
        pytest.param("resident-1", 4, 4096, marks=pytest.mark.smoke),
        pytest.param("resident-2", 4, 4096, marks=pytest.mark.full),
        pytest.param("streamed-2", 4, 4096, marks=pytest.mark.full),
        pytest.param("streamed-4", 4, 4096, marks=pytest.mark.full),
        # Columns no thread count divides, in both kernels.
        pytest.param("resident-1", 4, 1000, marks=pytest.mark.full),
        pytest.param("streamed-4", 4, 1000, marks=pytest.mark.full),
        # Four column vectors a thread held; two tokens of them are past the register budget,
        # so that count streams one token a CTA.
        pytest.param("resident-1", 4, 7168, marks=pytest.mark.full),
        pytest.param("resident-2", 4, 7168, marks=pytest.mark.full),
        # Columns read 2 and 1 a vector.
        pytest.param("resident-1", 4, 998, marks=pytest.mark.full),
        pytest.param("streamed-4", 4, 999, marks=pytest.mark.full),
        # Other stream counts: none, one, three, and the largest served.
        pytest.param("resident-1", 0, 64, marks=pytest.mark.full),
        pytest.param("resident-1", 1, 256, marks=pytest.mark.full),
        pytest.param("streamed-4", 3, 384, marks=pytest.mark.full),
        pytest.param("resident-1", 8, 4096, marks=pytest.mark.full),
    ],
)
def test_hyper_connection_head_op(program, n, c) -> None:
    test = HyperConnectionHeadTest((_head_tokens(program),), n, c)
    test.check(HyperConnectionHeadFwdOp(1e-6, 1e-6), *test.gen_inputs())


@pytest.mark.smoke
def test_hyper_connection_head_folds_leading_axes() -> None:
    test = HyperConnectionHeadTest((2, 3), 4, 512)
    test.check(HyperConnectionHeadFwdOp(1e-6, 1e-6), *test.gen_inputs())


@pytest.mark.smoke
def test_hyper_connection_head_reads_strided_inputs() -> None:
    """Views that are not contiguous: every other token of the streams and every other
    stream bias, and hc_fn stored transposed."""
    test = HyperConnectionHeadTest((6,), 4, 512)
    hidden_states, hc_fn, hc_scale, hc_base = test.gen_inputs()
    device = run_device()
    wide = torch.randn(12, 4, 512, dtype=hidden_states.dtype, device=device)
    wide[::2] = hidden_states
    base_wide = torch.randn(8, device=device)
    base_wide[::2] = hc_base
    hc_fn_t = hc_fn.t().contiguous().t()
    views = (wide[::2], hc_fn_t, hc_scale, base_wide[::2])
    assert not any(view.is_contiguous() for view in views if view is not hc_scale)
    test.check(HyperConnectionHeadFwdOp(1e-6, 1e-6), *views)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "hidden_dtype, fn_dtype, reason",
    [
        (torch.float16, torch.float32, "hidden_states dtype is not bfloat16"),
        (torch.bfloat16, torch.bfloat16, "hc_fn dtype is not float32"),
    ],
    ids=["float16-streams", "bfloat16-hc-fn"],
)
def test_hyper_connection_head_rejects_dtypes_outside_the_signature(
    hidden_dtype, fn_dtype, reason
) -> None:
    device = run_device()
    hidden_states = torch.randn(2, 4, 64, dtype=hidden_dtype, device=device)
    hc_fn = torch.randn(4, 256, device=device).to(fn_dtype)
    hc_scale = torch.ones(1, device=device)
    hc_base = torch.zeros(4, device=device)
    with pytest.raises(ValueError, match=reason):
        HyperConnectionHeadFwdOp(1e-6, 1e-6)(hidden_states, hc_fn, hc_scale, hc_base)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_hyper_connection_head_serves_new_token_counts_from_one_kernel() -> None:
    """Token counts reaching the resident and the streamed programs share one kernel."""
    op = HyperConnectionHeadFwdOp(1e-6, 1e-6)
    for tokens in (3, _head_tokens("streamed-4")):
        test = HyperConnectionHeadTest((tokens,), 4, 1024)
        test.check(op, *test.gen_inputs())
    assert len(list(op.iter_kernels())) == 1


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_hyper_connection_head_refuses_streams_past_the_register_budget() -> None:
    """Nine streams give a thread hc_fn vectors of nine streams to hold."""
    from tileops.backend import OpNotAvailableError

    test = HyperConnectionHeadTest((2,), 9, 64)
    with pytest.raises(OpNotAvailableError, match="register budget"):
        HyperConnectionHeadFwdOp(1e-6, 1e-6)(*test.gen_inputs())
