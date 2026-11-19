"""Public HyperConnectionPreFwdOp semantics: each token count's path, leading shapes, a
broadcast residual, one kernel serving every token count, and the calls no kernel serves."""

import pytest

from tests.workload_test_base import TestBase, smoke_first
from tileops.sequence_modeling import HyperConnectionPreFwdOp
from workloads.sequence_modeling import HyperConnectionPreCall
from workloads.workload_base import manifest_call


class HyperConnectionPreTest(HyperConnectionPreCall, TestBase):
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
