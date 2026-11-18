"""Public HyperConnectionPreFwdOp semantics: each token count's path, leading shapes, a
broadcast residual, and the DeepSeek-V4-Flash and GLM-5.3-Flash constants."""

import pytest

from tests.workload_test_base import TestBase, smoke_first
from tileops.sequence_modeling import HyperConnectionPreFwdOp
from workloads.sequence_modeling import HyperConnectionPreCall
from workloads.workload_base import manifest_call


class HyperConnectionPreTest(HyperConnectionPreCall, TestBase):
    pass


_SMOKE = pytest.mark.smoke
_FULL = pytest.mark.full
_NIGHTLY = pytest.mark.nightly
_DSV4 = {
    "rms_eps": 1e-6,
    "hc_pre_eps": 1e-6,
    "hc_sinkhorn_eps": 1e-6,
    "hc_post_mult_value": 2.0,
    "sinkhorn_repeat": 20,
}


def _case(leading, mark, case_id, c=4096, **params):
    return pytest.param(leading, c, {**_DSV4, **params}, marks=mark, id=case_id)


def _call(leading, params, c=4096):
    return manifest_call("HyperConnectionPreFwdOp", S=leading, n=4, C=c, **params)


_CASES = (
    # One token: a 64-token tile, the reduction split 128 ways.
    _case([1], _SMOKE, "decode"),
    # A leading shape of two axes over a partial 64-token tile.
    _case([2, 31], _FULL, "leading-dims"),
    # 128-token tiles, the last one partial, fn split in the projection.
    _case([300], _FULL, "128-token-tiles"),
    # Past 512 tokens, behind the launch that splits fn.
    _case([2500], _FULL, "split-launch"),
    _case([3000], _FULL, "glm-5.3-flash", rms_eps=1e-5),
    # The softmax and one column normalization.
    _case([5], _FULL, "one-sinkhorn-iteration", sinkhorn_repeat=1),
    # Streams staged 128 channels at a time.
    _case([5], _FULL, "640-channels", c=640),
    _case([32768], _NIGHTLY, "dsv4-flash-prefill"),
)


@pytest.mark.parametrize("leading, c, params", smoke_first(list(_CASES)))
def test_hyper_connection_pre(leading, c, params) -> None:
    call = _call(leading, params, c)
    test = HyperConnectionPreTest(call)
    test.check(HyperConnectionPreFwdOp(**call.arguments()), *test.gen_inputs())


@pytest.mark.parametrize(
    "leading", [pytest.param([9], marks=_SMOKE), pytest.param([3000], marks=_FULL)]
)
def test_hyper_connection_pre_reads_a_broadcast_residual(leading) -> None:
    """The first block's residual, one stream expanded to all four along a stride-0 axis,
    gives what its materialized streams give."""
    call = _call(leading, _DSV4)
    test = HyperConnectionPreTest(call)
    residual, fn, hc_scale, hc_base = test.gen_inputs()
    residual = residual[..., :1, :].expand_as(residual)
    test.check(HyperConnectionPreFwdOp(**call.arguments()), residual, fn, hc_scale, hc_base)
