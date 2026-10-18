import pytest

from tests.workload_test_base import TestBase
from tileops.kernels.mamba import SelectiveScanCall
from tileops.mamba import SelectiveScanFwdOp
from workloads.device import run_device
from workloads.mamba import SelectiveScanFwdCall
from workloads.workload_base import manifest_call


class SelectiveScanFwdTest(SelectiveScanFwdCall, TestBase):
    pass


_SMOKE = pytest.mark.smoke
_FULL = pytest.mark.full
_MAMBA = ["D_in", "z", "delta_bias"]

# The chunk kernel walks runs of up to 512 steps, a lane up to 16 steps of each; the step
# kernel serves up to 48 steps. One case per path through them.
_CASES = (
    pytest.param(
        2, 1024, 64, 16, 1, "bfloat16", _MAMBA, True, marks=_SMOKE, id="chunk-full-runs-bf16"
    ),
    # 256 steps: one run of 8-step lane segments.
    pytest.param(
        1, 256, 32, 16, 1, "float16", _MAMBA, True, marks=_SMOKE, id="chunk-fp16-short-runs"
    ),
    pytest.param(
        1,
        640,
        16,
        16,
        1,
        "float32",
        ["D_in", "delta_bias"],
        True,
        marks=_SMOKE,
        id="chunk-tail-run-fp32",
    ),
    # 100 steps: rows off the 16-byte vector boundary, one partial run.
    pytest.param(
        2, 100, 16, 16, 1, "bfloat16", _MAMBA, True, marks=_SMOKE, id="chunk-unaligned-rows"
    ),
    pytest.param(
        2, 1024, 32, 16, 1, "bfloat16", [*_MAMBA, "seq_idx"], True, marks=_SMOKE, id="chunk-packed"
    ),
    pytest.param(
        2,
        600,
        32,
        16,
        1,
        "bfloat16",
        [*_MAMBA, "initial_states"],
        True,
        marks=_SMOKE,
        id="chunk-carried-state",
    ),
    pytest.param(
        1, 512, 32, 16, 1, "bfloat16", ["z"], False, marks=_SMOKE, id="chunk-bare-no-softplus"
    ),
    pytest.param(
        4,
        1,
        64,
        16,
        1,
        "bfloat16",
        [*_MAMBA, "initial_states"],
        True,
        marks=_SMOKE,
        id="step-decode",
    ),
    pytest.param(
        2,
        1,
        32,
        16,
        1,
        "float32",
        ["D_in", "delta_bias", "initial_states"],
        True,
        marks=_SMOKE,
        id="step-decode-fp32",
    ),
    # 48 states: a second, partial slot of lane-held states; two groups of B and C.
    pytest.param(
        2, 512, 64, 48, 2, "bfloat16", _MAMBA, True, marks=_FULL, id="chunk-two-slots-two-groups"
    ),
    pytest.param(
        1, 512, 16, 8, 1, "bfloat16", _MAMBA, True, marks=_FULL, id="chunk-fewer-states-than-lanes"
    ),
    pytest.param(1, 49, 16, 16, 1, "bfloat16", _MAMBA, True, marks=_FULL, id="chunk-shortest"),
    pytest.param(
        2,
        48,
        32,
        16,
        1,
        "float16",
        [*_MAMBA, "seq_idx"],
        True,
        marks=_FULL,
        id="step-longest-packed",
    ),
    # 12 states do not split over a power of two of lanes: one thread holds all of them.
    pytest.param(
        3,
        4,
        16,
        12,
        1,
        "bfloat16",
        ["z", "initial_states"],
        False,
        marks=_FULL,
        id="step-unsplit-states",
    ),
    pytest.param(2, 8, 64, 32, 4, "bfloat16", _MAMBA, True, marks=_FULL, id="step-groups"),
)


@pytest.mark.parametrize("batch, seqlen, dim, dstate, groups, dtype, some, softplus", _CASES)
def test_selective_scan(batch, seqlen, dim, dstate, groups, dtype, some, softplus) -> None:
    call = manifest_call(
        "SelectiveScanFwdOp",
        {"T": dtype},
        B=batch,
        S=seqlen,
        D=dim,
        N=dstate,
        G=groups,
        delta_softplus=softplus,
        some=some,
    )
    test = SelectiveScanFwdTest(call)
    op = SelectiveScanFwdOp(**call.arguments({}))
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("seqlen, key", [(48, "selective_scan_step"), (49, "selective_scan_chunk")])
def test_selective_scan_dispatch_range(seqlen, key) -> None:
    """The step kernel serves sequences up to its length; the chunk kernel serves the rest."""
    call = SelectiveScanCall(dim=64, seqlen=seqlen, dstate=16, groups=1, device=run_device())
    assert SelectiveScanFwdOp().key_for("selective_scan", call) == key
