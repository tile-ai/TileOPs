import pytest
import torch

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


@pytest.mark.parametrize(
    "packed, carried, sign, drive, rate",
    [
        pytest.param(True, False, 1.0, 0.0, 6.0, marks=_SMOKE, id="restart"),
        pytest.param(False, False, 1.0, 0.0, 6.0, marks=_FULL, id="zero-state"),
        pytest.param(False, True, 1.0, 0.0, 6.0, marks=_FULL, id="carried-state"),
        pytest.param(False, True, -1.0, 0.0, 6.0, marks=_FULL, id="carried-state-negative-steps"),
        pytest.param(True, False, 1.0, 1e-30, 6.0, marks=_FULL, id="overflowed-state-restart"),
        # A zero rate: the input alone overflows the state, on the shuffle scan's path.
        pytest.param(True, False, 1.0, 1e38, 0.0, marks=_FULL, id="driven-overflow-restart"),
    ],
)
def test_selective_scan_growing_rates(packed, carried, sign, drive, rate) -> None:
    """Regression: where ``dt * A > 0``, a rising rate or, without the softplus, a negative
    step under a falling rate, the decay product over a run overflows float32 before the
    state it scales does; and a restart follows a state its input overflowed. The scan must
    keep the step-by-step recurrence's values: zero for a zero state, a carried state finite
    until the recurrence itself overflows, and every step after a restart exact, not NaN."""
    some = (["seq_idx"] if packed else []) + (["initial_states"] if carried else [])
    call = manifest_call(
        "SelectiveScanFwdOp",
        {"T": "float32"},
        B=1,
        S=257,
        D=8,
        N=1,
        G=1,
        delta_softplus=False,
        some=some,
    )
    test = SelectiveScanFwdTest(call)
    u, delta, A, B_in, C_in, D_in, z, delta_bias, seq_idx, initial_states = test.gen_inputs()
    # exp(6) a step: a lane's 16-step segment grows by exp(96), past float32. A state of
    # 1e-10 overflows at step 19; one driven by 1e-30 a step, at step 26; one driven by
    # 1e38 a step under a zero rate, at step 2.
    u = torch.ones_like(u)
    delta = torch.full_like(delta, sign)
    A = torch.full_like(A, rate * sign)
    B_in = torch.full_like(B_in, drive)
    C_in = torch.ones_like(C_in)
    if packed:
        # A sequence starts at step 1, or, driven, at step 100, after the state overflows.
        seq_idx = torch.ones_like(seq_idx)
        seq_idx[:, : 100 if drive else 1] = 0
    if carried:
        initial_states = torch.full_like(initial_states, 1e-10)
    op = SelectiveScanFwdOp(**call.arguments({}))
    test.check(op, u, delta, A, B_in, C_in, D_in, z, delta_bias, seq_idx, initial_states)


@pytest.mark.smoke
@pytest.mark.parametrize("seqlen, key", [(48, "selective_scan_step"), (49, "selective_scan_chunk")])
def test_selective_scan_dispatch_range(seqlen, key) -> None:
    """The step kernel serves sequences up to its length; the chunk kernel serves the rest."""
    call = SelectiveScanCall(dim=64, seqlen=seqlen, dstate=16, groups=1, device=run_device())
    assert SelectiveScanFwdOp().key_for("selective_scan", call) == key
