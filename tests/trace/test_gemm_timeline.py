"""The instrumented SM90 GEMM traced end to end through the op layer."""

import pytest
import torch

from tileops.backend import BUILTIN
from tileops.gemm import GemmFwdOp
from tileops.trace import trace
from workloads.device import run_device

pytestmark = [pytest.mark.smoke, pytest.mark.sm90, pytest.mark.in_tree_kernels]


@pytest.fixture
def trace_output(tmp_path):
    """Tracing on, dumping under *tmp_path*; the switch and the output directory are restored."""
    enabled, output = trace.enabled, trace.output
    trace.enable(str(tmp_path))
    yield tmp_path
    trace.enable(output)
    if not enabled:
        trace.disable()


def test_a_traced_gemm_matches_the_untraced_one_and_times_both_warp_groups(trace_output):
    """One instance follows the switch: warmed untraced, it runs the traced build once
    tracing is on, the untraced one again once it is off, and the traced one once more."""
    a = torch.randn(512, 512, dtype=torch.float16, device=run_device())
    b = torch.randn(512, 512, dtype=torch.float16, device=run_device())
    op = GemmFwdOp(trans_b=True, target=BUILTIN)
    trace.disable()
    untraced = op(a, b)
    trace.enable(str(trace_output))

    traced = op(a, b)
    trace.disable()
    again = op(a, b)
    trace.enable(str(trace_output))
    op(a, b)
    trace.disable()

    assert torch.equal(traced, untraced) and torch.equal(again, untraced)
    timeline, _ = sorted(trace_output.glob("*.html"))
    html = timeline.read_text()
    for lane in ("producer / tma", "producer / barrier", "consumer / barrier", "consumer / wgmma"):
        assert lane in html
    assert '"showarrow": true' in html  # the declared arrive -> wait flow
