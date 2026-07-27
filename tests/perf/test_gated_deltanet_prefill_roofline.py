"""Cross-layout contract for the Gated DeltaNet prefill roofline."""

import pytest

from tileops.perf.formulas import gated_deltanet_prefill_fwd_roofline

pytestmark = pytest.mark.smoke


def test_gated_deltanet_prefill_roofline_layout_equivalence() -> None:
    """bthd and bhtd bindings of the same problem yield identical costs."""
    bthd = gated_deltanet_prefill_fwd_roofline(
        q_shape=[1, 512, 16, 128],
        v_shape=[1, 512, 16, 128],
        chunk_size=64,
        layout="bthd",
        dtype="float16",
    )
    bhtd = gated_deltanet_prefill_fwd_roofline(
        q_shape=[1, 16, 512, 128],
        v_shape=[1, 16, 512, 128],
        chunk_size=64,
        layout="bhtd",
        dtype="float16",
    )
    assert bthd == bhtd
    assert bthd[0] > 0 and bthd[1] > 0
