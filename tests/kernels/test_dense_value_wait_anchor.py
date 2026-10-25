"""Unit tests for the dense prefill kernel's value-product wait anchor rewrite."""

import pytest

from tileops._kernels.attention.gqa import dense

_STEP = """\
        tl::wait_wgmma<1>();
        for (int i_11 = 0; i_11 < 64; ++i_11) {
          acc_s[i_11] = exp2f(((acc_s[i_11] * 0x1p+0f) - (sm[((i_11 & 3) >> 1)] * 0x1p+0f)));
        }
        tileops::value_wait_anchor_slot((&(((float*)anchor)[128])));
        tl::wait_wgmma<0>();"""


@pytest.mark.smoke
def test_anchor_slot_becomes_a_store_of_the_step_probabilities() -> None:
    """Each slot stores the step's first upper-row probabilities; other code stays."""
    rewritten = dense._anchor_value_waits(_STEP)

    assert "value_wait_anchor_slot" not in rewritten
    assert (
        "tileops::value_wait_anchor((float*)((&(((float*)anchor)[128]))), "
        "acc_s[0] + acc_s[1] + acc_s[4] + acc_s[5]);" in rewritten
    )
    assert rewritten.endswith("tl::wait_wgmma<0>();")
    assert rewritten.startswith("        tl::wait_wgmma<1>();")
    plain = _STEP.replace("tileops::value_wait_anchor_slot", "tileops::other_call")
    assert dense._anchor_value_waits(plain) == plain


@pytest.mark.smoke
def test_anchor_slot_without_exponentials_raises() -> None:
    """A slot with no exponential loop since its step's QK wait is refused."""
    with pytest.raises(RuntimeError, match="no exponential loop"):
        dense._anchor_value_waits(_STEP.replace("exp2f", "expf"))
