"""Guard that attention ops with manifest `is_causal: true` default expose the
same default on `__init__`. Locks the C3 ctor-defaults invariant for these ops.
"""
import inspect

import pytest

from tileops.ops import (
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionFwdOp,
    GroupedQueryAttentionPrefillFwdOp,
    GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
    GroupedQueryAttentionPrefillWithKVCacheFwdOp,
    GroupedQueryAttentionSlidingWindowFwdOp,
    GroupedQueryAttentionSlidingWindowVarlenFwdOp,
    MultiHeadAttentionBwdOp,
    MultiHeadAttentionFwdOp,
)


@pytest.mark.parametrize(
    "op_cls",
    [
        pytest.param(MultiHeadAttentionFwdOp, marks=pytest.mark.smoke),
        pytest.param(MultiHeadAttentionBwdOp, marks=pytest.mark.smoke),
        pytest.param(GroupedQueryAttentionFwdOp, marks=pytest.mark.smoke),
        pytest.param(GroupedQueryAttentionBwdOp, marks=pytest.mark.smoke),
        pytest.param(GroupedQueryAttentionSlidingWindowFwdOp, marks=pytest.mark.smoke),
        pytest.param(GroupedQueryAttentionSlidingWindowVarlenFwdOp, marks=pytest.mark.smoke),
        pytest.param(GroupedQueryAttentionPrefillFwdOp, marks=pytest.mark.smoke),
        pytest.param(GroupedQueryAttentionPrefillVarlenFwdOp, marks=pytest.mark.smoke),
        pytest.param(GroupedQueryAttentionPrefillWithKVCacheFwdOp, marks=pytest.mark.smoke),
        pytest.param(
            GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
            marks=pytest.mark.smoke,
        ),
    ],
)
def test_is_causal_default_true_on_init(op_cls):
    sig = inspect.signature(op_cls.__init__)
    param = sig.parameters.get("is_causal")
    assert param is not None, f"{op_cls.__name__}: missing is_causal param"
    assert param.default is True, (
        f"{op_cls.__name__}: is_causal default = {param.default!r}, expected True")
