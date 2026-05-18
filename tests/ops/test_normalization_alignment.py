"""Behavior-level conformance tests for the ``normalization`` family.

These tests anchor the manifest-aligned ctor surface (``normalized_shape``,
``num_groups``, ``training``-in-ctor) and verify forward calls produce
real tensors. No ``inspect.signature`` parsing or
``scripts.validate_manifest`` re-imports — those metadata-pinning tests
were retired together with the same-PR-rename legacy aliases.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.smoke
def test_rms_norm_accepts_normalized_shape() -> None:
    from tileops.ops.norm.rms_norm import RMSNormFwdOp

    op = RMSNormFwdOp(normalized_shape=(4096,), eps=None, dtype=torch.float16)
    assert op.N == 4096
    assert op.normalized_shape == (4096,)


@pytest.mark.smoke
def test_layer_norm_accepts_normalized_shape() -> None:
    from tileops.ops.norm.layer_norm import LayerNormFwdOp

    op = LayerNormFwdOp(normalized_shape=[4096], dtype=torch.float16)
    assert op.N == 4096
    assert op.normalized_shape == (4096,)


@pytest.mark.smoke
def test_rms_norm_accepts_tuple_normalized_shape_runtime() -> None:
    """Multi-axis ``normalized_shape`` is the manifest contract; reduction
    runs over the trailing ``len(normalized_shape)`` axes and ``weight``
    must match ``tuple(normalized_shape)``."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for forward call")

    from tileops.ops.norm.rms_norm import RMSNormFwdOp

    op = RMSNormFwdOp(normalized_shape=(2, 3), dtype=torch.float16)
    assert op.N == 6
    assert op.normalized_shape == (2, 3)
    x = torch.randn(4, 2, 3, dtype=torch.float16, device="cuda")
    w = torch.randn(2, 3, dtype=torch.float16, device="cuda")
    y = op(x, w)
    assert y.shape == x.shape


@pytest.mark.smoke
def test_layer_norm_accepts_tuple_normalized_shape_runtime() -> None:
    """Multi-axis ``normalized_shape`` is the manifest contract; weight/bias
    must match ``tuple(normalized_shape)``."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for forward call")

    from tileops.ops.norm.layer_norm import LayerNormFwdOp

    op = LayerNormFwdOp(normalized_shape=(2, 3), dtype=torch.float16)
    assert op.N == 6
    assert op.normalized_shape == (2, 3)
    x = torch.randn(4, 2, 3, dtype=torch.float16, device="cuda")
    w = torch.randn(2, 3, dtype=torch.float16, device="cuda")
    b = torch.randn(2, 3, dtype=torch.float16, device="cuda")
    y = op(x, w, b)
    assert y.shape == x.shape


@pytest.mark.smoke
def test_group_norm_uses_num_groups_kwarg() -> None:
    from tileops.ops.norm.group_norm import GroupNormFwdOp

    op = GroupNormFwdOp(
        N=2, C=32, spatial=(8, 8), num_groups=8, dtype=torch.float16,
    )
    assert op.num_groups == 8
