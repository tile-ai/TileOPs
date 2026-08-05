"""One op instance, two element types.

Dtype used to be fixed at construction, so an instance served exactly one
element type and per-instance state could safely hold anything derived from
it. Now the tensors decide, and the family caches are hand-written per
family — this covers the invariant they all have to satisfy.

The rest of the suite parametrizes dtype but builds a fresh op per case, so
it never exercises a second dtype through an instance that already has an
entry.
"""

import pytest
import torch

from tileops.ops.norm.layer_norm import LayerNormFwdOp
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from tileops.ops.reduction.reduce import SumFwdOp

_DTYPES = (torch.float16, torch.bfloat16)


def _assert_two_entries(op, cache_name="_kernel_cache"):
    cache = getattr(op, cache_name)
    assert len(cache) == 2, f"expected one entry per dtype, got {list(cache)}"
    kernels = list(cache.values())
    assert kernels[0] is not kernels[1], "both dtypes reused one kernel"


@pytest.mark.smoke
def test_reduction_serves_two_dtypes_from_one_instance():
    op = SumFwdOp(dim=-1)
    for dtype in _DTYPES:
        x = torch.randn(8, 128, dtype=dtype, device="cuda")
        y = op(x)
        assert y.dtype == dtype
        torch.testing.assert_close(y, x.sum(-1), atol=2e-2, rtol=2e-2)
    _assert_two_entries(op)


@pytest.mark.smoke
def test_rms_norm_serves_two_dtypes_from_one_instance():
    n = 256
    op = RMSNormFwdOp(normalized_shape=(n,))
    for dtype in _DTYPES:
        x = torch.randn(16, n, dtype=dtype, device="cuda")
        w = torch.randn(n, dtype=dtype, device="cuda")
        y = op(x, w)
        assert y.dtype == dtype
    _assert_two_entries(op)


@pytest.mark.smoke
def test_layer_norm_keys_on_both_shape_and_dtype():
    """A second dtype at the same shape must not reuse the first entry."""
    n = 256
    op = LayerNormFwdOp(normalized_shape=(n,))
    for dtype in _DTYPES:
        x = torch.randn(16, n, dtype=dtype, device="cuda")
        w = torch.randn(n, dtype=dtype, device="cuda")
        b = torch.randn(n, dtype=dtype, device="cuda")
        assert op(x, w, b).dtype == dtype
    assert set(op._kernel_cache) == {(16, dt) for dt in _DTYPES}


@pytest.mark.smoke
def test_roofline_reports_the_most_recent_forward():
    """`self.dtype` is most-recent-forward, so bytes follow the last call."""
    op = SumFwdOp(dim=-1)
    op(torch.randn(8, 128, dtype=torch.float32, device="cuda"))
    _, bytes_fp32 = op.eval_roofline()
    op(torch.randn(8, 128, dtype=torch.float16, device="cuda"))
    _, bytes_fp16 = op.eval_roofline()
    assert bytes_fp16 < bytes_fp32


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
