"""Regression: ``SoftplusFwdOp`` opts out of the shared inplace path.

The parametric activation base shares an inplace path with siblings
(LeakyReLU/ELU/Hardtanh) that declare ``inplace`` in the manifest.
Softplus's manifest signature does not declare ``inplace``, so the
leaf opts out via ``_SUPPORTS_INPLACE = False``. Pin the opt-out:
flipping ``self.inplace`` after construction must not change tensor
identity or mutate the input.
"""

import pytest
import torch


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_softplus_ignores_post_construction_inplace() -> None:
    """Softplus must not honor ``op.inplace = True`` set after construction."""
    import tileops.ops.elementwise as mod

    n_total = 64
    dtype = torch.float16
    op = mod.SoftplusFwdOp(N_total=n_total, dtype=dtype)
    x = torch.randn(n_total, dtype=dtype, device="cuda")
    x_before = x.clone()

    # Baseline: default ``inplace=False`` returns a fresh tensor.
    y_default = op(x)
    assert y_default is not x
    assert torch.allclose(x, x_before), "default path must not mutate input"

    # Force the flag on; with ``_SUPPORTS_INPLACE = False`` the forward
    # flow ignores it and behavior matches the default path.
    op.inplace = True
    y_forced = op(x)
    assert y_forced is not x, (
        "SoftplusFwdOp.inplace = True must not alias the input "
        "(manifest does not declare inplace)"
    )
    assert torch.allclose(x, x_before), (
        "SoftplusFwdOp.inplace = True must not mutate the input "
        "(manifest does not declare inplace)"
    )
