"""Norm kernels validate the config they are built with."""

import pytest
import torch

from tileops.kernels.norm import FusedAddRMSNormKernel


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_fused_add_rms_norm_rejects_partial_access_width() -> None:
    """A width leaving a partial 16-byte access is refused at construction.

    It truncates the kernel's per-CTA loop bounds to zero, so the row comes back
    untouched rather than wrong in a way a tolerance would catch.
    """
    with pytest.raises(ValueError, match="whole 16-byte accesses"):
        FusedAddRMSNormKernel(4096, 1e-6, torch.bfloat16, config={"threads": 768})
