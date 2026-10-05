"""GQA kernels' autotune spaces."""

import pytest
import torch

from tileops.kernels.attention import GQADecodeKernel
from tileops.utils import get_sm_version
from workloads.device import run_device_available


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("seqlen_kv", [1, 63, 128, 1024])
def test_gqa_decode_autotune_configs_keep_full_tiles_per_split(seqlen_kv: int) -> None:
    """Every swept config leaves each split one full KV tile; num_split=1 stays comparable."""
    if not run_device_available() or get_sm_version() not in (80, 89, 90):
        pytest.skip("GQA decode requires SM80/89/90")
    kernel = GQADecodeKernel(2, 8, 2, seqlen_kv, 128, dtype=torch.float16)
    configs = kernel.autotune_configs
    assert configs, "the sweep must stay non-empty for any positive sequence length"
    for config in configs:
        assert config["num_split"] <= max(1, seqlen_kv // config["block_N"])
    assert any(config["num_split"] == 1 for config in configs)
