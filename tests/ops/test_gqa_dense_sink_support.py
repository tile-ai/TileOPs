"""Dense sinks are unavailable on SM89, for prefill and decode alike."""

import pytest
import torch

from tileops.attention import GQADenseFwdOp
from tileops.backend import BUILTIN
from workloads.device import run_device


@pytest.mark.in_tree_kernels
@pytest.mark.sm89
@pytest.mark.parametrize(
    "batch,seq_len_q,heads,heads_kv",
    [
        pytest.param(2, 1, 8, 2, id="decode", marks=pytest.mark.smoke),
        pytest.param(1, 1, 32, 4, id="long-decode", marks=pytest.mark.full),
        pytest.param(2, 16, 8, 2, id="prefill", marks=pytest.mark.full),
    ],
)
def test_gqa_dense_sinks_refused_on_sm89(
    batch: int, seq_len_q: int, heads: int, heads_kv: int
) -> None:
    """SM89 has dense kernels, but their sink paths are not supported."""
    q = torch.zeros(batch, seq_len_q, heads, 128, device=run_device(), dtype=torch.float16)
    k = torch.zeros(batch, 1024, heads_kv, 128, device=run_device(), dtype=torch.float16)
    v = torch.zeros_like(k)
    sinks = torch.zeros(heads, device=run_device(), dtype=torch.float32)
    with pytest.raises(ValueError, match="attention sinks require SM90"):
        GQADenseFwdOp(is_causal=True, target=BUILTIN)(q, k, v, sinks=sinks)
