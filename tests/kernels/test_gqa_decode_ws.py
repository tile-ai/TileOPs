"""Host-side split and overlap selection for warp-specialized GQA decode."""

import pytest
import torch

from tileops._kernels.attention.gqa.decode_ws import GQADecodeWSKernel


@pytest.mark.parametrize(
    "shape, options, plan",
    [
        pytest.param((16, 8, 4096, 64, 8, 128), {}, (1, True), id="alone", marks=pytest.mark.smoke),
        pytest.param(
            (16, 8, 4160, 64, 8, 128), {}, (2, False), id="past-64-tiles", marks=pytest.mark.smoke
        ),
        pytest.param(
            (8, 8, 8192, 64, 8, 128), {}, (2, True), id="alone-two-splits", marks=pytest.mark.full
        ),
        pytest.param(
            (8, 9, 4096, 64, 8, 128), {}, (2, False), id="two-row-blocks", marks=pytest.mark.full
        ),
        pytest.param(
            (17, 8, 4096, 64, 8, 128),
            {},
            (1, False),
            id="more-units-than-sms",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (2, 4, 1025, 32, 8, 128), {}, (4, True), id="tile-capped-grid", marks=pytest.mark.full
        ),
        pytest.param(
            (5, 1, 8192, 32, 8, 128),
            {},
            (4, False),
            id="under-two-thirds-of-sms",
            marks=pytest.mark.full,
        ),
        # Three key tiles: one split holds every tile, so the overlapped program serves.
        pytest.param(
            (16, 8, 129, 64, 8, 128), {}, (1, True), id="short-context", marks=pytest.mark.full
        ),
        # No key tile: the overlapped consumer would wait on a tile no producer loads.
        pytest.param((16, 8, 0, 64, 8, 128), {}, (1, False), id="no-keys", marks=pytest.mark.full),
        pytest.param((16, 8, 4096, 64, 8, 256), {}, (1, False), id="d256", marks=pytest.mark.full),
        pytest.param(
            (16, 8, 4096, 64, 8, 128),
            {"softcap": 2.0},
            (2, False),
            id="softcap",
            marks=pytest.mark.full,
        ),
    ],
)
def test_decode_ws_plan(shape: tuple, options: dict, plan: tuple) -> None:
    """Each case guards a split or overlap boundary on a 132-SM device."""
    batch, seq_len_q, seq_len_kv, heads, groups, dim = shape
    kernel = object.__new__(GQADecodeWSKernel)
    kernel.batch, kernel.seq_len_q = batch, seq_len_q
    kernel.heads, kernel.groups, kernel.dim = heads, groups, dim
    kernel.softcap = options.get("softcap", 0.0)
    kernel.dtype = torch.float16
    kernel.config = {"block_m": 64, "block_n": 64}
    kernel._sm_count, kernel._smem_budget = 132, 232448
    assert kernel.plan(seq_len_kv) == plan
