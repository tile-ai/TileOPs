import pytest
import torch

from tileops.ops import DSAPagedFwdOp
from workloads.attention.dsa import DSAPagedCall
from workloads.numerics import Request, verify
from workloads.workload_base import manifest_call


def _call(**row):
    return manifest_call("DSAPagedFwdOp", **row)


def _check(call):
    """Compare the op with the reference; the verification's negative controls must fail.

    Every other dead slot holds a position at or past its request's cache_seqlens instead
    of -1: both are dead.
    """
    workload = DSAPagedCall(call)
    q, kv_cache, block_table, cache_seqlens, indices = workload.gen_inputs()
    length = cache_seqlens.long()[:, None, None]
    room = block_table.shape[1] * kv_cache.shape[1] - length
    flat = torch.arange(indices.numel(), device=indices.device).view_as(indices)
    past = (length + flat // 2 % room).to(indices.dtype)
    indices = torch.where((indices < 0) & (flat % 2 == 0), past, indices)
    inputs = (q, kv_cache, block_table, cache_seqlens, indices)
    verify(
        workload.ref_program,
        inputs,
        evidence=workload.verification(*inputs),
        requests={"tileops": Request(DSAPagedFwdOp(**call.arguments({})), inputs)},
    )


@pytest.mark.sm90
@pytest.mark.parametrize(
    "row",
    [
        # Cluster kernel with key splits.
        pytest.param(
            dict(S_q=1, H=128, K=256, NP=64, PS=64, W=8, cache_lens="repeat(500, 4)"),
            id="h128",
            marks=pytest.mark.smoke,
        ),
        # Single-CTA kernel, two query rows, dead trailing slots, negative scale.
        pytest.param(
            dict(
                S_q=2,
                H=64,
                K=256,
                NP=40,
                PS=64,
                W=4,
                cache_lens="balanced_sizes(1001, 6)",
                sm_scale=-0.05,
            ),
            id="h64-mtp-short",
            marks=pytest.mark.full,
        ),
        # No split; 8 blocks per CTA pair cycle every barrier phase.
        pytest.param(
            dict(S_q=1, H=128, K=512, NP=480, PS=64, W=10, cache_lens="repeat(600, 48)"),
            id="h128-unsplit",
            marks=pytest.mark.full,
        ),
        # Mostly empty splits; 32-row pages.
        pytest.param(
            dict(S_q=1, H=128, K=1024, NP=8, PS=32, W=4, cache_lens="repeat(100, 2)"),
            id="h128-empty-splits-page32",
            marks=pytest.mark.full,
        ),
    ],
)
def test_dsa_paged(row) -> None:
    _check(_call(**row))


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "row, reason",
    [
        pytest.param(
            dict(S_q=1, H=32, K=128, NP=8, PS=64, W=2, cache_lens="repeat(128, 2)"),
            "64 heads",
            id="heads-below-64",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            dict(S_q=1, H=64, K=96, NP=8, PS=64, W=2, cache_lens="repeat(128, 2)"),
            "topk",
            id="topk-not-multiple-of-64",
            marks=pytest.mark.full,
        ),
    ],
)
def test_dsa_paged_refuses(row, reason) -> None:
    """The in-tree kernel serves 64-head blocks and 64-slot key blocks only."""
    call = _call(**row)
    inputs = DSAPagedCall(call).gen_inputs()
    with pytest.raises(Exception, match=reason):
        DSAPagedFwdOp(**call.arguments({}))(*inputs)
