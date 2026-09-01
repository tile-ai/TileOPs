"""The autograd wrappers over the DeltaNet forward and backward ops.

What a wrapper owes is the wiring: its forward is the forward op's `o`, and its backward
produces what the backward op produces from the tensors the forward saved. The numbers
themselves are checked against a torch reference in the forward and backward op's own
tests, so these do not check them twice.
"""

import pytest
import torch

from tileops.linear_attention import (
    DeltaNetAutogradOp,
    DeltaNetBwdOp,
    DeltaNetFwdOp,
    GatedDeltaNetAutogradOp,
    GatedDeltaNetBHTDFwdOp,
    GatedDeltaNetBwdOp,
)

B, H, S, DK, DV, BC = 1, 2, 256, 64, 64, 64


def _inputs(dtype: torch.dtype, gated: bool) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(42)
    scale = 0.1
    q = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * scale
    k = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * scale
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * scale
    beta = torch.rand(B, H, S, device="cuda", dtype=dtype) * 0.5
    if not gated:
        return q, k, v, beta
    g = torch.rand(B, H, S, device="cuda", dtype=dtype) * -0.1
    return q, k, v, g, beta


@pytest.mark.smoke
def test_deltanet_autograd_matches_the_ops_it_wraps() -> None:
    dtype = torch.float16
    q, k, v, beta = _inputs(dtype, gated=False)
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1

    o_ref, s, aw, au, w, u = DeltaNetFwdOp(chunk_size=BC).forward(q, k, v, beta)
    grads_ref = DeltaNetBwdOp(chunk_size=BC).forward(do, q, k, v, beta, s, aw, au, w, u)

    leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, beta)]
    o = DeltaNetAutogradOp(chunk_size=BC)(*leaves)
    o.backward(do)

    torch.testing.assert_close(o, o_ref)
    for name, leaf, ref in zip(("dq", "dk", "dv", "dbeta"), leaves, grads_ref, strict=True):
        torch.testing.assert_close(leaf.grad, ref, msg=lambda m, n=name: f"{n}: {m}")


@pytest.mark.smoke
def test_gated_deltanet_autograd_matches_the_ops_it_wraps() -> None:
    dtype = torch.float16
    q, k, v, g, beta = _inputs(dtype, gated=True)
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1

    o_ref, state = GatedDeltaNetBHTDFwdOp(chunk_size=BC).forward(q, k, v, g, beta)[:2]
    grads_ref = GatedDeltaNetBwdOp(chunk_size=BC).forward(do, q, k, v, g, beta, state)

    leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
    o = GatedDeltaNetAutogradOp(chunk_size=BC)(*leaves)
    o.backward(do)

    torch.testing.assert_close(o, o_ref)
    for name, leaf, ref in zip(("dq", "dk", "dv", "dg", "dbeta"), leaves, grads_ref, strict=True):
        torch.testing.assert_close(leaf.grad, ref, msg=lambda m, n=name: f"{n}: {m}")
