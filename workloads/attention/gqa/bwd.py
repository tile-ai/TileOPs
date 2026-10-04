import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from workloads.attention.gqa.call_metadata import _dtype
from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = ["GroupedQueryAttentionBwdCall", "GroupedQueryAttentionBwdWorkload"]


def _compute_gqa_square_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
) -> torch.Tensor:
    groups = heads // heads_kv
    seq_len = q.shape[1]
    q_bhsd = q.transpose(1, 2).float()
    k_bhsd = k.repeat_interleave(groups, dim=2).transpose(1, 2).float()
    scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * (dim**-0.5)
    if is_causal:
        pos = torch.arange(seq_len, device=q.device)
        mask = pos[None, :] <= pos[:, None]
        scores = scores.masked_fill(~mask.view(1, 1, seq_len, seq_len), float("-inf"))
    return torch.logsumexp(scores, dim=-1) * math.log2(math.e)


class GroupedQueryAttentionBwdWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device=run_device(),
            requires_grad=True,
        )
        k = torch.randn(
            self.batch,
            self.seq_len,
            self.heads_kv,
            self.dim,
            dtype=self.dtype,
            device=run_device(),
            requires_grad=True,
        )
        v = torch.randn(
            self.batch,
            self.seq_len,
            self.heads_kv,
            self.dim,
            dtype=self.dtype,
            device=run_device(),
            requires_grad=True,
        )
        grad_output = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, dtype=self.dtype, device=run_device()
        )

        with torch.no_grad():
            o = (
                F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=self.is_causal,
                    enable_gqa=True,
                )
                .transpose(1, 2)
                .contiguous()
            )
            lse = _compute_gqa_square_lse(
                q,
                k,
                heads=self.heads,
                heads_kv=self.heads_kv,
                dim=self.dim,
                is_causal=self.is_causal,
            )

        return q, k, v, o, grad_output, lse

    def ref_program(self, q, k, v, o, grad_output, lse):
        """FP32 SDPA gradients in BSHD layout, rounded once to the input dtype."""
        dtype = q.dtype
        with torch.enable_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
            q, k, v = (t.detach().float().requires_grad_(True) for t in (q, k, v))
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=self.is_causal,
                enable_gqa=True,
            ).transpose(1, 2)
            gradients = torch.autograd.grad(out, (q, k, v), grad_output.float())
            return tuple(grad.to(dtype) for grad in gradients)

    def verification(self, *inputs):
        from workloads.numerics import Exact, zeroed_input

        return Exact(controls=(zeroed_input(0, "first-input-zeroed"),), atol=5e-3, rtol=1e-5)


class GroupedQueryAttentionBwdCall(CallWorkload, GroupedQueryAttentionBwdWorkload):
    """A manifest call of GroupedQueryAttentionBwdOp; ``o`` and ``lse`` are the forward's."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix = call.ix
        GroupedQueryAttentionBwdWorkload.__init__(
            self, ix["B"], ix["H"], ix["H_kv"], ix["S"], ix["D"], ix["is_causal"], _dtype(call, "q")
        )

    gen_inputs = GroupedQueryAttentionBwdWorkload.gen_inputs
