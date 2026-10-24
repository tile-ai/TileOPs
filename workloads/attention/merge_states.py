"""Attention-state inputs and independent FP32 stable merge reference."""

import torch

from workloads.device import run_device
from workloads.numerics import Exact
from workloads.workload_base import CallWorkload, WorkloadBase


class MergeAttentionStatesWorkload(WorkloadBase):
    """Two partial attention states with a stable FP32 natural-log merge reference."""

    def __init__(
        self,
        tokens,
        heads,
        dim,
        dtype,
        *,
        edge_states=False,
        strided=False,
        misaligned=False,
        finite_underflow=False,
    ):
        self.tokens, self.heads, self.dim, self.dtype = tokens, heads, dim, dtype
        self.edge_states, self.strided = edge_states, strided
        self.misaligned, self.finite_underflow = misaligned, finite_underflow

    def gen_inputs(self):
        device = run_device()
        inputs = []
        for name in ("a", "b"):
            v = torch.randn(
                (self.tokens, self.heads, self.dim),
                dtype=self.dtype,
                device=device,
                generator=self.rng(f"v_{name}", device=device),
            )
            s = (
                torch.randn(
                    (self.tokens, self.heads),
                    dtype=torch.float32,
                    device=device,
                    generator=self.rng(f"s_{name}", device=device),
                )
                * 4
                + 5
            )
            inputs.extend((v, s))
        if self.edge_states and self.tokens * self.heads >= 5:
            va, sa, vb, sb = inputs
            sa.view(-1)[:5] = torch.tensor(
                [-float("inf"), 2, -float("inf"), 10000, -10000], device=device
            )
            sb.view(-1)[:5] = torch.tensor(
                [2, -float("inf"), -float("inf"), -10000, -10000], device=device
            )
            va.view(self.tokens * self.heads, self.dim)[[0, 2]] = float("nan") if self.dim else 0
            vb.view(self.tokens * self.heads, self.dim)[[1, 2]] = float("nan") if self.dim else 0
        if self.strided:
            inputs = [x.repeat_interleave(2, dim=-1)[..., ::2] for x in inputs]
        if self.finite_underflow:
            va, sa, vb, sb = inputs
            sa.view(-1)[:2] = torch.tensor([0.0, -200.0], device=device)
            sb.view(-1)[:2] = torch.tensor([-200.0, 0.0], device=device)
            vb.view(-1, self.dim)[0] = float("nan")
            va.view(-1, self.dim)[1] = float("nan")
        if self.misaligned:
            shifted = []
            for x in inputs:
                buffer = torch.empty(x.numel() + 1, dtype=x.dtype, device=device)
                value = buffer[1:].view_as(x)
                value.copy_(x)
                shifted.append(value)
            inputs = shifted
        return tuple(inputs)

    def ref_program(self, v_a, s_a, v_b, s_b):
        maximum = torch.maximum(s_a, s_b)
        maximum = torch.where(maximum == -float("inf"), 0, maximum)
        a, b = torch.exp(s_a - maximum), torch.exp(s_b - maximum)
        total = a + b
        va = torch.where((s_a == -float("inf"))[..., None], 0, v_a.float())
        vb = torch.where((s_b == -float("inf"))[..., None], 0, v_b.float())
        v = va * (a / total.clamp_min(1))[..., None] + vb * (b / total.clamp_min(1))[..., None]
        return v.to(v_a.dtype), total.log() + maximum

    def verification(self, *inputs):
        return Exact()


class MergeAttentionStatesCall(CallWorkload, MergeAttentionStatesWorkload):
    def __init__(self, call):
        CallWorkload.__init__(self, call)
        ix = call.indices
        MergeAttentionStatesWorkload.__init__(
            self, ix["T_q"], ix["H"], ix["D"], getattr(torch, call.specs["v_a"].dtype)
        )

    gen_inputs = MergeAttentionStatesWorkload.gen_inputs
