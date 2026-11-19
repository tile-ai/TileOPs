"""Torch references of the hyper-connection ops that no library reference covers."""

from typing import Any

import torch

from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = ["HyperConnectionPostWorkload", "HyperConnectionPreCall", "hc_head", "hc_post", "hc_pre"]


def hc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``HyperConnectionPreFwdOp``'s reference, vLLM ``mhc_pre_torch`` in float32: per token,
    the ``n`` streams flattened give ``mixes = (x @ fn^T) * rsqrt(mean(x^2) + rms_eps)``, split
    into the pre, post and comb logits, and returns ``(post_mix, comb_mix, layer_input)``."""
    n = residual.shape[-2]
    x = residual.flatten(-2).float()
    mixes = (x @ fn.T) * torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(mixes[..., :n] * hc_scale[0] + hc_base[:n]) + hc_pre_eps
    post = torch.sigmoid(mixes[..., n : 2 * n] * hc_scale[1] + hc_base[n : 2 * n])
    comb = mixes[..., 2 * n :].unflatten(-1, (n, n)) * hc_scale[2] + hc_base[2 * n :].view(n, n)
    comb = comb.softmax(-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(-2).to(torch.bfloat16)
    return (post * hc_post_mult_value).unsqueeze(-1), comb, layer_input


def hc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """``HyperConnectionHeadFwdOp``'s reference, DeepSeek-V4-Flash ``inference/model.py``
    ``ParallelHead.hc_head``: per token, the ``n`` streams flattened in float32 give
    ``pre = sigmoid((x @ hc_fn^T) * rsqrt(mean(x^2) + rms_norm_eps) * hc_scale + hc_base)
    + hc_eps``, and the output is ``sum_i pre_i * hidden_states_i`` rounded once to bfloat16.
    """
    n, c = hidden_states.shape[-2:]
    if hc_fn.shape != (n, n * c) or hc_scale.shape != (1,) or hc_base.shape != (n,):
        raise ValueError("hc_fn, hc_scale and hc_base must match the streams")
    x = hidden_states.flatten(-2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
    pre = torch.sigmoid((x @ hc_fn.T) * rsqrt * hc_scale + hc_base) + hc_eps
    return (pre.unsqueeze(-1) * hidden_states.float()).sum(-2).to(torch.bfloat16)


class HyperConnectionPreCall(CallWorkload):
    """One manifest call of ``HyperConnectionPreFwdOp``; inputs ``(residual, fn, hc_scale,
    hc_base)``, ``fn`` scaled by ``1 / sqrt(n * C)`` so the logits are of order one."""

    def gen_inputs(self):
        residual, fn, hc_scale, hc_base = super().gen_inputs()
        return residual, fn / fn.shape[-1] ** 0.5, hc_scale, hc_base

    def ref_program(self, residual, fn, hc_scale, hc_base):
        return hc_pre(residual, fn, hc_scale, hc_base, **self.call.arguments())

    torch_program = ref_program


def hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """``HyperConnectionPostFwdOp``'s reference, vLLM ``mhc_post_torch``: per token,
    ``comb_res_mix`` applied by column to the residual streams in float32, plus
    ``post_layer_mix`` times ``x``, rounded once to bfloat16."""
    mixed = torch.einsum("...ij,...ic->...jc", comb_res_mix.float(), residual.float())
    return (post_layer_mix.float() * x.float().unsqueeze(-2) + mixed).to(torch.bfloat16)


class HyperConnectionPostWorkload(WorkloadBase):
    """``HyperConnectionPostFwdOp`` over tokens of ``shape``: ``n`` residual streams of ``c``
    columns, the mixes in ``mix_dtype``. ``post_layer_mix`` is twice a sigmoid, as the pre
    step makes it, and each row of ``comb_res_mix`` a softmax."""

    def __init__(
        self, shape: tuple[int, ...], n: int, c: int, mix_dtype: torch.dtype = torch.float32
    ):
        self.shape = tuple(shape)
        self.n = n
        self.c = c
        self.mix_dtype = mix_dtype

    @classmethod
    def from_call(cls, call: Any) -> "HyperConnectionPostWorkload":
        """The workload of one manifest call of ``HyperConnectionPostFwdOp``."""
        shape, _dtype = call.tensors["residual"]
        _shape, mix_dtype = call.tensors["comb_res_mix"]
        return cls(tuple(shape[:-2]), shape[-2], shape[-1], getattr(torch, mix_dtype))

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        device = run_device()
        x = torch.randn((*self.shape, self.c), dtype=torch.bfloat16, device=device)
        residual = torch.randn((*self.shape, self.n, self.c), dtype=torch.bfloat16, device=device)
        post = 2 * torch.sigmoid(torch.randn((*self.shape, self.n, 1), device=device))
        comb = torch.softmax(torch.randn((*self.shape, self.n, self.n), device=device), dim=-1)
        return x, residual, post.to(self.mix_dtype), comb.to(self.mix_dtype)

    def ref_program(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return hc_post(x, residual, post_layer_mix, comb_res_mix)
