"""Torch references of the hyper-connection ops that no library reference covers."""

from dataclasses import dataclass
from typing import Any

import torch

from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = [
    "HyperConnectionHeadWorkload",
    "HyperConnectionPostWorkload",
    "HyperConnectionPreCall",
    "hc_head",
    "hc_post",
    "hc_pre",
    "hc_split_mixes",
]


def _normalized_projection(streams: torch.Tensor, fn: torch.Tensor, eps: float) -> torch.Tensor:
    """``(x @ fn^T) * rsqrt(mean(x^2) + eps)``, ``x`` each token's streams flattened to
    float32."""
    x = streams.flatten(-2).float()
    return (x @ fn.T) * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def _weighted_stream_sum(weights: torch.Tensor, streams: torch.Tensor) -> torch.Tensor:
    """``sum_i weights_i * streams_i`` per token in float32, rounded once to bfloat16."""
    return (weights.unsqueeze(-1) * streams.float()).sum(-2).to(torch.bfloat16)


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
    mixes = _normalized_projection(residual, fn, rms_eps)
    pre, post, comb = hc_split_mixes(
        mixes, hc_scale, hc_base, n, hc_pre_eps, hc_sinkhorn_eps, sinkhorn_repeat
    )
    layer_input = _weighted_stream_sum(pre, residual)
    return (post * hc_post_mult_value).unsqueeze(-1), comb, layer_input


def hc_split_mixes(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    n: int,
    pre_eps: float,
    sinkhorn_eps: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The ``(2 + n) * n`` mix logits, scaled and biased, split into ``pre`` (sigmoid plus
    ``pre_eps``), ``post`` (sigmoid) and the ``n x n`` comb: softmax over its last axis,
    normalized over columns, then ``sinkhorn_repeat - 1`` times over rows and columns, each
    step adding ``sinkhorn_eps``."""
    pre = torch.sigmoid(mixes[..., :n] * hc_scale[0] + hc_base[:n]) + pre_eps
    post = torch.sigmoid(mixes[..., n : 2 * n] * hc_scale[1] + hc_base[n : 2 * n])
    comb = mixes[..., 2 * n :].unflatten(-1, (n, n)) * hc_scale[2] + hc_base[2 * n :].view(n, n)
    comb = comb.softmax(-1) + sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
    return pre, post, comb


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
    logits = _normalized_projection(hidden_states, hc_fn, rms_norm_eps)
    pre = torch.sigmoid(logits * hc_scale + hc_base) + hc_eps
    return _weighted_stream_sum(pre, hidden_states)


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


@dataclass(eq=False)
class HyperConnectionPostWorkload(WorkloadBase):
    """``HyperConnectionPostFwdOp`` over tokens of ``shape``: ``n`` residual streams of ``c``
    columns, the mixes in ``mix_dtype``. ``post_layer_mix`` is twice a sigmoid, as the pre
    step makes it, and each row of ``comb_res_mix`` a softmax."""

    shape: tuple[int, ...]
    n: int
    c: int
    mix_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.shape = tuple(self.shape)

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


@dataclass(eq=False)
class HyperConnectionHeadWorkload(WorkloadBase):
    """``HyperConnectionHeadFwdOp`` over tokens of ``shape``: ``n`` streams of ``c`` columns.
    ``hc_fn`` is scaled so that a normalized token's projections are of order one, where the
    sigmoid is steepest."""

    shape: tuple[int, ...]
    n: int
    c: int
    rms_norm_eps: float = 1e-6
    hc_eps: float = 1e-6

    def __post_init__(self) -> None:
        self.shape = tuple(self.shape)

    @classmethod
    def from_call(cls, call: Any) -> "HyperConnectionHeadWorkload":
        """The workload of one manifest call of ``HyperConnectionHeadFwdOp``."""
        shape, _dtype = call.tensors["hidden_states"]
        return cls(
            tuple(shape[:-2]),
            shape[-2],
            shape[-1],
            call.params["rms_norm_eps"],
            call.params["hc_eps"],
        )

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        device = run_device()
        k = self.n * self.c
        hidden_states = torch.randn(
            (*self.shape, self.n, self.c), dtype=torch.bfloat16, device=device
        )
        hc_fn = torch.randn((self.n, k), device=device) / k**0.5
        hc_scale = 1 + torch.rand(1, device=device)
        hc_base = torch.randn(self.n, device=device) / 4
        return hidden_states, hc_fn, hc_scale, hc_base

    def ref_program(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        return hc_head(hidden_states, hc_fn, hc_scale, hc_base, self.rms_norm_eps, self.hc_eps)
