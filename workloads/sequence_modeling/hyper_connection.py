"""Torch references of the hyper-connection ops that no library reference covers."""

import torch

__all__ = ["hc_head"]


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
