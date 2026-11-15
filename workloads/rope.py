"""Workload definitions for the RoPE op family."""

import math

import torch

from workloads.workload_base import CallWorkload


def rope_verification():
    from workloads.numerics import Exact

    return Exact()


def _yarn_correction_range(params: dict) -> tuple[float, float]:
    def dim(rotations):
        return (
            params["rotary_dim"]
            * math.log(params["max_position_embeddings"] / (rotations * 2 * math.pi))
            / (2 * math.log(params["base"]))
        )

    low, high = dim(params["beta_fast"]), dim(params["beta_slow"])
    if params["truncate"]:
        low, high = math.floor(low), math.ceil(high)
    return max(low, 0), min(high, params["rotary_dim"] - 1)


def rope_cache(name: str, params: dict) -> tuple[torch.Tensor, torch.Tensor, float]:
    """The float32 cos and sin caches of RoPE op *name* with *params*, and their magnitude.

    The caches are ``(rows, rotary_dim // 2)``, by the formulas of the vLLM class the op
    follows.
    """
    rotary, base = params["rotary_dim"], params["base"]
    pos_freqs = base ** (torch.arange(0, rotary, 2, dtype=torch.float32) / rotary)
    inv = 1.0 / pos_freqs
    rows, mscale = params["max_position_embeddings"], 1.0
    if name == "RoPELlama31FwdOp":
        low, high = params["low_freq_factor"], params["high_freq_factor"]
        orig, factor = params["orig_max_position"], params["scaling_factor"]
        wave_len = 2 * math.pi / inv
        smooth = (orig / wave_len - low) / (high - low) if low != high else 0
        inv = torch.where(
            wave_len < orig / high,
            inv,
            torch.where(
                wave_len > orig / low, inv / factor, (1 - smooth) * inv / factor + smooth * inv
            ),
        )
    elif name == "YaRNFwdOp":
        factor = params["scaling_factor"]
        low, high = _yarn_correction_range(params)
        if low == high:
            high += 0.001
        ramp = torch.clamp(
            (torch.arange(rotary // 2, dtype=torch.float32) - low) / (high - low), 0, 1
        )
        mask = (1 - ramp) * params["extrapolation_factor"]
        inv = (1.0 / (factor * pos_freqs)) * (1 - mask) + inv * mask
        rows = math.ceil(rows * factor)
        yarn = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0
        mscale = params["attn_factor"] * (yarn if params["apply_yarn_scaling"] else 1.0)
    elif name == "LongRoPEFwdOp":
        if params["rescale_factors"] is not None:
            inv = inv / torch.tensor(params["rescale_factors"], dtype=torch.float32)
        scale = rows / params["original_max_position_embeddings"]
        default = (
            1.0
            if scale <= 1
            else math.sqrt(
                1 + math.log(scale) / math.log(params["original_max_position_embeddings"])
            )
        )
        mscale = default if params["mscale"] is None else params["mscale"]
    angles = torch.outer(torch.arange(rows, dtype=torch.float32), inv)
    return angles.cos(), angles.sin(), mscale


def ref_rotate(
    x: torch.Tensor,
    index: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    neox: bool,
) -> torch.Tensor:
    """*x* ``[*S, head_size * H]`` with each head's first *rotary_dim* columns rotated.

    Each token reads cache row *index*; the rotation is float32, rounded once.
    """
    tokens = index.numel()
    heads = x.reshape(tokens, -1, head_size)
    c, s = cos[index.reshape(-1)].unsqueeze(1).float(), sin[index.reshape(-1)].unsqueeze(1).float()
    part = heads[..., :rotary_dim].float()
    if neox:
        x1, x2 = part.chunk(2, dim=-1)
        rotated = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
    else:
        x1, x2 = part[..., 0::2], part[..., 1::2]
        rotated = torch.stack((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).flatten(-2)
    out = torch.cat((rotated.to(x.dtype), heads[..., rotary_dim:]), dim=-1)
    return out.reshape(x.shape)


class RoPECall(CallWorkload):
    """A manifest rotation checked against independently computed cos/sin caches."""

    def verification(self, *inputs):
        return rope_verification()

    def ref_program(self, positions, query, key=None, offsets=None):
        name, params = self.call.signature.name, self.arguments()
        angles_cos, angles_sin, mscale = rope_cache(name, params)
        cos, sin = (
            (t * mscale).to(params["dtype"]).to(device=query.device, dtype=query.dtype)
            for t in (angles_cos, angles_sin)
        )
        index = positions if offsets is None else positions + offsets

        def rotate(x):
            head_size, rotary_dim = params["head_size"], params["rotary_dim"]
            return ref_rotate(x, index, cos, sin, head_size, rotary_dim, params["is_neox_style"])

        return rotate(query), None if key is None else rotate(key)

    def __init__(self, call, device=None):
        CallWorkload.__init__(self, call, device)
        self.tensors = call.materialize(self.device)

    def gen_inputs(self):
        return tuple(self.tensors[name] for name in self.call.signature.inputs)
