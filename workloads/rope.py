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


def rope_tables(name: str, params: dict, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The cos and sin caches of RoPE op *name* with *params*, as the op holds them.

    Each is scaled by the magnitude, rounded to the ``dtype`` parameter, then to the query's
    dtype, on the query's device.
    """
    angles_cos, angles_sin, mscale = rope_cache(name, params)
    return tuple(
        (t * mscale).to(params["dtype"]).to(device=query.device, dtype=query.dtype)
        for t in (angles_cos, angles_sin)
    )


def _rotate(
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


def rope_reference(
    params: dict,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(query_out, key_out)`` of a RoPE op with *params*, reading *cos* and *sin* (from
    :func:`rope_tables`) at row ``positions + offsets`` (``positions`` without offsets)."""
    index = positions if offsets is None else positions + offsets
    hs, rd, neox = params["head_size"], params["rotary_dim"], params["is_neox_style"]
    query_out = _rotate(query, index, cos, sin, hs, rd, neox)
    return query_out, None if key is None else _rotate(key, index, cos, sin, hs, rd, neox)


def rope_cos_sin_reference(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    seqlen_offsets: int | torch.Tensor = 0,
    *,
    interleaved: bool = False,
    inplace: bool = False,
) -> torch.Tensor:
    """``RoPECosSinFwdOp`` as ``flash_attn.layers.rotary.apply_rotary_emb`` computes it.

    Token ``s`` of sequence ``b`` reads row ``s + seqlen_offsets[b]`` of *cos* and *sin*
    (one offset for every sequence when it is an int) and rotates the first
    ``2 * cos.shape[-1]`` columns of each head in float32, rounded once; the rest are copied.
    A row outside the tables reads its nearest row, as the op does unchecked. With *inplace*
    the result is written into *x*, which is returned.
    """
    if cos.shape != sin.shape or 2 * cos.shape[-1] > x.shape[-1]:
        raise ValueError("cos and sin must share a shape of at most half the head")
    r = 2 * cos.shape[-1]
    # Each token's sequence b and position s in it, as tensors: read on the host, the
    # contents would break a torch.compile graph of this reference.
    if cu_seqlens is None:
        b = torch.arange(x.shape[0], device=x.device)[:, None]
        s = torch.arange(x.shape[1], device=x.device)[None, :]
    else:
        tokens = torch.arange(x.shape[0], device=x.device)
        # The last sequence starting at or before the token, past any empty one.
        b = torch.searchsorted(cu_seqlens[1:], tokens, right=True)
        s = tokens - cu_seqlens[b]
    offset = seqlen_offsets[b] if isinstance(seqlen_offsets, torch.Tensor) else seqlen_offsets
    rows = (s + offset).clamp(0, cos.shape[0] - 1)
    c, s = cos[rows][..., None, :].float(), sin[rows][..., None, :].float()
    h = x[..., :r].float()
    if interleaved:
        x1, x2 = h[..., 0::2], h[..., 1::2]
        rotated = torch.stack((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).flatten(-2)
    else:
        x1, x2 = h.chunk(2, dim=-1)
        rotated = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
    out = x if inplace else x.clone()
    out[..., :r] = rotated.to(x.dtype)
    return out


class RoPECosSinCall(CallWorkload):
    """A manifest call of ``RoPECosSinFwdOp`` checked against :func:`rope_cos_sin_reference`."""

    def verification(self, *inputs):
        return rope_verification()

    def ref_program(self, x, cos, sin, cu_seqlens=None, seqlen_offsets=0):
        params = self.arguments()
        return rope_cos_sin_reference(
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            interleaved=params["interleaved"],
            inplace=params["inplace"],
        )


class RoPECosSinViewCall(RoPECosSinCall):
    """The call with ``x`` a column view of heads *width* columns wide: ``x``'s values sit in
    *columns* of each wider head, whose other columns hold their own values."""

    def __init__(self, call, width: int, columns: slice, device=None):
        RoPECosSinCall.__init__(self, call, device)
        self.width, self.columns = width, columns

    def gen_inputs(self):
        x, *rest = RoPECosSinCall.gen_inputs(self)
        wide = torch.randn(*x.shape[:-1], self.width, dtype=x.dtype, device=x.device)
        view = wide[..., self.columns]
        view.copy_(x)
        return (view, *rest)


class RoPECall(CallWorkload):
    """A manifest rotation checked against independently computed cos/sin caches."""

    def verification(self, *inputs):
        return rope_verification()

    def ref_program(self, positions, query, key=None, offsets=None):
        name, params = self.call.signature.name, self.arguments()
        cos, sin = rope_tables(name, params, query)
        return rope_reference(params, cos, sin, positions, query, key, offsets)

    def __init__(self, call, device=None):
        CallWorkload.__init__(self, call, device)
        self.tensors = call.materialize(self.device)

    def gen_inputs(self):
        return tuple(self.tensors[name] for name in self.call.signature.inputs)
