import torch

__all__ = ["apply_dense_rope", "apply_packed_rope"]


def apply_dense_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rotary_dim: int,
    layout: str,
) -> torch.Tensor:
    """Rotate the first *rotary_dim* channels of a BSHD tensor at *positions*.

    ``neox`` pairs channel ``i`` with ``i + rotary_dim // 2``, ``interleaved``
    pairs adjacent channels, and channels past *rotary_dim* pass through.
    """
    half = rotary_dim // 2
    x_rot = x[..., :rotary_dim].float()
    c = cos[positions].view(1, x.shape[1], 1, half).float()
    s = sin[positions].view(1, x.shape[1], 1, half).float()
    if layout == "neox":
        x0, x1 = x_rot[..., :half], x_rot[..., half:]
    else:
        x0, x1 = x_rot[..., 0::2], x_rot[..., 1::2]
    y0, y1 = x0 * c - x1 * s, x1 * c + x0 * s
    rotated = (
        torch.cat((y0, y1), dim=-1)
        if layout == "neox"
        else torch.stack((y0, y1), dim=-1).flatten(-2)
    )
    return torch.cat((rotated.to(x.dtype), x[..., rotary_dim:]), dim=-1).contiguous()


def apply_packed_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rotary_dim: int,
    layout: str,
) -> torch.Tensor:
    """Rotate the first *rotary_dim* channels of a packed THD tensor at *positions*.

    ``neox`` pairs channel ``i`` with ``i + rotary_dim // 2``, ``interleaved``
    pairs adjacent channels, and channels past *rotary_dim* pass through.
    """
    half = rotary_dim // 2
    x_rot = x[..., :rotary_dim].float()
    c = cos[positions].view(x.shape[0], 1, half).float()
    s = sin[positions].view(x.shape[0], 1, half).float()
    if layout == "neox":
        x0, x1 = x_rot[..., :half], x_rot[..., half:]
    else:
        x0, x1 = x_rot[..., 0::2], x_rot[..., 1::2]
    y0, y1 = x0 * c - x1 * s, x1 * c + x0 * s
    rotated = (
        torch.cat((y0, y1), dim=-1)
        if layout == "neox"
        else torch.stack((y0, y1), dim=-1).flatten(-2)
    )
    return torch.cat((rotated, x[..., rotary_dim:].float()), dim=-1)
