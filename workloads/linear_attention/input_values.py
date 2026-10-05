import torch

# Manifest calls: shapes, dtypes and presence come from a workload row; these
# classes condition the values the row does not determine and carry the reference.


def _step_sizes(like: torch.Tensor) -> torch.Tensor:
    """Delta-rule step sizes in ``[0, 0.5)``, with *like*'s shape and dtype."""
    return torch.rand(like.shape, device=like.device).to(like.dtype) * 0.5


def _log_gates(like: torch.Tensor) -> torch.Tensor:
    """Log-space forget gates in ``(-1, 0]``, with *like*'s shape and dtype."""
    return -torch.rand(like.shape, device=like.device).to(like.dtype)


def _small(t: torch.Tensor | None, scale: float = 0.1) -> torch.Tensor | None:
    return None if t is None else t * scale
