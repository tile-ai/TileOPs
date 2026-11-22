"""Bounds of a model's distance from its reference by the reference's own rounding noise.

A model rounds to bfloat16 at other points than its reference (fused ops, other accumulation
orders), an error of the same size as the reference's own rounding; a factor of 2 lets the two
add.
"""

import torch

__all__ = ["check_argmax_within_rounding", "check_within_rounding"]

_NOISE_FACTOR = 2


def check_within_rounding(
    actual: list[torch.Tensor], expected: list[torch.Tensor], rounded: list[torch.Tensor], what: str
) -> None:
    """Assert ``actual`` is at most twice as far from ``expected`` as ``rounded`` is.

    Each list holds one tensor per step; distances are L2 over all of them. ``expected`` comes
    from the reference and ``rounded`` from the same reference rounding to bfloat16 where the
    model does: their distance is the reference's own rounding noise.
    """

    def distance(xs: list[torch.Tensor]) -> float:
        return (
            sum(
                (x.float().cpu() - e.float().cpu()).square().sum().item()
                for x, e in zip(xs, expected, strict=True)
            )
            ** 0.5
        )

    error, noise = distance(actual), distance(rounded)
    assert error <= _NOISE_FACTOR * noise, (
        f"{what} differ from the reference by {error:.4g} in L2, its bfloat16 rounding by "
        f"{noise:.4g}"
    )


def check_argmax_within_rounding(
    token: int, expected: torch.Tensor, rounded: torch.Tensor, what: str
) -> None:
    """Assert ``token``'s logit in ``expected`` (the reference's) is the maximum to within twice
    the reference's largest bfloat16 logit error, ``rounded - expected``."""
    slack = _NOISE_FACTOR * (rounded - expected).abs().max()
    assert expected[token] >= expected.max() - slack, (
        f"{what}: token {token} is {expected.max() - expected[token]:.3f} below the reference's "
        f"maximum, its bfloat16 rounding moves logits by {slack / _NOISE_FACTOR:.3f}"
    )
