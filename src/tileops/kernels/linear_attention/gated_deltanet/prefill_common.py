"""Gated DeltaNet private helpers shared by the prefill stages."""

import tilelang.language as T

# What the comparator's L2 normalization adds under the square root before taking the
# reciprocal, from `fla.modules.l2norm.l2norm_fwd`. The block solve and the forward both
# form a reciprocal norm and must add the same thing.
L2NORM_EPS: float = 1e-6


def step_size(raw, beta_sigmoid: bool, allow_neg_eigval: bool):
    """The delta-rule step size, from what the op handed the kernel.

    The sigmoid runs in float32 and the result returns to *raw*'s dtype, which is what a
    caller transforming ``beta`` itself would hand every stage. Keeping the wider value
    would give the triangular solve and the recurrence two different step sizes, since the
    solve stores it at the activation dtype and the recurrence in float32.

    Args:
        raw: The value read from ``beta``, already indexed.
        beta_sigmoid: ``beta`` carries raw logits rather than the step size.
        allow_neg_eigval: The transform is ``2 * sigmoid`` rather than ``sigmoid``.

    Returns:
        An expression in *raw*'s dtype, which is *raw* itself where no sigmoid applies.
    """
    if not beta_sigmoid:
        return raw
    scaled = T.sigmoid(T.cast(raw, "float32")) * (2.0 if allow_neg_eigval else 1.0)
    return T.Cast(raw.dtype, scaled)
