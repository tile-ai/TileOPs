from collections.abc import Sequence


def per_axis_values(value: "int | Sequence[int]", ndim: int) -> tuple[int, ...]:
    """A pooling parameter as one value per spatial axis, as ``per_axis`` reads it."""
    return (value,) * ndim if isinstance(value, int) else tuple(value)
