import torch


def reduction_tolerance(dtype: torch.dtype) -> dict[str, float]:
    """Return ``atol``/``rtol`` for a reduction test.

    Reductions accumulate in fp32, but the narrowing cast back to a half-precision
    storage dtype still rounds.
    """
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    return {"atol": 1e-2, "rtol": 1e-2}
