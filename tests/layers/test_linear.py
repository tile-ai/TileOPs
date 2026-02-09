import pytest
import torch

from top.layers import LinearLayer


@pytest.mark.parametrize(
    "m, n, k, dtype, tune",
    [
        (1024, 1024, 1024, torch.float16, False),
    ],
)
def test_linear(m: int, n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    linear_layer = LinearLayer(m, n, k, dtype=dtype, tune=tune)
    input_tensor = torch.randn(m, k, dtype=dtype, device='cuda', requires_grad=True)

    output = linear_layer(input_tensor)

    loss = output.sum()
    loss.backward()

    print("Output shape:", output.shape)
    print("Gradient shape:", input_tensor.grad.shape)


if __name__ == "__main__":
    import sys

    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
