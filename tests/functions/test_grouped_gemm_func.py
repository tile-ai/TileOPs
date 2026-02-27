import torch
import pytest

from tests.test_base import FixtureBase
from tests.ops.test_grouped_gemm import GroupedGemmCompleteTest
from tileops.functions import GroupedGemmFunc


class GroupedGemmFuncFixture(FixtureBase):
    PARAMS = [
        ("batch_sum, batch_count, N, K, dtype, tune", [
            (16384, 4, 4864, 8192, torch.float16, False),
        ]),
    ]


@GroupedGemmFuncFixture
def test_grouped_gemm_fn(batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype,
                         tune: bool) -> None:
    test = GroupedGemmCompleteTest(batch_sum, batch_count, N, K, dtype)
    fn = GroupedGemmFunc(batch_sum, batch_count, N, K, dtype, tune=tune)
    inputs = test.gen_inputs()

    # Test forward pass (no grad)
    with torch.no_grad():
        output = fn(*inputs)
    assert output is not None, "Forward output should not be None"

    # Test forward + backward pass
    A, B, batch_sizes, batch_offsets, batch_padded_offsets = inputs
    A = A.clone().detach().requires_grad_(True)
    B = B.clone().detach().requires_grad_(True)
    output = fn(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    assert A.grad is not None, "Gradient for A should not be None"
    assert B.grad is not None, "Gradient for B should not be None"
    print(f"A.grad shape: {A.grad.shape}")
    print(f"B.grad shape: {B.grad.shape}")
    print(f"A.grad range: [{A.grad.min():.6f}, {A.grad.max():.6f}]")
    print(f"B.grad range: [{B.grad.min():.6f}, {B.grad.max():.6f}]")
    print("Function test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
