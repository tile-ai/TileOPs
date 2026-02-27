import pytest
import torch

from tests.ops.test_grouped_gemm import GroupedGemmCompleteTest
from tests.test_base import FixtureBase
from tileops.layers import GroupedGemmLayer


class GroupedGemmLayerFixture(FixtureBase):
    PARAMS = [
        ("batch_sum, batch_count, N, K, dtype", [
            (16384, 4, 4864, 8192, torch.float16),
        ]),
    ]


@GroupedGemmLayerFixture
def test_grouped_gemm_layer(batch_sum: int, batch_count: int, N: int, K: int,
                            dtype: torch.dtype):
    grouped_gemm = GroupedGemmLayer(batch_sum, batch_count, N, K, dtype)
    test = GroupedGemmCompleteTest(batch_sum, batch_count, N, K, dtype)
    inputs = test.gen_inputs()
    # enable gradients for A and B
    inputs = list(inputs)
    inputs[0] = inputs[0].clone().detach().requires_grad_(True)
    inputs[1] = inputs[1].clone().detach().requires_grad_(True)
    inputs = tuple(inputs)
    test.check_fn(grouped_gemm, *inputs)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
