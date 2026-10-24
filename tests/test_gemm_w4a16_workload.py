"""CPU contracts for affine INT4 quantization and W4A16 repacking."""

import pytest
import torch

from workloads.gemm import quantize_weight_int4, repack_w4a16_weight


@pytest.mark.smoke
def test_quantize_weight_int4_keeps_one_sided_groups_in_range() -> None:
    weight = torch.tensor(
        [
            [0.25, 0.50, 0.75, 1.00],
            [-1.00, -0.75, -0.50, -0.25],
        ],
        dtype=torch.float32,
    )

    _, scale, zero, dequantized = quantize_weight_int4(weight, group_size=4)

    assert torch.equal(zero, torch.tensor([[0], [15]], dtype=torch.uint8))
    assert torch.all(scale > 0)
    # Both groups span one unit including zero: their 15-step scale is known exactly.
    expected_scale = torch.full_like(scale, 1 / 15)
    assert torch.equal(scale, expected_scale)
    assert dequantized[0].max() == 15 * expected_scale[0, 0].float()
    assert dequantized[1].min() == -15 * expected_scale[1, 0].float()


@pytest.mark.smoke
def test_repack_w4a16_weight_permutes_nibbles_inside_a_step() -> None:
    packed = torch.randint(0, 256, (7, 256), dtype=torch.uint8)

    repacked = repack_w4a16_weight(packed)

    assert repacked.shape == packed.shape
    assert repacked.is_contiguous()

    def nibbles(tile: torch.Tensor) -> torch.Tensor:
        return torch.cat([tile & 0xF, tile >> 4], dim=1).sort(dim=1).values

    step = 64
    for start in range(0, packed.shape[1], step):
        assert torch.equal(
            nibbles(repacked[:, start : start + step]),
            nibbles(packed[:, start : start + step]),
        )
