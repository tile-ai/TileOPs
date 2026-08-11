"""Common-input equivalence tests for the vLLM Marlin W4A16 adapter."""

import pytest
import torch

from benchmarks.ops.bench_gemm import _prepare_marlin_w4a16_baseline


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("use_fp32_reduce", [False, True], ids=["fp16-reduce", "fp32-reduce"])
def test_marlin_w4a16_conversion_preserves_nibbles_and_group_metadata(
    use_fp32_reduce: bool,
) -> None:
    pytest.importorskip("vllm")
    m, n, k = 1, 64, 256
    device = torch.device("cuda")

    # Different even/odd formulas catch a nibble swap. The offset after K=128
    # combines with distinct metadata to catch a group-boundary permutation.
    k_index = torch.arange(k, device=device, dtype=torch.int32)
    row_index = torch.arange(n, device=device, dtype=torch.int32)[:, None]
    even = (row_index + 3 * k_index) % 16
    odd = (5 * row_index + k_index + 7) % 16
    logical_q = torch.where((k_index % 2)[None, :] == 0, even, odd).to(torch.uint8)
    packed_weight = (logical_q[:, 0::2] | (logical_q[:, 1::2] << 4)).contiguous()

    weight_scale = torch.stack(
        (
            0.03125 + torch.arange(n, device=device, dtype=torch.float32) / 4096,
            0.125 + torch.arange(n, device=device, dtype=torch.float32) / 2048,
        ),
        dim=1,
    ).contiguous()
    weight_zero = torch.stack(
        (
            torch.arange(n, device=device, dtype=torch.uint8) % 5,
            11 + torch.arange(n, device=device, dtype=torch.uint8) % 5,
        ),
        dim=1,
    ).contiguous()
    activation = torch.linspace(-1, 1, k, device=device, dtype=torch.float16)[None, :]

    expanded_scale = weight_scale.repeat_interleave(128, dim=1)
    expanded_zero = weight_zero.float().repeat_interleave(128, dim=1)
    dequantized = ((logical_q.float() - expanded_zero) * expanded_scale).half()
    expected = activation @ dequantized.T

    marlin, marlin_inputs = _prepare_marlin_w4a16_baseline(
        m,
        n,
        k,
        use_fp32_reduce,
        activation,
        packed_weight,
        weight_scale,
        weight_zero,
    )
    actual = marlin(*marlin_inputs)

    torch.testing.assert_close(actual, expected, atol=7e-2, rtol=5e-2)
