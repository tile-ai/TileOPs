"""Test for MoeGroupedGemmPersistent3WGFusedActKernel."""
import pytest
import torch
import torch.nn.functional as F

from tileops.kernels.moe import MoeGroupedGemmPersistent3WGFusedActKernel

# Skip if not SM90
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires SM90+ (Hopper or newer)",
)


def _ref_grouped_gemm_fused_act(A, B, true_sizes, true_offsets, activation):
    """Reference implementation: grouped GEMM + fused activation.

    Args:
        A: [numel, K]
        B: [E, 2N, K] (gate_up concatenated)
        true_sizes: [E] int32
        true_offsets: [E] int32
        activation: "silu_and_mul" or "gelu_and_mul"

    Returns:
        C: [numel, N]
    """
    numel, K = A.shape
    E, two_N, _ = B.shape
    N = two_N // 2

    # Move to CPU to avoid CUBLAS issues with small batch sizes
    A_cpu = A.cpu().float()
    B_cpu = B.cpu().float()
    C_cpu = torch.zeros(numel, N, dtype=torch.float32, device="cpu")

    for e in range(E):
        offset = true_offsets[e].item()
        size = true_sizes[e].item()
        if size == 0:
            continue

        # A_slice: [size, K], B_e: [2N, K]
        A_slice = A_cpu[offset:offset + size]
        B_e = B_cpu[e]  # [2N, K]

        # GEMM: [size, K] @ [2N, K]^T -> [size, 2N]
        gate_up = torch.matmul(A_slice, B_e.T)  # [size, 2N]

        # Split and apply activation
        gate = gate_up[:, :N]
        up = gate_up[:, N:]

        if activation == "silu_and_mul":
            # silu(gate) * up
            act_out = F.silu(gate) * up
        elif activation == "gelu_and_mul":
            # gelu(gate) * up
            act_out = F.gelu(gate) * up
        else:
            raise ValueError(f"Unknown activation: {activation}")

        C_cpu[offset:offset + size] = act_out

    # Single transfer to GPU at the end
    return C_cpu.to(A.dtype).cuda()


@pytest.mark.smoke
@pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_and_mul"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block_m", [64, 128])
def test_moe_grouped_gemm_3wg_fused_act_basic(activation, dtype, block_m):
    """Test basic correctness of fused activation kernel.

    Note: Current implementation requires N <= 128 due to TMA boxDim limitation (256).
    """
    torch.manual_seed(42)
    device = "cuda"

    # Small test case with N <= 128
    numel = 512
    num_experts = 8
    N = 128  # Output width (ffn_size), limited to 128 so 2N <= 256
    K = 128  # Hidden size

    # Create inputs
    A = torch.randn(numel, K, dtype=dtype, device=device) * 0.02
    B = torch.randn(num_experts, N * 2, K, dtype=dtype, device=device) * 0.02

    # Uniform distribution for simplicity
    per_expert = numel // num_experts
    true_sizes = torch.full((num_experts,), per_expert, dtype=torch.int32, device=device)
    true_sizes[-1] = numel - per_expert * (num_experts - 1)
    true_offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
    true_offsets[1:] = torch.cumsum(true_sizes[:-1], dim=0)

    # Run kernel
    # Note: block_n should cover the full B width (2N), not just output width (N)
    # For pingpong (block_m=64), use num_stages=2 to fit in SMEM
    num_stages = 2 if block_m == 64 else 3
    config = {
        "block_m": block_m,
        "block_n": N * 2,  # B matrix width is 2N (gate_up concatenated)
        "block_k": 64,
        "num_stages": num_stages,
        "threads": 384,
        "group_size_m": 1,
    }
    kernel = MoeGroupedGemmPersistent3WGFusedActKernel(
        numel=numel, num_experts=num_experts, N=N, K=K,
        dtype=dtype, activation=activation, config=config,
    )
    output = kernel(A, B, true_sizes, true_offsets)

    # Reference
    ref_output = _ref_grouped_gemm_fused_act(A, B, true_sizes, true_offsets, activation)

    # Compare
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print(f"✓ {activation}, {dtype}, block_m={block_m}: PASSED")


@pytest.mark.nightly
@pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_and_mul"])
def test_moe_grouped_gemm_3wg_fused_act_large(activation):
    """Test with larger, more realistic dimensions."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    # Realistic MoE dimensions (N must be <= 128 due to TMA boxDim constraint)
    numel = 4096  # T * top_k
    num_experts = 64
    N = 128  # ffn_size (max allowed for fused activation)
    K = 1024  # hidden_size

    A = torch.randn(numel, K, dtype=dtype, device=device) * 0.02
    B = torch.randn(num_experts, N * 2, K, dtype=dtype, device=device) * 0.02

    # Skewed distribution (more realistic)
    true_sizes = torch.randint(32, 128, (num_experts,), dtype=torch.int32, device=device)
    true_sizes = (true_sizes * numel // true_sizes.sum()).int()
    true_sizes[-1] = numel - true_sizes[:-1].sum()
    true_offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
    true_offsets[1:] = torch.cumsum(true_sizes[:-1], dim=0)

    kernel = MoeGroupedGemmPersistent3WGFusedActKernel(
        numel=numel, num_experts=num_experts, N=N, K=K,
        dtype=dtype, activation=activation,
    )
    output = kernel(A, B, true_sizes, true_offsets)

    ref_output = _ref_grouped_gemm_fused_act(A, B, true_sizes, true_offsets, activation)

    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print(f"✓ {activation} (large): PASSED")


if __name__ == "__main__":
    # Quick smoke test
    test_moe_grouped_gemm_3wg_fused_act_basic("silu_and_mul", torch.bfloat16, 128)
    test_moe_grouped_gemm_3wg_fused_act_basic("gelu_and_mul", torch.bfloat16, 128)
    test_moe_grouped_gemm_3wg_fused_act_large("silu_and_mul")
    print("\n✅ All tests passed!")
