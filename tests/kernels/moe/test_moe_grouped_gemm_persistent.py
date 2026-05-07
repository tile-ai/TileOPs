"""Correctness tests for MoeGroupedGemmPersistentKernel.

Verifies that the persistent kernel produces the same output as
MoeGroupedGemmNopadKernel across shapes, distributions, and dtypes.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from tileops.kernels.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadKernel
from tileops.kernels.moe.moe_grouped_gemm_persistent import MoeGroupedGemmPersistentKernel


def make_inputs(T: int, E: int, top_k: int, N: int, K: int,
                dtype: torch.dtype, distribution: str = "uniform",
                seed: int = 42):
    """Generate nopad GEMM inputs with specified token distribution.

    Args:
        T: Number of tokens.
        E: Number of experts.
        top_k: Top-k routing.
        N: Output feature dimension.
        K: Input feature dimension.
        dtype: Activation dtype.
        distribution: "uniform" or "skewed" (most tokens to first expert).
        seed: Random seed.

    Returns:
        A, B, true_sizes, true_offsets, numel
    """
    torch.manual_seed(seed)
    numel = T * top_k
    dev = "cuda"

    if distribution == "uniform":
        tokens_per_expert = max(1, numel // E)
        sizes = torch.zeros(E, dtype=torch.int32, device=dev)
        sizes[:] = tokens_per_expert
        sizes[-1] = numel - tokens_per_expert * (E - 1)
    else:  # skewed: 80% tokens go to first 20% experts
        sizes = torch.ones(E, dtype=torch.int32, device=dev)
        extra = numel - E
        top_experts = max(1, E // 5)
        per_top = extra // top_experts
        sizes[:top_experts] += per_top
        sizes[0] += extra - per_top * top_experts  # remainder to first expert

    offsets = torch.zeros(E, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)

    A = torch.randn(numel, K, dtype=dtype, device=dev) * 0.02
    B = torch.randn(E, N, K, dtype=dtype, device=dev) * 0.02
    return A, B, sizes, offsets, numel


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_import():
    """MoeGroupedGemmPersistentKernel can be imported."""
    from tileops.kernels.moe.moe_grouped_gemm_persistent import MoeGroupedGemmPersistentKernel  # noqa: F401


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_output_shape(dtype):
    """Persistent kernel output has shape [numel, N]."""
    T, E, top_k, N, K = 32, 4, 2, 128, 64
    numel = T * top_k
    A, B, sizes, offsets, _ = make_inputs(T, E, top_k, N, K, dtype)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    kernel = MoeGroupedGemmPersistentKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=dtype, sm_count=sm_count)
    C = kernel(A, B, sizes, offsets)
    assert C.shape == (numel, N), f"Expected ({numel}, {N}), got {C.shape}"
    assert C.dtype == dtype


# ---------------------------------------------------------------------------
# Correctness tests vs MoeGroupedGemmNopadKernel
# ---------------------------------------------------------------------------

@pytest.mark.nightly
@pytest.mark.parametrize("distribution", ["uniform", "skewed"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("T,E,top_k,N,K", [
    (32,  4,  2,  128,  64),   # small, K aligned
    (64,  8,  4,  256,  128),  # medium, K aligned
    (128, 16, 4,  512,  256),  # larger
    (64,  8,  4,  256,  96),   # K NOT block_k-aligned (non-TMA path)
], ids=["small", "medium", "larger", "k_unaligned"])
def test_matches_nopad_kernel(T, E, top_k, N, K, dtype, distribution):
    """Persistent kernel output matches MoeGroupedGemmNopadKernel."""
    numel = T * top_k
    A, B, sizes, offsets, _ = make_inputs(T, E, top_k, N, K, dtype, distribution)

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count

    ref_kernel = MoeGroupedGemmNopadKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=dtype)
    C_ref = ref_kernel(A, B, sizes, offsets)

    persistent_kernel = MoeGroupedGemmPersistentKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=dtype, sm_count=sm_count)
    C_out = persistent_kernel(A, B, sizes, offsets)

    torch.testing.assert_close(
        C_out.float(), C_ref.float(),
        atol=1e-2, rtol=1e-2,
        msg=f"Mismatch: T={T} E={E} top_k={top_k} N={N} K={K} {dtype} dist={distribution}")


@pytest.mark.nightly
@pytest.mark.parametrize("E", [128, 256], ids=["E128", "E256"])
def test_large_expert_count(E):
    """Persistent kernel correctness with large expert counts (warp scan multi-round)."""
    T, top_k, N, K = 512, 8, 256, 128
    numel = T * top_k
    dtype = torch.bfloat16
    A, B, sizes, offsets, _ = make_inputs(T, E, top_k, N, K, dtype, "uniform")

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    ref_kernel = MoeGroupedGemmNopadKernel(numel=numel, num_experts=E, N=N, K=K, dtype=dtype)
    C_ref = ref_kernel(A, B, sizes, offsets)

    persistent_kernel = MoeGroupedGemmPersistentKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=dtype, sm_count=sm_count)
    C_out = persistent_kernel(A, B, sizes, offsets)

    torch.testing.assert_close(C_out.float(), C_ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
def test_zero_tokens_some_experts():
    """Some experts have zero tokens — persistent kernel handles empty experts correctly."""
    T, top_k, N, K, E = 16, 2, 128, 64, 8
    numel = T * top_k
    dtype = torch.bfloat16
    A, B, sizes, offsets, _ = make_inputs(T, E, top_k, N, K, dtype)
    # Force experts 2, 5 to have 0 tokens (transfer their tokens to expert 0)
    extra = sizes[2].item() + sizes[5].item()
    sizes[0] += extra
    sizes[2] = 0
    sizes[5] = 0
    offsets = torch.zeros(E, dtype=torch.int32, device="cuda")
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    ref_kernel = MoeGroupedGemmNopadKernel(numel=numel, num_experts=E, N=N, K=K, dtype=dtype)
    C_ref = ref_kernel(A, B, sizes, offsets)

    persistent_kernel = MoeGroupedGemmPersistentKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=dtype, sm_count=sm_count)
    C_out = persistent_kernel(A, B, sizes, offsets)

    torch.testing.assert_close(C_out.float(), C_ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
def test_all_zero_tokens():
    """total_tiles=0: all experts have zero tokens — kernel returns zero C without hanging."""
    E, N, K = 8, 128, 64
    numel = 16  # constructor value; actual numel is 0 but we need non-zero for allocation
    dtype = torch.bfloat16
    # All experts have 0 tokens
    sizes = torch.zeros(E, dtype=torch.int32, device="cuda")
    offsets = torch.zeros(E, dtype=torch.int32, device="cuda")
    # A and B with valid shapes (kernel should not read A at all since total_tiles=0)
    A = torch.randn(numel, K, dtype=dtype, device="cuda")
    B = torch.randn(E, N, K, dtype=dtype, device="cuda")
    C_expected = torch.zeros(numel, N, dtype=dtype, device="cuda")

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    persistent_kernel = MoeGroupedGemmPersistentKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=dtype, sm_count=sm_count)
    C_out = persistent_kernel(A, B, sizes, offsets)

    # With total_tiles=0, the kernel should exit immediately and C should be all-zero.
    assert C_out.shape == (numel, N)
    assert torch.all(C_out == 0), f"Expected all-zero output when total_tiles=0, got max={C_out.abs().max()}"
