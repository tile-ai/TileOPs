"""Correctness tests for GroupedGemmPersistent3WGKernel.

Verifies that the 3-WG kernel produces the same output as the 2-WG
GroupedGemmPersistentKernel across shapes and distributions.
The 3-WG kernel is K-aligned only, so all test shapes satisfy K % block_k == 0.
"""

import pytest
import torch

from tileops.kernels.grouped_gemm import (
    GroupedGemmPersistent3WGKernel,
    GroupedGemmPersistentKernel,
)


def make_inputs(T: int, E: int, top_k: int, N: int, K: int,
                dtype: torch.dtype, distribution: str = "uniform",
                seed: int = 42):
    torch.manual_seed(seed)
    numel = T * top_k
    dev = "cuda"

    if distribution == "uniform":
        tokens_per_expert = max(1, numel // E)
        sizes = torch.zeros(E, dtype=torch.int32, device=dev)
        sizes[:] = tokens_per_expert
        sizes[-1] = numel - tokens_per_expert * (E - 1)
    else:
        sizes = torch.ones(E, dtype=torch.int32, device=dev)
        extra = numel - E
        top_experts = max(1, E // 5)
        per_top = extra // top_experts
        sizes[:top_experts] += per_top
        sizes[0] += extra - per_top * top_experts

    offsets = torch.zeros(E, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)

    A = torch.randn(numel, K, dtype=dtype, device=dev) * 0.02
    B = torch.randn(E, N, K, dtype=dtype, device=dev) * 0.02
    return A, B, sizes, offsets, numel


@pytest.mark.smoke
def test_import():
    from tileops.kernels.grouped_gemm import GroupedGemmPersistent3WGKernel  # noqa: F401


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_output_shape(dtype):
    T, E, top_k, N, K = 32, 4, 2, 256, 64
    numel = T * top_k
    A, B, sizes, offsets, _ = make_inputs(T, E, top_k, N, K, dtype)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    kernel = GroupedGemmPersistent3WGKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=dtype, sm_count=sm)
    C = kernel(A, B, sizes, offsets)
    assert C.shape == (numel, N)
    assert C.dtype == dtype


@pytest.mark.nightly
@pytest.mark.parametrize("distribution", ["uniform", "skewed"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("T,E,top_k,N,K", [
    (32,  4,  2,  256,  64),
    (64,  8,  4,  256,  128),
    (128, 16, 4,  512,  256),
], ids=["small", "medium", "larger"])
def test_matches_2wg_kernel(T, E, top_k, N, K, dtype, distribution):
    numel = T * top_k
    A, B, sizes, offsets, _ = make_inputs(T, E, top_k, N, K, dtype, distribution)
    sm = torch.cuda.get_device_properties(0).multi_processor_count

    ref = GroupedGemmPersistentKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=dtype, sm_count=sm)
    C_ref = ref(A, B, sizes, offsets)

    k3 = GroupedGemmPersistent3WGKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=dtype, sm_count=sm)
    C_out = k3(A, B, sizes, offsets)

    torch.testing.assert_close(
        C_out.float(), C_ref.float(),
        atol=1e-2, rtol=1e-2,
        msg=f"Mismatch: T={T} E={E} top_k={top_k} N={N} K={K} {dtype} dist={distribution}")


@pytest.mark.nightly
@pytest.mark.parametrize("E", [128, 256], ids=["E128", "E256"])
def test_large_expert_count(E):
    T, top_k, N, K = 512, 8, 256, 128
    numel = T * top_k
    dtype = torch.bfloat16
    A, B, sizes, offsets, _ = make_inputs(T, E, top_k, N, K, dtype, "uniform")
    sm = torch.cuda.get_device_properties(0).multi_processor_count

    ref = GroupedGemmPersistentKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=dtype, sm_count=sm)
    C_ref = ref(A, B, sizes, offsets)

    k3 = GroupedGemmPersistent3WGKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=dtype, sm_count=sm)
    C_out = k3(A, B, sizes, offsets)

    torch.testing.assert_close(C_out.float(), C_ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
def test_skewed_decode_shape():
    """Decode-like shape with skewed distribution — the case that previously hung."""
    T, E, top_k, N, K = 64, 16, 4, 256, 128
    numel = T * top_k
    dtype = torch.bfloat16
    A, B, sizes, offsets, _ = make_inputs(T, E, top_k, N, K, dtype, "skewed")
    sm = torch.cuda.get_device_properties(0).multi_processor_count

    ref = GroupedGemmPersistentKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=dtype, sm_count=sm)
    C_ref = ref(A, B, sizes, offsets)

    k3 = GroupedGemmPersistent3WGKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=dtype, sm_count=sm)
    C_out = k3(A, B, sizes, offsets)

    torch.testing.assert_close(C_out.float(), C_ref.float(), atol=1e-2, rtol=1e-2)
