"""Correctness tests for GroupedGemmPersistentV2Kernel.

Verifies V2 produces same output as the 2WG reference and current 3WG kernel.
"""
import pytest
import torch

from tileops.kernels.grouped_gemm import (
    GroupedGemmPersistentKernel,
    GroupedGemmPersistent3WGKernel,
    GroupedGemmPersistentV2Kernel,
)


def make_inputs(T, E, top_k, N, K, dtype, distribution="uniform", seed=42):
    torch.manual_seed(seed)
    numel = T * top_k
    dev = "cuda"
    if distribution == "uniform":
        tpe = max(1, numel // E)
        sizes = torch.zeros(E, dtype=torch.int32, device=dev)
        sizes[:] = tpe
        sizes[-1] = numel - tpe * (E - 1)
    else:
        sizes = torch.ones(E, dtype=torch.int32, device=dev)
        extra = numel - E
        top = max(1, E // 5)
        per = extra // top
        sizes[:top] += per
        sizes[0] += extra - per * top
    offsets = torch.zeros(E, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    A = torch.randn(numel, K, dtype=dtype, device=dev) * 0.02
    B = torch.randn(E, N, K, dtype=dtype, device=dev) * 0.02
    return A, B, sizes, offsets, numel


@pytest.mark.smoke
def test_import():
    assert GroupedGemmPersistentV2Kernel is not None


@pytest.mark.smoke
def test_output_shape():
    T, E, top_k, N, K = 32, 4, 2, 256, 64
    numel = T * top_k
    A, B, sizes, offsets, _ = make_inputs(T, E, top_k, N, K, torch.bfloat16)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    kernel = GroupedGemmPersistentV2Kernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm)
    C = kernel(A, B, sizes, offsets)
    assert C.shape == (numel, N)


@pytest.mark.nightly
@pytest.mark.parametrize("dist", ["uniform", "skewed"])
def test_against_2wg_reference(dist):
    T, E, top_k, N, K = 512, 8, 2, 256, 128
    numel = T * top_k
    A, B, sizes, offsets, _ = make_inputs(T, E, top_k, N, K, torch.bfloat16, dist)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    ref = GroupedGemmPersistentKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm)
    v2 = GroupedGemmPersistentV2Kernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm)
    C_ref = ref(A, B, sizes, offsets)
    C_v2 = v2(A, B, sizes, offsets)
    torch.testing.assert_close(C_v2, C_ref, rtol=2e-2, atol=2e-2)
