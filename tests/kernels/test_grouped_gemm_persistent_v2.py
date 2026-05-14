"""Correctness tests for GroupedGemmPersistentV2Kernel.

Verifies V2 produces same output as the 2WG reference. 3WG cross-check will be added in a later phase.
"""
import pytest
import torch

from tileops.kernels.grouped_gemm import (
    GroupedGemmPersistentKernel,
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


@pytest.mark.nightly
@pytest.mark.parametrize("num_stages", [2, 3, 4])
def test_correctness_stages(num_stages):
    T_count, E, top_k, N, K = 512, 8, 2, 256, 128
    numel = T_count * top_k
    A, B, sizes, offsets, _ = make_inputs(T_count, E, top_k, N, K, torch.bfloat16, "uniform")
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    cfg = {"block_m": 64, "block_n": 128, "block_k": 64,    # block_n reduced from 256 to fit num_stages>=3
           "num_stages": num_stages, "threads": 384, "group_size_m": 1}
    ref = GroupedGemmPersistentKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm)
    v2 = GroupedGemmPersistentV2Kernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm, config=cfg)
    C_ref = ref(A, B, sizes, offsets)
    C_v2 = v2(A, B, sizes, offsets)
    torch.testing.assert_close(C_v2, C_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.nightly
def test_max_waves_edge_case():
    """Validate static-wave slack: cross the 2*sm CTA-pair boundary AND
    end on a partial last wave with an odd tile count, exercising the
    `if valid_1:` guard (WG1-OOB on the final pair claim)."""
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    block_m, block_n, K, N = 64, 256, 128, 768
    E, top_k = 4, 1
    # numel = 3 * sm * 64 + 31; per-expert ceil-rounded -> ~3*sm M-tiles plus a few; num_pid_n=3
    # total tiles ~= 3 * (3*sm + small) ~ 9*sm; CTA pairs ~= 4.5*sm -> _max_waves ~ 5+slack
    # odd numel ensures a trailing tile triggers `valid_1=False` on the last claim
    numel = 3 * sm * block_m + 31
    T_count = numel  # numel = T * top_k with top_k=1
    A, B, sizes, offsets, _ = make_inputs(T_count, E, top_k, N, K, torch.bfloat16, "uniform")
    ref = GroupedGemmPersistentKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm)
    v2 = GroupedGemmPersistentV2Kernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm)
    C_ref = ref(A, B, sizes, offsets)
    C_v2 = v2(A, B, sizes, offsets)
    torch.testing.assert_close(C_v2, C_ref, rtol=2e-2, atol=2e-2)
