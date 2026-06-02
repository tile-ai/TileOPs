"""Correctness tests for MoeGroupedGemmPersistent3WGFusedActKernel."""
import pytest
import torch
import torch.nn.functional as F

from tileops.kernels.moe import MoeGroupedGemmPersistent3WGFusedActKernel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires SM90 (Hopper)",
)


def make_inputs(T, E, top_k, ffn, K, dtype, distribution="uniform", seed=42):
    """A[numel,K], B[E, 2*ffn, K] (gate||up), int32 sizes/offsets."""
    torch.manual_seed(seed)
    numel = T * top_k
    dev = "cuda"
    if distribution == "uniform":
        tpe = max(1, numel // E)
        sizes = torch.full((E,), tpe, dtype=torch.int32, device=dev)
        sizes[-1] = numel - tpe * (E - 1)
    else:  # skewed: a few fat experts, many size-0/1 (exercises many waves)
        sizes = torch.zeros(E, dtype=torch.int32, device=dev)
        top = max(1, E // 8)
        per = numel // top
        sizes[:top] = per
        sizes[0] += numel - per * top
    offsets = torch.zeros(E, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    A = torch.randn(numel, K, dtype=dtype, device=dev) * 0.02
    B = torch.randn(E, 2 * ffn, K, dtype=dtype, device=dev) * 0.02
    return A, B, sizes, offsets, numel


@pytest.mark.smoke
def test_import():
    assert MoeGroupedGemmPersistent3WGFusedActKernel is not None


@pytest.mark.smoke
def test_output_shape():
    T, E, top_k, ffn, K = 32, 4, 2, 256, 64
    A, B, sizes, offsets, numel = make_inputs(T, E, top_k, ffn, K, torch.bfloat16)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    kernel = MoeGroupedGemmPersistent3WGFusedActKernel(
        numel=numel, num_experts=E, N=ffn, K=K, dtype=torch.bfloat16,
        activation="silu_and_mul", sm_count=sm)
    C = kernel(A, B, sizes, offsets)
    assert C.shape == (numel, ffn)
