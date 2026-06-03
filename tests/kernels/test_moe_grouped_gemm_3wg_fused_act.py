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
        sizes = torch.full((E,), numel // E, dtype=torch.int32, device=dev)
        sizes[:numel % E] += 1  # spread remainder; safe when numel < E
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


def _ref_fused_act(A, B, sizes, offsets, ffn, activation):
    """Per-expert ground truth: act(gate) * up over tight rows."""
    numel = A.shape[0]
    out = torch.zeros(numel, ffn, dtype=A.dtype, device=A.device)
    act_fn = {
        "silu_and_mul": lambda g, u: F.silu(g) * u,
        "gelu_and_mul": lambda g, u: F.gelu(g, approximate="none") * u,
    }[activation]
    for e in range(B.shape[0]):
        n = int(sizes[e])
        o = int(offsets[e])
        if n == 0:
            continue
        gate_up = A[o:o + n].float() @ B[e].float().t()  # [n, 2*ffn]
        out[o:o + n] = act_fn(gate_up[:, :ffn], gate_up[:, ffn:]).to(A.dtype)
    return out


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


@pytest.mark.nightly
@pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_and_mul"])
def test_pingpong_against_reference(activation):
    T_count, E, top_k, ffn, K = 256, 8, 2, 256, 128
    A, B, sizes, offsets, numel = make_inputs(T_count, E, top_k, ffn, K, torch.bfloat16, "uniform")
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    # num_stages=2: the dual-B ring (gate+up) at block_n=128 makes ns>=3
    # exceed the H100/H200 ~227 KB dynamic-SMEM opt-in cap (ns=3 -> 272 KB),
    # which the kernel's own autotune_configs SMEM formula also prunes.
    cfg = {"block_m": 64, "block_n": 128, "block_k": 64,
           "num_stages": 2, "threads": 384, "group_size_m": 1}
    k = MoeGroupedGemmPersistent3WGFusedActKernel(
        numel=numel, num_experts=E, N=ffn, K=K, dtype=torch.bfloat16,
        activation=activation, sm_count=sm, config=cfg)
    C = k(A, B, sizes, offsets)
    ref = _ref_fused_act(A, B, sizes, offsets, ffn, activation)
    torch.testing.assert_close(C, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.nightly
def test_pingpong_partial_m_tile():
    """Force arows < block_m so the predicated STG fallback path runs."""
    # Per-expert sizes deliberately NOT multiples of block_m (64): 50, 30, 70, 90
    # -> each expert's last M-tile is partial.
    E, _, ffn, K = 4, 1, 256, 128
    sizes_list = [50, 30, 70, 90]
    numel = sum(sizes_list)
    dev = "cuda"
    torch.manual_seed(7)
    sizes = torch.tensor(sizes_list, dtype=torch.int32, device=dev)
    offsets = torch.zeros(E, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    A = torch.randn(numel, K, dtype=torch.bfloat16, device=dev) * 0.02
    B = torch.randn(E, 2 * ffn, K, dtype=torch.bfloat16, device=dev) * 0.02
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    cfg = {"block_m": 64, "block_n": 128, "block_k": 64,
           "num_stages": 2, "threads": 384, "group_size_m": 1}
    k = MoeGroupedGemmPersistent3WGFusedActKernel(
        numel=numel, num_experts=E, N=ffn, K=K, dtype=torch.bfloat16,
        activation="silu_and_mul", sm_count=sm, config=cfg)
    C = k(A, B, sizes, offsets)
    ref = _ref_fused_act(A, B, sizes, offsets, ffn, "silu_and_mul")
    torch.testing.assert_close(C, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.nightly
@pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_and_mul"])
@pytest.mark.parametrize("dist", ["uniform", "skewed"])
def test_cooperative_against_reference(activation, dist):
    T_count, E, top_k, ffn, K = 512, 16, 2, 1536, 128   # real-scale ffn, multiple N-tiles
    A, B, sizes, offsets, numel = make_inputs(T_count, E, top_k, ffn, K, torch.bfloat16, dist)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    k = MoeGroupedGemmPersistent3WGFusedActKernel(   # default config: block_m=128 cooperative, ns=3
        numel=numel, num_experts=E, N=ffn, K=K, dtype=torch.bfloat16,
        activation=activation, sm_count=sm)
    assert k.config["block_m"] >= 128, "expected cooperative template (block_m>=128)"
    C = k(A, B, sizes, offsets)
    ref = _ref_fused_act(A, B, sizes, offsets, ffn, activation)
    torch.testing.assert_close(C, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.nightly
def test_many_zero_experts():
    """Many zero-size experts -> many waves, tight C_shared reuse (race regression)."""
    T_count, E, top_k, ffn, K = 256, 64, 2, 768, 128
    A, B, sizes, offsets, numel = make_inputs(T_count, E, top_k, ffn, K, torch.bfloat16, "skewed")
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    k = MoeGroupedGemmPersistent3WGFusedActKernel(
        numel=numel, num_experts=E, N=ffn, K=K, dtype=torch.bfloat16,
        activation="silu_and_mul", sm_count=sm)
    for _ in range(5):  # repeat: race is intermittent
        C = k(A, B, sizes, offsets)
        ref = _ref_fused_act(A, B, sizes, offsets, ffn, "silu_and_mul")
        assert not torch.isnan(C).any(), "NaN in output (C_shared reuse race)"
        torch.testing.assert_close(C, ref, rtol=2e-2, atol=2e-2)
