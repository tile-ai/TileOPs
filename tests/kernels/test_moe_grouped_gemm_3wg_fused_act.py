"""Correctness tests for MoeGroupedGemmPersistent3WGFusedActKernel."""
import pytest
import torch
import torch.nn.functional as F

from tileops.kernels.grouped_gemm import GroupedGemmCall
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
def test_output_shape():
    T, E, top_k, ffn, K = 32, 4, 2, 256, 64
    A, B, sizes, offsets, numel = make_inputs(T, E, top_k, ffn, K, torch.bfloat16)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    kernel = MoeGroupedGemmPersistent3WGFusedActKernel(
        numel=numel, num_experts=E, N=ffn, K=K, dtype=torch.bfloat16,
        activation="silu_and_mul", sm_count=sm)
    C = kernel(A, B, sizes, offsets)
    assert C.shape == (numel, ffn)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "numel,num_experts,expected_block_m,expected_block_k",
    [
        pytest.param(2048, 64, 128, 128, id="dense-decode-cooperative"),
        pytest.param(2048, 128, 64, 128, id="sparse-decode-pingpong"),
        pytest.param(2048, 8, 128, 64, id="prefill-keeps-the-default"),
    ],
)
def test_row_count_selects_the_schedule(numel, num_experts, expected_block_m,
                                        expected_block_k):
    """Rows per expert pick the schedule; the decode ones are the BK128 pair."""
    kernel = MoeGroupedGemmPersistent3WGFusedActKernel(
        numel=numel, num_experts=num_experts, N=256, K=256,
        dtype=torch.bfloat16, activation="silu_and_mul",
    )
    assert kernel.config["block_m"] == expected_block_m
    assert kernel.config["block_k"] == expected_block_k


@pytest.mark.smoke
@pytest.mark.parametrize(
    "numel,num_experts,n,k,expected",
    [
        pytest.param(2048, 128, 2048, 7168, True, id="decode-shaped"),
        pytest.param(2048, 8, 2048, 7168, False, id="too-many-rows-per-expert"),
        pytest.param(2048, 128, 2048, 64, False, id="k-cannot-use-the-decode-schedule"),
        pytest.param(128, 128, 256, 7168, True, id="too-small-to-fill-the-device"),
    ],
)
def test_which_shapes_want_the_fused_epilogue(numel, num_experts, n, k, expected):
    """The region the gate_up stage asks about before it picks this pipeline."""
    call = GroupedGemmCall(numel=numel, num_experts=num_experts, n=n, k=k,
                           dtype=torch.bfloat16, activation="silu_and_mul")
    assert MoeGroupedGemmPersistent3WGFusedActKernel.applies(call) is expected


@pytest.mark.smoke
def test_an_activation_it_cannot_carry_is_outside_the_region():
    """The activation is part of the call, so the region covers it too."""
    call = GroupedGemmCall(numel=2048, num_experts=128, n=2048, k=7168,
                           dtype=torch.bfloat16, activation="unknown_act")
    assert MoeGroupedGemmPersistent3WGFusedActKernel.applies(call) is False


@pytest.mark.smoke
def test_cooperative_sparse_bottom_half_paths():
    """Exercise both partial and empty WG1 halves in one cooperative launch."""
    E, ffn, K = 2, 256, 128
    sizes = torch.tensor([96, 32], dtype=torch.int32, device="cuda")
    offsets = torch.tensor([0, 96], dtype=torch.int32, device="cuda")
    numel = int(sizes.sum())
    torch.manual_seed(17)
    A = torch.randn(numel, K, dtype=torch.bfloat16, device="cuda") * 0.02
    B = torch.randn(E, 2 * ffn, K, dtype=torch.bfloat16, device="cuda") * 0.02
    config = {
        "block_m": 128,
        "block_n": 128,
        "block_k": 128,
        "num_stages": 2,
        "threads": 384,
        "group_size_m": 1,
    }
    kernel = MoeGroupedGemmPersistent3WGFusedActKernel(
        numel=numel,
        num_experts=E,
        N=ffn,
        K=K,
        dtype=torch.bfloat16,
        activation="silu_and_mul",
        config=config,
    )
    actual = kernel(A, B, sizes, offsets)
    expected = _ref_fused_act(A, B, sizes, offsets, ffn, "silu_and_mul")
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


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
