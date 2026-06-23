"""Correctness tests for GroupedGemmPersistent3WGKernel.

Verifies 3WG produces same output as the 2WG reference.
"""
import pytest
import torch

from tileops.kernels.grouped_gemm import (
    GroupedGemmPersistent3WGKernel,
    GroupedGemmPersistentKernel,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires SM90 (Hopper)",
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
    assert GroupedGemmPersistent3WGKernel is not None


@pytest.mark.smoke
def test_output_shape():
    T, E, top_k, N, K = 32, 4, 2, 256, 64
    numel = T * top_k
    A, B, sizes, offsets, _ = make_inputs(T, E, top_k, N, K, torch.bfloat16)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    kernel = GroupedGemmPersistent3WGKernel(
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
    v2 = GroupedGemmPersistent3WGKernel(
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
    v2 = GroupedGemmPersistent3WGKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm, config=cfg)
    C_ref = ref(A, B, sizes, offsets)
    C_v2 = v2(A, B, sizes, offsets)
    torch.testing.assert_close(C_v2, C_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.nightly
def test_partial_m_tile():
    """Exercise the STG-fallback slow path: per-expert size that is NOT a
    multiple of block_m, so the trailing m-tile of each expert has
    arows < block_m and must take the predicated direct-STG epilogue.

    The TMA-store fast path triggers for the leading full tiles; this test
    guarantees both paths are exercised in a single run."""
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    E, N, K = 4, 256, 128
    block_m = 64
    # Make every expert end on a partial tile (size = 1 full block_m tile + 17 rows).
    per_expert = block_m + 17
    numel = E * per_expert
    sizes = torch.full((E,), per_expert, dtype=torch.int32, device="cuda")
    offsets = torch.zeros(E, dtype=torch.int32, device="cuda")
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    torch.manual_seed(0)
    A = torch.randn(numel, K, dtype=torch.bfloat16, device="cuda") * 0.02
    B = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.02
    ref = GroupedGemmPersistentKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm)
    v2 = GroupedGemmPersistent3WGKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm)
    C_ref = ref(A, B, sizes, offsets)
    C_v2 = v2(A, B, sizes, offsets)
    torch.testing.assert_close(C_v2, C_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.nightly
def test_max_waves_edge_case():
    """Validate static-wave slack: cross the 2*sm CTA-pair boundary AND
    end on a partial last wave with an odd tile count, exercising the
    `if valid_1:` guard (WG1-OOB on the final pair claim)."""
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    block_m, K, N = 64, 128, 768
    E = 4
    # numel = 3 * sm * 64 + 31; per-expert ceil-rounded -> ~3*sm M-tiles plus a few; num_pid_n=3
    # total tiles ~= 3 * (3*sm + small) ~ 9*sm; CTA pairs ~= 4.5*sm -> _max_waves ~ 5+slack
    # odd numel ensures a trailing tile triggers `valid_1=False` on the last claim
    numel = 3 * sm * block_m + 31
    T_count = numel
    A, B, sizes, offsets, _ = make_inputs(T_count, E, 1, N, K, torch.bfloat16, "uniform")
    ref = GroupedGemmPersistentKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm)
    v2 = GroupedGemmPersistent3WGKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm)
    C_ref = ref(A, B, sizes, offsets)
    C_v2 = v2(A, B, sizes, offsets)
    torch.testing.assert_close(C_v2, C_ref, rtol=2e-2, atol=2e-2)


# ════════════════════════════════════════════════════════════════════════
# Cooperative template (block_m >= 128) — phase 5
# ════════════════════════════════════════════════════════════════════════

_COOP_CFGS = [
    {"block_m": 128, "block_n": 128, "block_k": 64, "num_stages": 2, "threads": 384, "group_size_m": 1},
    {"block_m": 128, "block_n": 128, "block_k": 64, "num_stages": 3, "threads": 384, "group_size_m": 1},
    {"block_m": 128, "block_n": 256, "block_k": 64, "num_stages": 2, "threads": 384, "group_size_m": 1},
    {"block_m": 128, "block_n": 256, "block_k": 64, "num_stages": 3, "threads": 384, "group_size_m": 1},
]


@pytest.mark.nightly
@pytest.mark.parametrize("cfg", _COOP_CFGS, ids=lambda c: f"bm{c['block_m']}_bn{c['block_n']}_ns{c['num_stages']}")
@pytest.mark.parametrize("dist", ["uniform", "skewed"])
def test_cooperative_correctness(cfg, dist):
    """Cooperative path (bm>=128, split-A) vs 2WG reference across the
    surviving autotune configs and both routing distributions.
    Aligned per-expert sizes so every tile is a full block_m=128 tile,
    isolating the WGMMA + barrier-set + epilogue-fast-path correctness.
    """
    T_count, E, top_k, N, K = 512, 8, 2, 256, 128
    numel = T_count * top_k
    A, B, sizes, offsets, _ = make_inputs(T_count, E, top_k, N, K, torch.bfloat16, dist)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    ref = GroupedGemmPersistentKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm)
    v2 = GroupedGemmPersistent3WGKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm, config=cfg)
    C_ref = ref(A, B, sizes, offsets)
    C_v2 = v2(A, B, sizes, offsets)
    torch.testing.assert_close(C_v2, C_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.nightly
def test_cooperative_partial_m_tile():
    """Cooperative path: each math WG owns half_m=64 rows of a bm=128 tile,
    so partial-m has THREE distinct boundary classes:
      * arows >= 128: both WGs full → both TMA-store
      * 64 <= arows < 128: WG0 full TMA, WG1 partial STG (arows-64 rows)
      * arows < 64: WG0 partial STG (arows rows), WG1 skips entirely

    Pick per-expert sizes that hit all three:
      sizes = [40, 90, 128, 200] →
        * expert 0 has 40 rows: a single tile with arows=40 (WG0 partial,  WG1 skip)
        * expert 1 has 90 rows: a single tile with arows=90 (WG0 full,     WG1 partial 26)
        * expert 2 has 128 rows: a single tile with arows=128 (both full)
        * expert 3 has 200 rows: tile 0 full (128), tile 1 partial 72
                                 (WG0 full 64, WG1 partial 8)
    """
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    cfg = {"block_m": 128, "block_n": 128, "block_k": 64,
           "num_stages": 2, "threads": 384, "group_size_m": 1}
    sizes = torch.tensor([40, 90, 128, 200], dtype=torch.int32, device="cuda")
    numel = int(sizes.sum().item())
    offsets = torch.zeros(4, dtype=torch.int32, device="cuda")
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    E, N, K = 4, 256, 128
    torch.manual_seed(0)
    A = torch.randn(numel, K, dtype=torch.bfloat16, device="cuda") * 0.02
    B = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.02
    ref = GroupedGemmPersistentKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm)
    v2 = GroupedGemmPersistent3WGKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm, config=cfg)
    C_ref = ref(A, B, sizes, offsets)
    C_v2 = v2(A, B, sizes, offsets)
    torch.testing.assert_close(C_v2, C_ref, rtol=2e-2, atol=2e-2)


# ════════════════════════════════════════════════════════════════════════
# Calling-cost parity with DeepGEMM: out= reuse
# ════════════════════════════════════════════════════════════════════════
def _aligned_coop_inputs(seed=42):
    """Cooperative-path inputs with every expert a multiple of block_m=128."""
    E, N, K = 4, 256, 128
    per = 256  # 2 full block_m=128 tiles per expert
    numel = E * per
    torch.manual_seed(seed)
    sizes = torch.full((E,), per, dtype=torch.int32, device="cuda")
    offsets = torch.zeros(E, dtype=torch.int32, device="cuda")
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    A = torch.randn(numel, K, dtype=torch.bfloat16, device="cuda") * 0.02
    B = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.02
    return A, B, sizes, offsets, numel, N, K, E


@pytest.mark.nightly
def test_out_param_reuse():
    """Caller-provided ``out`` buffer is written in place and returned, with
    no internal allocation, matching the result of the allocating path."""
    A, B, sizes, offsets, numel, N, K, E = _aligned_coop_inputs()
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    k3 = GroupedGemmPersistent3WGKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm)
    C_alloc = k3(A, B, sizes, offsets)
    out = torch.empty(numel, N, dtype=torch.bfloat16, device="cuda")
    C_out = k3(A, B, sizes, offsets, out=out)
    assert C_out.data_ptr() == out.data_ptr()
    torch.testing.assert_close(C_out, C_alloc, rtol=2e-2, atol=2e-2)


@pytest.mark.nightly
def test_out_buffer_validation():
    """A caller-provided ``out`` with the wrong shape, dtype, device, layout, or
    one that overlaps A/B in memory is rejected with a clear ValueError instead
    of corrupting silently; disjoint slices of a shared workspace are accepted."""
    A, B, sizes, offsets, numel, N, K, E = _aligned_coop_inputs()
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    k3 = GroupedGemmPersistent3WGKernel(
        numel=numel, num_experts=E, N=N, K=K, dtype=torch.bfloat16, sm_count=sm)

    with pytest.raises(ValueError, match="out shape"):
        k3(A, B, sizes, offsets,
           out=torch.empty(numel, N + 1, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(ValueError, match="out dtype"):
        k3(A, B, sizes, offsets,
           out=torch.empty(numel, N, dtype=torch.float16, device="cuda"))
    with pytest.raises(ValueError, match="out device"):
        k3(A, B, sizes, offsets,
           out=torch.empty(numel, N, dtype=torch.bfloat16, device="cpu"))
    # Non-contiguous view with the right logical shape (transpose of [N, numel]):
    # passes shape/dtype/device but must be rejected on layout.
    with pytest.raises(ValueError, match="contiguous"):
        k3(A, B, sizes, offsets,
           out=torch.empty(N, numel, dtype=torch.bfloat16, device="cuda").t())
    # out overlapping the input: A and out are the same storage region. Passes
    # shape/dtype/device/contiguity but must be rejected (read/write race).
    shared = torch.empty(numel * max(N, K), dtype=torch.bfloat16, device="cuda")
    a_alias = shared[:numel * K].view(numel, K)
    out_alias = shared[:numel * N].view(numel, N)  # overlaps a_alias from byte 0
    with pytest.raises(ValueError, match="overlap"):
        k3(a_alias, B, sizes, offsets, out=out_alias)

    # Disjoint slices of one workspace must be ACCEPTED (vLLM-style): a_ws and
    # out_ws share storage but their byte intervals do not overlap.
    C_ref = k3(A, B, sizes, offsets)
    ws = torch.empty(numel * K + numel * N, dtype=torch.bfloat16, device="cuda")
    a_ws = ws[:numel * K].view(numel, K)
    out_ws = ws[numel * K:].view(numel, N)
    a_ws.copy_(A)
    res = k3(a_ws, B, sizes, offsets, out=out_ws)
    torch.testing.assert_close(res, C_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.nightly
def test_no_host_pad_on_unaligned(monkeypatch):
    """OOB zero-fill: forward must not call F.pad even when experts are unaligned.

    Guards T-A — the last expert's partial-tile A over-read is hardware
    zero-filled (TMA globalDim=numel), so no physical guard rows are needed.
    """
    torch.manual_seed(0)
    num_experts, N, K = 4, 256, 128          # default block_m=128 (cooperative)
    sizes = torch.tensor([130, 70, 200, 33], dtype=torch.int32, device="cuda")
    numel = int(sizes.sum())
    offsets = torch.zeros(num_experts, dtype=torch.int32, device="cuda")
    offsets[1:] = torch.cumsum(sizes[:-1], 0)
    A = torch.randn(numel, K, dtype=torch.bfloat16, device="cuda") * 0.02
    B = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda") * 0.02

    calls = {"n": 0}
    real_pad = torch.nn.functional.pad

    def spy_pad(*args, **kwargs):
        calls["n"] += 1
        return real_pad(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "pad", spy_pad)
    kernel = GroupedGemmPersistent3WGKernel(numel, num_experts, N, K)
    kernel(A, B, sizes, offsets)
    assert calls["n"] == 0, "forward still calls F.pad; OOB zero-fill not in effect"
