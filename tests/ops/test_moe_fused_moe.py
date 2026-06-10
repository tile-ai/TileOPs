"""Tests for FusedMoe — unified routed MoE FFN operator.

Covers:
  - Qwen3 config: softmax, renormalize=False/True
  - Kimi K2 config: sigmoid, correction_bias, routed_scaling_factor=2.827
  - expert_map local filtering (EP simulation without All-to-All)
  - vLLM correctness (optional, skipped when vLLM is not installed)
  - correction_bias routing precision (weights from original sigmoid, not biased)
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase
from tileops.ops.moe import (
    FusedMoe,
    FusedMoeFwdCbFwdOp,
    FusedMoeFwdOp,
    FusedTopKOp,
)

# ---------------------------------------------------------------------------
# vLLM optional import
# ---------------------------------------------------------------------------

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts as _vllm_fused_experts,
    )
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _ref_moe_ffn(
    hidden_states: torch.Tensor,   # [T, H]
    w_gate_up: torch.Tensor,       # [E, 2*F, H]
    w_down: torch.Tensor,          # [E, H, F]
    topk_weights: torch.Tensor,    # [T, K] float32
    topk_ids: torch.Tensor,        # [T, K] int64
) -> torch.Tensor:
    """PyTorch reference: per-expert GEMM (memory-efficient, no O(T*K*2F*H) alloc)."""
    T, H = hidden_states.shape
    E = w_gate_up.shape[0]
    ffn_size = w_gate_up.shape[1] // 2

    output = torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)
    for e in range(E):
        mask = (topk_ids == e)
        if not mask.any():
            continue
        t_idx, k_idx = mask.nonzero(as_tuple=True)
        h = hidden_states[t_idx].float()
        gate_up = h @ w_gate_up[e].float().t()
        act = F.silu(gate_up[:, :ffn_size]) * gate_up[:, ffn_size:]
        down = act @ w_down[e].float().t()
        weights = topk_weights[t_idx, k_idx].float().unsqueeze(-1)
        output.index_add_(0, t_idx, down * weights)
    return output.to(hidden_states.dtype)


def _ref_kimi_routing(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch Kimi K2 routing: sigmoid → biased topk → gather original weights."""
    scores = gating_output.float().sigmoid()
    tmp = (scores + correction_bias.float().unsqueeze(0)) if correction_bias is not None else scores
    topk_ids = tmp.topk(top_k, dim=-1, sorted=False).indices
    topk_weights = scores.gather(1, topk_ids)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


# ---------------------------------------------------------------------------
# Test fixture — Qwen3 config
# ---------------------------------------------------------------------------


class Qwen3Fixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
            " scoring_func, renormalize, dtype",
            [
                pytest.param(
                    32, 8, 2, 64, 32, "softmax", False, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-softmax-bf16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, "softmax", True, torch.float16,
                    marks=pytest.mark.smoke, id="smoke-softmax-renorm-fp16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, "sigmoid", True, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-sigmoid-renorm-bf16",
                ),
                pytest.param(
                    512, 128, 8, 2048, 1024, "softmax", False, torch.bfloat16,
                    marks=pytest.mark.full, id="qwen3-small",
                ),
                pytest.param(
                    2048, 128, 8, 2048, 1024, "softmax", False, torch.bfloat16,
                    marks=pytest.mark.full, id="qwen3-medium",
                ),
                pytest.param(
                    512, 256, 8, 2048, 1024, "softmax", True, torch.bfloat16,
                    marks=pytest.mark.full, id="qwen35-small",
                ),
                pytest.param(
                    512, 256, 8, 2048, 1024, "sigmoid", True, torch.bfloat16,
                    marks=pytest.mark.full, id="deepseek-small",
                ),
            ],
        )
    ]


@Qwen3Fixture
def test_fused_moe_qwen3(
    num_tokens, num_experts, top_k, hidden_size, ffn_size,
    scoring_func, renormalize, dtype,
) -> None:
    torch.manual_seed(42)
    dev = "cuda"
    hidden = torch.randn(num_tokens, hidden_size, dtype=dtype, device=dev)
    gating = torch.randn(num_tokens, num_experts, dtype=dtype, device=dev)
    w_gate_up = torch.randn(
        num_experts, ffn_size * 2, hidden_size, dtype=dtype, device=dev
    ) * 0.02
    w_down = torch.randn(
        num_experts, hidden_size, ffn_size, dtype=dtype, device=dev
    ) * 0.02

    op_nopad = FusedMoe(
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
        hidden_size=hidden_size, ffn_size=ffn_size,
        scoring_func=scoring_func, renormalize=renormalize,
        dtype=dtype,
    )

    out_nopad = op_nopad(hidden, gating, w_gate_up, w_down)

    assert out_nopad.shape == (num_tokens, hidden_size)
    assert out_nopad.dtype == dtype

    # Reference using the same FusedTopKOp routing
    fk = FusedTopKOp(num_tokens, num_experts, top_k, scoring_func, renormalize)
    topk_weights, topk_ids = fk(gating)
    ref = _ref_moe_ffn(hidden, w_gate_up, w_down, topk_weights, topk_ids.long())

    torch.testing.assert_close(out_nopad.float(), ref.float(), rtol=1e-2, atol=1e-2)

    if _VLLM_AVAILABLE and scoring_func == "softmax":
        out_vllm = _vllm_fused_experts(
            hidden.float(), w_gate_up.float(), w_down.float(),
            topk_weights, topk_ids,
        ).to(dtype)
        torch.testing.assert_close(out_nopad.float(), out_vllm.float(), rtol=1e-2, atol=1e-2)

    print(
        f"PASS [T={num_tokens}, E={num_experts}, K={top_k}, "
        f"H={hidden_size}, F={ffn_size}, fn={scoring_func}, "
        f"renorm={renormalize}, {dtype}]"
    )


# Cases for the FusedMoe non-determinism regression. The cooperative 3WG
# grouped-GEMM TMA-store epilogue race is intermittent (~3% of calls at the
# qwen3-medium scale) and only manifests inside the full FusedMoe pipeline
# (adjacent activation/permute kernels keep the timing window open) — an
# isolated grouped-GEMM loop never trips it, so this must run the op. The
# nightly case repeats many times: at 3% per call, 200 repeats give >99%
# detection. The smoke case is a fast path-coverage check.
_DET_CASES = [
    pytest.param(
        dict(num_tokens=32, num_experts=8, top_k=2, hidden_size=64,
             ffn_size=32, reps=3),
        marks=pytest.mark.smoke, id="smoke",
    ),
    pytest.param(
        dict(num_tokens=2048, num_experts=128, top_k=8, hidden_size=2048,
             ffn_size=1024, reps=200),
        marks=pytest.mark.nightly, id="qwen3-medium",
    ),
]


@pytest.mark.parametrize("case", _DET_CASES)
def test_fused_moe_deterministic(case):
    """Regression for the cooperative 3WG grouped-GEMM TMA-store epilogue race.

    The fast-path epilogue staged each full tile through a per-WG ``C_shared``
    SMEM buffer without ordering the register→SMEM write before the async TMA
    read, so the store could read a half-written ``C_shared`` on a small
    fraction of calls, corrupting a sub-tile of the output non-deterministically.
    The race only manifests inside the full FusedMoe pipeline; qwen3-medium
    (E=128, ~128 rows/expert) drives the cooperative full-tile fast path heavily.
    Run the op many times on fixed seed-42 inputs and assert every call matches
    the PyTorch reference and is bitwise-identical to the first. Pre-fix the
    qwen3-medium case trips within ~100 repeats; post-fix it is deterministic.
    """
    torch.manual_seed(42)
    dev = "cuda"
    dtype = torch.bfloat16
    nt, ne, tk = case["num_tokens"], case["num_experts"], case["top_k"]
    hs, ff, reps = case["hidden_size"], case["ffn_size"], case["reps"]
    hidden = torch.randn(nt, hs, dtype=dtype, device=dev)
    gating = torch.randn(nt, ne, dtype=dtype, device=dev)
    w_gate_up = torch.randn(ne, ff * 2, hs, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(ne, hs, ff, dtype=dtype, device=dev) * 0.02

    op = FusedMoe(
        num_tokens=nt, num_experts=ne, top_k=tk, hidden_size=hs,
        ffn_size=ff, scoring_func="softmax", renormalize=False, dtype=dtype,
    )
    fk = FusedTopKOp(nt, ne, tk, "softmax", False)
    topk_weights, topk_ids = fk(gating)
    ref = _ref_moe_ffn(hidden, w_gate_up, w_down, topk_weights, topk_ids.long())

    first = op(hidden, gating, w_gate_up, w_down)
    torch.testing.assert_close(first.float(), ref.float(), rtol=1e-2, atol=1e-2)
    for i in range(reps):
        out = op(hidden, gating, w_gate_up, w_down)
        assert torch.equal(out, first), (
            f"non-deterministic FusedMoe output on call {i + 1}/{reps}: "
            "3WG grouped-GEMM TMA-store epilogue write→read race regressed"
        )


# ---------------------------------------------------------------------------
# Test fixture — Kimi K2 config
# ---------------------------------------------------------------------------


class KimiFixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
            " routed_scaling_factor, with_correction_bias, dtype",
            [
                pytest.param(
                    32, 8, 2, 64, 32, 1.0, False, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-nobias-bf16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, 1.0, True, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-bias-bf16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, 2.827, True, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-bias-scale-bf16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, 2.827, True, torch.float16,
                    marks=pytest.mark.smoke, id="smoke-bias-scale-fp16",
                ),
                pytest.param(
                    64, 384, 8, 64, 32, 2.827, True, torch.bfloat16,
                    marks=pytest.mark.smoke, id="kimi-k2-smoke-bf16",
                ),
                pytest.param(
                    512, 384, 8, 256, 128, 2.827, True, torch.bfloat16,
                    marks=pytest.mark.full, id="kimi-k2-small-bf16",
                ),
                pytest.param(
                    2048, 384, 8, 256, 128, 2.827, True, torch.bfloat16,
                    marks=pytest.mark.full, id="kimi-k2-medium-bf16",
                ),
            ],
        )
    ]


@KimiFixture
def test_fused_moe_kimi(
    num_tokens, num_experts, top_k, hidden_size, ffn_size,
    routed_scaling_factor, with_correction_bias, dtype,
) -> None:
    torch.manual_seed(42)
    dev = "cuda"
    hidden = torch.randn(num_tokens, hidden_size, dtype=dtype, device=dev)
    gating = torch.randn(num_tokens, num_experts, dtype=dtype, device=dev)
    correction_bias = (
        torch.randn(num_experts, dtype=torch.float32, device=dev) * 0.1
        if with_correction_bias else None
    )
    w_gate_up = torch.randn(
        num_experts, ffn_size * 2, hidden_size, dtype=dtype, device=dev
    ) * 0.02
    w_down = torch.randn(
        num_experts, hidden_size, ffn_size, dtype=dtype, device=dev
    ) * 0.02

    op_nopad = FusedMoe(
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
        hidden_size=hidden_size, ffn_size=ffn_size,
        scoring_func="sigmoid", renormalize=True,
        with_correction_bias=with_correction_bias,
        routed_scaling_factor=routed_scaling_factor,
        dtype=dtype,
    )

    out_nopad = op_nopad(hidden, gating, w_gate_up, w_down, correction_bias)

    assert out_nopad.shape == (num_tokens, hidden_size)
    assert out_nopad.dtype == dtype

    # Reference using FusedTopKOp for consistent routing
    fk = FusedTopKOp(
        num_tokens, num_experts, top_k, "sigmoid", True,
        with_correction_bias=with_correction_bias,
    )
    topk_weights, topk_ids = fk(gating, correction_bias)
    ref = _ref_moe_ffn(hidden, w_gate_up, w_down, topk_weights, topk_ids.long())
    if routed_scaling_factor != 1.0:
        ref = ref * routed_scaling_factor

    torch.testing.assert_close(out_nopad.float(), ref.float(), rtol=1e-2, atol=1e-2)

    print(
        f"PASS [T={num_tokens}, E={num_experts}, K={top_k}, "
        f"H={hidden_size}, F={ffn_size}, scale={routed_scaling_factor}, "
        f"bias={with_correction_bias}, {dtype}]"
    )


# ---------------------------------------------------------------------------
# expert_map local filtering test (EP simulation without All-to-All)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_expert_map_local_filter() -> None:
    """Simulate EP=2 on a single GPU: each rank owns half the experts.

    Verification: sum of outputs from rank-0 and rank-1 (with their disjoint
    expert_maps) equals the output without expert_map.
    """
    torch.manual_seed(42)
    dev = "cuda"
    T, E, K, H, F = 32, 8, 2, 64, 32
    dtype = torch.bfloat16

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    w_gate_up = torch.randn(E, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F, dtype=dtype, device=dev) * 0.02

    # Rank 0 owns experts 0..3, rank 1 owns experts 4..7
    expert_map_rank0 = torch.full((E,), -1, dtype=torch.int32, device=dev)
    expert_map_rank0[:E // 2] = torch.arange(E // 2, dtype=torch.int32, device=dev)

    expert_map_rank1 = torch.full((E,), -1, dtype=torch.int32, device=dev)
    expert_map_rank1[E // 2:] = torch.arange(E // 2, dtype=torch.int32, device=dev)

    # Full output (no expert_map)
    op_full = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F, dtype=dtype,
    )
    out_full = op_full(hidden, gating, w_gate_up, w_down)

    # Rank-0 partial output (local experts 0..3)
    op_r0 = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F, dtype=dtype,
        expert_map=expert_map_rank0,
    )
    out_r0 = op_r0(hidden, gating, w_gate_up[:E // 2], w_down[:E // 2])

    # Rank-1 partial output (local experts 4..7)
    op_r1 = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F, dtype=dtype,
        expert_map=expert_map_rank1,
    )
    out_r1 = op_r1(hidden, gating, w_gate_up[E // 2:], w_down[E // 2:])

    # Sum of partial outputs should match full output
    out_sum = (out_r0.float() + out_r1.float()).to(dtype)
    torch.testing.assert_close(out_sum.float(), out_full.float(), rtol=1e-2, atol=1e-2)
    print(f"PASS expert_map local filter [T={T}, E={E}, K={K}, H={H}, F={F}]")


# ---------------------------------------------------------------------------
# correction_bias routing precision
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_correction_bias_routing_precision() -> None:
    """Verify correction_bias is used only for top-k selection, not for weights.

    Constructs a case where biased and unbiased top-k differ, then checks:
      - topk_ids match the biased-sort order
      - topk_weights use ORIGINAL (unbiased) sigmoid scores, renormalized
    """
    torch.manual_seed(7)
    T, E, K = 4, 8, 2
    dev = "cuda"

    logits = torch.randn(T, E, dtype=torch.float32, device=dev)
    bias = torch.randn(E, dtype=torch.float32, device=dev)

    ref_weights, ref_ids = _ref_kimi_routing(logits, bias, K)

    op = FusedTopKOp(
        num_tokens=T, num_experts=E, top_k=K,
        scoring_func="sigmoid", renormalize=True, with_correction_bias=True,
    )
    tw, ti = op(logits, bias)

    ref_ids_sorted = ref_ids.sort(dim=-1).values
    tile_ids_sorted = ti.long().sort(dim=-1).values
    assert torch.equal(ref_ids_sorted, tile_ids_sorted), (
        f"Expert selection mismatch:\n  ref={ref_ids_sorted}\n  got={tile_ids_sorted}"
    )
    torch.testing.assert_close(tw, ref_weights, rtol=1e-4, atol=1e-4)
    print("PASS correction_bias precision test")


# ---------------------------------------------------------------------------
# vLLM alignment (optional)
# ---------------------------------------------------------------------------


class VllmFixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
            " routed_scaling_factor, dtype",
            [
                pytest.param(
                    32, 8, 2, 64, 32, 1.0, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-bf16",
                ),
                pytest.param(
                    32, 8, 2, 64, 32, 2.827, torch.bfloat16,
                    marks=pytest.mark.smoke, id="smoke-scale-bf16",
                ),
                pytest.param(
                    64, 384, 8, 64, 32, 2.827, torch.bfloat16,
                    marks=pytest.mark.smoke, id="kimi-k2-smoke-bf16",
                ),
                pytest.param(
                    512, 384, 8, 256, 128, 2.827, torch.bfloat16,
                    marks=pytest.mark.full, id="kimi-k2-small-bf16",
                ),
            ],
        )
    ]


@VllmFixture
def test_fused_moe_vs_vllm(
    num_tokens, num_experts, top_k, hidden_size, ffn_size,
    routed_scaling_factor, dtype,
) -> None:
    if not _VLLM_AVAILABLE:
        pytest.skip("vllm not installed")

    torch.manual_seed(42)
    dev = "cuda"
    hidden = torch.randn(num_tokens, hidden_size, dtype=dtype, device=dev)
    gating = torch.randn(num_tokens, num_experts, dtype=dtype, device=dev)
    correction_bias = torch.randn(num_experts, dtype=torch.float32, device=dev) * 0.1
    w_gate_up = torch.randn(
        num_experts, ffn_size * 2, hidden_size, dtype=dtype, device=dev
    ) * 0.02
    w_down = torch.randn(
        num_experts, hidden_size, ffn_size, dtype=dtype, device=dev
    ) * 0.02

    fk = FusedTopKOp(
        num_tokens, num_experts, top_k, "sigmoid", True, with_correction_bias=True,
    )
    topk_weights, topk_ids = fk(gating, correction_bias)

    op = FusedMoe(
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
        hidden_size=hidden_size, ffn_size=ffn_size,
        scoring_func="sigmoid", renormalize=True, with_correction_bias=True,
        routed_scaling_factor=routed_scaling_factor,
        dtype=dtype,
    )
    out_tileops = op(hidden, gating, w_gate_up, w_down, correction_bias)

    out_vllm = _vllm_fused_experts(hidden, w_gate_up, w_down, topk_weights, topk_ids)
    if routed_scaling_factor != 1.0:
        out_vllm = out_vllm * routed_scaling_factor

    torch.testing.assert_close(
        out_tileops.float(), out_vllm.float(), rtol=1e-2, atol=1e-2,
    )
    print(
        f"PASS vs vLLM [T={num_tokens}, E={num_experts}, K={top_k}, "
        f"H={hidden_size}, F={ffn_size}, scale={routed_scaling_factor}, {dtype}]"
    )


# ---------------------------------------------------------------------------
# prepare_finalize / experts contract
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_fused_moe_fwd_op_identity() -> None:
    """`FusedMoeFwdOp` (no correction bias) matches the `FusedMoe` reference output."""
    torch.manual_seed(42)
    dev = "cuda"
    T, E, K, H, F_ = 32, 8, 2, 64, 32
    dtype = torch.bfloat16

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    w_gate_up = torch.randn(E, F_ * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F_, dtype=dtype, device=dev) * 0.02

    op = FusedMoeFwdOp(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F_, dtype=dtype,
    )
    out = op(hidden, gating, w_gate_up, w_down)
    assert out.shape == (T, H)
    assert out.dtype == dtype

    ref_op = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F_, dtype=dtype,
    )
    ref = ref_op(hidden, gating, w_gate_up, w_down)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)

    flops, nbytes = op.eval_roofline()
    assert flops > 0 and nbytes > 0


@pytest.mark.smoke
def test_fused_moe_fwd_cb_op_identity() -> None:
    """`FusedMoeFwdCbFwdOp` (with correction bias) end-to-end smoke."""
    torch.manual_seed(7)
    dev = "cuda"
    T, E, K, H, F_ = 32, 8, 2, 64, 32
    dtype = torch.bfloat16

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    correction_bias = torch.randn(E, dtype=torch.float32, device=dev) * 0.1
    w_gate_up = torch.randn(E, F_ * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F_, dtype=dtype, device=dev) * 0.02

    op = FusedMoeFwdCbFwdOp(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F_, renormalize=True, dtype=dtype,
    )
    out = op(hidden, gating, correction_bias, w_gate_up, w_down)
    assert out.shape == (T, H)
    assert out.dtype == dtype

    ref_op = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F_,
        scoring_func="sigmoid", renormalize=True, with_correction_bias=True,
        dtype=dtype,
    )
    ref = ref_op(hidden, gating, w_gate_up, w_down, correction_bias)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)

    flops, nbytes = op.eval_roofline()
    assert flops > 0 and nbytes > 0


@pytest.mark.smoke
def test_prepare_finalize_without_experts_raises() -> None:
    """Supplying prepare_finalize= without experts= must raise ValueError.

    prepare_finalize can change the dispatched token count T'; the default
    experts instance is JIT-compiled for the original T and cannot be reused.
    """
    from tileops.ops.moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEP

    with pytest.raises(ValueError, match="experts="):
        FusedMoe(
            num_tokens=16, num_experts=4, top_k=2,
            hidden_size=64, ffn_size=32,
            prepare_finalize=MoEPrepareAndFinalizeNoDPEP(),
        )
