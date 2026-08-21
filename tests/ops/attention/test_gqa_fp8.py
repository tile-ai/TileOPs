import pytest
import torch

from tileops.kernels.attention import GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel
from tileops.ops import GroupedQueryAttentionDenseFwdOp
from workloads.gqa_fp8_utils import (
    quantize_kv_fa3_descale,
    quantize_q_fa3_gqa_descale,
)


def _has_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _run_canonical_fp8_prefill(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    out_dtype: torch.dtype,
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    v_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> torch.Tensor:
    op = GroupedQueryAttentionDenseFwdOp(
        dtype=out_dtype,
        is_causal=False,
    )
    return op(
        q_fp8.contiguous(),
        k_fp8.contiguous(),
        v_fp8.contiguous(),
        q_scale,
        k_scale,
        v_scale,
    )


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.skipif(not _has_sm90(), reason="requires Hopper FP8 WGMMA")
@pytest.mark.smoke
def test_gqa_fp8_bn224_kernel_accepts_fa3_descale_contract() -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 896, 8, 2, 128
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16) * 0.25
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25
    v = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25

    q_fp8, q_descale = quantize_q_fa3_gqa_descale(q, heads_kv)
    k_fp8, k_descale = quantize_kv_fa3_descale(k)
    v_fp8, v_descale = quantize_kv_fa3_descale(v)

    kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel(
        batch, heads, heads_kv, seq_len, seq_len, dim, False, torch.float16
    )
    # Native Dense kernels consume BSHD directly. A log-sum-exp the
    # implementation computes stays internal to the semantic output contract.
    out = kernel(
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
    )

    assert tuple(q_descale.shape) == (batch, heads_kv)
    assert tuple(k_descale.shape) == (batch, heads_kv)
    assert tuple(v_descale.shape) == (batch, heads_kv)
    assert out.shape == (batch, seq_len, heads, dim)
    assert torch.isfinite(out.float()).all()


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.skipif(not _has_sm90(), reason="requires Hopper FP8 WGMMA")
@pytest.mark.parametrize(
    ("seq_len", "out_dtype", "input_scale"),
    [
        pytest.param(896, torch.float16, 0.25, id="s896-fp16-scale025"),
        pytest.param(896, torch.bfloat16, 0.25, id="s896-bf16-scale025"),
        pytest.param(1792, torch.float16, 0.75, id="s1792-fp16-scale075"),
    ],
)
@pytest.mark.smoke
def test_gqa_prefill_canonical_fp8_accepts_fa3_descale_contract(
    seq_len: int,
    out_dtype: torch.dtype,
    input_scale: float,
) -> None:
    batch, heads, heads_kv, dim = 1, 8, 2, 128
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16) * input_scale
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * input_scale
    v = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * input_scale

    q_fp8, q_descale = quantize_q_fa3_gqa_descale(q, heads_kv)
    k_fp8, k_descale = quantize_kv_fa3_descale(k)
    v_fp8, v_descale = quantize_kv_fa3_descale(v)

    out = _run_canonical_fp8_prefill(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        out_dtype=out_dtype,
        q_fp8=q_fp8,
        k_fp8=k_fp8,
        v_fp8=v_fp8,
        q_scale=q_descale,
        k_scale=k_descale,
        v_scale=v_descale,
    )

    assert out.shape == (batch, seq_len, heads, dim)
    assert out.dtype == out_dtype
    assert torch.isfinite(out.float()).all()


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.skipif(not _has_sm90(), reason="requires Hopper FP8 WGMMA")
@pytest.mark.smoke
def test_gqa_prefill_canonical_op_dispatches_fp8_tensor_core_path() -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 896, 8, 2, 128
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16) * 0.25
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25
    v = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25

    q_fp8, q_scale = quantize_q_fa3_gqa_descale(q, heads_kv)
    k_fp8, k_scale = quantize_kv_fa3_descale(k)
    v_fp8, v_scale = quantize_kv_fa3_descale(v)
    op = GroupedQueryAttentionDenseFwdOp(
        dtype=torch.float16,
        is_causal=False,
    )
    out = op(
        q_fp8.contiguous(),
        k_fp8.contiguous(),
        v_fp8.contiguous(),
        q_scale,
        k_scale,
        v_scale,
    )

    assert out.shape == (batch, seq_len, heads, dim)
    assert out.dtype == torch.float16
    assert torch.isfinite(out.float()).all()


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.skipif(not _has_sm90(), reason="requires Hopper FP8 WGMMA")
@pytest.mark.parametrize(
    (
        "seq_len_q",
        "seq_len_kv",
        "is_causal",
        "window_size_left",
        "window_size_right",
        "softcap",
        "sm_scale",
        "out_dtype",
    ),
    [
        pytest.param(
            193,
            193,
            True,
            -1,
            -1,
            0.0,
            None,
            torch.float16,
            id="square-causal-tail",
        ),
        pytest.param(
            191,
            191,
            False,
            48,
            16,
            2.0,
            0.125,
            torch.bfloat16,
            id="square-bidirectional-window-softcap-tail",
        ),
        pytest.param(
            129,
            257,
            True,
            96,
            -1,
            2.0,
            0.11,
            torch.float16,
            id="rectangular-causal-window-softcap-tail",
        ),
    ],
)
@pytest.mark.smoke
def test_gqa_prefill_fp8_native_general_semantic_matrix(
    seq_len_q: int,
    seq_len_kv: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    sm_scale: float | None,
    out_dtype: torch.dtype,
) -> None:
    """The native general schedule owns semantics outside the BN224 fast region."""
    batch, heads, heads_kv, dim = 1, 8, 2, 128
    group_size = heads // heads_kv
    torch.manual_seed(321)
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=torch.float16) * 0.2
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.2
    v = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.2
    q_fp8, q_scale = quantize_q_fa3_gqa_descale(q, heads_kv)
    k_fp8, k_scale = quantize_kv_fa3_descale(k)
    v_fp8, v_scale = quantize_kv_fa3_descale(v)
    op = GroupedQueryAttentionDenseFwdOp(
        dtype=out_dtype,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        softcap=softcap,
        sm_scale=sm_scale,
    )
    out = op(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale)

    q_deq = q_fp8.float().reshape(batch, seq_len_q, heads_kv, group_size, dim)
    q_deq = (q_deq * q_scale[:, None, :, None, None]).reshape(batch, seq_len_q, heads, dim)
    k_deq = k_fp8.float() * k_scale[:, None, :, None]
    v_deq = v_fp8.float() * v_scale[:, None, :, None]
    offset = seq_len_kv - seq_len_q
    q_pos = torch.arange(seq_len_q, device="cuda")[:, None]
    k_pos = torch.arange(seq_len_kv, device="cuda")[None, :]
    center = q_pos + offset
    visible = torch.ones((seq_len_q, seq_len_kv), dtype=torch.bool, device="cuda")
    if is_causal:
        visible &= k_pos <= center
    if window_size_left >= 0:
        visible &= k_pos >= center - window_size_left
    if window_size_right >= 0:
        visible &= k_pos <= center + window_size_right
    ref_heads = []
    for head in range(heads):
        kv_head = head // group_size
        score = torch.matmul(q_deq[0, :, head], k_deq[0, :, kv_head].T)
        score *= dim**-0.5 if sm_scale is None else sm_scale
        if softcap > 0:
            score = softcap * torch.tanh(score / softcap)
        score = score.masked_fill(~visible, float("-inf"))
        ref_heads.append(torch.matmul(torch.softmax(score, dim=-1), v_deq[0, :, kv_head]))
    ref = torch.stack(ref_heads, dim=1).unsqueeze(0)
    assert out.dtype == out_dtype
    torch.testing.assert_close(out.float(), ref, atol=8e-2, rtol=8e-2)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.skipif(not _has_sm90(), reason="requires Hopper FP8 WGMMA")
@pytest.mark.parametrize("sm_scale", [0.0, -0.125])
@pytest.mark.smoke
def test_gqa_fp8_general_empty_window_rows_are_zero(sm_scale: float) -> None:
    batch, seq_len_q, seq_len_kv = 1, 65, 1
    heads, heads_kv, dim = 8, 2, 128
    q = (
        torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=torch.float16)
        .clamp(-2, 2)
        .to(torch.float8_e4m3fn)
    )
    k = (
        torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16)
        .clamp(-2, 2)
        .to(torch.float8_e4m3fn)
    )
    v = (
        torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16)
        .clamp(-2, 2)
        .to(torch.float8_e4m3fn)
    )
    scales = torch.ones((batch, heads_kv), device="cuda", dtype=torch.float32)
    op = GroupedQueryAttentionDenseFwdOp(
        is_causal=False,
        window_size_right=0,
        sm_scale=sm_scale,
        dtype=torch.float16,
    )

    output = op(q, k, v, scales, scales, scales)

    assert torch.equal(output[:, :-1], torch.zeros_like(output[:, :-1]))
    expected = v[:, 0].to(torch.float16).repeat_interleave(4, dim=1)
    torch.testing.assert_close(output[:, -1], expected, atol=2e-2, rtol=1e-3)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.skipif(not _has_sm90(), reason="requires Hopper FP8 WGMMA")
@pytest.mark.smoke
def test_gqa_prefill_fp8_tensor_core_matches_dequantized_reference() -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 896, 8, 2, 128
    group_size = heads // heads_kv
    torch.manual_seed(123)
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16) * 0.25
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25
    v = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25

    q_fp8, q_descale = quantize_q_fa3_gqa_descale(q, heads_kv)
    k_fp8, k_descale = quantize_kv_fa3_descale(k)
    v_fp8, v_descale = quantize_kv_fa3_descale(v)

    out = _run_canonical_fp8_prefill(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        out_dtype=torch.float16,
        q_fp8=q_fp8,
        k_fp8=k_fp8,
        v_fp8=v_fp8,
        q_scale=q_descale,
        k_scale=k_descale,
        v_scale=v_descale,
    )

    q_deq = q_fp8.float().reshape(batch, seq_len, heads_kv, group_size, dim)
    q_deq = (q_deq * q_descale[:, None, :, None, None]).reshape(batch, seq_len, heads, dim)
    k_deq = k_fp8.float() * k_descale[:, None, :, None]
    v_deq = v_fp8.float() * v_descale[:, None, :, None]

    scale = dim**-0.5
    ref_heads = []
    for head in range(heads):
        head_kv = head // group_size
        scores = (
            torch.matmul(
                q_deq[0, :, head, :],
                k_deq[0, :, head_kv, :].T,
            )
            * scale
        )
        probs = torch.softmax(scores, dim=-1)
        ref_heads.append(torch.matmul(probs, v_deq[0, :, head_kv, :]))
    ref = torch.stack(ref_heads, dim=1).unsqueeze(0)

    torch.testing.assert_close(
        out.reshape(batch, seq_len, heads, dim).float(), ref, atol=5e-2, rtol=5e-2
    )


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.skipif(not _has_sm90(), reason="requires Hopper FP8 WGMMA")
@pytest.mark.parametrize("seq_len", [6272, 7168, 8064])
@pytest.mark.smoke
def test_gqa_prefill_fp8_row_reduction_boundary_is_live_and_repeatable(
    seq_len: int,
) -> None:
    """Exercise 28/32/36 BN224 tiles across the deferred-reduction boundary."""
    batch, heads, heads_kv, dim = 1, 64, 8, 128
    fp8 = torch.float8_e4m3fn
    q_fp8 = torch.zeros((batch, seq_len, heads, dim), device="cuda", dtype=fp8)
    k_fp8 = torch.zeros((batch, seq_len, heads_kv, dim), device="cuda", dtype=fp8)
    v_fp8 = torch.full((batch, seq_len, heads_kv, dim), 0.5, device="cuda", dtype=fp8)
    q_scale = torch.ones((batch, heads_kv), device="cuda", dtype=torch.float32)
    k_scale = torch.ones_like(q_scale)
    v_scale = torch.full_like(q_scale, 0.25)

    outputs = [
        _run_canonical_fp8_prefill(
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            out_dtype=torch.float16,
            q_fp8=q_fp8,
            k_fp8=k_fp8,
            v_fp8=v_fp8,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        for _ in range(2)
    ]

    expected = torch.full_like(outputs[0], 0.125)
    for output in outputs:
        torch.testing.assert_close(output, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
