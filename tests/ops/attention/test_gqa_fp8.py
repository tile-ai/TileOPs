import pytest
import torch

from tileops.kernels.attention import GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel
from tileops.ops import GroupedQueryAttentionPrefillFP8TensorCoreFwdOp
from tileops.testing.gqa_fp8_utils import (
    quantize_kv_fa3_descale,
    quantize_q_fa3_gqa_descale,
)


def _has_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


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
        batch, heads, heads_kv, seq_len, dim, torch.float16
    )
    out, lse = kernel(q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale)

    assert tuple(q_descale.shape) == (batch, heads_kv)
    assert tuple(k_descale.shape) == (batch, heads_kv)
    assert tuple(v_descale.shape) == (batch, heads_kv)
    assert out.shape == (batch, seq_len, heads, dim)
    assert lse.shape == (batch, heads, seq_len)
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(lse.float()).all()


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
def test_gqa_prefill_fp8_tensor_core_op_accepts_fa3_descale_contract(
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

    op = GroupedQueryAttentionPrefillFP8TensorCoreFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        seq_len=seq_len,
        dim=dim,
        is_causal=False,
        dtype=out_dtype,
    )
    out = op(q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale)

    assert out.shape == (batch, seq_len, heads, dim)
    assert out.dtype == out_dtype
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("seq_len", [224, 672])
@pytest.mark.smoke
def test_gqa_prefill_fp8_tensor_core_rejects_unaligned_q_tiles(seq_len: int) -> None:
    with pytest.raises(ValueError, match="seq_len % 128 == 0"):
        GroupedQueryAttentionPrefillFP8TensorCoreFwdOp(
            batch=1,
            heads=8,
            heads_kv=2,
            seq_len=seq_len,
            dim=128,
            is_causal=False,
            dtype=torch.float16,
        )

    with pytest.raises(ValueError, match="seq_len % 128 == 0"):
        GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel(1, 8, 2, seq_len, 128, torch.float16)


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

    op = GroupedQueryAttentionPrefillFP8TensorCoreFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        seq_len=seq_len,
        dim=dim,
        is_causal=False,
        dtype=torch.float16,
    )
    out = op(q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale)

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

    torch.testing.assert_close(out.float(), ref, atol=5e-2, rtol=5e-2)
