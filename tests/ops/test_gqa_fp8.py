import pytest
import torch

from tileops.ops import GQADenseFwdOp
from workloads.attention.gqa.dense import GQADensePrefillWorkload
from workloads.device import run_device
from workloads.numerics import compare_outputs


def _quantize_kv_fa3_descale(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize K/V and return FA3-interface descales with shape ``[B, H_kv]``.

    The current FA3 FP8 GQA Python API accepts one scale per batch/KV-head pair.
    TileOps broadcasts this public 2D contract to its internal per-128-token-block
    scale layout before launch.
    """
    descale = x.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 448.0
    x_fp8 = (
        torch.clamp(x / descale[:, None, :, None], -448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    return x_fp8, descale.float().contiguous()


def _quantize_q_fa3_gqa_descale(
    x: torch.Tensor,
    heads_kv: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize grouped Q and return FA3-interface descales with shape ``[B, H_kv]``."""
    batch, seq_len, heads, dim = x.shape
    group_size = heads // heads_kv
    x_grouped = x.reshape(batch, seq_len, heads_kv, group_size, dim)
    descale = x_grouped.abs().amax(dim=(1, 3, 4)).clamp(min=1e-4) / 448.0
    x_fp8 = torch.clamp(x_grouped / descale[:, None, :, None, None], -448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    return x_fp8.reshape(batch, seq_len, heads, dim).contiguous(), descale.float().contiguous()


def _run_fp8_prefill_kernel(
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
    is_causal: bool = False,
) -> torch.Tensor:
    workload = GQADensePrefillWorkload(
        batch,
        seq_len,
        seq_len,
        heads,
        heads_kv,
        dim,
        q_fp8.dtype,
        out_dtype=out_dtype,
        is_causal=is_causal,
    )
    op = GQADenseFwdOp(out_dtype=out_dtype, is_causal=is_causal)
    inputs = (q_fp8.contiguous(), k_fp8.contiguous(), v_fp8.contiguous(), q_scale, k_scale, v_scale)
    output = op(*inputs)
    compare_outputs(output, workload.ref_program(*inputs), workload.verification(*inputs))
    return output


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.sm90
@pytest.mark.parametrize(
    ("seq_len", "out_dtype", "input_scale"),
    [
        pytest.param(896, torch.float16, 0.25, id="s896-fp16-scale025"),
        pytest.param(896, torch.bfloat16, 0.25, id="s896-bf16-scale025"),
        pytest.param(1792, torch.float16, 0.75, id="s1792-fp16-scale075"),
    ],
)
@pytest.mark.smoke
def test_gqa_prefill_fp8_kernel_accepts_fa3_descale_contract(
    seq_len: int,
    out_dtype: torch.dtype,
    input_scale: float,
) -> None:
    batch, heads, heads_kv, dim = 1, 8, 2, 128
    q = (
        torch.randn(batch, seq_len, heads, dim, device=run_device(), dtype=torch.float16)
        * input_scale
    )
    k = (
        torch.randn(batch, seq_len, heads_kv, dim, device=run_device(), dtype=torch.float16)
        * input_scale
    )
    v = (
        torch.randn(batch, seq_len, heads_kv, dim, device=run_device(), dtype=torch.float16)
        * input_scale
    )

    q_fp8, q_descale = _quantize_q_fa3_gqa_descale(q, heads_kv)
    k_fp8, k_descale = _quantize_kv_fa3_descale(k)
    v_fp8, v_descale = _quantize_kv_fa3_descale(v)

    out = _run_fp8_prefill_kernel(
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
@pytest.mark.sm90
@pytest.mark.parametrize("seq_len", [225, 897])
@pytest.mark.smoke
def test_gqa_prefill_fp8_tensor_core_handles_tail_tiles(seq_len: int) -> None:
    batch, heads, heads_kv, dim = 1, 8, 2, 128
    fp8 = torch.float8_e4m3fn
    q = torch.zeros((batch, seq_len, heads, dim), device=run_device(), dtype=fp8)
    k = torch.zeros((batch, seq_len, heads_kv, dim), device=run_device(), dtype=fp8)
    v = torch.ones_like(k)
    scale = torch.ones((batch, heads_kv), device=run_device(), dtype=torch.float32)

    out = _run_fp8_prefill_kernel(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        out_dtype=torch.float16,
        q_fp8=q,
        k_fp8=k,
        v_fp8=v,
        q_scale=scale,
        k_scale=scale,
        v_scale=scale,
    )

    torch.testing.assert_close(out.float(), torch.ones_like(out, dtype=torch.float32))


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize("is_causal", [False, True], ids=["full", "causal"])
def test_gqa_prefill_fp8_tensor_core_matches_dequantized_reference(is_causal: bool) -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 897, 8, 2, 128
    torch.manual_seed(123)
    q = torch.randn(batch, seq_len, heads, dim, device=run_device(), dtype=torch.float16) * 0.25
    k = torch.randn(batch, seq_len, heads_kv, dim, device=run_device(), dtype=torch.float16) * 0.25
    v = torch.randn(batch, seq_len, heads_kv, dim, device=run_device(), dtype=torch.float16) * 0.25

    q_fp8, q_descale = _quantize_q_fa3_gqa_descale(q, heads_kv)
    k_fp8, k_descale = _quantize_kv_fa3_descale(k)
    v_fp8, v_descale = _quantize_kv_fa3_descale(v)

    _run_fp8_prefill_kernel(
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
        is_causal=is_causal,
    )
