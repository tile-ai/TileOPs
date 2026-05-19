import pytest
import torch
import torch.nn.functional as F

from tileops.kernels.attention import (
    GQAFwdFP8Fa3ContractKernel,
    GQAFwdFP8Fa3ContractPtxAccDirectStoreKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongCorrectedKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongCorrectedPreRescaleKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaEmitterK224Kernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaSharedVKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPPlainWaitKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVInplaceKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVInplaceBarrierKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel,
    GQAFwdFP8Fa3ContractPtxAccFa3EpilogueStoreKernel,
    GQAFwdFP8Fa3ContractPtxAccFa3EpilogueReuseVSmemKernel,
    GQAFwdFP8Fa3ContractPtxAccKernel,
    GQAFwdFP8WgmmaKernel,
    GQAFwdFP8WsPersistentKernel,
)


def _has_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _block_quant_128(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, heads, dim = x.shape
    x_blocked = x.reshape(batch, seq_len // 128, 128, heads,
                          dim).permute(0, 3, 1, 2, 4).contiguous()
    amax = x_blocked.abs().amax(dim=(3, 4)).clamp(min=1e-4)
    scale = amax / 448.0
    x_fp8 = torch.clamp(x_blocked / scale[..., None, None], -448.0,
                        448.0).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.permute(0, 2, 3, 1, 4).reshape(batch, seq_len, heads, dim).contiguous()
    return x_fp8, scale.contiguous()


def _block_dequant_128(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    batch, seq_len, heads, dim = x_fp8.shape
    x_blocked = x_fp8.float().reshape(batch, seq_len // 128, 128, heads,
                                      dim).permute(0, 3, 1, 2, 4)
    x = x_blocked * scale[..., None, None]
    return x.permute(0, 2, 3, 1, 4).reshape(batch, seq_len, heads, dim).contiguous()


def _quantize_kv_fa3_scale(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, heads, dim = x.shape
    descale = x.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 448.0
    x_fp8 = torch.clamp(x / descale[:, None, :, None], -448.0,
                        448.0).to(torch.float8_e4m3fn).contiguous()
    tileops_scale = descale[:, :, None].expand(batch, heads, seq_len // 128).contiguous()
    return x_fp8, tileops_scale.float()


def _quantize_q_fa3_gqa_scale(
    x: torch.Tensor,
    heads_kv: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, heads, dim = x.shape
    group_size = heads // heads_kv
    x_grouped = x.reshape(batch, seq_len, heads_kv, group_size, dim)
    descale = x_grouped.abs().amax(dim=(1, 3, 4)).clamp(min=1e-4) / 448.0
    x_fp8 = torch.clamp(x_grouped / descale[:, None, :, None, None], -448.0,
                        448.0).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.reshape(batch, seq_len, heads, dim).contiguous()
    q_head_scale = descale.repeat_interleave(group_size, dim=1)
    tileops_scale = q_head_scale[:, :, None].expand(batch, heads, seq_len // 128).contiguous()
    return x_fp8, tileops_scale.float()


def _quantize_kv_fa3_descale(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    descale = x.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 448.0
    x_fp8 = torch.clamp(x / descale[:, None, :, None], -448.0,
                        448.0).to(torch.float8_e4m3fn).contiguous()
    return x_fp8, descale.float().contiguous()


def _quantize_q_fa3_gqa_descale(
    x: torch.Tensor,
    heads_kv: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, heads, dim = x.shape
    group_size = heads // heads_kv
    x_grouped = x.reshape(batch, seq_len, heads_kv, group_size, dim)
    descale = x_grouped.abs().amax(dim=(1, 3, 4)).clamp(min=1e-4) / 448.0
    x_fp8 = torch.clamp(x_grouped / descale[:, None, :, None, None], -448.0,
                        448.0).to(torch.float8_e4m3fn)
    return x_fp8.reshape(batch, seq_len, heads, dim).contiguous(), descale.float().contiguous()

@pytest.mark.smoke
@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.skipif(not _has_sm90(), reason="requires Hopper FP8 WGMMA")
@pytest.mark.parametrize(
    "kernel_cls,use_register_p,extra_config",
    [
        pytest.param(GQAFwdFP8WgmmaKernel, False, {}, id="wgmma_shared_p"),
        pytest.param(GQAFwdFP8WgmmaKernel, True, {}, id="wgmma_register_p"),
        pytest.param(GQAFwdFP8WsPersistentKernel, False, {}, id="ws_shared_p"),
        pytest.param(GQAFwdFP8WsPersistentKernel, True, {}, id="ws_register_p"),
        pytest.param(
            GQAFwdFP8WsPersistentKernel,
            False,
            {"use_full_pv_gemm": True},
            id="ws_full_pv_gemm",
        ),
        pytest.param(
            GQAFwdFP8WsPersistentKernel,
            False,
            {"use_full_pv_gemm": True, "use_tl_full_v_transpose": True},
            id="ws_full_pv_gemm_tl_full_v",
        ),
        pytest.param(
            GQAFwdFP8WsPersistentKernel,
            False,
            {"use_full_pv_gemm": True, "use_tl_pack_ldsm_v_transpose": True},
            id="ws_full_pv_gemm_tl_pack_ldsm_v",
        ),
        pytest.param(
            GQAFwdFP8WsPersistentKernel,
            False,
            {"use_fa3_pv_extern": True},
            id="ws_fa3_pv_extern",
        ),
        pytest.param(
            GQAFwdFP8WsPersistentKernel,
            False,
            {"use_fa3_pv_extern": True, "use_fa3_v_tma_direct": True},
            id="ws_fa3_pv_extern_tma_direct",
        ),
        pytest.param(
            GQAFwdFP8Fa3ContractKernel,
            False,
            {},
            id="ws_fa3_contract_kernel",
        ),
        pytest.param(
            GQAFwdFP8Fa3ContractKernel,
            False,
            {"use_ptx_pv": True},
            id="ws_fa3_contract_kernel_ptx_pv",
        ),
        pytest.param(
            GQAFwdFP8Fa3ContractPtxAccKernel,
            False,
            {},
            id="ws_fa3_contract_kernel_ptx_acc",
        ),
        pytest.param(
            GQAFwdFP8Fa3ContractPtxAccDirectStoreKernel,
            False,
            {},
            id="ws_fa3_contract_kernel_ptx_acc_direct_store",
        ),
        pytest.param(
            GQAFwdFP8Fa3ContractPtxAccFa3EpilogueStoreKernel,
            False,
            {},
            id="ws_fa3_contract_kernel_ptx_acc_fa3_epilogue_store",
        ),
        pytest.param(
            GQAFwdFP8Fa3ContractPtxAccFa3EpilogueReuseVSmemKernel,
            False,
            {},
            id="ws_fa3_contract_kernel_ptx_acc_fa3_epilogue_reuse_v_smem",
        ),
    ],
)
def test_gqa_fwd_fp8_wgmma_baseline(kernel_cls, use_register_p: bool,
                                    extra_config: dict) -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 128, 2, 1, 128
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16) * 0.25
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25
    v = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25

    q_fp8, q_scale = _block_quant_128(q)
    k_fp8, k_scale = _block_quant_128(k)
    v_fp8, v_scale = _block_quant_128(v)

    kernel = kernel_cls(
        batch,
        heads,
        heads_kv,
        seq_len,
        dim,
        torch.float16,
        config={"use_register_p": use_register_p, **extra_config},
    )
    out, _lse = kernel(q_fp8, k_fp8, v_fp8, q_scale.float(), k_scale.float(), v_scale.float())

    q_ref = _block_dequant_128(q_fp8, q_scale)
    k_ref = _block_dequant_128(k_fp8, k_scale)
    v_ref = _block_dequant_128(v_fp8, v_scale)
    refs = []
    for h in range(heads):
        h_kv = h // (heads // heads_kv)
        scores = (q_ref[0, :, h, :] @ k_ref[0, :, h_kv, :].T) * (dim**-0.5)
        probs_fp8 = torch.softmax(scores, dim=-1).to(torch.float8_e4m3fn).float()
        refs.append(probs_fp8 @ v_ref[0, :, h_kv, :])
    ref = torch.stack(refs, dim=1).unsqueeze(0)

    assert torch.isfinite(out.float()).all()
    cos = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
    assert cos >= 0.997
    torch.testing.assert_close(out.float(), ref.float(), atol=8e-3, rtol=1e-2)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.skipif(not _has_sm90(), reason="requires Hopper FP8 WGMMA")
@pytest.mark.full
@pytest.mark.xfail(
    reason=(
        "Single Vt->Vtc in-place transform is byte-exact, but the full WS pipeline "
        "still has a V stage lifetime/overlap hazard."
    ),
    strict=False,
)
def test_gqa_fwd_fp8_bn224_tma_v_inplace_matches_out_of_place() -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 896, 2, 1, 128
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16) * 0.25
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25
    v = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25

    q_fp8, q_scale = _block_quant_128(q)
    k_fp8, k_scale = _block_quant_128(k)
    v_fp8, v_scale = _block_quant_128(v)
    inputs = (q_fp8, k_fp8, v_fp8, q_scale.float(), k_scale.float(), v_scale.float())

    baseline = GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel(
        batch, heads, heads_kv, seq_len, dim, torch.float16)
    candidate = GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVInplaceKernel(
        batch, heads, heads_kv, seq_len, dim, torch.float16)

    out_base, lse_base = baseline(*inputs)
    out_candidate, lse_candidate = candidate(*inputs)

    assert torch.isfinite(out_candidate.float()).all()
    assert torch.isfinite(lse_candidate.float()).all()
    torch.testing.assert_close(out_candidate.float(), out_base.float(), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lse_candidate.float(), lse_base.float(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.skipif(not _has_sm90(), reason="requires Hopper FP8 WGMMA")
@pytest.mark.smoke
def test_gqa_fwd_fp8_bn224_tma_v_inplace_barrier_matches_out_of_place() -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 896, 2, 1, 128
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16) * 0.25
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25
    v = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25

    q_fp8, q_scale = _block_quant_128(q)
    k_fp8, k_scale = _block_quant_128(k)
    v_fp8, v_scale = _block_quant_128(v)
    inputs = (q_fp8, k_fp8, v_fp8, q_scale.float(), k_scale.float(), v_scale.float())

    baseline = GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel(
        batch, heads, heads_kv, seq_len, dim, torch.float16)
    candidate = GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVInplaceBarrierKernel(
        batch, heads, heads_kv, seq_len, dim, torch.float16)

    out_base, lse_base = baseline(*inputs)
    out_candidate, lse_candidate = candidate(*inputs)

    assert torch.isfinite(out_candidate.float()).all()
    assert torch.isfinite(lse_candidate.float()).all()
    torch.testing.assert_close(out_candidate.float(), out_base.float(), atol=3e-3, rtol=1e-3)
    torch.testing.assert_close(lse_candidate.float(), lse_base.float(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.skipif(not _has_sm90(), reason="requires Hopper FP8 WGMMA")
@pytest.mark.smoke
def test_gqa_fwd_fp8_bn224_pingpong_matches_serial_tma_v() -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 896, 2, 1, 128
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16) * 0.25
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25
    v = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25

    q_fp8, q_scale = _block_quant_128(q)
    k_fp8, k_scale = _block_quant_128(k)
    v_fp8, v_scale = _block_quant_128(v)
    inputs = (q_fp8, k_fp8, v_fp8, q_scale.float(), k_scale.float(), v_scale.float())

    baseline = GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel(
        batch, heads, heads_kv, seq_len, dim, torch.float16)
    candidate = GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongKernel(
        batch, heads, heads_kv, seq_len, dim, torch.float16)

    out_base, lse_base = baseline(*inputs)
    out_candidate, lse_candidate = candidate(*inputs)

    assert torch.isfinite(out_candidate.float()).all()
    assert torch.isfinite(lse_candidate.float()).all()
    torch.testing.assert_close(out_candidate.float(), out_base.float(), atol=3e-3, rtol=1e-3)
    torch.testing.assert_close(lse_candidate.float(), lse_base.float(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.skipif(not _has_sm90(), reason="requires Hopper FP8 WGMMA")
@pytest.mark.smoke
def test_gqa_fwd_fp8_bn224_pingpong_corrected_handles_late_max_rescale() -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 896, 2, 1, 128
    q = torch.zeros(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.zeros(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.zeros(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)

    q[..., 0] = 1.0
    k[:, 0:224, :, 0] = 1.0
    v[:, 0:224, :, 0] = 4.0
    v[:, 0:224, :, 1] = -3.0
    k[:, 224:448, :, 0] = 80.0
    v[:, 224:448, :, 0] = -1.0
    v[:, 224:448, :, 1] = 2.0
    k[:, 448:, :, 0] = -2.0
    v[:, 448:, :, 0] = 0.5
    v[:, 448:, :, 1] = 0.25

    q_fp8, q_scale = _quantize_q_fa3_gqa_scale(q, heads_kv)
    k_fp8, k_scale = _quantize_kv_fa3_scale(k)
    v_fp8, v_scale = _quantize_kv_fa3_scale(v)
    inputs = (q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale)

    baseline = GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel(
        batch, heads, heads_kv, seq_len, dim, torch.float16)
    out_base, lse_base = baseline(*inputs)
    for kernel_cls in (
            GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongCorrectedKernel,
            GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongCorrectedPreRescaleKernel,
            GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaKernel,
            GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaEmitterK224Kernel,
            GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaSharedVKernel,
    ):
        candidate = kernel_cls(batch, heads, heads_kv, seq_len, dim, torch.float16)
        out_candidate, lse_candidate = candidate(*inputs)

        assert torch.isfinite(out_candidate.float()).all()
        assert torch.isfinite(lse_candidate.float()).all()
        torch.testing.assert_close(out_candidate.float(), out_base.float(), atol=3e-3, rtol=1e-3)
        torch.testing.assert_close(lse_candidate.float(), lse_base.float(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.skipif(not _has_sm90(), reason="requires Hopper FP8 WGMMA")
@pytest.mark.smoke
def test_gqa_fwd_fp8_bn224_direct_overlap_fa3_scale_matches_serial_tma_v() -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 896, 2, 1, 128
    baseline = GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel(
        batch, heads, heads_kv, seq_len, dim, torch.float16)
    candidates = (
        GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapKernel(
            batch, heads, heads_kv, seq_len, dim, torch.float16),
        GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPKernel(
            batch, heads, heads_kv, seq_len, dim, torch.float16),
        GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPPlainWaitKernel(
            batch, heads, heads_kv, seq_len, dim, torch.float16),
    )

    torch.manual_seed(0)
    cases = [
        (
            torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16) * 0.25,
            torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25,
            torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25,
        )
    ]

    q = torch.zeros(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.zeros(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.zeros(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    q[..., 0] = 1.0
    k[:, 0:224, :, 0] = 1.0
    v[:, 0:224, :, 0] = 4.0
    v[:, 0:224, :, 1] = -3.0
    k[:, 224:448, :, 0] = 80.0
    v[:, 224:448, :, 0] = -1.0
    v[:, 224:448, :, 1] = 2.0
    k[:, 448:, :, 0] = -2.0
    v[:, 448:, :, 0] = 0.5
    v[:, 448:, :, 1] = 0.25
    cases.append((q, k, v))

    for q_case, k_case, v_case in cases:
        q_fp8, q_scale = _quantize_q_fa3_gqa_scale(q_case, heads_kv)
        k_fp8, k_scale = _quantize_kv_fa3_scale(k_case)
        v_fp8, v_scale = _quantize_kv_fa3_scale(v_case)
        inputs = (q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale)

        out_base, lse_base = baseline(*inputs)
        for candidate in candidates:
            out_candidate, lse_candidate = candidate(*inputs)

            assert torch.isfinite(out_candidate.float()).all()
            assert torch.isfinite(lse_candidate.float()).all()
            torch.testing.assert_close(out_candidate.float(), out_base.float(), atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(lse_candidate.float(), lse_base.float(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.skipif(not _has_sm90(), reason="requires Hopper FP8 WGMMA")
@pytest.mark.smoke
def test_gqa_fwd_fp8_bn224_tma_v_gqa_heads_are_finite() -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 896, 32, 8, 128
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16) * 0.25
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25
    v = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25

    q_fp8, q_scale = _block_quant_128(q)
    k_fp8, k_scale = _block_quant_128(k)
    v_fp8, v_scale = _block_quant_128(v)

    kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel(
        batch, heads, heads_kv, seq_len, dim, torch.float16)
    out, lse = kernel(
        q_fp8, k_fp8, v_fp8, q_scale.float(), k_scale.float(), v_scale.float())

    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(lse.float()).all()


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
@pytest.mark.skipif(not _has_sm90(), reason="requires Hopper FP8 WGMMA")
@pytest.mark.smoke
def test_gqa_fwd_fp8_bn224_tma_v_accepts_fa3_descale_contract() -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 896, 8, 2, 128
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16) * 0.25
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25
    v = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16) * 0.25

    q_fp8, q_descale = _quantize_q_fa3_gqa_descale(q, heads_kv)
    k_fp8, k_descale = _quantize_kv_fa3_descale(k)
    v_fp8, v_descale = _quantize_kv_fa3_descale(v)

    kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel(
        batch, heads, heads_kv, seq_len, dim, torch.float16)
    out, lse = kernel(
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    assert tuple(q_descale.shape) == (batch, heads_kv)
    assert tuple(k_descale.shape) == (batch, heads_kv)
    assert tuple(v_descale.shape) == (batch, heads_kv)
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(lse.float()).all()
