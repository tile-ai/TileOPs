from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.attention import (
    GQADecodeBs1Kernel,
    GQADecodeKernel,
    GQADenseCausalWsKernel,
    GQADenseSlidingWindowKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.ops import (
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionDenseFwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
)
from tileops.utils import get_sm_version
from workloads.attention.gqa import GroupedQueryAttentionBwdWorkload


class GroupedQueryAttentionBwdTest(GroupedQueryAttentionBwdWorkload, TestBase):
    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        grad_output: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True
            )
        output = output_bhsd.transpose(1, 2).contiguous()

        output.backward(grad_output)
        return q.grad, k.grad, v.grad


def _gqa_prefill_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    heads: int,
    heads_kv: int,
    is_causal: bool,
    sm_scale: Optional[float] = None,
    softcap: Optional[float] = None,
    window_size_left: int = -1,
    window_size_right: int = -1,
) -> torch.Tensor:
    batch, seq_len_q, _, dim = q.shape
    seq_len_kv = k.shape[1]
    groups = heads // heads_kv
    q_bhsd = q.transpose(1, 2).float()
    k_bhsd = k.repeat_interleave(groups, dim=2).transpose(1, 2).float()
    v_bhsd = v.repeat_interleave(groups, dim=2).transpose(1, 2).float()
    scale = dim**-0.5 if sm_scale is None else sm_scale
    scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * scale
    if softcap is not None and softcap > 0:
        scores = softcap * torch.tanh(scores / softcap)
    offset = seq_len_kv - seq_len_q
    q_pos = torch.arange(seq_len_q, device=q.device)[:, None] + offset
    k_pos = torch.arange(seq_len_kv, device=q.device)[None, :]
    mask = torch.ones((seq_len_q, seq_len_kv), device=q.device, dtype=torch.bool)
    if is_causal:
        mask &= k_pos <= q_pos
    if window_size_left >= 0:
        mask &= k_pos >= q_pos - window_size_left
    if window_size_right >= 0:
        mask &= k_pos <= q_pos + window_size_right
    if is_causal or window_size_left >= 0 or window_size_right >= 0:
        scores = scores.masked_fill(~mask.view(1, 1, seq_len_q, seq_len_kv), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v_bhsd)
    assert output.shape == (batch, heads, seq_len_q, dim)
    return output.transpose(1, 2).to(q.dtype).contiguous()


def _apply_dense_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rotary_dim: int,
    layout: str,
) -> torch.Tensor:
    half = rotary_dim // 2
    x_rot = x[..., :rotary_dim].float()
    c = cos[positions].view(1, x.shape[1], 1, half).float()
    s = sin[positions].view(1, x.shape[1], 1, half).float()
    if layout == "neox":
        x0, x1 = x_rot[..., :half], x_rot[..., half:]
    else:
        x0, x1 = x_rot[..., 0::2], x_rot[..., 1::2]
    y0, y1 = x0 * c - x1 * s, x1 * c + x0 * s
    rotated = (
        torch.cat((y0, y1), dim=-1)
        if layout == "neox"
        else torch.stack((y0, y1), dim=-1).flatten(-2)
    )
    return torch.cat((rotated.to(x.dtype), x[..., rotary_dim:]), dim=-1).contiguous()


@pytest.mark.parametrize(
    "rope_layout, rotary_dim, dtype",
    [
        (None, None, torch.float16),
        ("neox", 64, torch.float16),
        ("interleaved", 64, torch.float16),
        ("neox", None, torch.bfloat16),
    ],
)
@pytest.mark.smoke
def test_gqa_dense_sm90_main_kernel_matches_reference(
    rope_layout: Optional[str], rotary_dim: Optional[int], dtype: torch.dtype
) -> None:
    if not torch.cuda.is_available() or get_sm_version() != 90:
        pytest.skip("Dense warp-specialized prefill requires SM90")
    batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 160, 270, 8, 2, 128
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)

    if rope_layout is None:
        op = GroupedQueryAttentionDenseFwdOp()
        output = op(q, k, v)
        q_ref, k_ref = q, k
    else:
        resolved_rotary_dim = dim if rotary_dim is None else rotary_dim
        angles = torch.randn(seq_len_kv, resolved_rotary_dim // 2, device="cuda") * 0.1
        rope_cos, rope_sin = angles.cos().to(dtype), angles.sin().to(dtype)
        op = GroupedQueryAttentionDenseFwdOp(
            pos_encoding_mode="rope", rotary_dim=rotary_dim, rope_layout=rope_layout
        )
        output = op(q, k, v, rope_cos=rope_cos, rope_sin=rope_sin)
        q_positions = torch.arange(seq_len_kv - seq_len_q, seq_len_kv, device="cuda")
        k_positions = torch.arange(seq_len_kv, device="cuda")
        q_ref = _apply_dense_rope(
            q,
            q_positions,
            rope_cos,
            rope_sin,
            rotary_dim=resolved_rotary_dim,
            layout=rope_layout,
        )
        k_ref = _apply_dense_rope(
            k,
            k_positions,
            rope_cos,
            rope_sin,
            rotary_dim=resolved_rotary_dim,
            layout=rope_layout,
        )

    torch.testing.assert_close(
        output,
        _gqa_prefill_ref(q_ref, k_ref, v, heads=heads, heads_kv=heads_kv, is_causal=True),
        atol=2e-2 if dtype == torch.bfloat16 else 5e-3,
        rtol=1e-5,
    )
    assert isinstance(next(iter(op.iter_kernels())), GQADenseCausalWsKernel)


@pytest.mark.smoke
@pytest.mark.parametrize("batch", [1, 2])
def test_gqa_dense_reuses_one_kernel_across_sequence_lengths(batch: int) -> None:
    if not torch.cuda.is_available() or get_sm_version() != 90:
        pytest.skip("Dense warp-specialized prefill requires SM90")
    heads, heads_kv, dim = 8, 2, 128
    op = GroupedQueryAttentionDenseFwdOp()

    for seq_len_q, seq_len_kv in (
        (2, 270),
        (2, 271),
        (160, 270),
        (896, 896),
        (2, 270),
    ):
        q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        output = op(q, k, v)
        torch.testing.assert_close(
            output,
            _gqa_prefill_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True),
            atol=5e-3,
            rtol=1e-5,
        )

    assert len(list(op.iter_kernels())) == 1


@pytest.mark.parametrize(
    "batch, dtype, seq_lens_kv, kernel_type",
    [
        pytest.param(
            1,
            torch.float16,
            (257, 1057),
            GQADecodeBs1Kernel,
            id="bs1-fp16",
        ),
        pytest.param(
            2,
            torch.bfloat16,
            (257, 2051),
            GQADecodeKernel,
            id="batched-bf16",
        ),
    ],
)
@pytest.mark.smoke
def test_gqa_dense_decode_dispatch_and_dynamic_sequence_lengths(
    batch: int,
    dtype: torch.dtype,
    seq_lens_kv: tuple[int, ...],
    kernel_type: type[Kernel],
) -> None:
    if not torch.cuda.is_available() or get_sm_version() != 90:
        pytest.skip("Dense decode requires SM90")
    heads, heads_kv, dim = 8, 2, 128
    op = GroupedQueryAttentionDenseFwdOp()

    for seq_len_kv in seq_lens_kv:
        q = torch.randn(batch, 1, heads, dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        output = op(q, k, v)
        torch.testing.assert_close(
            output,
            _gqa_prefill_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True),
            atol=1.6e-2 if dtype == torch.bfloat16 else 5e-3,
            rtol=1.6e-2 if dtype == torch.bfloat16 else 1e-5,
        )

    kernels = list(op.iter_kernels())
    assert len(kernels) == 1
    assert isinstance(kernels[0], kernel_type)


@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.smoke
def test_gqa_dense_sm90_sliding_window_kernel_matches_reference(use_rope: bool) -> None:
    if not torch.cuda.is_available() or get_sm_version() != 90:
        pytest.skip("Dense sliding-window prefill requires SM90")
    batch, seq_len, heads, heads_kv, dim = 1, 256, 8, 2, 128
    window_size_left = 64
    sm_scale = 0.125
    softcap = 2.0
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    rotary_dim = 64
    if use_rope:
        angles = torch.randn(seq_len, rotary_dim // 2, device="cuda") * 0.1
        rope_cos, rope_sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
    else:
        rope_cos = rope_sin = None
    op = GroupedQueryAttentionDenseFwdOp(
        window_size_left=window_size_left,
        window_size_right=0,
        sm_scale=sm_scale,
        softcap=softcap,
        pos_encoding_mode="rope" if use_rope else "none",
        rotary_dim=rotary_dim if use_rope else None,
    )
    output = op(
        q,
        k,
        v,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
    )
    if use_rope:
        assert rope_cos is not None and rope_sin is not None
        positions = torch.arange(seq_len, device="cuda")
        q = _apply_dense_rope(
            q, positions, rope_cos, rope_sin, rotary_dim=rotary_dim, layout="neox"
        )
        k = _apply_dense_rope(
            k, positions, rope_cos, rope_sin, rotary_dim=rotary_dim, layout="neox"
        )

    torch.testing.assert_close(
        output,
        _gqa_prefill_ref(
            q,
            k,
            v,
            heads=heads,
            heads_kv=heads_kv,
            is_causal=True,
            sm_scale=sm_scale,
            softcap=softcap,
            window_size_left=window_size_left,
            window_size_right=0,
        ),
        atol=5e-3,
        rtol=1e-5,
    )
    assert isinstance(next(iter(op.iter_kernels())), GQADenseSlidingWindowKernel)


class GroupedQueryAttentionBwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, heads_kv, dim, causal, dtype, tune",
            [
                pytest.param(
                    1, 1024, 8, 4, 64, False, torch.float16, False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    1, 1024, 8, 4, 64, False, torch.bfloat16, False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    4, 2048, 64, 4, 128, False, torch.float16, False, marks=pytest.mark.full
                ),
                pytest.param(
                    4, 2048, 64, 4, 128, False, torch.bfloat16, False, marks=pytest.mark.full
                ),
            ],
        ),
    ]


@pytest.mark.smoke
def test_gqa_prefill_varlen_rejects_bad_contract_inputs() -> None:
    q_lens, kv_lens = [64, 32], [128, 96]
    heads, heads_kv, dim = 8, 2, 64
    q = torch.randn(sum(q_lens), heads, dim, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(sum(kv_lens), heads_kv, dim, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn_like(k)
    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )
    cu_kv = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )

    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        max(q_lens), max(kv_lens), True, validate_inputs=True
    )
    with pytest.raises(ValueError, match="Expected k shape"):
        op(q, k[:, :, :-1].contiguous(), v, cu_q, cu_kv)
    with pytest.raises(ValueError, match="cu_seqlens_q\\[-1\\].*must equal"):
        op(q[:-1], k, v, cu_q, cu_kv)
    with pytest.raises(ValueError, match="max_seqlen_q"):
        bad_op = GroupedQueryAttentionPrefillVarlenFwdOp(
            max(q_lens) - 1, max(kv_lens), True, validate_inputs=True
        )
        bad_op(q, k, v, cu_q, cu_kv)
    bad_cu = torch.tensor([0, 128, 96], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="cu_seqlens_q must be non-decreasing"):
        op(q, k, v, bad_cu, cu_kv)


@pytest.mark.smoke
def test_gqa_prefill_varlen_rejects_unsupported_dtype() -> None:
    """The element type now arrives with the tensors, so the rejection does too."""
    op = GroupedQueryAttentionPrefillVarlenFwdOp(max_seqlen_q=64, max_seqlen_kv=128)
    kwargs = {"dtype": torch.float32, "device": "cuda"}
    q = torch.randn(64, 8, 64, **kwargs)
    k = torch.randn(128, 2, 64, **kwargs)
    v = torch.randn(128, 2, 64, **kwargs)
    cu_q = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="Expected dtype torch.float16 or torch.bfloat16"):
        op(q, k, v, cu_q, cu_kv)


@GroupedQueryAttentionBwdFixture
def test_gqa_bwd(
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    causal: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GroupedQueryAttentionBwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    op = GroupedQueryAttentionBwdOp(batch, heads, heads_kv, seq_len, dim, causal, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)
