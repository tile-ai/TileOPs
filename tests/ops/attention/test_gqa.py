import dataclasses
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.test_base import FixtureBase, TestBase
from tileops.ops import (
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionPrefillDenseFwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
)
from tileops.ops.attention.selection import DENSE_PREFILL_KEYS
from tileops.ops.op_base import Op
from workloads.attention.gqa import (
    GroupedQueryAttentionBwdWorkload,
    GroupedQueryAttentionFwdWorkload,
)

_PREFILL_TOLERANCE = {
    torch.float16: (5e-3, 1e-5),
    torch.bfloat16: (8e-2, 1e-2),
}


def _rope_tables(
    max_position: int, rotary_dim: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, rotary_dim, 2, device="cuda", dtype=torch.float32) / rotary_dim)
    )
    freqs = torch.outer(torch.arange(max_position, device="cuda", dtype=torch.float32), inv_freq)
    return freqs.cos().to(dtype).contiguous(), freqs.sin().to(dtype).contiguous()


def _apply_test_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    rope_layout: str,
) -> torch.Tensor:
    selected_cos = cos[position_ids]
    selected_sin = sin[position_ids]
    while selected_cos.ndim < x.ndim:
        selected_cos = selected_cos.unsqueeze(-2)
        selected_sin = selected_sin.unsqueeze(-2)
    out = x.clone()
    if rope_layout == "neox":
        half = rotary_dim // 2
        first = x[..., :half]
        second = x[..., half:rotary_dim]
        out[..., :half] = first * selected_cos - second * selected_sin
        out[..., half:rotary_dim] = second * selected_cos + first * selected_sin
    elif rope_layout == "interleaved":
        even = x[..., :rotary_dim:2]
        odd = x[..., 1:rotary_dim:2]
        rotated = torch.stack(
            (even * selected_cos - odd * selected_sin, odd * selected_cos + even * selected_sin),
            dim=-1,
        )
        out[..., :rotary_dim] = rotated.flatten(-2)
    else:
        raise ValueError(f"unsupported test RoPE layout: {rope_layout}")
    return out


#: The shipped implementations of the Dense-prefill slot, by dispatch key.
_SHIPPED_PREFILL_MAP = GroupedQueryAttentionPrefillDenseFwdOp.default_kernel_map.fget(
    object.__new__(GroupedQueryAttentionPrefillDenseFwdOp)
)


def _stand_in(real: type) -> type:
    """A replacement for *real* that answers selection but compiles nothing.

    ``refusal`` / ``general`` / ``supported_archs`` come from the class it
    stands in for, so selection reaches the same key it would have; only the
    instance is cheap, returning the semantic output shape a prefill kernel
    returns.
    """

    class StandIn(real):
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

        def forward(self, q: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            return torch.empty_like(q)

    return StandIn


def _stand_in_prefill_map() -> dict:
    """A ``kernel_map=`` replacing every Dense-prefill key with a stand-in."""
    return {key: _stand_in(cls) for key, cls in _SHIPPED_PREFILL_MAP.items()}


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


class GroupedQueryAttentionFwdTest(GroupedQueryAttentionFwdWorkload, TestBase):
    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return _gqa_prefill_ref(
            q,
            k,
            v,
            heads=self.heads,
            heads_kv=self.heads_kv,
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
            softcap=self.softcap,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
        )


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
    visible = torch.ones((seq_len_q, seq_len_kv), device=q.device, dtype=torch.bool)
    if is_causal:
        visible &= k_pos <= q_pos
    if window_size_left >= 0:
        visible &= k_pos >= q_pos - window_size_left
    if window_size_right >= 0:
        visible &= k_pos <= q_pos + window_size_right
    scores = scores.masked_fill(~visible.view(1, 1, seq_len_q, seq_len_kv), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v_bhsd)
    assert output.shape == (batch, heads, seq_len_q, dim)
    return output.transpose(1, 2).to(q.dtype).contiguous()


def _gqa_prefill_varlen_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    *,
    batch: int,
    heads: int,
    heads_kv: int,
    is_causal: bool,
    sm_scale: Optional[float] = None,
    softcap: Optional[float] = None,
    window_size_left: int = -1,
    window_size_right: int = -1,
) -> torch.Tensor:
    groups = heads // heads_kv
    dim = q.shape[-1]
    scale = dim**-0.5 if sm_scale is None else sm_scale
    outputs = []
    for b in range(batch):
        q_start = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b + 1].item())
        kv_start = int(cu_seqlens_kv[b].item())
        kv_end = int(cu_seqlens_kv[b + 1].item())
        q_bhsd = q[q_start:q_end].transpose(0, 1).float()
        k_i = k[kv_start:kv_end].repeat_interleave(groups, dim=1).permute(1, 0, 2).float()
        v_i = v[kv_start:kv_end].repeat_interleave(groups, dim=1).permute(1, 0, 2).float()
        q_len = q_end - q_start
        kv_len = kv_end - kv_start
        scores = torch.matmul(q_bhsd, k_i.transpose(-2, -1)) * scale
        if softcap is not None and softcap > 0:
            scores = softcap * torch.tanh(scores / softcap)
        offset = kv_len - q_len
        center = torch.arange(q_len, device=q.device)[:, None] + offset
        kv_pos = torch.arange(kv_len, device=q.device)[None, :]
        visible = torch.ones((q_len, kv_len), device=q.device, dtype=torch.bool)
        if is_causal:
            visible &= kv_pos <= center
        if window_size_left >= 0:
            visible &= kv_pos >= center - window_size_left
        if window_size_right >= 0:
            visible &= kv_pos <= center + window_size_right
        scores = scores.masked_fill(~visible.view(1, q_len, kv_len), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(probs, v_i).transpose(0, 1).to(q.dtype).contiguous())
    return torch.cat(outputs, dim=0)


class GroupedQueryAttentionFwdFixture(FixtureBase):
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
                    4, 512, 64, 4, 128, False, torch.float16, False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    4, 512, 64, 4, 128, True, torch.float16, False, marks=pytest.mark.smoke
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


@GroupedQueryAttentionFwdFixture
def test_gqa_fwd(
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    causal: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GroupedQueryAttentionFwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        batch, heads, heads_kv, seq_len, dim, causal, tune=tune
    )
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_gqa_fwd_output_matches_the_declared_shape() -> None:
    """``H % H_kv`` keeps the validator's mocks away, so assert parity here."""
    batch, seq_len, heads, heads_kv, dim = 1, 128, 8, 2, 64
    op = GroupedQueryAttentionPrefillDenseFwdOp(batch, heads, heads_kv, seq_len, dim, False)
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    o = op(q, k, v)

    assert o.shape == q.shape
    assert o.dtype == q.dtype
    assert op._infer_output_shapes(tuple(q.shape), tuple(k.shape), tuple(v.shape)) == {
        "o": (batch, seq_len, heads, dim),
    }


@pytest.mark.smoke
@pytest.mark.parametrize("rope_layout", ["neox", "interleaved"])
def test_gqa_prefill_dense_fused_rope(rope_layout: str) -> None:
    batch, seq_len_q, seq_len_kv = 2, 48, 80
    heads, heads_kv, dim, rotary_dim = 8, 2, 64, 32
    dtype = torch.float16
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    cos, sin = _rope_tables(seq_len_kv, rotary_dim, dtype)
    q_positions = torch.arange(
        seq_len_kv - seq_len_q, seq_len_kv, device="cuda", dtype=torch.long
    ).expand(batch, -1)
    k_positions = torch.arange(seq_len_kv, device="cuda", dtype=torch.long).expand(batch, -1)
    q_rot = _apply_test_rope(q, q_positions, cos, sin, rotary_dim, rope_layout)
    k_rot = _apply_test_rope(k, k_positions, cos, sin, rotary_dim, rope_layout)
    ref = _gqa_prefill_ref(q_rot, k_rot, v, heads=heads, heads_kv=heads_kv, is_causal=True)
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        batch,
        heads,
        heads_kv,
        seq_len_q,
        dim,
        True,
        seq_len_kv=seq_len_kv,
        fuse_rope=True,
        rotary_dim=rotary_dim,
        rope_layout=rope_layout,
    )
    output = op(q, k, v, rope_cos=cos, rope_sin=sin)
    torch.testing.assert_close(output, ref, atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_gqa_prefill_dense_native_fp8_fused_rope() -> None:
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("native FP8 prefill requires SM90 and float8_e4m3fn")
    batch, seq_len_q, seq_len_kv = 1, 65, 97
    heads, heads_kv, dim, rotary_dim = 8, 2, 128, 64
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda").clamp(-2, 2).to(fp8)
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda").clamp(-2, 2).to(fp8)
    v = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda").clamp(-2, 2).to(fp8)
    cos, sin = _rope_tables(seq_len_kv, rotary_dim, torch.float16)
    q_positions = torch.arange(
        seq_len_kv - seq_len_q, seq_len_kv, device="cuda", dtype=torch.long
    ).expand(batch, -1)
    k_positions = torch.arange(seq_len_kv, device="cuda", dtype=torch.long).expand(batch, -1)
    q_rot = (
        _apply_test_rope(q.to(torch.float16), q_positions, cos, sin, rotary_dim, "interleaved")
        .to(fp8)
        .to(torch.float16)
    )
    k_rot = (
        _apply_test_rope(k.to(torch.float16), k_positions, cos, sin, rotary_dim, "interleaved")
        .to(fp8)
        .to(torch.float16)
    )
    ref = _gqa_prefill_ref(
        q_rot, k_rot, v.to(torch.float16), heads=heads, heads_kv=heads_kv, is_causal=True
    )
    scale = torch.ones((batch, heads_kv), device="cuda", dtype=torch.float32)
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        batch,
        heads,
        heads_kv,
        seq_len_q,
        dim,
        True,
        seq_len_kv=seq_len_kv,
        dtype=torch.float16,
        fuse_rope=True,
        rotary_dim=rotary_dim,
        rope_layout="interleaved",
    )
    output = op(q, k, v, scale, scale, scale, cos, sin)
    torch.testing.assert_close(output, ref, atol=8e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "q_lens, kv_lens, heads, heads_kv, dim, causal, dtype",
    [
        pytest.param(
            [64, 128],
            [64, 128],
            8,
            2,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="uniform-causal-fp16",
        ),
        pytest.param(
            [33, 96, 129],
            [64, 128, 256],
            8,
            2,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="mixed-tail-causal-fp16",
        ),
        pytest.param(
            [64, 96],
            [128, 160],
            8,
            2,
            64,
            False,
            torch.float16,
            marks=pytest.mark.smoke,
            id="mixed-noncausal-fp16",
        ),
        pytest.param(
            [33, 65],
            [33, 65],
            8,
            2,
            64,
            False,
            torch.float16,
            marks=pytest.mark.smoke,
            id="equal-tail-noncausal-fp16",
        ),
        pytest.param(
            [64, 96],
            [128, 160],
            8,
            1,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="mqa-causal-fp16",
        ),
        pytest.param(
            [96],
            [160],
            8,
            2,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="batch1-causal-fp16",
        ),
        pytest.param(
            [64, 128],
            [128, 256],
            8,
            2,
            64,
            True,
            torch.bfloat16,
            marks=pytest.mark.smoke,
            id="mixed-causal-bf16",
        ),
    ],
)
def test_gqa_prefill_varlen_fwd(
    q_lens: list[int],
    kv_lens: list[int],
    heads: int,
    heads_kv: int,
    dim: int,
    causal: bool,
    dtype: torch.dtype,
) -> None:
    batch = len(q_lens)
    total_q = sum(q_lens)
    total_kv = sum(kv_lens)
    q = torch.randn(total_q, heads, dim, device="cuda", dtype=dtype).contiguous()
    k = torch.randn(total_kv, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    v = torch.randn(total_kv, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )
    cu_kv = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )
    ref = _gqa_prefill_varlen_ref(
        q, k, v, cu_q, cu_kv, batch=batch, heads=heads, heads_kv=heads_kv, is_causal=causal
    )

    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        is_causal=causal,
    )
    output = op(q, k, v, cu_q, cu_kv)

    atol, rtol = _PREFILL_TOLERANCE[dtype]
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.parametrize("rope_layout", ["neox", "interleaved"])
def test_gqa_prefill_varlen_fused_rope(rope_layout: str) -> None:
    q_lens, kv_lens = [33, 48], [65, 80]
    batch, heads, heads_kv, dim, rotary_dim = 2, 8, 2, 64, 32
    dtype = torch.float16
    q = torch.randn(sum(q_lens), heads, dim, device="cuda", dtype=dtype).contiguous()
    k = torch.randn(sum(kv_lens), heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    v = torch.randn_like(k)
    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )
    cu_kv = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )
    cos, sin = _rope_tables(max(kv_lens), rotary_dim, dtype)
    q_positions = torch.cat(
        [
            torch.arange(kv_len - q_len, kv_len, device="cuda", dtype=torch.long)
            for q_len, kv_len in zip(q_lens, kv_lens, strict=True)
        ]
    )
    k_positions = torch.cat(
        [torch.arange(kv_len, device="cuda", dtype=torch.long) for kv_len in kv_lens]
    )
    q_rot = _apply_test_rope(q, q_positions, cos, sin, rotary_dim, rope_layout)
    k_rot = _apply_test_rope(k, k_positions, cos, sin, rotary_dim, rope_layout)
    ref = _gqa_prefill_varlen_ref(
        q_rot, k_rot, v, cu_q, cu_kv, batch=batch, heads=heads, heads_kv=heads_kv, is_causal=True
    )
    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        is_causal=True,
        fuse_rope=True,
        rotary_dim=rotary_dim,
        rope_layout=rope_layout,
    )
    output = op(q, k, v, cu_q, cu_kv, rope_cos=cos, rope_sin=sin)
    torch.testing.assert_close(output, ref, atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_gqa_prefill_varlen_native_fp8_fused_rope() -> None:
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("native FP8 prefill requires SM90 and float8_e4m3fn")
    q_lens, kv_lens = [33, 48], [65, 80]
    batch, heads, heads_kv, dim, rotary_dim = 2, 8, 2, 128, 64
    q = torch.randn(sum(q_lens), heads, dim, device="cuda").clamp(-2, 2).to(fp8)
    k = torch.randn(sum(kv_lens), heads_kv, dim, device="cuda").clamp(-2, 2).to(fp8)
    v = torch.randn(sum(kv_lens), heads_kv, dim, device="cuda").clamp(-2, 2).to(fp8)
    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )
    cu_kv = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )
    cos, sin = _rope_tables(max(kv_lens), rotary_dim, torch.float16)
    q_positions = torch.cat(
        [
            torch.arange(kv_len - q_len, kv_len, device="cuda", dtype=torch.long)
            for q_len, kv_len in zip(q_lens, kv_lens, strict=True)
        ]
    )
    k_positions = torch.cat(
        [torch.arange(kv_len, device="cuda", dtype=torch.long) for kv_len in kv_lens]
    )
    q_rot = (
        _apply_test_rope(q.to(torch.float16), q_positions, cos, sin, rotary_dim, "neox")
        .to(fp8)
        .to(torch.float16)
    )
    k_rot = (
        _apply_test_rope(k.to(torch.float16), k_positions, cos, sin, rotary_dim, "neox")
        .to(fp8)
        .to(torch.float16)
    )
    ref = _gqa_prefill_varlen_ref(
        q_rot,
        k_rot,
        v.to(torch.float16),
        cu_q,
        cu_kv,
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        is_causal=True,
    )
    scale = torch.ones((batch, heads_kv), device="cuda", dtype=torch.float32)
    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        is_causal=True,
        dtype=torch.float16,
        fuse_rope=True,
        rotary_dim=rotary_dim,
        rope_layout="neox",
    )
    output = op(q, k, v, cu_q, cu_kv, scale, scale, scale, cos, sin)
    torch.testing.assert_close(output, ref, atol=8e-2, rtol=2e-2)


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", ["dense", "varlen", "paged"])
def test_gqa_prefill_rejects_unknown_rope_layout(op_kind: str) -> None:
    common = dict(heads=8, heads_kv=2, dim=64, fuse_rope=True, rope_layout="unknown")
    with pytest.raises(ValueError, match="rope_layout"):
        if op_kind == "dense":
            GroupedQueryAttentionPrefillDenseFwdOp(batch=1, seq_len=32, is_causal=True, **common)
        elif op_kind == "varlen":
            GroupedQueryAttentionPrefillVarlenFwdOp(
                batch=1, max_seqlen_q=32, max_seqlen_kv=32, is_causal=True, **common
            )
        else:
            from tileops.ops import GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp

            GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
                batch=1,
                max_pages_per_req=2,
                page_size=64,
                max_seqlen_q=32,
                is_causal=True,
                **common,
            )


@pytest.mark.smoke
def test_gqa_prefill_rope_tables_are_required_exactly_when_enabled() -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 32, 8, 2, 64
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    fused = GroupedQueryAttentionPrefillDenseFwdOp(
        batch, heads, heads_kv, seq_len, dim, True, fuse_rope=True
    )
    with pytest.raises(ValueError, match="requires both"):
        fused(q, k, v)
    plain = GroupedQueryAttentionPrefillDenseFwdOp(batch, heads, heads_kv, seq_len, dim, True)
    cos, sin = _rope_tables(seq_len, dim, torch.float16)
    with pytest.raises(ValueError, match="require fuse_rope=True"):
        plain(q, k, v, rope_cos=cos, rope_sin=sin)


@pytest.mark.smoke
def test_gqa_prefill_dense_fused_rope_cuda_graph_replay() -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 32, 8, 2, 64
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    cos, sin = _rope_tables(seq_len, dim, torch.float16)
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        batch, heads, heads_kv, seq_len, dim, True, fuse_rope=True, rope_layout="interleaved"
    )
    op(q, k, v, rope_cos=cos, rope_sin=sin)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(q, k, v, rope_cos=cos, rope_sin=sin)
    first = captured.clone()
    q.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(captured).all()
    assert not torch.equal(captured, first)


@pytest.mark.smoke
def test_gqa_prefill_varlen_native_fp8_causal_window_softcap_tail() -> None:
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None:
        pytest.skip("torch.float8_e4m3fn is unavailable")
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        pytest.skip("native FP8 varlen prefill requires SM90")

    q_lens, kv_lens = [65, 97], [129, 193]
    batch, heads, heads_kv, dim = 2, 8, 2, 128
    groups = heads // heads_kv
    q_scale = torch.tensor([[0.08, 0.11], [0.07, 0.13]], device="cuda", dtype=torch.float32)
    k_scale = torch.tensor([[0.09, 0.12], [0.10, 0.075]], device="cuda", dtype=torch.float32)
    v_scale = torch.tensor([[0.06, 0.14], [0.085, 0.105]], device="cuda", dtype=torch.float32)
    q_parts, k_parts, v_parts = [], [], []
    q_ref_parts, k_ref_parts, v_ref_parts = [], [], []
    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens, strict=True)):
        q_ref = torch.randn(q_len, heads, dim, device="cuda", dtype=torch.float16)
        k_ref = torch.randn(kv_len, heads_kv, dim, device="cuda", dtype=torch.float16)
        v_ref = torch.randn_like(k_ref)
        q_head_scale = q_scale[b].repeat_interleave(groups).view(1, heads, 1)
        q_quant = (q_ref.float() / q_head_scale).to(fp8)
        k_quant = (k_ref.float() / k_scale[b].view(1, heads_kv, 1)).to(fp8)
        v_quant = (v_ref.float() / v_scale[b].view(1, heads_kv, 1)).to(fp8)
        q_parts.append(q_quant)
        k_parts.append(k_quant)
        v_parts.append(v_quant)
        q_ref_parts.append((q_quant.float() * q_head_scale).to(torch.float16))
        k_ref_parts.append((k_quant.float() * k_scale[b].view(1, heads_kv, 1)).to(torch.float16))
        v_ref_parts.append((v_quant.float() * v_scale[b].view(1, heads_kv, 1)).to(torch.float16))

    q, k, v = map(lambda parts: torch.cat(parts).contiguous(), (q_parts, k_parts, v_parts))
    q_ref, k_ref, v_ref = map(
        lambda parts: torch.cat(parts).contiguous(),
        (q_ref_parts, k_ref_parts, v_ref_parts),
    )
    cu_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cu_kv = torch.tensor(
        [0, *torch.tensor(kv_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    softcap, window_size_left = 2.0, 80
    ref = _gqa_prefill_varlen_ref(
        q_ref,
        k_ref,
        v_ref,
        cu_q,
        cu_kv,
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        is_causal=True,
        softcap=softcap,
        window_size_left=window_size_left,
    )
    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        is_causal=True,
        softcap=softcap,
        window_size_left=window_size_left,
        dtype=torch.float16,
    )

    output = op(q, k, v, cu_q, cu_kv, q_scale, k_scale, v_scale)

    assert op._get_kernel(fp8).__class__.__name__ == ("GQAPrefillVarlenFP8TensorCoreFwdKernel")
    torch.testing.assert_close(output, ref, atol=0.12, rtol=0.10)


@pytest.mark.smoke
def test_gqa_prefill_varlen_respects_sm_scale() -> None:
    q_lens, kv_lens = [64, 96], [128, 160]
    batch, heads, heads_kv, dim = len(q_lens), 8, 2, 64
    sm_scale = 0.125
    q = torch.randn(sum(q_lens), heads, dim, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(sum(kv_lens), heads_kv, dim, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn_like(k)
    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )
    cu_kv = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )
    ref = _gqa_prefill_varlen_ref(
        q,
        k,
        v,
        cu_q,
        cu_kv,
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        is_causal=True,
        sm_scale=sm_scale,
    )

    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        is_causal=True,
        sm_scale=sm_scale,
    )
    output = op(q, k, v, cu_q, cu_kv)

    torch.testing.assert_close(output, ref, atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_gqa_prefill_varlen_respects_softcap() -> None:
    q_lens, kv_lens = [64, 96], [128, 160]
    batch, heads, heads_kv, dim = len(q_lens), 8, 2, 64
    softcap = 2.0
    q = torch.randn(sum(q_lens), heads, dim, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(sum(kv_lens), heads_kv, dim, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn_like(k)
    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )
    cu_kv = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )
    ref = _gqa_prefill_varlen_ref(
        q,
        k,
        v,
        cu_q,
        cu_kv,
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        is_causal=True,
        softcap=softcap,
    )

    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        is_causal=True,
        softcap=softcap,
    )
    output = op(q, k, v, cu_q, cu_kv)

    torch.testing.assert_close(output, ref, atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_gqa_prefill_varlen_combines_window_and_softcap() -> None:
    q_lens, kv_lens = [64, 96], [128, 160]
    batch, heads, heads_kv, dim = len(q_lens), 8, 2, 64
    softcap, window_size_left = 2.0, 48
    q = torch.randn(sum(q_lens), heads, dim, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(sum(kv_lens), heads_kv, dim, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn_like(k)
    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )
    cu_kv = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32
    )
    ref = _gqa_prefill_varlen_ref(
        q,
        k,
        v,
        cu_q,
        cu_kv,
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        is_causal=True,
        softcap=softcap,
        window_size_left=window_size_left,
    )

    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        is_causal=True,
        softcap=softcap,
        window_size_left=window_size_left,
    )
    output = op(q, k, v, cu_q, cu_kv)

    torch.testing.assert_close(output, ref, atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_gqa_prefill_varlen_rejects_bad_contract_inputs() -> None:
    q_lens, kv_lens = [64, 32], [128, 96]
    batch, heads, heads_kv, dim = len(q_lens), 8, 2, 64
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
        batch, heads, heads_kv, dim, max(q_lens), max(kv_lens), True, validate_inputs=True
    )
    with pytest.raises(ValueError, match="Expected k shape"):
        op(q, k[:, :, :-1].contiguous(), v, cu_q, cu_kv)
    with pytest.raises(ValueError, match="cu_seqlens_q\\[-1\\].*must equal"):
        op(q[:-1], k, v, cu_q, cu_kv)
    with pytest.raises(ValueError, match="max_seqlen_q"):
        bad_op = GroupedQueryAttentionPrefillVarlenFwdOp(
            batch, heads, heads_kv, dim, max(q_lens) - 1, max(kv_lens), True, validate_inputs=True
        )
        bad_op(q, k, v, cu_q, cu_kv)
    bad_cu = torch.tensor([0, 128, 96], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="cu_seqlens_q must be non-decreasing"):
        op(q, k, v, bad_cu, cu_kv)


@pytest.mark.smoke
def test_gqa_prefill_varlen_rejects_unsupported_dtype() -> None:
    """The element type now arrives with the tensors, so the rejection does too."""
    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        batch=1,
        heads=8,
        heads_kv=2,
        dim=64,
        max_seqlen_q=64,
        max_seqlen_kv=128,
    )
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


# The square BSHD wrapper reaches a dense-prefill kernel by itself: it states
# its call, selects a key, and builds through the shared step. These pin that
# it holds no child op, builds once, and passes exactly what the packed op does.


def _holds_op(value: object, depth: int = 0) -> bool:
    """Whether *value* is or reaches an ``Op``, descending the same containers
    and depth as the L1 kernel walk."""
    if isinstance(value, Op):
        return True
    if depth >= 2:
        return False
    if isinstance(value, dict):
        return any(_holds_op(item, depth + 1) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_holds_op(item, depth + 1) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(
            _holds_op(getattr(value, field.name), depth + 1) for field in dataclasses.fields(value)
        )
    return False


def _op_valued_attrs(op: Op) -> list:
    """Names of *op*'s attributes that are or reach an ``Op``; scans ``dir``
    so properties and class-bound attributes are covered."""
    return sorted(
        name for name in dir(op) if not name.startswith("__") and _holds_op(getattr(op, name, None))
    )


def _record_kernel_builds(op: Op) -> list:
    """Replace each of *op*'s kernel slots with a recorder of its build call.

    The recorder still answers ``refusal``, delegating to the class it stands
    in for, so selection runs exactly as it would have.

    Returns the list the recorders append ``(slot, args, kwargs)`` to.
    """
    calls: list = []

    def recorder(slot: str, real: type) -> type:
        class Recorder:
            supported_archs = real.supported_archs
            general = real.general

            @classmethod
            def refusal(cls, call: object) -> "str | None":
                return real.refusal(call)

            def __new__(cls, *args: object, **kwargs: object) -> str:
                calls.append((slot, args, kwargs))
                return f"built:{slot}"

        return Recorder

    for slot in op.kernel_map:
        op.kernel_map[slot] = recorder(slot, op.kernel_map[slot])
    return calls


@pytest.mark.smoke
def test_gqa_fwd_bshd_wrapper_ctor_rejects_non_positive_dims() -> None:
    """Nothing downstream validates; a zero ``heads_kv`` would surface as
    ``ZeroDivisionError`` at ``heads % heads_kv``."""
    with pytest.raises(ValueError, match="heads_kv must be positive"):
        GroupedQueryAttentionPrefillDenseFwdOp(1, 8, 0, 64, 64, True)
    with pytest.raises(ValueError, match="batch must be positive"):
        GroupedQueryAttentionPrefillDenseFwdOp(0, 8, 2, 64, 64, True)
    with pytest.raises(ValueError, match="seq_len must be positive"):
        GroupedQueryAttentionPrefillDenseFwdOp(1, 8, 2, 0, 64, True)


@pytest.mark.smoke
def test_dense_prefill_path_rejects_unsupported_dtype() -> None:
    """Every region declines rather than raising, so an unguarded element type
    would land on the general dense implementation. Both entry points into the
    dense-prefill build must refuse it instead."""
    with pytest.raises(ValueError, match="float16 or torch.bfloat16"):
        GroupedQueryAttentionPrefillDenseFwdOp(1, 8, 2, 128, 128, True)._get_kernel(torch.float32)


@pytest.mark.smoke
def test_gqa_fwd_bshd_wrapper_caches_its_own_kernel_and_holds_no_child_op() -> None:
    """The wrapper builds its kernel once and holds no op to build it for one.

    Two calls leave one entry under the key selection reached: the registry is
    what "built once" means, so nothing has to count constructor calls.
    """
    batch, seq_len, heads, heads_kv, dim = 2, 64, 8, 2, 64
    q = torch.empty(batch, seq_len, heads, dim, dtype=torch.float16)
    k = torch.empty(batch, seq_len, heads_kv, dim, dtype=torch.float16)
    v = torch.empty_like(k)
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        batch, heads, heads_kv, seq_len, dim, True, kernel_map=_stand_in_prefill_map()
    )

    assert op(q, k, v).shape == q.shape
    assert op(q, k, v).shape == q.shape

    assert list(op.built_kernels("gqa_prefill_dense")) == [(q.device, torch.float16, None, None)]
    assert _op_valued_attrs(op) == []


@pytest.mark.smoke
def test_dense_window_dispatch_keeps_scale_and_softcap_in_specialization() -> None:
    """Scale and softcap are semantics of the retained SM90 pipeline, not
    reasons to fall back to the general dense implementation."""
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        batch=1,
        heads=8,
        heads_kv=2,
        seq_len=128,
        dim=64,
        is_causal=False,
        sm_scale=0.125,
        softcap=3.0,
        window_size_left=48,
        window_size_right=16,
    )
    call = dataclasses.replace(op.attention_call(torch.float16), arch=90, h200=True)

    assert op.select_kernel_key(DENSE_PREFILL_KEYS, call) == (
        "gqa_prefill_dense_sliding_fwd_kernel"
    )


@pytest.mark.smoke
def test_dense_window_dispatch_falls_back_off_sm90() -> None:
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        batch=1,
        heads=8,
        heads_kv=2,
        seq_len=128,
        dim=64,
        window_size_left=48,
    )
    call = dataclasses.replace(op.attention_call(torch.float16), arch=80, h200=False)

    assert op.select_kernel_key(DENSE_PREFILL_KEYS, call) == "gqa_prefill_fwd_kernel"


@pytest.mark.smoke
def test_dense_window_with_scale_and_softcap_matches_reference() -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 128, 8, 2, 64
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    kwargs = {
        "is_causal": False,
        "sm_scale": 0.125,
        "softcap": 3.0,
        "window_size_left": 48,
        "window_size_right": 16,
    }
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        seq_len=seq_len,
        dim=dim,
        **kwargs,
    )
    ref = _gqa_prefill_ref(q, k, v, heads=heads, heads_kv=heads_kv, **kwargs)

    torch.testing.assert_close(op(q, k, v), ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
