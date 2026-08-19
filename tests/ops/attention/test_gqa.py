import dataclasses
import inspect
import re
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.attention import GQAPrefillDenseFwdKernel, GQAPrefillVarlenFwdKernel
from tileops.kernels.attention.prefill import DensePrefillKernel, PackedPrefillKernel
from tileops.ops import (
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionPrefillDenseFwdOp,
    GroupedQueryAttentionPrefillFwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
)
from tileops.ops.attention.selection import PACKED_PREFILL_KEYS
from tileops.ops.op_base import Op
from workloads.attention.gqa import (
    GroupedQueryAttentionBwdWorkload,
    GroupedQueryAttentionFwdWorkload,
    uniform_packed_prefill_inputs,
)

_PREFILL_TOLERANCE = {
    torch.float16: (5e-3, 1e-5),
    torch.bfloat16: (8e-2, 1e-2),
}


def _rope_tables(
    max_position: int,
    rotary_dim: int,
    dtype: torch.dtype,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, device="cuda", dtype=torch.float32) / rotary_dim)
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


def _selected_prefill_kernel_cls(op: GroupedQueryAttentionPrefillFwdOp) -> type:
    """Kernel class selection picks for a uniform, non-FP8 packed prefill call."""
    call = op.attention_call(is_fp8=False)
    return op.kernel_map[op.select_kernel_key(PACKED_PREFILL_KEYS, call)]


#: The shipped implementations of the packed-prefill slot, by dispatch key.
_SHIPPED_PREFILL_MAP = GroupedQueryAttentionPrefillFwdOp.default_kernel_map.fget(
    object.__new__(GroupedQueryAttentionPrefillFwdOp)
)
_SHIPPED_DENSE_PREFILL_MAP = GroupedQueryAttentionPrefillDenseFwdOp.default_kernel_map.fget(
    object.__new__(GroupedQueryAttentionPrefillDenseFwdOp)
)


def _stand_in(real: type, *, allow_cpu: bool = False) -> type:
    """A replacement for *real* that answers selection but compiles nothing.

    ``refusal`` / ``general`` / ``supported_archs`` come from the class it
    stands in for, so selection reaches the same key it would have.  CPU may be
    admitted explicitly for wrapper-only tests that never launch the real CUDA
    implementation.  The instance itself is cheap and only returns the semantic
    output shape a prefill kernel returns.
    """

    class StandIn(real):
        supported_archs = [*real.supported_archs, 0] if allow_cpu else real.supported_archs

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

        def forward(self, q: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            return torch.empty_like(q)

    return StandIn


def _stand_in_prefill_map() -> dict:
    """A ``kernel_map=`` replacing every packed-prefill key with a stand-in."""
    return {key: _stand_in(cls) for key, cls in _SHIPPED_PREFILL_MAP.items()}


def _stand_in_dense_prefill_map() -> dict:
    """Replace every Dense-prefill implementation with a non-compiling stand-in."""
    return {key: _stand_in(cls, allow_cpu=True) for key, cls in _SHIPPED_DENSE_PREFILL_MAP.items()}


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
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True
            )
        output = output_bhsd.transpose(1, 2).contiguous()
        return output


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
    probs = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    output = torch.matmul(probs, v_bhsd)
    assert output.shape == (batch, heads, seq_len_q, dim)
    return output.transpose(1, 2).to(q.dtype).contiguous()


def _ones_prefill_scales(
    batch: int,
    heads_kv: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_scale = torch.ones((batch, heads_kv), device=device, dtype=torch.float32)
    return q_scale, torch.ones_like(q_scale), torch.ones_like(q_scale)


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
        if is_causal:
            offset = kv_len - q_len
            q_pos = torch.arange(q_len, device=q.device)[:, None] + offset
            kv_pos = torch.arange(kv_len, device=q.device)[None, :]
            mask = kv_pos <= q_pos
            scores = scores.masked_fill(~mask.view(1, q_len, kv_len), float("-inf"))
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
    op = GroupedQueryAttentionPrefillDenseFwdOp(is_causal=causal, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_gqa_fwd_output_matches_the_declared_shape() -> None:
    """``H % H_kv`` keeps the validator's mocks away, so assert parity here."""
    batch, seq_len, heads, heads_kv, dim = 1, 128, 8, 2, 64
    op = GroupedQueryAttentionPrefillDenseFwdOp(is_causal=False)
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
@pytest.mark.parametrize(
    ("rope_layout", "rope_base"),
    [("neox", 10000.0), ("interleaved", 500000.0)],
)
def test_gqa_prefill_dense_fused_rope(rope_layout: str, rope_base: float) -> None:
    batch, seq_len_q, seq_len_kv = 2, 48, 80
    heads, heads_kv, dim, rotary_dim = 8, 2, 64, 32
    dtype = torch.float16
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    cos, sin = _rope_tables(seq_len_kv, rotary_dim, dtype, rope_base)
    q_positions = torch.arange(
        seq_len_kv - seq_len_q, seq_len_kv, device="cuda", dtype=torch.long
    ).expand(batch, -1)
    k_positions = torch.arange(seq_len_kv, device="cuda", dtype=torch.long).expand(batch, -1)
    q_rot = _apply_test_rope(q, q_positions, cos, sin, rotary_dim, rope_layout)
    k_rot = _apply_test_rope(k, k_positions, cos, sin, rotary_dim, rope_layout)
    ref = _gqa_prefill_ref(q_rot, k_rot, v, heads=heads, heads_kv=heads_kv, is_causal=True)
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        is_causal=True,
        pos_encoding_mode="rope",
        rotary_dim=rotary_dim,
        rope_layout=rope_layout,
        rope_base=rope_base,
    )

    output = op(q, k, v)

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
        q_rot,
        k_rot,
        v.to(torch.float16),
        heads=heads,
        heads_kv=heads_kv,
        is_causal=True,
    )
    scale = torch.ones((batch, heads_kv), device="cuda", dtype=torch.float32)
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        is_causal=True,
        dtype=torch.float16,
        pos_encoding_mode="rope",
        rotary_dim=rotary_dim,
        rope_layout="interleaved",
    )

    output = op(q, k, v, scale, scale, scale)

    torch.testing.assert_close(output, ref, atol=8e-2, rtol=2e-2)


@pytest.mark.smoke
def test_gqa_prefill_dense_rope_tables_are_constructor_owned_and_cached() -> None:
    batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 32, 80, 8, 2, 64
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        is_causal=True,
        pos_encoding_mode="rope",
    )

    # 需要先调用 forward 来设置 seq_len_kv
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    op(q, k, v)

    cos, sin = op._rope_tables(torch.device("cuda"), torch.float16)
    cached_cos, cached_sin = op._rope_tables(torch.device("cuda"), torch.float16)

    assert cos.shape == sin.shape == (seq_len_kv, dim // 2)
    assert cached_cos is cos
    assert cached_sin is sin


@pytest.mark.smoke
def test_gqa_prefill_dense_validates_constructor_rope_mode() -> None:
    with pytest.raises(ValueError, match="pos_encoding_mode must be"):
        GroupedQueryAttentionPrefillDenseFwdOp(pos_encoding_mode="alibi")
    with pytest.raises(ValueError, match="requires pos_encoding_mode='rope'"):
        GroupedQueryAttentionPrefillDenseFwdOp(rotary_dim=32)
    with pytest.raises(ValueError, match="rope_base must be finite and positive"):
        GroupedQueryAttentionPrefillDenseFwdOp(pos_encoding_mode="rope", rope_base=0.0)


@pytest.mark.smoke
def test_gqa_prefill_dense_rejects_scales_for_16_bit_input() -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 32, 8, 2, 64
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    scale = torch.ones(batch, heads_kv, device="cuda", dtype=torch.float32)
    op = GroupedQueryAttentionPrefillDenseFwdOp(is_causal=False)

    with pytest.raises(ValueError, match="only valid for FP8 input"):
        op(q, k, v, scale, scale, scale)


@pytest.mark.smoke
def test_gqa_prefill_dense_fused_rope_rejects_negative_q_positions() -> None:
    batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 80, 48, 8, 2, 64
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        is_causal=False,
        pos_encoding_mode="rope",
    )

    with pytest.raises(ValueError, match="requires seq_len <= seq_len_kv"):
        op(q, k, v)


@pytest.mark.smoke
def test_gqa_prefill_dense_fused_rope_cuda_graph_replay() -> None:
    batch, seq_len, heads, heads_kv, dim = 1, 32, 8, 2, 64
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        is_causal=True,
        pos_encoding_mode="rope",
        rope_layout="interleaved",
    )
    op(q, k, v)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(q, k, v)
    first = captured.clone()
    q.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(captured).all()
    assert not torch.equal(captured, first)


@pytest.mark.parametrize(
    (
        "seq_len_q",
        "seq_len_kv",
        "is_causal",
        "window_size_left",
        "window_size_right",
        "sm_scale",
        "softcap",
        "dim",
        "dtype",
    ),
    [
        pytest.param(
            97, 193, True, -1, -1, None, None, 64, torch.float16, id="rectangular-causal-tail"
        ),
        pytest.param(
            129,
            257,
            True,
            96,
            -1,
            0.11,
            2.0,
            64,
            torch.float16,
            id="rectangular-causal-left-window-softcap",
        ),
        pytest.param(
            127,
            191,
            False,
            48,
            16,
            0.125,
            50.0,
            64,
            torch.float16,
            id="bidirectional-window-softcap",
        ),
        pytest.param(
            96, 160, False, -1, 32, None, None, 64, torch.bfloat16, id="right-window-bf16"
        ),
        pytest.param(
            97,
            193,
            True,
            -1,
            -1,
            None,
            None,
            128,
            torch.float16,
            id="rectangular-causal-dim128-tail",
        ),
        pytest.param(
            191, 191, True, 96, -1, None, None, 128, torch.float16, id="sliding-dim128-tail"
        ),
    ],
)
@pytest.mark.smoke
def test_gqa_prefill_dense_semantic_matrix(
    seq_len_q: int,
    seq_len_kv: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    sm_scale: Optional[float],
    softcap: Optional[float],
    dim: int,
    dtype: torch.dtype,
) -> None:
    batch, heads, heads_kv = 1, 8, 2
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    ref = _gqa_prefill_ref(
        q,
        k,
        v,
        heads=heads,
        heads_kv=heads_kv,
        is_causal=is_causal,
        sm_scale=sm_scale,
        softcap=softcap,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        is_causal=is_causal,
        sm_scale=sm_scale,
        softcap=softcap,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )

    output = op(q, k, v)

    atol, rtol = _PREFILL_TOLERANCE[dtype]
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("sm_scale", [0.0, -0.125])
@pytest.mark.smoke
def test_gqa_prefill_dense_empty_window_rows_are_zero(sm_scale: float) -> None:
    batch, seq_len_q, seq_len_kv = 1, 65, 1
    heads, heads_kv, dim = 8, 2, 64
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        is_causal=False,
        window_size_right=0,
        sm_scale=sm_scale,
    )

    output = op(q, k, v)

    assert torch.equal(output[:, :-1], torch.zeros_like(output[:, :-1]))
    torch.testing.assert_close(output[:, -1], v[:, 0].repeat_interleave(4, dim=1))


@pytest.mark.parametrize("sm_scale", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.smoke
def test_gqa_prefill_dense_rejects_non_finite_scale(sm_scale: float) -> None:
    with pytest.raises(ValueError, match="sm_scale must be finite"):
        GroupedQueryAttentionPrefillDenseFwdOp(
            sm_scale=sm_scale,
        )


@pytest.mark.parametrize(
    "batch, seq_len_q, seq_len_kv, heads, heads_kv, dim, causal, dtype",
    [
        pytest.param(
            1,
            128,
            128,
            8,
            2,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="gqa_ratio4_q_eq_kv_fp16",
        ),
        pytest.param(
            1,
            128,
            384,
            8,
            2,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="gqa_ratio4_q_lt_kv_fp16",
        ),
        pytest.param(
            1,
            96,
            150,
            8,
            2,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="gqa_tail_q_and_kv_fp16",
        ),
        pytest.param(
            2,
            128,
            256,
            16,
            4,
            128,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="gqa_ratio4_batch2_dim128_fp16",
        ),
        pytest.param(
            1,
            128,
            256,
            8,
            2,
            64,
            False,
            torch.float16,
            marks=pytest.mark.smoke,
            id="gqa_noncausal_fp16",
        ),
        pytest.param(
            1,
            128,
            256,
            8,
            2,
            64,
            True,
            torch.bfloat16,
            marks=pytest.mark.smoke,
            id="gqa_ratio4_bf16",
        ),
    ],
)
def test_gqa_prefill_fwd(
    batch: int,
    seq_len_q: int,
    seq_len_kv: int,
    heads: int,
    heads_kv: int,
    dim: int,
    causal: bool,
    dtype: torch.dtype,
) -> None:
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=dtype).contiguous()
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    v = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    ref = _gqa_prefill_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=causal)

    packed_inputs = uniform_packed_prefill_inputs(q, k, v)
    op = GroupedQueryAttentionPrefillFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=seq_len_q,
        max_seqlen_kv=seq_len_kv,
        is_causal=causal,
        dtype=dtype,
    )
    output = op(*packed_inputs).view(batch, seq_len_q, heads, dim)

    atol, rtol = _PREFILL_TOLERANCE[dtype]
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_gqa_prefill_fwd_auto_uses_varlen_for_uniform_input() -> None:
    batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 128, 256, 8, 2, 64
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(
        batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16
    ).contiguous()
    v = torch.randn(
        batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16
    ).contiguous()
    ref = _gqa_prefill_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True)

    packed_inputs = uniform_packed_prefill_inputs(q, k, v)
    op = GroupedQueryAttentionPrefillFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=seq_len_q,
        max_seqlen_kv=seq_len_kv,
        is_causal=True,
        dtype=torch.float16,
        backend="auto",
    )
    output = op(*packed_inputs).view(batch, seq_len_q, heads, dim)

    torch.testing.assert_close(output, ref, atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_gqa_prefill_fwd_uses_bottom_right_causal_mask() -> None:
    batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 128, 256, 4, 2, 64
    q = torch.zeros(batch, seq_len_q, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.zeros(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.zeros(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16)
    q[..., 0] = 1
    k[..., 0] = 1
    v[:, :128, :, 0] = 1
    v[:, 128:, :, 0] = 100

    packed_inputs = uniform_packed_prefill_inputs(q.contiguous(), k.contiguous(), v.contiguous())
    op = GroupedQueryAttentionPrefillFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=seq_len_q,
        max_seqlen_kv=seq_len_kv,
        is_causal=True,
        dtype=torch.float16,
    )
    output = op(*packed_inputs).view(batch, seq_len_q, heads, dim)

    assert output[0, 0, 0, 0] < 2
    assert output[0, -1, 0, 0] > 40


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.smoke
def test_gqa_prefill_fwd_uniform_uses_varlen(dtype: torch.dtype) -> None:
    op = GroupedQueryAttentionPrefillFwdOp(
        batch=4,
        heads=32,
        heads_kv=8,
        dim=128,
        max_seqlen_q=512,
        max_seqlen_kv=512,
        is_causal=True,
        dtype=dtype,
        backend="auto",
    )

    assert _selected_prefill_kernel_cls(op) is GQAPrefillVarlenFwdKernel


@pytest.mark.parametrize(
    "sm_scale, softcap",
    [
        pytest.param(0.125, None, id="custom-scale"),
        pytest.param(None, 2.0, id="softcap"),
    ],
)
@pytest.mark.smoke
def test_gqa_prefill_fwd_uniform_feature_variants_use_varlen(
    sm_scale: Optional[float],
    softcap: Optional[float],
) -> None:
    op = GroupedQueryAttentionPrefillFwdOp(
        batch=4,
        heads=32,
        heads_kv=8,
        dim=128,
        max_seqlen_q=512,
        max_seqlen_kv=512,
        is_causal=True,
        dtype=torch.float16,
        sm_scale=sm_scale,
        softcap=softcap,
        backend="auto",
    )

    assert _selected_prefill_kernel_cls(op) is GQAPrefillVarlenFwdKernel


@pytest.mark.parametrize(
    "seq_len_q, seq_len_kv, sm_scale, softcap",
    [
        pytest.param(512, 4096, None, None, id="q-lt-kv"),
    ],
)
@pytest.mark.smoke
def test_gqa_prefill_fwd_q_lt_kv_uses_varlen_kernel(
    seq_len_q: int,
    seq_len_kv: int,
    sm_scale: Optional[float],
    softcap: Optional[float],
) -> None:
    op = GroupedQueryAttentionPrefillFwdOp(
        batch=2,
        heads=32,
        heads_kv=8,
        dim=128,
        max_seqlen_q=seq_len_q,
        max_seqlen_kv=seq_len_kv,
        is_causal=True,
        dtype=torch.float16,
        sm_scale=sm_scale,
        softcap=softcap,
        backend="auto",
    )

    assert _selected_prefill_kernel_cls(op) is GQAPrefillVarlenFwdKernel


@pytest.mark.smoke
@pytest.mark.parametrize("backend", ["varlen", "sliding_window"])
def test_gqa_prefill_fwd_explicit_varlen_backends_accept_ragged_input(
    backend: str,
) -> None:
    """A backend that packs ragged requests takes them: the observable contract.

    Whether the op compared the ranges to a uniform one is its own business; the
    promise is that a ragged ``cu_seqlens`` is served rather than refused.
    """
    batch, seq_len, heads, heads_kv, dim = 2, 64, 8, 2, 64
    q = torch.randn(batch * seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch * seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, seq_len // 2, batch * seq_len], device="cuda", dtype=torch.int32)
    cu_kv = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * seq_len
    q_scale, k_scale, v_scale = _ones_prefill_scales(batch, heads_kv, device=q.device)
    op = GroupedQueryAttentionPrefillFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=seq_len,
        max_seqlen_kv=seq_len,
        is_causal=backend != "fp8",
        dtype=torch.float16,
        backend=backend,
        window_size_left=16 if backend == "sliding_window" else -1,
        kernel_map=_stand_in_prefill_map(),
    )

    out = op(q, k, v, cu_q, cu_kv, q_scale, k_scale, v_scale)
    assert out.shape == q.shape


@pytest.mark.smoke
@pytest.mark.parametrize("backend", ["dense", "fp8"])
def test_gqa_prefill_fwd_explicit_dense_backends_refuse_ragged_input(
    backend: str,
) -> None:
    """A backend that packs uniform requests refuses a ragged one, by name."""
    batch, seq_len, heads, heads_kv, dim = 2, 64, 8, 2, 64
    # backend='fp8' is reached by handing it FP8 tensors, not by telling the op
    # its inputs are FP8: the element type is what makes the request one.
    element_type = torch.float8_e4m3fn if backend == "fp8" else torch.float16
    q = torch.zeros(batch * seq_len, heads, dim, device="cuda", dtype=element_type)
    k = torch.zeros(batch * seq_len, heads_kv, dim, device="cuda", dtype=element_type)
    v = torch.zeros_like(k)
    cu_q = torch.tensor([0, seq_len // 2, batch * seq_len], device="cuda", dtype=torch.int32)
    cu_kv = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * seq_len
    q_scale, k_scale, v_scale = _ones_prefill_scales(batch, heads_kv, device=q.device)
    op = GroupedQueryAttentionPrefillFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=seq_len,
        max_seqlen_kv=seq_len,
        is_causal=backend != "fp8",
        dtype=torch.float16,
        backend=backend,
    )

    # The caller asked for something the request is not, which is a better
    # answer than the list of implementations that declined it. Each backend is
    # refused in its own words, so the FP8 case cannot pass by landing on the
    # dense message.
    expected = (
        "Packed FP8 prefill moved to GroupedQueryAttentionPrefillDenseFwdOp"
        if backend == "fp8"
        else "backend='dense' moved to GroupedQueryAttentionPrefillDenseFwdOp"
    )
    with pytest.raises(ValueError, match=re.escape(expected)):
        op(q, k, v, cu_q, cu_kv, q_scale, k_scale, v_scale)


@pytest.mark.smoke
def test_gqa_prefill_fwd_auto_backend_serves_uniform_input_with_varlen() -> None:
    batch, seq_len, heads, heads_kv, dim = 2, 64, 8, 2, 64
    q = torch.randn(batch * seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch * seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    cu_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * seq_len
    cu_kv = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * seq_len
    q_scale, k_scale, v_scale = _ones_prefill_scales(batch, heads_kv, device=q.device)
    op = GroupedQueryAttentionPrefillFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=seq_len,
        max_seqlen_kv=seq_len,
        is_causal=True,
        dtype=torch.float16,
        backend="auto",
        kernel_map=_stand_in_prefill_map(),
    )

    out = op(q, k, v, cu_q, cu_kv, q_scale, k_scale, v_scale)

    assert out.shape == q.shape
    assert list(op.built_kernels("gqa_prefill_varlen_fwd_kernel")) == [torch.float16]


@pytest.mark.smoke
def test_gqa_prefill_fwd_respects_sm_scale() -> None:
    batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 128, 256, 8, 2, 64
    sm_scale = 0.125
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(
        batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16
    ).contiguous()
    v = torch.randn(
        batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16
    ).contiguous()
    ref = _gqa_prefill_ref(
        q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True, sm_scale=sm_scale
    )

    packed_inputs = uniform_packed_prefill_inputs(q, k, v)
    op = GroupedQueryAttentionPrefillFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=seq_len_q,
        max_seqlen_kv=seq_len_kv,
        is_causal=True,
        dtype=torch.float16,
        sm_scale=sm_scale,
    )
    output = op(*packed_inputs).view(batch, seq_len_q, heads, dim)

    torch.testing.assert_close(output, ref, atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_gqa_prefill_fwd_respects_softcap() -> None:
    batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 128, 256, 8, 2, 64
    softcap = 2.0
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(
        batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16
    ).contiguous()
    v = torch.randn(
        batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16
    ).contiguous()
    ref = _gqa_prefill_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True, softcap=softcap)

    packed_inputs = uniform_packed_prefill_inputs(q, k, v)
    op = GroupedQueryAttentionPrefillFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=seq_len_q,
        max_seqlen_kv=seq_len_kv,
        is_causal=True,
        dtype=torch.float16,
        softcap=softcap,
    )
    output = op(*packed_inputs).view(batch, seq_len_q, heads, dim)

    torch.testing.assert_close(output, ref, atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype, sm_scale, softcap, atol, rtol",
    [
        pytest.param(torch.bfloat16, None, None, 8e-2, 1e-2, id="bf16-default-scale"),
        pytest.param(torch.float16, 0.125, None, 5e-3, 1e-5, id="fp16-custom-scale"),
        pytest.param(torch.float16, None, 2.0, 5e-3, 1e-5, id="fp16-softcap"),
    ],
)
def test_gqa_prefill_fwd_uniform_varlen_matches_reference(
    dtype: torch.dtype,
    sm_scale: Optional[float],
    softcap: Optional[float],
    atol: float,
    rtol: float,
) -> None:
    batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 128, 256, 8, 2, 128
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=dtype).contiguous()
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    v = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    ref = _gqa_prefill_ref(
        q,
        k,
        v,
        heads=heads,
        heads_kv=heads_kv,
        is_causal=True,
        sm_scale=sm_scale,
        softcap=softcap,
    )

    packed_inputs = uniform_packed_prefill_inputs(q, k, v)
    op = GroupedQueryAttentionPrefillFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=seq_len_q,
        max_seqlen_kv=seq_len_kv,
        is_causal=True,
        dtype=dtype,
        sm_scale=sm_scale,
        softcap=softcap,
    )
    assert _selected_prefill_kernel_cls(op) is GQAPrefillVarlenFwdKernel
    output = op(*packed_inputs).view(batch, seq_len_q, heads, dim)

    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


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

    q_scale, k_scale, v_scale = _ones_prefill_scales(batch, heads_kv, device=q.device)
    op = GroupedQueryAttentionPrefillFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        is_causal=causal,
        dtype=dtype,
    )
    output = op(q, k, v, cu_q, cu_kv, q_scale, k_scale, v_scale)

    atol, rtol = _PREFILL_TOLERANCE[dtype]
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


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

    q_scale, k_scale, v_scale = _ones_prefill_scales(batch, heads_kv, device=q.device)
    op = GroupedQueryAttentionPrefillFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        is_causal=True,
        dtype=torch.float16,
        sm_scale=sm_scale,
    )
    output = op(q, k, v, cu_q, cu_kv, q_scale, k_scale, v_scale)

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

    q_scale, k_scale, v_scale = _ones_prefill_scales(batch, heads_kv, device=q.device)
    op = GroupedQueryAttentionPrefillFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        is_causal=True,
        dtype=torch.float16,
        softcap=softcap,
    )
    output = op(q, k, v, cu_q, cu_kv, q_scale, k_scale, v_scale)

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
    """Shape validation now happens in forward(), not constructor.
    This test verifies that invalid shapes are caught during forward."""
    op = GroupedQueryAttentionPrefillDenseFwdOp(is_causal=True)

    # Test with zero heads_kv dimension
    q = torch.randn(1, 64, 8, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 64, 0, 64, device="cuda", dtype=torch.float16)  # heads_kv = 0
    v = torch.randn_like(k)
    with pytest.raises((ValueError, RuntimeError), match="heads_kv|divisible|positive"):
        op(q, k, v)

    # Test with zero batch dimension
    q = torch.randn(0, 64, 8, 64, device="cuda", dtype=torch.float16)  # batch = 0
    k = torch.randn(0, 64, 2, 64, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    with pytest.raises((ValueError, RuntimeError), match="batch|positive"):
        op(q, k, v)

    # Test with zero seq_len dimension
    q = torch.randn(1, 0, 8, 64, device="cuda", dtype=torch.float16)  # seq_len = 0
    k = torch.randn(1, 0, 2, 64, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    with pytest.raises((ValueError, RuntimeError), match="seq_len|positive"):
        op(q, k, v)


@pytest.mark.smoke
def test_dense_prefill_path_rejects_unsupported_dtype() -> None:
    """Every region declines rather than raising, so an unguarded element type
    would land on the general dense implementation. Both entry points into the
    dense-prefill build must refuse it instead."""
    with pytest.raises(ValueError, match="float16 or torch.bfloat16"):
        GroupedQueryAttentionPrefillDenseFwdOp(is_causal=True)._get_kernel(
            torch.float32, device=torch.device("cuda")
        )
    with pytest.raises(ValueError, match="float16 or torch.bfloat16"):
        GroupedQueryAttentionPrefillFwdOp(
            batch=1,
            heads=8,
            heads_kv=2,
            dim=128,
            max_seqlen_q=128,
            max_seqlen_kv=128,
            is_causal=True,
            dtype=torch.float32,
            backend="dense",
        )


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
        is_causal=True,
        kernel_map=_stand_in_dense_prefill_map(),
    )

    assert op(q, k, v).shape == q.shape
    assert op(q, k, v).shape == q.shape

    entries = op.built_kernels("gqa_prefill_dense")
    assert len(entries) == 1
    assert isinstance(next(iter(entries.values())).kernel, GQAPrefillDenseFwdKernel)
    assert not hasattr(op, "_cu_seqlens")
    assert _op_valued_attrs(op) == []


@pytest.mark.smoke
def test_dense_and_packed_prefill_have_distinct_runtime_abis() -> None:
    """Fixed-shape kernels are BSHD-native; only ragged kernels consume ranges."""
    assert all(
        issubclass(kernel_cls, DensePrefillKernel)
        for kernel_cls in _SHIPPED_DENSE_PREFILL_MAP.values()
    )
    dense_abi = {
        "q",
        "k",
        "v",
        "q_scale",
        "k_scale",
        "v_scale",
        "rope_cos",
        "rope_sin",
    }
    for kernel_cls in _SHIPPED_DENSE_PREFILL_MAP.values():
        assert dense_abi <= set(inspect.signature(kernel_cls.forward).parameters)
    assert all(
        issubclass(kernel_cls, PackedPrefillKernel) for kernel_cls in _SHIPPED_PREFILL_MAP.values()
    )


@pytest.mark.smoke
def test_gqa_prefill_dense_build_threads_q_and_kv_lengths_apart() -> None:
    """A non-square geometry reaches the kernel with q and kv lengths unswapped."""
    batch, heads, heads_kv, dim = 1, 8, 2, 128
    max_seqlen_q, max_seqlen_kv = 128, 256
    dense = GroupedQueryAttentionPrefillDenseFwdOp(
        is_causal=True,
        kernel_map=_stand_in_dense_prefill_map(),
    )

    # 需要先调用 forward 来设置 shape
    q = torch.randn(batch, max_seqlen_q, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, max_seqlen_kv, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    dense(q, k, v)

    kernel = dense._get_kernel(torch.float16, device=torch.device("cuda"))

    assert kernel.kwargs["max_seqlen_q"] == max_seqlen_q
    assert kernel.kwargs["max_seqlen_kv"] == max_seqlen_kv


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
