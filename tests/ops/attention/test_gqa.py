import re
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.attention import (
    GQAFwdWsPersistentCausalKernel,
    GQAPrefillFwdWsPersistentCausalKernel,
)
from tileops.ops import (
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionPrefillFwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
)
from tileops.ops.attention.selection import PACKED_PREFILL_KEYS
from tileops.utils import is_h200
from workloads.attention.gqa import (
    GroupedQueryAttentionBwdWorkload,
    uniform_packed_prefill_inputs,
)

_PREFILL_TOLERANCE = {
    torch.float16: (5e-3, 1e-5),
    torch.bfloat16: (8e-2, 1e-2),
}


def _selected_prefill_kernel_cls(op: GroupedQueryAttentionPrefillFwdOp) -> type:
    """Kernel class selection picks for a uniform, non-FP8 packed prefill call."""
    call = op.attention_call(is_fp8=False, is_uniform=True)
    return op.kernel_map[op.select_kernel_key(PACKED_PREFILL_KEYS, call)]


#: The shipped implementations of the packed-prefill slot, by dispatch key.
_SHIPPED_PREFILL_MAP = GroupedQueryAttentionPrefillFwdOp.default_kernel_map.fget(
    object.__new__(GroupedQueryAttentionPrefillFwdOp)
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
    """A ``kernel_map=`` replacing every packed-prefill key with a stand-in."""
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
    if is_causal:
        offset = seq_len_kv - seq_len_q
        q_pos = torch.arange(seq_len_q, device=q.device)[:, None] + offset
        k_pos = torch.arange(seq_len_kv, device=q.device)[None, :]
        mask = k_pos <= q_pos
        scores = scores.masked_fill(~mask.view(1, 1, seq_len_q, seq_len_kv), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
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
def test_gqa_prefill_fwd_dense_backend_matches_reference() -> None:
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
        backend="dense",
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
def test_gqa_prefill_fwd_square_uses_square_fast_path(dtype: torch.dtype) -> None:
    if not is_h200():
        pytest.skip("square fast path requires H200")

    op = GroupedQueryAttentionPrefillFwdOp(
        batch=4,
        heads=32,
        heads_kv=8,
        dim=128,
        max_seqlen_q=512,
        max_seqlen_kv=512,
        is_causal=True,
        dtype=dtype,
        backend="dense",
    )

    assert _selected_prefill_kernel_cls(op) is GQAFwdWsPersistentCausalKernel


@pytest.mark.parametrize(
    "sm_scale, softcap",
    [
        pytest.param(0.125, None, id="custom-scale"),
        pytest.param(None, 2.0, id="softcap"),
    ],
)
@pytest.mark.smoke
def test_gqa_prefill_fwd_square_feature_variants_use_square_fast_path(
    sm_scale: Optional[float],
    softcap: Optional[float],
) -> None:
    if not is_h200():
        pytest.skip("square fast path requires H200")

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
        backend="dense",
    )

    assert _selected_prefill_kernel_cls(op) is GQAFwdWsPersistentCausalKernel


@pytest.mark.parametrize(
    "seq_len_q, seq_len_kv, sm_scale, softcap",
    [
        pytest.param(512, 4096, None, None, id="q-lt-kv"),
    ],
)
@pytest.mark.smoke
def test_gqa_prefill_fwd_q_lt_kv_uses_prefill_ws_kernel(
    seq_len_q: int,
    seq_len_kv: int,
    sm_scale: Optional[float],
    softcap: Optional[float],
) -> None:
    if not is_h200():
        pytest.skip("warp-specialized dense prefill dispatch is validated on H200")

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
        backend="dense",
    )

    assert _selected_prefill_kernel_cls(op) is GQAPrefillFwdWsPersistentCausalKernel


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
        "FP8 prefill requires uniform packed cu_seqlens."
        if backend == "fp8"
        else "backend='dense' requires uniform packed cu_seqlens."
    )
    with pytest.raises(ValueError, match=re.escape(expected)):
        op(q, k, v, cu_q, cu_kv, q_scale, k_scale, v_scale)


@pytest.mark.smoke
def test_gqa_prefill_fwd_explicit_dense_can_skip_uniform_validation() -> None:
    """Opting out of the range check still serves a uniform dense request."""
    batch, seq_len, heads, heads_kv, dim = 2, 64, 8, 2, 64
    q = torch.randn(batch * seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch * seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * seq_len
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
        backend="dense",
        validate_uniform_cu_seqlens=False,
        kernel_map=_stand_in_prefill_map(),
    )

    out = op(q, k, v, cu, cu, q_scale, k_scale, v_scale)
    assert out.shape == q.shape


@pytest.mark.smoke
def test_gqa_prefill_fwd_auto_backend_requires_uniform_validation() -> None:
    with pytest.raises(
        ValueError, match="backend='auto' requires validate_uniform_cu_seqlens=True"
    ):
        GroupedQueryAttentionPrefillFwdOp(
            batch=2,
            heads=8,
            heads_kv=2,
            dim=64,
            max_seqlen_q=64,
            max_seqlen_kv=64,
            dtype=torch.float16,
            backend="auto",
            validate_uniform_cu_seqlens=False,
        )


@pytest.mark.smoke
def test_gqa_prefill_fwd_auto_backend_serves_uniform_input_dense() -> None:
    """``backend='auto'`` reads the ranges and lands on the dense key for uniform ones."""
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
    # Uniform ranges put an automatic request on a dense implementation, not on
    # the ragged one — the outcome the range reading exists to produce.
    assert list(op.built_kernels("gqa_prefill_fwd_kernel")) == [torch.float16]
    assert not op.built_kernels("gqa_prefill_varlen_fwd_kernel")


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
def test_gqa_prefill_fwd_ws_path_matches_reference(
    dtype: torch.dtype,
    sm_scale: Optional[float],
    softcap: Optional[float],
    atol: float,
    rtol: float,
) -> None:
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("warp-specialized prefill path requires Hopper")

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
    assert _selected_prefill_kernel_cls(op) is GQAPrefillFwdWsPersistentCausalKernel
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


@pytest.mark.smoke
def test_packed_dense_prefill_rejects_unsupported_dtype() -> None:
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
def test_gqa_prefill_dense_build_threads_q_and_kv_lengths_apart() -> None:
    """A non-square geometry reaches the kernel with q and kv lengths unswapped."""
    batch, heads, heads_kv, dim = 1, 8, 2, 128
    max_seqlen_q, max_seqlen_kv = 128, 256
    packed = GroupedQueryAttentionPrefillFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        is_causal=True,
        dtype=torch.float16,
        backend="dense",
        kernel_map=_stand_in_prefill_map(),
    )

    call = packed.attention_call(is_fp8=False, is_uniform=True)
    kernel = packed._kernel_for((), packed.select_kernel_key(PACKED_PREFILL_KEYS, call), call)

    assert kernel.kwargs["max_seqlen_q"] == max_seqlen_q
    assert kernel.kwargs["max_seqlen_kv"] == max_seqlen_kv
