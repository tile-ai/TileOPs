
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.attention import (
    GQAFwdKernel,
    GQAFwdWgmmaPipelinedKernel,
    GQAFwdWsPersistentCausalKernel,
    GQAFwdWsPersistentKernel,
)
from tileops.ops import (
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionFwdOp,
    GroupedQueryAttentionPrefillFwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
)
from tileops.ops.attention.gqa import (
    _select_gqa_fwd_kernel_cls,
    _select_gqa_paged_prefill_kernel_keys,
    _select_gqa_prefill_fwd_kernel_cls,
    _select_gqa_prefill_kernel_key,
)
from workloads.attention.gqa import (
    GroupedQueryAttentionBwdTest as _GroupedQueryAttentionBwdTestWorkload,
)
from workloads.attention.gqa import (
    GroupedQueryAttentionFwdTest as _GroupedQueryAttentionFwdTestWorkload,
)

_PREFILL_TOLERANCE = {
    torch.float16: (5e-3, 1e-5),
    torch.bfloat16: (8e-2, 1e-2),
}


class GroupedQueryAttentionBwdTest(_GroupedQueryAttentionBwdTestWorkload, TestBase):
    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                    grad_output: torch.Tensor,
                    lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True)
        output = output_bhsd.transpose(1, 2).contiguous()

        output.backward(grad_output)
        return q.grad, k.grad, v.grad


class GroupedQueryAttentionFwdTest(_GroupedQueryAttentionFwdTestWorkload, TestBase):
    def ref_program(self, q: torch.Tensor, k: torch.Tensor,
                    v: torch.Tensor) -> torch.Tensor:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True)
        return output_bhsd.transpose(1, 2).contiguous()


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


def _uniform_packed_prefill_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    batch, seq_len_q, _, _ = q.shape
    _, seq_len_kv, heads_kv, _ = k.shape
    cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * seq_len_q
    cu_kv = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * seq_len_kv
    q_scale = torch.ones((batch, heads_kv), device=q.device, dtype=torch.float32)
    k_scale = torch.ones_like(q_scale)
    v_scale = torch.ones_like(q_scale)
    return (
        q.reshape(batch * seq_len_q, q.shape[2], q.shape[3]).contiguous(),
        k.reshape(batch * seq_len_kv, heads_kv, k.shape[3]).contiguous(),
        v.reshape(batch * seq_len_kv, heads_kv, v.shape[3]).contiguous(),
        cu_q,
        cu_kv,
        q_scale,
        k_scale,
        v_scale,
    )


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
        ("batch, seq_len, heads, heads_kv, dim, causal, dtype, tune", [
            pytest.param(1, 1024, 8, 4, 64, False, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 1024, 8, 4, 64, False, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(4, 512, 64, 4, 128, False, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(4, 512, 64, 4, 128, True, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(4, 2048, 64, 4, 128, False, torch.float16, False, marks=pytest.mark.full),
            pytest.param(4, 2048, 64, 4, 128, False, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


class GroupedQueryAttentionBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, heads_kv, dim, causal, dtype, tune", [
            pytest.param(1, 1024, 8, 4, 64, False, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 1024, 8, 4, 64, False, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(4, 2048, 64, 4, 128, False, torch.float16, False, marks=pytest.mark.full),
            pytest.param(4, 2048, 64, 4, 128, False, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@GroupedQueryAttentionFwdFixture
def test_gqa_fwd(batch: int, seq_len: int, heads: int, heads_kv: int, dim: int, causal: bool,
                 dtype: torch.dtype, tune: bool) -> None:
    test = GroupedQueryAttentionFwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    op = GroupedQueryAttentionFwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


@pytest.mark.parametrize("batch, seq_len_q, seq_len_kv, heads, heads_kv, dim, causal, dtype", [
    pytest.param(1, 128, 128, 8, 2, 64, True, torch.float16, marks=pytest.mark.smoke,
                 id="gqa_ratio4_q_eq_kv_fp16"),
    pytest.param(1, 128, 384, 8, 2, 64, True, torch.float16, marks=pytest.mark.smoke,
                 id="gqa_ratio4_q_lt_kv_fp16"),
    pytest.param(1, 96, 150, 8, 2, 64, True, torch.float16, marks=pytest.mark.smoke,
                 id="gqa_tail_q_and_kv_fp16"),
    pytest.param(2, 128, 256, 16, 4, 128, True, torch.float16, marks=pytest.mark.smoke,
                 id="gqa_ratio4_batch2_dim128_fp16"),
    pytest.param(1, 128, 256, 8, 2, 64, False, torch.float16, marks=pytest.mark.smoke,
                 id="gqa_noncausal_fp16"),
    pytest.param(1, 128, 256, 8, 2, 64, True, torch.bfloat16, marks=pytest.mark.smoke,
                 id="gqa_ratio4_bf16"),
])
def test_gqa_prefill_fwd(batch: int, seq_len_q: int, seq_len_kv: int, heads: int,
                         heads_kv: int, dim: int, causal: bool, dtype: torch.dtype) -> None:
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=dtype).contiguous()
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    v = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    ref = _gqa_prefill_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=causal)

    packed_inputs = _uniform_packed_prefill_inputs(q, k, v)
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
def test_gqa_prefill_fwd_uses_bottom_right_causal_mask() -> None:
    batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 128, 256, 4, 2, 64
    q = torch.zeros(batch, seq_len_q, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.zeros(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.zeros(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16)
    q[..., 0] = 1
    k[..., 0] = 1
    v[:, :128, :, 0] = 1
    v[:, 128:, :, 0] = 100

    packed_inputs = _uniform_packed_prefill_inputs(q.contiguous(), k.contiguous(),
                                                   v.contiguous())
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


@pytest.mark.smoke
def test_gqa_prefill_fwd_respects_sm_scale() -> None:
    batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 128, 256, 8, 2, 64
    sm_scale = 0.125
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda",
                    dtype=torch.float16).contiguous()
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda",
                    dtype=torch.float16).contiguous()
    v = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda",
                    dtype=torch.float16).contiguous()
    ref = _gqa_prefill_ref(
        q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True, sm_scale=sm_scale)

    packed_inputs = _uniform_packed_prefill_inputs(q, k, v)
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
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda",
                    dtype=torch.float16).contiguous()
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda",
                    dtype=torch.float16).contiguous()
    v = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda",
                    dtype=torch.float16).contiguous()
    ref = _gqa_prefill_ref(
        q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True, softcap=softcap)

    packed_inputs = _uniform_packed_prefill_inputs(q, k, v)
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
@pytest.mark.parametrize("dtype, sm_scale, softcap, atol, rtol", [
    pytest.param(torch.bfloat16, None, None, 8e-2, 1e-2, id="bf16-default-scale"),
    pytest.param(torch.float16, 0.125, None, 5e-3, 1e-5, id="fp16-custom-scale"),
    pytest.param(torch.float16, None, 2.0, 5e-3, 1e-5, id="fp16-softcap"),
])
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

    packed_inputs = _uniform_packed_prefill_inputs(q, k, v)
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
    assert op._get_dense_kernel().__class__.__name__ == "GQAPrefillFwdWsPersistentCausalKernel"
    output = op(*packed_inputs).view(batch, seq_len_q, heads, dim)

    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("q_lens, kv_lens, heads, heads_kv, dim, causal, dtype", [
    pytest.param([64, 128], [64, 128], 8, 2, 64, True, torch.float16,
                 marks=pytest.mark.smoke, id="uniform-causal-fp16"),
    pytest.param([33, 96, 129], [64, 128, 256], 8, 2, 64, True, torch.float16,
                 marks=pytest.mark.smoke, id="mixed-tail-causal-fp16"),
    pytest.param([64, 96], [128, 160], 8, 2, 64, False, torch.float16,
                 marks=pytest.mark.smoke, id="mixed-noncausal-fp16"),
    pytest.param([33, 65], [33, 65], 8, 2, 64, False, torch.float16,
                 marks=pytest.mark.smoke, id="equal-tail-noncausal-fp16"),
    pytest.param([64, 96], [128, 160], 8, 1, 64, True, torch.float16,
                 marks=pytest.mark.smoke, id="mqa-causal-fp16"),
    pytest.param([96], [160], 8, 2, 64, True, torch.float16,
                 marks=pytest.mark.smoke, id="batch1-causal-fp16"),
    pytest.param([64, 128], [128, 256], 8, 2, 64, True, torch.bfloat16,
                 marks=pytest.mark.smoke, id="mixed-causal-bf16"),
])
def test_gqa_prefill_varlen_fwd(q_lens: list[int], kv_lens: list[int], heads: int,
                                heads_kv: int, dim: int, causal: bool,
                                dtype: torch.dtype) -> None:
    batch = len(q_lens)
    total_q = sum(q_lens)
    total_kv = sum(kv_lens)
    q = torch.randn(total_q, heads, dim, device="cuda", dtype=dtype).contiguous()
    k = torch.randn(total_kv, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    v = torch.randn(total_kv, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    ref = _gqa_prefill_varlen_ref(
        q, k, v, cu_q, cu_kv, batch=batch, heads=heads, heads_kv=heads_kv,
        is_causal=causal)

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
    k = torch.randn(sum(kv_lens), heads_kv, dim, device="cuda",
                    dtype=torch.float16).contiguous()
    v = torch.randn_like(k)
    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    ref = _gqa_prefill_varlen_ref(
        q, k, v, cu_q, cu_kv, batch=batch, heads=heads, heads_kv=heads_kv,
        is_causal=True, sm_scale=sm_scale)

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
    k = torch.randn(sum(kv_lens), heads_kv, dim, device="cuda",
                    dtype=torch.float16).contiguous()
    v = torch.randn_like(k)
    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    ref = _gqa_prefill_varlen_ref(
        q, k, v, cu_q, cu_kv, batch=batch, heads=heads, heads_kv=heads_kv,
        is_causal=True, softcap=softcap)

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
    k = torch.randn(sum(kv_lens), heads_kv, dim, device="cuda",
                    dtype=torch.float16).contiguous()
    v = torch.randn_like(k)
    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32)

    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        batch, heads, heads_kv, dim, max(q_lens), max(kv_lens), True, torch.float16,
        validate_inputs=True)
    with pytest.raises(ValueError, match="Expected k shape"):
        op(q, k[:, :, :-1].contiguous(), v, cu_q, cu_kv)
    with pytest.raises(ValueError, match="cu_seqlens_q\\[-1\\].*must equal"):
        op(q[:-1], k, v, cu_q, cu_kv)
    with pytest.raises(ValueError, match="max_seqlen_q"):
        bad_op = GroupedQueryAttentionPrefillVarlenFwdOp(
            batch, heads, heads_kv, dim, max(q_lens) - 1, max(kv_lens), True,
            torch.float16, validate_inputs=True)
        bad_op(q, k, v, cu_q, cu_kv)
    bad_cu = torch.tensor([0, 128, 96], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="cu_seqlens_q must be non-decreasing"):
        op(q, k, v, bad_cu, cu_kv)


@pytest.mark.smoke
def test_gqa_prefill_varlen_rejects_unsupported_dtype() -> None:
    with pytest.raises(ValueError, match="Expected dtype torch.float16 or torch.bfloat16"):
        GroupedQueryAttentionPrefillVarlenFwdOp(
            batch=1,
            heads=8,
            heads_kv=2,
            dim=64,
            max_seqlen_q=64,
            max_seqlen_kv=128,
            dtype=torch.float32,
        )



@GroupedQueryAttentionBwdFixture
def test_gqa_bwd(batch: int, seq_len: int, heads: int, heads_kv: int, dim: int, causal: bool,
                 dtype: torch.dtype, tune: bool) -> None:
    test = GroupedQueryAttentionBwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    op = GroupedQueryAttentionBwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_gqa_fwd_dispatch_selects_ws_noncausal_on_h200() -> None:
    kernel_cls = _select_gqa_fwd_kernel_cls(
        4, 64, 4, 512, 128, False, torch.float16, hopper=True, h200=True)
    assert kernel_cls is GQAFwdWsPersistentKernel


@pytest.mark.smoke
def test_gqa_fwd_dispatch_selects_ws_causal_on_h200() -> None:
    kernel_cls = _select_gqa_fwd_kernel_cls(
        4, 64, 4, 512, 128, True, torch.float16, hopper=True, h200=True)
    assert kernel_cls is GQAFwdWsPersistentCausalKernel


@pytest.mark.smoke
def test_gqa_fwd_dispatch_falls_back_for_small_causal_shape() -> None:
    kernel_cls = _select_gqa_fwd_kernel_cls(
        1, 32, 8, 1024, 128, True, torch.float16, hopper=True, h200=True)
    assert kernel_cls is GQAFwdWgmmaPipelinedKernel


@pytest.mark.smoke
def test_gqa_fwd_dispatch_falls_back_off_h200() -> None:
    hopper_cls = _select_gqa_fwd_kernel_cls(
        4, 64, 4, 512, 128, False, torch.float16, hopper=True, h200=False)
    assert hopper_cls is GQAFwdWgmmaPipelinedKernel

    non_hopper_cls = _select_gqa_fwd_kernel_cls(
        4, 64, 4, 512, 128, False, torch.float16, hopper=False, h200=False)
    assert non_hopper_cls is GQAFwdKernel


@pytest.mark.smoke
@pytest.mark.parametrize("backend,is_fp8,uses_window,is_uniform,expected", [
    ("auto", False, False, True, "gqa_prefill_fwd_kernel"),
    ("auto", False, False, False, "gqa_prefill_varlen_fwd_kernel"),
    ("varlen", False, False, True, "gqa_prefill_varlen_fwd_kernel"),
    ("auto", False, True, True, "gqa_sliding_window_varlen_fwd"),
    ("auto", True, False, True, "gqa_prefill_fp8_tensor_core_fwd_kernel"),
])
def test_gqa_prefill_dispatch_key_selector(
    backend: str,
    is_fp8: bool,
    uses_window: bool,
    is_uniform: bool,
    expected: str,
) -> None:
    assert _select_gqa_prefill_kernel_key(
        backend=backend,
        is_fp8=is_fp8,
        uses_sliding_window=uses_window,
        is_uniform=is_uniform,
    ) == expected


@pytest.mark.smoke
def test_gqa_prefill_dispatch_key_selector_rejects_forced_dense_ragged() -> None:
    with pytest.raises(ValueError, match="backend='dense' requires uniform"):
        _select_gqa_prefill_kernel_key(
            backend="dense",
            is_fp8=False,
            uses_sliding_window=False,
            is_uniform=False,
        )


@pytest.mark.smoke
def test_gqa_prefill_dense_selector_widens_ws_capability() -> None:
    assert _select_gqa_prefill_fwd_kernel_cls(
        128,
        True,
        torch.float16,
        sm_scale=0.25,
        softcap=2.0,
        hopper=True,
    ).__name__ == "GQAPrefillFwdWsPersistentCausalKernel"
    assert _select_gqa_prefill_fwd_kernel_cls(
        128,
        True,
        torch.bfloat16,
        sm_scale=128**-0.5,
        softcap=0.0,
        hopper=True,
    ).__name__ == "GQAPrefillFwdWsPersistentCausalKernel"


@pytest.mark.smoke
def test_gqa_paged_prefill_selector_keys() -> None:
    assert _select_gqa_paged_prefill_kernel_keys(
        cache_dtype=torch.float16,
        attention_dtype=torch.float16,
        fuse_rope=False,
    ) == ("gqa_prefill_paged_with_kv_cache_fwd_kernel",)
    assert _select_gqa_paged_prefill_kernel_keys(
        cache_dtype=torch.float16,
        attention_dtype=torch.float16,
        fuse_rope=True,
    ) == (
        "gqa_prefill_paged_with_kv_cache_rope_append_kernel",
        "gqa_prefill_paged_with_kv_cache_rope_fwd_kernel",
    )
    if hasattr(torch, "float8_e4m3fn"):
        assert _select_gqa_paged_prefill_kernel_keys(
            cache_dtype=torch.float8_e4m3fn,
            attention_dtype=torch.float16,
            fuse_rope=False,
        ) == ("gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel",)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
