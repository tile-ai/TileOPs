"""The MLA prefill backward ops against the float32 gradients of decompressed MLA attention."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.attention import MLADenseBwdOp, MLAVarlenBwdOp, MLAVarlenFwdOp
from workloads.attention.mla import (
    MLADenseBwdWorkload,
    MLAVarlenBwdWorkload,
    mla_varlen_inputs,
)
from workloads.numerics import reference_tolerance

# Both MLA backward kernels are SM90 only.
pytestmark = pytest.mark.sm90


def _check(workload, op) -> None:
    TestBase.check(workload, op, *workload.gen_inputs())


class MLADenseBwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seqlen_q, seqlen_kv, is_causal, sm_scale, dtype",
            [
                pytest.param(2, 256, 256, True, None, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(2, 256, 256, True, None, torch.float16, marks=pytest.mark.smoke),
                # A length that no power-of-two tile divides.
                pytest.param(1, 200, 200, True, None, torch.bfloat16, marks=pytest.mark.full),
                # Fewer queries than keys: the causal mask aligns to the bottom right.
                pytest.param(1, 128, 320, True, None, torch.bfloat16, marks=pytest.mark.full),
                # More queries than keys: the first rows see no key and get zero gradients.
                pytest.param(1, 320, 128, True, None, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(1, 256, 256, False, None, torch.bfloat16, marks=pytest.mark.full),
                # A caller's score scale in place of D_qk ** -0.5.
                pytest.param(1, 256, 256, True, 0.05, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@MLADenseBwdFixture
def test_mla_dense_bwd_op(batch, seqlen_q, seqlen_kv, is_causal, sm_scale, dtype) -> None:
    # DeepSeek-V3's decompressed head: D_qk 192, D_v 128.
    workload = MLADenseBwdWorkload(
        batch, seqlen_q, seqlen_kv, 4, 192, 128, dtype, is_causal, sm_scale
    )
    _check(workload, MLADenseBwdOp(is_causal=is_causal, sm_scale=sm_scale))


class MLAVarlenBwdFixture(FixtureBase):
    PARAMS = [
        (
            "q_lens, kv_lens, is_causal, sm_scale, dtype",
            [
                pytest.param(
                    [256, 384], [256, 384], True, None, torch.bfloat16, marks=pytest.mark.smoke
                ),
                pytest.param(
                    [256, 384], [256, 384], True, None, torch.float16, marks=pytest.mark.smoke
                ),
                # Request lengths that no tile divides, one shorter than any tile.
                pytest.param(
                    [200, 17, 91], [200, 17, 91], True, None, torch.bfloat16, marks=pytest.mark.full
                ),
                # Fewer queries than keys in one request: its mask aligns to the bottom right.
                pytest.param(
                    [128, 256], [320, 256], True, None, torch.bfloat16, marks=pytest.mark.full
                ),
                # More queries than keys: the first rows of that request see no key.
                pytest.param(
                    [320, 64], [128, 64], True, None, torch.bfloat16, marks=pytest.mark.full
                ),
                pytest.param(
                    [256, 384], [256, 384], False, None, torch.bfloat16, marks=pytest.mark.full
                ),
                # A caller's score scale in place of D_qk ** -0.5.
                pytest.param(
                    [256, 384], [256, 384], True, 0.05, torch.bfloat16, marks=pytest.mark.full
                ),
            ],
        ),
    ]


@MLAVarlenBwdFixture
def test_mla_varlen_bwd_op(q_lens, kv_lens, is_causal, sm_scale, dtype) -> None:
    # DeepSeek-V3's decompressed head: D_qk 192, D_v 128.
    workload = MLAVarlenBwdWorkload(q_lens, kv_lens, 4, 192, 128, dtype, is_causal, sm_scale)
    _check(workload, MLAVarlenBwdOp(is_causal=is_causal, sm_scale=sm_scale))


@pytest.mark.smoke
def test_mla_dense_bwd_general_kernel() -> None:
    """A head other than D_qk 192, D_v 128 runs the general kernel."""
    _check(
        MLADenseBwdWorkload(1, 256, 256, 4, 128, 64, torch.bfloat16), MLADenseBwdOp(is_causal=True)
    )


@pytest.mark.smoke
def test_mla_varlen_bwd_general_kernel() -> None:
    """A head other than D_qk 192, D_v 128 runs the general kernel."""
    workload = MLAVarlenBwdWorkload([256, 128], [256, 128], 4, 128, 64, torch.bfloat16)
    _check(workload, MLAVarlenBwdOp(is_causal=True))


@pytest.mark.smoke
def test_mla_dense_bwd_rejects_a_token_major_lse() -> None:
    """``lse`` is ``(B, H, S_q)``, as FA3 returns it, not the token-major layout."""
    inputs = list(MLADenseBwdWorkload(1, 64, 64, 4, 192, 128, torch.bfloat16).gen_inputs())
    inputs[5] = inputs[5].transpose(1, 2).contiguous()
    with pytest.raises(ValueError, match="lse"):
        MLADenseBwdOp(is_causal=True)(*inputs)


@pytest.mark.smoke
def test_mla_varlen_bwd_rejects_a_token_major_lse() -> None:
    """``lse`` is head-major ``(H, T_q)``, as FA3 and MLAVarlenFwdOp return it for packed
    requests, not token-major ``(T_q, H)``."""
    workload = MLAVarlenBwdWorkload([64, 32], [64, 32], 4, 192, 128, torch.bfloat16)
    inputs = list(workload.gen_inputs())
    inputs[5] = inputs[5].T.contiguous()
    with pytest.raises(ValueError, match="lse"):
        MLAVarlenBwdOp(is_causal=True)(*inputs)


def _causal_mla(q, k_nope, k_pe, v, seq_lens):
    """Float32 causal MLA prefill: each request attends to its own tokens, the shared
    ``k_pe`` beside every head's ``k_nope``."""
    scale = q.shape[-1] ** -0.5
    outputs, start = [], 0
    for length in seq_lens:
        rows = slice(start, start + length)
        key = torch.cat([k_nope[rows], k_pe[rows, None].expand(-1, q.shape[1], -1)], dim=-1)
        scores = torch.einsum("shd,nhd->hsn", q[rows], key) * scale
        visible = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
        probs = scores.masked_fill(~visible, float("-inf")).softmax(-1)
        outputs.append(torch.einsum("hsn,nhd->shd", probs, v[rows]))
        start += length
    return torch.cat(outputs)


@pytest.mark.smoke
def test_mla_varlen_fwd_feeds_the_varlen_bwd() -> None:
    """MLAVarlenFwdOp's ``o`` and head-major ``lse`` go into MLAVarlenBwdOp as they are; with
    the key expanded to every head, the gradients are torch autograd's of the forward."""
    torch.manual_seed(0)
    seq_lens, heads, dim_nope, dim_pe, dim_v = [256, 128], 4, 128, 64, 128
    dtype = torch.bfloat16
    q, k_nope, k_pe, v, cu_seqlens = mla_varlen_inputs(
        seq_lens, heads, dim_nope, dim_pe, dim_v, dtype
    )
    o, lse = MLAVarlenFwdOp(is_causal=True)(q, k_nope, k_pe, v, cu_seqlens)
    do = torch.randn_like(o)
    k = torch.cat([k_nope, k_pe[:, None].expand(-1, heads, -1)], dim=-1).contiguous()
    dq, dk, dv = MLAVarlenBwdOp(is_causal=True)(do, q, k, v, o, lse, cu_seqlens, cu_seqlens)

    leaves = [t.detach().float().requires_grad_() for t in (q, k_nope, k_pe, v)]
    _causal_mla(*leaves, seq_lens).backward(do.float())
    tolerance = reference_tolerance(dtype)
    atol, rtol = max(5e-3, tolerance["atol"]), tolerance["rtol"]
    got = (dq, dk[..., :dim_nope], dk[..., dim_nope:].float().sum(1), dv)
    for name, value, leaf in zip(("dq", "dk_nope", "dk_pe", "dv"), got, leaves, strict=True):
        # dk_pe sums a rounded gradient over the heads.
        scale = heads if name == "dk_pe" else 1
        torch.testing.assert_close(
            value.float(), leaf.grad, atol=atol * scale, rtol=rtol * scale, msg=name
        )


@pytest.mark.smoke
def test_mla_varlen_ops_check_offsets_with_validate_inputs() -> None:
    """``validate_inputs`` rejects offsets that do not span the packed tensors."""
    q, k_nope, k_pe, v, cu_seqlens = mla_varlen_inputs(
        [64, 32], 4, 128, 64, 128, dtype=torch.bfloat16
    )
    forward = MLAVarlenFwdOp(is_causal=True, validate_inputs=True)
    with pytest.raises(ValueError, match=r"cu_seqlens\[-1\] must equal"):
        forward(q[:-1], k_nope[:-1], k_pe[:-1], v[:-1], cu_seqlens)
    inputs = list(
        MLAVarlenBwdWorkload([64, 32], [64, 32], 4, 192, 128, torch.bfloat16).gen_inputs()
    )
    inputs[6] = torch.tensor([0, 100, 96], dtype=torch.int32, device=q.device)
    with pytest.raises(ValueError, match="cu_seqlens_q must be non-decreasing"):
        MLAVarlenBwdOp(is_causal=True, validate_inputs=True)(*inputs)
