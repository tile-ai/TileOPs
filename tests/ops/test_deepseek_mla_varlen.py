"""Packed-varlen MLA prefill tests against a float32 per-request reference.

Input construction and the reference belong in ``workloads/`` once an entry has
workload rows, so that a benchmark reads the same definition. This entry is
still ``spec-only`` and has neither, so they are module-level helpers here:
there is no second consumer to drift from yet, and promoting the entry moves
them to ``workloads/deepseek_attention.py`` along with the benchmark that will
read them. The reference itself is the semantics
``tests/test_spec_reference.py`` already states for the entry -- expand ``k_pe``
to every head, then attend per request in float32.
"""

import pytest
import torch

from tests.test_base import FixtureBase
from tileops.kernels.attention import (
    MLAVarlenPrefillFwdKernel,
    MLAVarlenPrefillWSFwdKernel,
)
from tileops.ops import MultiHeadLatentAttentionVarlenFwdOp
from workloads.device import run_device

_REF_HEADS = 8
_REF_ROWS = 1024


def _gen_inputs(seq_lens, heads, dim_nope, dim_pe, dim_v, dtype):
    """The packed tensors one call takes, for ``seq_lens`` requests."""
    total = sum(seq_lens)
    device = run_device()
    return (
        torch.randn(total, heads, dim_nope + dim_pe, dtype=dtype, device=device),
        torch.randn(total, heads, dim_nope, dtype=dtype, device=device),
        torch.randn(total, dim_pe, dtype=dtype, device=device),
        torch.randn(total, heads, dim_v, dtype=dtype, device=device),
        torch.tensor(
            [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        ),
    )


def _ref_program(q, k_nope, k_pe, v, cu_seqlens, *, is_causal, dim_v, sm_scale=None):
    """Expand the shared rope half to every head, then attend per request in float32.

    The score block of a whole request is quadratic in its length, so query rows
    and heads are taken a slab at a time; each row still sees its whole key
    axis, so the numbers are those of the unsplit form.
    """
    heads = q.shape[1]
    scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
    bounds = cu_seqlens.tolist()
    out = torch.empty_like(q[..., :dim_v])
    lse = torch.empty(q.shape[0], heads, dtype=torch.float32, device=q.device)
    for start, end in zip(bounds, bounds[1:], strict=False):
        span = end - start
        key = torch.cat(
            [k_nope[start:end], k_pe[start:end, None].expand(-1, heads, -1)], dim=-1
        ).float()
        value = v[start:end].float()
        rows = torch.arange(span, device=q.device)
        for h0 in range(0, heads, _REF_HEADS):
            h1 = min(h0 + _REF_HEADS, heads)
            for r0 in range(0, span, _REF_ROWS):
                r1 = min(r0 + _REF_ROWS, span)
                scores = (
                    torch.einsum(
                        "shd,nhd->hsn",
                        q[start + r0 : start + r1, h0:h1].float(),
                        key[:, h0:h1],
                    )
                    * scale
                )
                if is_causal:
                    visible = rows[None, :] <= rows[r0:r1, None]
                    scores = scores.masked_fill(~visible[None], float("-inf"))
                out[start + r0 : start + r1, h0:h1] = torch.einsum(
                    "hsn,nhd->shd", scores.softmax(-1), value[:, h0:h1]
                ).to(q.dtype)
                lse[start + r0 : start + r1, h0:h1] = scores.logsumexp(-1).T
    return out, lse


class MlaVarlenFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest

        return [
            (
                "seq_lens, heads, dim_nope, dim_pe, dim_v, is_causal, sm_scale, dtype",
                [
                    pytest.param(
                        [256, 384],
                        4,
                        128,
                        64,
                        128,
                        True,
                        None,
                        torch.bfloat16,
                        marks=pytest.mark.smoke,
                    ),
                    pytest.param(
                        [512, 512],
                        8,
                        128,
                        64,
                        128,
                        True,
                        None,
                        torch.float16,
                        marks=pytest.mark.smoke,
                    ),
                    # request lengths that are not multiples of any tile
                    pytest.param(
                        [200, 333, 91],
                        4,
                        128,
                        64,
                        128,
                        True,
                        None,
                        torch.bfloat16,
                        marks=pytest.mark.full,
                    ),
                    # one request shorter than a single query tile
                    pytest.param(
                        [17, 1024],
                        4,
                        128,
                        64,
                        128,
                        True,
                        None,
                        torch.bfloat16,
                        marks=pytest.mark.full,
                    ),
                    pytest.param(
                        [256, 256],
                        4,
                        128,
                        64,
                        128,
                        False,
                        None,
                        torch.bfloat16,
                        marks=pytest.mark.full,
                    ),
                    pytest.param(
                        [256, 256],
                        4,
                        128,
                        64,
                        128,
                        True,
                        0.05,
                        torch.bfloat16,
                        marks=pytest.mark.full,
                    ),
                    pytest.param(
                        [1024],
                        16,
                        128,
                        64,
                        128,
                        True,
                        None,
                        torch.bfloat16,
                        marks=pytest.mark.full,
                    ),
                ],
            ),
        ]


@MlaVarlenFwdFixture
def test_mla_varlen_fwd_op(
    seq_lens, heads, dim_nope, dim_pe, dim_v, is_causal, sm_scale, dtype
) -> None:
    inputs = _gen_inputs(seq_lens, heads, dim_nope, dim_pe, dim_v, dtype)
    op = MultiHeadLatentAttentionVarlenFwdOp(is_causal=is_causal, sm_scale=sm_scale)
    out, lse = op(*inputs)
    ref_out, ref_lse = _ref_program(*inputs, is_causal=is_causal, dim_v=dim_v, sm_scale=sm_scale)
    tolerance = 2e-2 if dtype is torch.bfloat16 else 4e-3
    torch.testing.assert_close(out.float(), ref_out.float(), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(lse, ref_lse, atol=2e-3, rtol=2e-3)


@pytest.mark.smoke
def test_mla_varlen_lse_merges_a_split_context() -> None:
    """``lse`` is the natural-base log-sum-exp its own outputs were divided by.

    A caller merging chunked-context partials weights each chunk by
    ``exp(lse_chunk - lse_all)``; that identity is what makes the returned
    ``lse`` usable, and it holds only if the scale is the one the manifest
    states.
    """
    torch.manual_seed(0)
    q, k_nope, k_pe, v, cu_seqlens = _gen_inputs([512], 4, 128, 64, 128, torch.float16)
    op = MultiHeadLatentAttentionVarlenFwdOp(is_causal=True)
    out, lse = op(q, k_nope, k_pe, v, cu_seqlens)

    heads = q.shape[1]
    scale = q.shape[-1] ** -0.5
    key = torch.cat([k_nope, k_pe[:, None].expand(-1, heads, -1)], dim=-1)
    scores = torch.einsum("shd,nhd->hsn", q.float(), key.float()) * scale
    rows = torch.arange(512, device=scores.device)[:, None]
    cols = torch.arange(512, device=scores.device)[None, :]
    scores = scores.masked_fill(~(cols <= rows)[None], float("-inf"))

    torch.testing.assert_close(lse, scores.logsumexp(-1).T, atol=2e-3, rtol=2e-3)
    assert out.dtype == q.dtype
    assert lse.dtype == torch.float32


@pytest.mark.parametrize(
    "kernel_cls",
    [
        pytest.param(MLAVarlenPrefillFwdKernel, id="general"),
        pytest.param(MLAVarlenPrefillWSFwdKernel, id="ws", marks=[pytest.mark.sm90]),
    ],
)
@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([256, 384], id="ragged", marks=pytest.mark.smoke),
        pytest.param([17, 1024], id="below-one-tile", marks=pytest.mark.full),
    ],
)
def test_each_implementation_matches_the_reference(kernel_cls, seq_lens) -> None:
    """Both implementations answer the same thing.

    The op dispatches to whichever one the device admits, so a test that only
    calls the op leaves the other unexercised on any given machine.
    """
    inputs = _gen_inputs(seq_lens, 4, 128, 64, 128, torch.bfloat16)
    kernel = kernel_cls(
        batch=len(seq_lens),
        heads=4,
        dim_nope=128,
        dim_pe=64,
        dim_v=128,
        is_causal=True,
        dtype=torch.bfloat16,
    )
    out, lse = kernel.forward(*inputs)
    ref_out, ref_lse = _ref_program(*inputs, is_causal=True, dim_v=128)
    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=2e-3, rtol=2e-3)
