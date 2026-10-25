"""The MLA prefill backward ops against the float32 gradients of decompressed MLA attention."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.attention import MLADenseBwdOp, MLAVarlenBwdOp
from tileops.backend import OpNotAvailableError
from workloads.attention.mla import MLADenseBwdWorkload, MLAVarlenBwdWorkload


def _check(workload, op) -> None:
    try:
        TestBase.check(workload, op, *workload.gen_inputs())
    except OpNotAvailableError as e:
        pytest.skip(str(e))


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
    _check(MLADenseBwdWorkload(1, 256, 256, 4, 128, 64, torch.bfloat16), MLADenseBwdOp())


@pytest.mark.smoke
def test_mla_varlen_bwd_general_kernel() -> None:
    """A head other than D_qk 192, D_v 128 runs the general kernel."""
    workload = MLAVarlenBwdWorkload([256, 128], [256, 128], 4, 128, 64, torch.bfloat16)
    _check(workload, MLAVarlenBwdOp())


@pytest.mark.smoke
def test_mla_dense_bwd_rejects_a_token_major_lse() -> None:
    """``lse`` is ``(B, H, S_q)``, as FA3 returns it, not the token-major layout."""
    inputs = list(MLADenseBwdWorkload(1, 64, 64, 4, 192, 128, torch.bfloat16).gen_inputs())
    inputs[5] = inputs[5].transpose(1, 2).contiguous()
    with pytest.raises(ValueError, match="lse"):
        MLADenseBwdOp()(*inputs)


@pytest.mark.smoke
def test_mla_varlen_bwd_rejects_a_token_major_lse() -> None:
    """``lse`` is head-major ``(H, T_q)``, as FA3 returns it for packed requests, not the
    ``(T_q, H)`` that MLAVarlenFwdOp returns."""
    workload = MLAVarlenBwdWorkload([64, 32], [64, 32], 4, 192, 128, torch.bfloat16)
    inputs = list(workload.gen_inputs())
    inputs[5] = inputs[5].T.contiguous()
    with pytest.raises(ValueError, match="lse"):
        MLAVarlenBwdOp()(*inputs)
