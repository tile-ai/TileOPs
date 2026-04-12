"""Test NativeSparseAttention operation."""


import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import NSAFwdVarlenOp
from workloads.ops.attention.deepseek_nsa import NsaFwdTest as _NsaFwdTestWorkload


class NsaFwdTest(_NsaFwdTestWorkload, TestBase):
    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    block_indices: torch.Tensor, block_counts: torch.Tensor,
                    offsets: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        _ = token_indices
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        block_indices = block_indices.unsqueeze(0)
        block_counts = block_counts.unsqueeze(0)
        return self.naive_nsa(
            q=q,
            k=k,
            v=v,
            g_slc=self.g_slc,
            g_swa=self.g_swa,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=self.block_size,
            window_size=0,
            scale=self.scale,
            cu_seqlens=offsets,
            head_first=False,
        )


class NsaFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, c_seq_len, dim, is_causal, scale, block_size, "
         "groups, selected_blocks, dtype, accum_dtype, tune", [
             pytest.param(
                 1, 16, 1024, 64, True, 0.1, 32, 16, 1, torch.float16, torch.float32, False,
                 marks=pytest.mark.smoke,
             ),
             pytest.param(
                 4, 16, 8192, 64, True, 0.1, 32, 16, 1, torch.float16, torch.float32, False,
                 marks=pytest.mark.full,
             ),
             pytest.param(
                 2, 16, 8192, 64, True, 0.1, 32, 16, 4, torch.float16, torch.float32, False,
                 marks=pytest.mark.full,
             ),
         ]),
    ]


@NsaFwdFixture
def test_nsa_varlen_op(
    batch: int,
    heads: int,
    c_seq_len: int,
    dim: int,
    is_causal: bool,
    scale: float,
    block_size: int,
    groups: int,
    selected_blocks: int,
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
    tune: bool,
) -> None:
    assert groups % 16 == 0, "Group size must be a multiple of 16 in NSA"

    test = NsaFwdTest(batch, heads, c_seq_len, dim, is_causal, scale, block_size, groups,
                      selected_blocks, dtype, accum_dtype)
    params = {
        "batch": batch,
        "heads": heads,
        "c_seq_len": c_seq_len,
        "dim": dim,
        "is_causal": is_causal,
        "scale": scale,
        "block_size": block_size,
        "groups": groups,
        "selected_blocks": selected_blocks,
        "dtype": dtype,
        "accum_dtype": accum_dtype,
        "tune": tune,
    }
    op = NSAFwdVarlenOp(**params)
    test.check(op, *test.gen_inputs(), atol=5e-4, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
