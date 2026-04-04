
import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import NSATopkVarlenOp
from workloads.ops.deepseek_nsa_topk import NsaTopkTest as _NsaTopkTestWorkload


class NsaTopkTest(_NsaTopkTestWorkload, TestBase):

    def check_topk(self, op, *inputs, threshold: float = 1e-3) -> None:
        """Custom check for topk indices (not floating point closeness)."""
        outputs_ref = self.ref_program(*inputs)
        outputs = op(*inputs)

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        for out, ref in zip(outputs, outputs_ref, strict=True):
            print("[Top-K Indices Comparison - TileLang vs PyTorch]")

            indices_match = torch.all(out == ref)
            if indices_match:
                print("Top-K Indices Matched!")
            else:
                mismatch_count = (out != ref).sum().item()
                total_count = out.numel()
                mismatch_ratio = mismatch_count / total_count

                assert mismatch_ratio <= threshold, \
                    f"Top-K mismatch ratio {mismatch_ratio:.3%} exceeds threshold {threshold:.3%}"
                print(f"Top-K Indices Mismatched slightly within threshold: "
                      f"{mismatch_ratio * 100:.3f}%")
        print(f"All checks passed for {op.__class__.__name__}.")


class NsaTopkFixture(FixtureBase):
    PARAMS = [
        ("seq_num, c_seq_len, heads, dim, group, scale, selected_block_num, bc, bs, bk, "
         "dtype, accum_dtype, tune", [
             pytest.param(
                 5, 1024, 32, 128, 16, 1, 16, 32, 32, 128, torch.float16, torch.float32, False,
                 marks=pytest.mark.smoke,
             ),
             pytest.param(
                 3, 512, 32, 128, 16, 1, 16, 32, 32, 128, torch.float16, torch.float32, False,
                 marks=pytest.mark.full,
             ),
         ]),
    ]


@NsaTopkFixture
def test_nsa_topk_varlen_op(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim: int,
    group: int,
    scale: float,
    selected_block_num: int,
    bc: int,
    bs: int,
    bk: int,
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
    tune: bool,
) -> None:
    assert group % 16 == 0, "Group size must be a multiple of 16 in NSA"

    test = NsaTopkTest(seq_num, c_seq_len, heads, dim, group, scale, selected_block_num, bc, bs,
                       bk, dtype, accum_dtype)
    inputs = test.gen_inputs()
    op = NSATopkVarlenOp(
        seq_num=seq_num, c_seq_len=c_seq_len, heads=heads, dim=dim, group=group, scale=scale,
        selected_block_num=selected_block_num, bc=bc, bs=bs, bk=bk, dtype=dtype,
        accum_dtype=accum_dtype, tune=tune, chunk_num=test.chunk_num)
    test.check_topk(op, *inputs)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
