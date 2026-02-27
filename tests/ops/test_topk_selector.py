from typing import Tuple

import torch
import pytest

from tests.test_base import TestBase, FixtureBase
from tileops.ops import TopkSelectorOp
from tileops.utils import str2dtype


class TopkSelectorFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, topk, in_dtype_str, out_dtype_str, tune", [
            (64, 32 * 1024, 1024, "float32", "int32", False),
            (64, 32 * 1024, 2048, "float32", "int32", False),
            (128, 64 * 1024, 1024, "float32", "int32", False),
            (128, 64 * 1024, 2048, "float32", "int32", False),
        ]),
    ]


class TopkSelectorTest(TestBase):

    def __init__(self, batch: int, seq_len: int, topk: int, in_dtype: torch.dtype,
                 out_dtype: torch.dtype):
        self.batch = batch
        self.seq_len = seq_len
        self.topk = topk
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index_score = torch.randn(self.batch, self.seq_len, dtype=self.in_dtype, device="cuda")
        starts = torch.zeros(self.batch, dtype=self.out_dtype, device="cuda")
        ends = torch.ones(self.batch, dtype=self.out_dtype, device="cuda") * self.seq_len
        return index_score, starts, ends

    def ref_program(self, index_score: torch.Tensor, starts: torch.Tensor,
                    ends: torch.Tensor) -> torch.Tensor:
        indexes_ref = torch.topk(index_score, self.topk, dim=-1)[1]
        return indexes_ref

    def check(self,
              op,
              *inputs: Tuple[torch.Tensor],
              atol: float = 1e-4,
              rtol: float = 1e-5) -> None:
        """Check using set intersection (topk indices may be in different order)."""
        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Skipped checking {self.__class__.__name__} due to OOM in ref: {e}")
                return
            raise e

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        elif not isinstance(outputs_ref, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs_ref)}")

        with torch.no_grad():
            outputs = op(*inputs)

        if isinstance(outputs, list):
            outputs = tuple(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif not isinstance(outputs, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs)}")

        assert len(outputs) == len(outputs_ref), \
            f"outputs: {len(outputs)} and outputs_ref: {len(outputs_ref)} have different size"
        for i, (output, output_ref) in enumerate(
                zip(outputs, outputs_ref, strict=True)):
            ref_np = output_ref.cpu().to(torch.int32).numpy()
            trt_np = output.cpu().to(torch.int32).numpy()

            ref_list = ref_np.flatten().tolist()
            trt_list = trt_np.flatten().tolist()

            set_ref = set(ref_list)
            set_trt = set(trt_list)
            intersection = set_ref & set_trt
            assert len(intersection) / len(set_ref) == 1.0, \
                f"outputs[{i}] is not close to outputs_ref[{i}]"

        print(f"All checks passed for {op.__class__.__name__}.")

    def check_fn(self,
                 fn: callable,
                 *inputs: Tuple[torch.Tensor],
                 atol: float = 1e-4,
                 rtol: float = 1e-5,
                 grad: bool = False) -> None:
        """Check function using set intersection."""
        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Skipped checking {self.__class__.__name__} due to OOM in ref: {e}")
                return
            raise e

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        elif not isinstance(outputs_ref, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs_ref)}")

        with torch.no_grad():
            outputs = fn(*inputs)

        if isinstance(outputs, list):
            outputs = tuple(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif not isinstance(outputs, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs)}")

        assert len(outputs) == len(outputs_ref), \
            f"outputs: {len(outputs)} and outputs_ref: {len(outputs_ref)} have different size"
        for i, (output, output_ref) in enumerate(
                zip(outputs, outputs_ref, strict=True)):
            ref_np = output_ref.cpu().to(torch.int32).numpy()
            trt_np = output.cpu().to(torch.int32).numpy()

            ref_list = ref_np.flatten().tolist()
            trt_list = trt_np.flatten().tolist()

            set_ref = set(ref_list)
            set_trt = set(trt_list)
            intersection = set_ref & set_trt
            assert len(intersection) / len(set_ref) == 1.0, \
                f"outputs[{i}] is not close to outputs_ref[{i}]"

        print(f"All checks passed for {fn.__class__.__name__}.")


@TopkSelectorFixture
def test_topk_selector_op(batch: int, seq_len: int, topk: int, in_dtype_str: str,
                          out_dtype_str: str, tune: bool) -> None:
    in_dtype = str2dtype[in_dtype_str]
    out_dtype = str2dtype[out_dtype_str]
    test = TopkSelectorTest(batch, seq_len, topk, in_dtype, out_dtype)
    op = TopkSelectorOp(batch, seq_len, topk, in_dtype, out_dtype, tune=tune)
    test.check(op, *test.gen_inputs())


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
