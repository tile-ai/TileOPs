"""Test ManifoldConstrainedHyperConnection Post operation."""

from typing import Tuple

import torch
import pytest

from tests.test_base import TestBase, FixtureBase
from tileops.ops import ManifoldConstrainedHyperConnectionPostOp


class MhcPostFixture(FixtureBase):
    PARAMS = [
        ("batch, n_expand, c_x, dtype, tune", [
            (1, 4, 1280, torch.bfloat16, False),
            (2, 4, 1920, torch.bfloat16, False),
            (4, 4, 2560, torch.bfloat16, False),
        ]),
    ]


class MhcPostTest(TestBase):

    def __init__(self, batch: int, n_expand: int, c_x: int, dtype: torch.dtype):
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        x_layer_out = torch.randn([batch, c_x], device="cuda", dtype=self.dtype)
        h_post = torch.randn([batch, n_expand], device="cuda", dtype=torch.float32)
        x_res = torch.randn([batch, n_expand * c_x], device="cuda", dtype=self.dtype)
        return x_layer_out, h_post, x_res

    def ref_program(self, x_layer_out: torch.Tensor, h_post: torch.Tensor,
                    x_res: torch.Tensor) -> torch.Tensor:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        x_out_ref = (h_post.unsqueeze(2).float() @ x_layer_out.unsqueeze(1).float()).reshape(
            batch, n_expand * c_x) + x_res.float()
        x_out_ref = x_out_ref.bfloat16()
        return x_out_ref

    def check(self,
              op,
              *inputs: Tuple[torch.Tensor],
              atol: float = 1e-2,
              rtol: float = 1e-2) -> None:
        """Check using cosine similarity (mhc post uses bf16 and needs looser checks)."""
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

        assert len(outputs) == len(outputs_ref), "outputs and outputs_ref have different size"
        for i, (output, output_ref) in enumerate(
                zip(outputs, outputs_ref, strict=True)):
            if output_ref is not None:
                cos_sim = torch.nn.functional.cosine_similarity(
                    output_ref, output, dim=-1, eps=1e-8)
                assert cos_sim.min() > 0.99, \
                    f"outputs[{i}]: cosine similarity too low: {cos_sim.min().item()}"

        print(f"All checks passed for {op.__class__.__name__}.")


@MhcPostFixture
def test_mhc_post_op(batch: int, n_expand: int, c_x: int, dtype: torch.dtype,
                     tune: bool) -> None:
    test = MhcPostTest(batch, n_expand, c_x, dtype)
    op = ManifoldConstrainedHyperConnectionPostOp(
        batch, n_expand, c_x, dtype=str(dtype).split('.')[-1])
    test.check(op, *test.gen_inputs())


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
