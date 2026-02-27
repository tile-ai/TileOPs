"""Test ManifoldConstrainedHyperConnection Pre operation."""

import math
from typing import Tuple

import torch
import pytest

from tests.test_base import TestBase, FixtureBase
from tileops.ops import ManifoldConstrainedHyperConnectionPreOp


class MhcPreFixture(FixtureBase):
    PARAMS = [
        ("batch, n_expand, c_x, dtype, tune", [
            (1, 4, 1280, torch.bfloat16, False),
            (2, 4, 1920, torch.bfloat16, False),
            (4, 4, 2560, torch.bfloat16, False),
        ]),
    ]


class MhcPreTest(TestBase):

    def __init__(self, batch: int, n_expand: int, c_x: int, dtype: torch.dtype):
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        phi = torch.randn([n_expand * c_x, n_expand * n_expand + 2 * n_expand],
                          device="cuda",
                          dtype=torch.float32)
        x = torch.randn([batch, n_expand * c_x], device="cuda", dtype=torch.bfloat16)
        b = torch.randn([n_expand * n_expand + 2 * n_expand], device="cuda", dtype=torch.float32)
        alpha_pre = torch.randn(())
        alpha_post = torch.randn(())
        alpha_res = torch.randn(())
        sinkhorn_repeat = 20
        eps = 0.02
        return phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat, eps

    def ref_program(self, phi: torch.Tensor, x: torch.Tensor, b: torch.Tensor,
                    alpha_pre, alpha_post, alpha_res,
                    sinkhorn_repeat: int, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        xsqr = x * x
        r_ref = torch.sqrt(xsqr.sum(dim=1)) / math.sqrt(n_expand * c_x)
        H = torch.zeros([batch, n_expand * n_expand + 2 * n_expand],
                        device="cuda", dtype=torch.float)
        for i in range(batch):
            H[i, :] = x[i, :].float() @ phi

        H_pre_ref = H[:, :n_expand]
        H_res_ref = H[:, 2 * n_expand:]
        H_res_ref = H_res_ref.reshape(batch, n_expand, n_expand)

        b_pre_ref = b[:n_expand]
        b_res_ref = b[2 * n_expand:]
        b_res_ref = b_res_ref.reshape([n_expand, n_expand])

        H_pre_ref = torch.sigmoid(alpha_pre * H_pre_ref / r_ref.unsqueeze(-1) + b_pre_ref)
        H_res_ref = alpha_res * H_res_ref / r_ref.unsqueeze(-1).unsqueeze(-1) + b_res_ref

        H_res_ref_tmp = H_res_ref.max(dim=-1, keepdim=True).values

        H_res_ref = torch.exp(H_res_ref - H_res_ref_tmp)
        for _i in range(sinkhorn_repeat):
            H_res_ref = H_res_ref / (H_res_ref.sum(dim=-1, keepdim=True) + eps)
            H_res_ref = H_res_ref / (H_res_ref.sum(dim=-2, keepdim=True) + eps)
        x_in_reshaped = x.reshape([batch, n_expand, c_x])
        x_res_ref = torch.zeros([batch, n_expand, c_x], device="cuda", dtype=torch.bfloat16)
        x_layer_ref = torch.zeros([batch, c_x], device="cuda", dtype=torch.bfloat16)

        h_res_ref = H_res_ref
        h_pre_ref = H_pre_ref
        for i in range(batch):
            h_res_tmp = h_res_ref[i, :, :].float()
            h_pre_tmp = h_pre_ref[i, :].float()
            x_in_reshaped_tmp = x_in_reshaped[i, :, :].float()
            x_res_ref[i, :, :] = h_res_tmp @ x_in_reshaped_tmp
            x_layer_ref[i, :] = h_pre_tmp @ x_in_reshaped_tmp

        x_res_ref = x_res_ref.reshape(batch, n_expand * c_x)

        x_res_ref = x_res_ref.bfloat16()
        x_layer_ref = x_layer_ref.bfloat16()
        return x_res_ref, x_layer_ref

    def check(self,
              op,
              *inputs: Tuple[torch.Tensor],
              atol: float = 1e-2,
              rtol: float = 1e-2) -> None:
        """Check using cosine similarity (mhc pre uses bf16 and needs looser checks)."""
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
        names = ["x_res", "x_layer"]
        for i, (output, output_ref) in enumerate(
                zip(outputs, outputs_ref, strict=True)):
            if output_ref is not None:
                cos_sim = torch.nn.functional.cosine_similarity(
                    output_ref, output, dim=-1, eps=1e-8)
                name = names[i] if i < len(names) else f"outputs[{i}]"
                assert cos_sim.min() > 0.99, \
                    f"{name}: cosine similarity too low: {cos_sim.min().item()}"

        print(f"All checks passed for {op.__class__.__name__}.")


@MhcPreFixture
def test_mhc_pre_op(batch: int, n_expand: int, c_x: int, dtype: torch.dtype,
                    tune: bool) -> None:
    test = MhcPreTest(batch, n_expand, c_x, dtype)
    op = ManifoldConstrainedHyperConnectionPreOp(batch, n_expand, c_x, dtype=torch.bfloat16)
    test.check(op, *test.gen_inputs())


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
