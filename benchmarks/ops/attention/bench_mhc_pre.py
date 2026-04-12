import math
from typing import Optional, Tuple

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import MHCPreOp
from workloads.ops.mhc_pre import MHCPreTest


class _MHCPreTestBaseline(MHCPreTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, phi: torch.Tensor, x: torch.Tensor, b: torch.Tensor,
                    alpha_pre, alpha_post, alpha_res,
                    sinkhorn_repeat: int, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        xsqr = x * x
        norm_eps = 0.0001
        r_ref = torch.sqrt(xsqr.sum(dim=1)) / math.sqrt(n_expand * c_x) + norm_eps
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


class MHCPreBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        flops = 2 * t.batch * (
            (t.n_expand * t.n_expand * t.c_x * t.c_x) *
            (t.n_expand * t.n_expand + 2 * t.n_expand) + t.n_expand * t.c_x)
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        return (t.n_expand * 3 + 1) * t.c_x + (t.n_expand * t.c_x) * (
            t.n_expand * t.n_expand + 2 * t.n_expand)


_MHC_PRE_BENCH_PARAMS = [
    pytest.param(1, 4, 1280, torch.bfloat16, True, id="small"),
    pytest.param(2, 4, 1920, torch.bfloat16, True, id="medium"),
    pytest.param(4, 4, 2560, torch.bfloat16, True, id="large"),
]


@pytest.mark.parametrize("batch, n_expand, c_x, dtype, tune", _MHC_PRE_BENCH_PARAMS)
def test_mhc_pre_bench(batch: int, n_expand: int, c_x: int, dtype: torch.dtype,
                       tune: bool) -> None:
    test = _MHCPreTestBaseline(batch, n_expand, c_x, dtype)
    bm = MHCPreBenchmark(test)
    inputs = test.gen_inputs()

    op = MHCPreOp(batch, n_expand, c_x, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
