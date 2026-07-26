"""Benchmarks for the MHC pre/post ops.

Workload shapes, dtypes, and the pre-op scaling params come from the ops
manifest; roofline FLOP and byte counts come from each op's
``eval_roofline()`` via :class:`ManifestBenchmark`.
"""

import math

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops import MHCPostOp, MHCPreOp
from workloads.mhc import MHCPostTest, MHCPreTest

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True

# Sinkhorn epsilon is not part of any manifest workload; use the manifest
# signature default.
_SINKHORN_EPS = 0.02


def _workload_params(workloads: list, keys: tuple) -> list:
    """Turn manifest workload dicts into pytest params.

    First workload is marked ``smoke``, the rest ``full``. Keys ending in
    ``dtype`` are resolved to ``torch.dtype`` values.
    """
    params = []
    for i, w in enumerate(workloads):
        args = [getattr(torch, w[k]) if k.endswith("dtype") else w[k] for k in keys]
        params.append(
            pytest.param(
                *args,
                marks=pytest.mark.smoke if i == 0 else pytest.mark.full,
                id=w["label"],
            )
        )
    return params


class _MHCPreTestBaseline(MHCPreTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, phi: torch.Tensor, x: torch.Tensor, b: torch.Tensor,
                    alpha_pre, alpha_post, alpha_res,
                    sinkhorn_repeat: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
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


_MHC_PRE_OP = "MHCPreOp"
_MHC_PRE_PARAMS = _workload_params(
    load_workloads(_MHC_PRE_OP),
    ("batch", "n_expand", "c_x", "dtype", "alpha_pre", "alpha_post", "alpha_res",
     "sinkhorn_repeat"),
)


@pytest.mark.parametrize(
    "batch, n_expand, c_x, dtype, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat",
    _MHC_PRE_PARAMS,
)
def test_mhc_pre_bench(batch: int, n_expand: int, c_x: int, dtype: torch.dtype,
                       alpha_pre: float, alpha_post: float, alpha_res: float,
                       sinkhorn_repeat: int) -> None:
    test = _MHCPreTestBaseline(batch, n_expand, c_x, dtype)
    phi, x, b = test.gen_inputs()[:3]
    # The shared workload generator draws its own scaling params; the
    # manifest workload is the authority for them.
    inputs = (phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat, _SINKHORN_EPS)

    op = MHCPreOp(tune=_TUNE)
    bm = ManifestBenchmark(_MHC_PRE_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


class _MHCPostTestBaseline(MHCPostTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, x_layer_out: torch.Tensor, h_post: torch.Tensor,
                    x_res: torch.Tensor) -> torch.Tensor:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        x_out_ref = (h_post.unsqueeze(2).float() @ x_layer_out.unsqueeze(1).float()).reshape(
            batch, n_expand * c_x) + x_res.float()
        x_out_ref = x_out_ref.bfloat16()
        return x_out_ref


_MHC_POST_OP = "MHCPostOp"
_MHC_POST_PARAMS = _workload_params(
    load_workloads(_MHC_POST_OP), ("batch", "n_expand", "c_x", "dtype"),
)


@pytest.mark.parametrize("batch, n_expand, c_x, dtype", _MHC_POST_PARAMS)
def test_mhc_post_bench(batch: int, n_expand: int, c_x: int, dtype: torch.dtype) -> None:
    test = _MHCPostTestBaseline(batch, n_expand, c_x, dtype)
    inputs = test.gen_inputs()

    op = MHCPostOp(tune=_TUNE)
    bm = ManifestBenchmark(_MHC_POST_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
