"""Benchmarks for the MHC pre/post ops.

Workload shapes, dtypes, and the pre-op scaling params come from the ops
manifest; roofline FLOP and byte counts come from each op's
``eval_roofline()`` via :class:`ManifestBenchmark`.
"""


import pytest
import torch

from benchmarks.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workload_field_params,
)
from tileops.manifest import load_workloads
from tileops.ops import MHCPostOp, MHCPreOp
from tileops.testing.mhc_reference import mhc_post_ref, mhc_pre_ref
from workloads.mhc import MHCPostWorkload, MHCPreWorkload

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True

# Sinkhorn epsilon is not part of any manifest workload; use the manifest
# signature default.
_SINKHORN_EPS = 0.02


class MHCPreTestBaseline(MHCPreWorkload):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, phi: torch.Tensor, x: torch.Tensor, b: torch.Tensor,
                    alpha_pre, alpha_post, alpha_res,
                    sinkhorn_repeat: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        return mhc_pre_ref(
            self.batch, self.n_expand, self.c_x,
            phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat, eps,
        )


_MHC_PRE_OP = "MHCPreOp"
_MHC_PRE_PARAMS = workload_field_params(
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
    test = MHCPreTestBaseline(batch, n_expand, c_x, dtype)
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


class MHCPostTestBaseline(MHCPostWorkload):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, x_layer_out: torch.Tensor, h_post: torch.Tensor,
                    x_res: torch.Tensor) -> torch.Tensor:
        return mhc_post_ref(
            self.batch, self.n_expand, self.c_x, x_layer_out, h_post, x_res,
        )


_MHC_POST_OP = "MHCPostOp"
_MHC_POST_PARAMS = workload_field_params(
    load_workloads(_MHC_POST_OP), ("batch", "n_expand", "c_x", "dtype"),
)


@pytest.mark.parametrize("batch, n_expand, c_x, dtype", _MHC_POST_PARAMS)
def test_mhc_post_bench(batch: int, n_expand: int, c_x: int, dtype: torch.dtype) -> None:
    test = MHCPostTestBaseline(batch, n_expand, c_x, dtype)
    inputs = test.gen_inputs()

    op = MHCPostOp(tune=_TUNE)
    bm = ManifestBenchmark(_MHC_POST_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
