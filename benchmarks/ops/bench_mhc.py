"""Benchmarks for the MHC pre/post ops.

Workload shapes, dtypes, and the pre-op scaling params come from the ops
manifest; roofline FLOP and byte counts come from each op's
``eval_roofline()`` via :class:`ManifestBenchmark`.
"""

import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import (
    ManifestBenchmark,
    fields,
    workload_params,
)
from tileops.manifest import load_workloads
from tileops.ops import MHCPostFwdOp, MHCPreFwdOp
from workloads.mhc import MHCPostWorkload, MHCPreWorkload

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True

# Sinkhorn epsilon is not part of any manifest workload; use the manifest
# signature default.
_SINKHORN_EPS = 0.02


_MHC_PRE_PARAMS = workload_params(
    load_workloads(MHCPreFwdOp),
    fields(
        "batch",
        "n_expand",
        "c_x",
        "dtype",
        "alpha_pre",
        "alpha_post",
        "alpha_res",
        "sinkhorn_repeat",
    ),
    smoke_first=True,
)


@pytest.mark.parametrize(
    "batch, n_expand, c_x, dtype, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat",
    _MHC_PRE_PARAMS,
)
def test_mhc_pre_bench(
    batch: int,
    n_expand: int,
    c_x: int,
    dtype: torch.dtype,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    sinkhorn_repeat: int,
) -> None:
    # The manifest workload is the authority for the scaling params, so the case
    # is built with them rather than with the ones the generator would draw.
    test = MHCPreWorkload(
        batch,
        n_expand,
        c_x,
        dtype,
        alpha_pre=alpha_pre,
        alpha_post=alpha_post,
        alpha_res=alpha_res,
        sinkhorn_repeat=sinkhorn_repeat,
        sinkhorn_eps=_SINKHORN_EPS,
    )
    inputs = test.gen_inputs()

    op = MHCPreFwdOp(tune=_TUNE)
    bm = ManifestBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
    )


_MHC_POST_PARAMS = workload_params(
    load_workloads(MHCPostFwdOp),
    fields("batch", "n_expand", "c_x", "dtype"),
    smoke_first=True,
)


@pytest.mark.parametrize("batch, n_expand, c_x, dtype", _MHC_POST_PARAMS)
def test_mhc_post_bench(batch: int, n_expand: int, c_x: int, dtype: torch.dtype) -> None:
    test = MHCPostWorkload(batch, n_expand, c_x, dtype)
    inputs = test.gen_inputs()

    op = MHCPostFwdOp(tune=_TUNE)
    bm = ManifestBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
    )
