"""Benchmark for BatchNormFwdOp and BatchNormBwdOp.

Compares TileOPs vs PyTorch cuDNN batch norm on common ResNet-style shapes.
"""

import pytest
import torch

from benchmarks.benchmark_base import ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.norm.batch_norm import BatchNormBwdOp, BatchNormFwdOp
from workloads.normalization import BatchNormBwdWorkload, BatchNormFwdWorkload

_FWD_OP_NAME = "BatchNormFwdOp"
_BWD_OP_NAME = "BatchNormBwdOp"


def _torch_bn_fwd(x, running_mean, running_var, weight, bias, *, training):
    return torch.nn.functional.batch_norm(
        x,
        running_mean,
        running_var,
        weight,
        bias,
        training=training,
    )


def _torch_bn_bwd(grad_out, x, weight, mean, rstd):
    """Run ATen BatchNorm backward with the contract's saved statistics."""
    return torch.ops.aten.native_batch_norm_backward.default(
        grad_out,
        x,
        weight,
        None,
        None,
        mean,
        rstd,
        True,
        1e-5,
        [True, True, True],
    )


# Manifest-driven params


def _manifest_fwd_params():
    params = []
    for w in load_workloads(_FWD_OP_NAME):
        shape = w["x_shape"]
        N, C, spatial = shape[0], shape[1], tuple(shape[2:])
        label = w.get("label", f"{N}x{C}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(N, C, spatial, dtype, True, False, id=f"{label}-{dtype_str}")
            )
    return params


def _manifest_bwd_params():
    params = []
    for w in load_workloads(_BWD_OP_NAME):
        shape = w["x_shape"]
        N, C, spatial = shape[0], shape[1], tuple(shape[2:])
        label = w.get("label", f"{N}x{C}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(N, C, spatial, dtype, id=f"{label}-{dtype_str}"))
    return params


# Benchmark tests


@pytest.mark.parametrize("N, C, spatial, dtype, training, tune", _manifest_fwd_params())
def test_batch_norm_fwd_bench(N, C, spatial, dtype, training, tune):
    test = BatchNormFwdWorkload(N, C, spatial, dtype, training)
    x, weight, bias, running_mean, running_var = test.gen_inputs()
    # Manifest input order: (x, running_mean, running_var, weight, bias).
    inputs = (x, running_mean, running_var, weight, bias)

    op = BatchNormFwdOp(training=training, tune=tune)
    bm = ManifestBenchmark(_FWD_OP_NAME, op, test)

    spatial = str(spatial)  # stringify tuple so it survives report parameter filtering

    def baseline_fn(x, running_mean, running_var, weight, bias):
        return _torch_bn_fwd(
            x,
            running_mean,
            running_var,
            weight,
            bias,
            training=training,
        )

    bm.compare(
        {"tileops": op, "torch-cudnn": baseline_fn},
        *inputs,
        record_as=op,
        params=locals(),
    )


@pytest.mark.parametrize("N, C, spatial, dtype", _manifest_bwd_params())
def test_batch_norm_bwd_bench(N, C, spatial, dtype):
    test = BatchNormBwdWorkload(N, C, spatial, dtype)
    inputs = test.gen_inputs()

    op = BatchNormBwdOp()
    bm = ManifestBenchmark(_BWD_OP_NAME, op, test)

    spatial = str(spatial)  # stringify tuple so it survives report parameter filtering

    bm.compare(
        {"tileops": op, "torch-aten": _torch_bn_bwd},
        *inputs,
        record_as=op,
        params=locals(),
    )
