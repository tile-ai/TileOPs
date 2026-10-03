"""Pooling benchmarks.

Every case is a manifest call of ``src/tileops/manifest/spec/pool.yaml``. The 2D cases model
vision-backbone downsampling patterns such as ResNet/ConvNeXt feature stages.
The 3D cases model video CNN spatiotemporal pooling patterns such as
I3D/SlowFast-style feature stages.
"""

from typing import Callable, Optional

import pytest
import torch

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    compiled_reference,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from benchmarks.cudnn_resample import cudnn_pool_fn
from tileops.ops import (
    AdaptiveAvgPool2dFwdOp,
    AdaptiveMaxPool2dFwdOp,
    AdaptiveMaxPool2dIndicesFwdOp,
    AvgPool1dFwdOp,
    AvgPool2dFwdOp,
    AvgPool3dFwdOp,
    MaxPool1dFwdOp,
    MaxPool1dIndicesFwdOp,
    MaxPool2dFwdOp,
    MaxPool2dIndicesFwdOp,
    MaxPool3dFwdOp,
    MaxPool3dIndicesFwdOp,
)
from tileops.pool import MeanPoolingFwdOp
from workloads.pool import (
    AdaptiveAvgPool2dCall,
    AdaptiveMaxPool2dCall,
    AvgPoolCall,
    MaxPoolCall,
    MeanPoolingCallWorkload,
    MeanPoolingWorkload,
)

# Which library serves an op, and the pooling kind and rank its adapter needs. An op absent
# here has none: no library covers 1D, adaptive pooling, or 3D max-pool indices. Every row
# is also timed against torch, eager and compiled, so this table is not the whole baseline.
_BASELINE: dict[str, tuple[str, str, int]] = {
    "AvgPool2dFwdOp": (FLAGGEMS_TAG, "avg", 2),
    "AvgPool3dFwdOp": ("cudnn", "avg", 3),
    "MaxPool2dFwdOp": (FLAGGEMS_TAG, "max", 2),
    "MaxPool2dIndicesFwdOp": (FLAGGEMS_TAG, "max", 2),
    "MaxPool3dFwdOp": ("cudnn", "max", 3),
}
# Autotuning is a bench-run policy; manifest workloads do not carry it.
_TUNE = True


def flaggems_pool_fn(
    kind: str,
    kernel_size: tuple,
    stride: tuple,
    padding: tuple,
    ceil_mode: bool,
    count_include_pad: bool = True,
    dilation: tuple = (1, 1),
    divisor_override: Optional[int] = None,
    return_indices: bool = False,
) -> Optional[Callable]:
    """Return a FlagGems 2D pooling callable, or None if unsupported.

    FlagGems 5.0.2's LibEntry cache misaligns kernel arguments under triton 3.7 and
    segfaults on the second launch, so this launches its triton kernels through triton's
    own Autotuner instead: same kernel, different host-side path.
    """
    if len(kernel_size) != 2:
        return None
    try:
        import importlib

        import triton
        from flag_gems.utils.libentry import LibEntry

        avg_mod = importlib.import_module("flag_gems.ops.avg_pool2d")
        max_mod = importlib.import_module("flag_gems.ops.max_pool2d_with_indices")
    except ImportError as exc:
        raise RuntimeError(
            "flag_gems is the selected baseline for 2D pooling; install it "
            "(constraints.txt pins the version) or the numbers are not what they claim"
        ) from exc

    def _raw_kernel(libentry_kernel):
        if isinstance(libentry_kernel, LibEntry):
            return libentry_kernel.fn
        return libentry_kernel

    try:
        avg_kernel = _raw_kernel(avg_mod.avg_pool2d_forward_kernel)
        max_kernel = _raw_kernel(max_mod.max_pool2d_forward_kernel)
    except AttributeError as exc:  # a bump moved the kernels this adapter launches
        raise RuntimeError(
            "flag_gems no longer exposes avg_pool2d_forward_kernel / "
            "max_pool2d_forward_kernel; this adapter was written against 5.0.2"
        ) from exc
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    def _grid(meta, in_n, in_c, out_h, out_w):
        return (
            in_n * in_c,
            triton.cdiv(out_h, meta["BLOCK_H"]) * triton.cdiv(out_w, meta["BLOCK_W"]),
        )

    if kind == "avg":
        divisor = float(divisor_override) if divisor_override is not None else 0.0

        def run_avg(x: torch.Tensor) -> torch.Tensor:
            x = x.contiguous()
            in_n, in_c, in_h, in_w = x.shape
            out_h = avg_mod.pool2d_output_size(in_h, kernel_h, stride_h, pad_h, 1, ceil_mode)
            out_w = avg_mod.pool2d_output_size(in_w, kernel_w, stride_w, pad_w, 1, ceil_mode)
            y = torch.empty((in_n, in_c, out_h, out_w), device=x.device, dtype=x.dtype)
            if y.numel() == 0:
                return y
            avg_kernel[lambda meta: _grid(meta, in_n, in_c, out_h, out_w)](
                x,
                y,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                x.stride(3),
                in_c,
                in_h,
                in_w,
                out_h,
                out_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                1,
                1,
                COUNT_INCLUDE_PAD=count_include_pad,
                divisor_override=divisor,
            )
            return y

        return run_avg
    if kind == "max":

        def run_max(x: torch.Tensor):
            x = x.contiguous()
            in_n, in_c, in_h, in_w = x.shape
            out_h = max_mod.max_pool2d_output_size(
                in_h, kernel_h, stride_h, pad_h, dil_h, ceil_mode
            )
            out_w = max_mod.max_pool2d_output_size(
                in_w, kernel_w, stride_w, pad_w, dil_w, ceil_mode
            )
            y = torch.empty((in_n, in_c, out_h, out_w), device=x.device, dtype=x.dtype)
            indices = torch.empty((in_n, in_c, out_h, out_w), device=x.device, dtype=torch.int64)
            if y.numel() == 0:
                return (y, indices) if return_indices else y
            max_kernel[lambda meta: _grid(meta, in_n, in_c, out_h, out_w)](
                x,
                y,
                indices,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                x.stride(3),
                in_c,
                in_h,
                in_w,
                out_h,
                out_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dil_h,
                dil_w,
            )
            return (y, indices) if return_indices else y

        return run_max
    return None


def _as_tuple(value, ndim: int) -> tuple:
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,) * ndim


def _assert_matches_reference(fn, workload, inputs: tuple) -> None:
    """A baseline that computes something else is worse than no baseline."""
    got, expected = fn(*inputs), workload.ref_program(*inputs)
    if isinstance(got, tuple):
        got, expected = got[0], expected[0]
    torch.testing.assert_close(got, expected)


def pool_baseline(op_name: str, workload, *inputs) -> tuple:
    """Return (tag, callable) for op_name's baseline.

    An op this table does not name, and a case the selected library cannot express,
    take the torch reference; the tag says so in the report. A selected library that
    is missing raises instead: silently reporting torch under a case that claims a
    library baseline is how a benchmark ends up measuring nothing it says it does.
    """
    selected = _BASELINE.get(op_name)
    if selected is None:
        return "torch-ref", workload.ref_program

    choice, kind, ndim = selected
    kernel = _as_tuple(workload.kernel_size, ndim)
    stride = kernel if workload.stride is None else _as_tuple(workload.stride, ndim)
    kwargs = dict(
        count_include_pad=getattr(workload, "count_include_pad", True),
        dilation=_as_tuple(getattr(workload, "dilation", 1), ndim),
        divisor_override=getattr(workload, "divisor_override", None),
    )
    if kind == "max":
        kwargs["return_indices"] = getattr(workload, "return_indices", False)

    factory = cudnn_pool_fn if choice == "cudnn" else flaggems_pool_fn
    fn = factory(
        kind, kernel, stride, _as_tuple(workload.padding, ndim), workload.ceil_mode, **kwargs
    )
    if fn is None:
        return "torch-ref", workload.ref_program
    _assert_matches_reference(fn, workload, inputs)
    return choice, fn


def _bench(op_cls: type, workload) -> None:
    inputs = workload.gen_inputs()
    op = op_cls(**workload.call.arguments({}), tune=True)
    bm = ManifestBenchmark(op, workload)

    _tag, _baseline_fn = pool_baseline(op_cls.__name__, workload, *inputs)
    # torch stays alongside the library baseline: it is what the nightly's ratio alert and
    # its history were measured against, and both numbers belong in the same row.
    bm.compare(
        {
            "tileops": op,
            _tag: _baseline_fn,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
    )


@pytest.mark.parametrize("call", manifest_calls(AvgPool1dFwdOp))
def test_avg_pool1d_bench(call) -> None:
    _bench(AvgPool1dFwdOp, AvgPoolCall(call))


@pytest.mark.parametrize("call", manifest_calls(AvgPool2dFwdOp))
def test_avg_pool2d_bench(call) -> None:
    _bench(AvgPool2dFwdOp, AvgPoolCall(call))


@pytest.mark.parametrize("call", manifest_calls(AvgPool3dFwdOp))
def test_avg_pool3d_bench(call) -> None:
    _bench(AvgPool3dFwdOp, AvgPoolCall(call))


@pytest.mark.parametrize("call", manifest_calls(MaxPool1dFwdOp))
def test_max_pool1d_bench(call) -> None:
    _bench(MaxPool1dFwdOp, MaxPoolCall(call))


@pytest.mark.parametrize("call", manifest_calls(MaxPool1dIndicesFwdOp))
def test_max_pool1d_indices_bench(call) -> None:
    _bench(MaxPool1dIndicesFwdOp, MaxPoolCall(call, return_indices=True))


@pytest.mark.parametrize("call", manifest_calls(MaxPool2dFwdOp))
def test_max_pool2d_bench(call) -> None:
    _bench(MaxPool2dFwdOp, MaxPoolCall(call))


@pytest.mark.parametrize("call", manifest_calls(MaxPool2dIndicesFwdOp))
def test_max_pool2d_indices_bench(call) -> None:
    _bench(MaxPool2dIndicesFwdOp, MaxPoolCall(call, return_indices=True))


@pytest.mark.parametrize("call", manifest_calls(MaxPool3dFwdOp))
def test_max_pool3d_bench(call) -> None:
    _bench(MaxPool3dFwdOp, MaxPoolCall(call))


@pytest.mark.parametrize("call", manifest_calls(MaxPool3dIndicesFwdOp))
def test_max_pool3d_indices_bench(call) -> None:
    _bench(MaxPool3dIndicesFwdOp, MaxPoolCall(call, return_indices=True))


@pytest.mark.parametrize("call", manifest_calls(AdaptiveAvgPool2dFwdOp))
def test_adaptive_avg_pool2d_bench(call) -> None:
    _bench(AdaptiveAvgPool2dFwdOp, AdaptiveAvgPool2dCall(call))


@pytest.mark.parametrize("call", manifest_calls(AdaptiveMaxPool2dFwdOp))
def test_adaptive_max_pool2d_bench(call) -> None:
    _bench(AdaptiveMaxPool2dFwdOp, AdaptiveMaxPool2dCall(call))


@pytest.mark.parametrize("call", manifest_calls(AdaptiveMaxPool2dIndicesFwdOp))
def test_adaptive_max_pool2d_indices_bench(call) -> None:
    _bench(AdaptiveMaxPool2dIndicesFwdOp, AdaptiveMaxPool2dCall(call, return_indices=True))


# MeanPoolingFwdOp, the chunked sequence mean.


def _torch_view_mean(workload: MeanPoolingWorkload):
    """The same mean over a reshaped view, or None where the chunks are ragged.

    ``ref_program`` averages one slice per chunk, a launch each. Where every chunk is full
    the chunk axis is a reshape away and the pooling is a single reduction.
    """
    if workload.seq_lens is not None or workload.seq_len % workload.chunk_size:
        return None

    chunks = workload.seq_len // workload.chunk_size

    def fn(x, *_):
        b, _, h, d = x.shape
        return x.view(b, chunks, workload.chunk_size, h, d).mean(dim=2)

    return fn


@pytest.mark.parametrize("call", manifest_calls(MeanPoolingFwdOp))
def test_mean_pooling_bench(call) -> None:
    workload = MeanPoolingCallWorkload(call)
    op = MeanPoolingFwdOp(**call.arguments({}), tune=_TUNE)

    inputs = workload.gen_inputs()
    bm = ManifestBenchmark(op, workload)

    reference = workload.ref_program
    if len(inputs) > 1 and inputs[1] is not None:
        # Fixed chunk bounds permit full-graph compilation of ragged workloads.
        offsets = inputs[1].tolist()
        slices = [
            (start, min(start + workload.chunk_size, end))
            for begin, end in zip(offsets[:-1], offsets[1:], strict=True)
            for start in range(begin, end, workload.chunk_size)
        ]

        def reference(x, *_metadata):
            return torch.stack([x[:, begin:end].mean(1) for begin, end in slices], dim=1)

    functors = {
        "tileops": op,
        "torch-ref": workload.ref_program,
        TORCH_COMPILE_TAG: compiled_reference(reference),
    }
    view_mean = _torch_view_mean(workload)
    if view_mean is not None:
        functors["torch-view-mean"] = view_mean

    bm.compare(functors, *inputs)
