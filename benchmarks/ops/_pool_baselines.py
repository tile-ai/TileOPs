"""Baselines for pool benchmarks: direct cuDNN graph API and FlagGems.

Private helper for ``bench_pool.py``. Follows the flash-attn pattern from
``bench_gqa.py``: libraries are imported lazily and any miss falls back to
the torch reference, so the bench file runs with or without the optional
dependencies.

The cuDNN side uses the v9 graph API's Resample node (the modern pooling
op). nvidia-cudnn-frontend 1.27 does not bind Resample to Python yet ("Python
API for resampling forward will be supported soon" — cuDNN FE docs), so this
module drives the same node through the backend C API via ctypes, mirroring
``cudnn_frontend/node/resample.h`` call for call.
"""

import atexit
import ctypes
import os
from typing import Callable, Optional

import torch

from tileops.kernels.pool.common import pool_output_dim

# v9 backend data types (cudnn_graph.h; note HALF=2/BF16=9, unlike legacy API).
_CUDNN_DATA_FLOAT = 0
_CUDNN_DATA_HALF = 2
_CUDNN_DATA_BFLOAT16 = 9
_CUDNN_DTYPES = {
    torch.float32: _CUDNN_DATA_FLOAT,
    torch.float16: _CUDNN_DATA_HALF,
    torch.bfloat16: _CUDNN_DATA_BFLOAT16,
}
_RESAMPLE_AVGPOOL_INCLUDE = 2
_RESAMPLE_MAXPOOL = 3
_RESAMPLE_AVGPOOL_EXCLUDE = 4
_ZERO_PAD = 0
_NEG_INF_PAD = 1
_PROPAGATE_NAN = 1

# cudnnBackendAttributeType_t.
_TYPE_HANDLE = 0
_TYPE_DATA_TYPE = 1
_TYPE_INT64 = 3
_TYPE_VOID_PTR = 6
_TYPE_HEUR_MODE = 8
_TYPE_NAN = 10
_TYPE_BACKEND_DESCRIPTOR = 15
_TYPE_RESAMPLE_MODE = 21
_TYPE_PADDING_MODE = 22
_TYPE_FRACTION = 26

# cudnnBackendDescriptorType_t.
_DESC_ENGINE = 2
_DESC_ENGINECFG = 3
_DESC_ENGINEHEUR = 4
_DESC_PLAN = 5
_DESC_OPGRAPH = 15
_DESC_VARPACK = 16
_DESC_TENSOR = 17
_DESC_RESAMPLE = 24
_DESC_OP_RESAMPLE_FWD = 25

_HEUR_MODE_A = 3

# cudnnBackendAttributeName_t.
_ATTR_TENSOR_BYTE_ALIGNMENT = 900
_ATTR_ENGINECFG_ENGINE = 300
_ATTR_ENGINEHEUR_MODE = 200
_ATTR_ENGINEHEUR_OPGRAPH = 201
_ATTR_ENGINEHEUR_RESULTS = 202
_ATTR_PLAN_HANDLE = 400
_ATTR_PLAN_ENGINECFG = 401
_ATTR_PLAN_WORKSPACE_SIZE = 402
_ATTR_OPGRAPH_HANDLE = 800
_ATTR_OPGRAPH_OPS = 801
_ATTR_OPGRAPH_ENGINE_COUNT = 802
_ATTR_TENSOR_DATA_TYPE = 901
_ATTR_TENSOR_DIMENSIONS = 902
_ATTR_TENSOR_STRIDES = 903
_ATTR_TENSOR_UNIQUE_ID = 906
_ATTR_VARPACK_UIDS = 1000
_ATTR_VARPACK_PTRS = 1001
_ATTR_VARPACK_WORKSPACE = 1003
_ATTR_ENGINE_OPGRAPH = 1300
_ATTR_ENGINE_GLOBAL_INDEX = 1301
_ATTR_RESAMPLE_MODE = 1700
_ATTR_RESAMPLE_COMP_TYPE = 1701
_ATTR_RESAMPLE_SPATIAL_DIMS = 1702
_ATTR_RESAMPLE_POST_PAD = 1703
_ATTR_RESAMPLE_PRE_PAD = 1704
_ATTR_RESAMPLE_STRIDES = 1705
_ATTR_RESAMPLE_WINDOW = 1706
_ATTR_RESAMPLE_NAN = 1707
_ATTR_RESAMPLE_PADDING_MODE = 1708
_ATTR_OP_RESAMPLE_XDESC = 1710
_ATTR_OP_RESAMPLE_YDESC = 1711
_ATTR_OP_RESAMPLE_DESC = 1716


class _Fraction(ctypes.Structure):
    """cudnnFraction_t."""

    _fields_ = [("numerator", ctypes.c_int64), ("denominator", ctypes.c_int64)]


_lib = None  # ctypes.CDLL, loaded on first use


def _cudnn_lib():
    """Load libcudnn.so.9 from the nvidia-cudnn pip package, once."""
    global _lib
    if _lib is None:
        import nvidia.cudnn

        path = os.path.join(next(iter(nvidia.cudnn.__path__)), "lib", "libcudnn.so.9")
        _lib = ctypes.CDLL(path)
    return _lib


def _check(status: int, what: str) -> None:
    if status != 0:  # CUDNN_STATUS_SUCCESS
        raise RuntimeError(f"cuDNN call failed: {what} (status {status})")


class _CudnnContext:
    """Process-wide cuDNN handle on torch's current stream; destroyed at exit."""

    _instance = None

    def __init__(self) -> None:
        lib = _cudnn_lib()
        handle = ctypes.c_void_p()
        _check(lib.cudnnCreate(ctypes.byref(handle)), "cudnnCreate")
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        _check(lib.cudnnSetStream(handle, stream), "cudnnSetStream")
        self._lib = lib
        self.handle = handle

    @classmethod
    def get(cls) -> "_CudnnContext":
        if cls._instance is None:
            cls._instance = cls()
            atexit.register(cls._destroy)
        return cls._instance

    @classmethod
    def _destroy(cls) -> None:
        if cls._instance is not None:
            cls._instance._lib.cudnnDestroy(cls._instance.handle)
            cls._instance = None


def _be_create(lib, desc_type: int) -> ctypes.c_void_p:
    desc = ctypes.c_void_p()
    _check(lib.cudnnBackendCreateDescriptor(desc_type, ctypes.byref(desc)), f"create({desc_type})")
    return desc


def _be_set(lib, desc, attr: int, atype: int, values: list, what: str) -> None:
    """SetAttribute; *values* is a list of python ints/floats/pointers."""
    if atype == _TYPE_INT64:
        arr = (ctypes.c_int64 * len(values))(*values)
    elif atype == _TYPE_FRACTION:
        arr = (_Fraction * len(values))(*[(v, 1) for v in values])
    elif atype in (_TYPE_VOID_PTR, _TYPE_BACKEND_DESCRIPTOR, _TYPE_HANDLE):
        arr = (ctypes.c_void_p * len(values))(*values)
    else:  # enum-typed scalars are C ints
        arr = (ctypes.c_int * len(values))(*values)
    _check(lib.cudnnBackendSetAttribute(desc, attr, atype, len(values), arr), what)


def _be_get_int64(lib, desc, attr: int, what: str) -> int:
    value = ctypes.c_int64(0)
    got = ctypes.c_int64(0)
    _check(
        lib.cudnnBackendGetAttribute(
            desc, attr, _TYPE_INT64, 1, ctypes.byref(got), ctypes.byref(value)
        ),
        what,
    )
    return value.value


def _be_finalize(lib, desc, what: str) -> bool:
    """Finalize; False (not raise) when the descriptor is simply unsupported."""
    status = lib.cudnnBackendFinalize(desc)
    if status == 0:
        return True
    if status == 3000:  # CUDNN_STATUS_NOT_SUPPORTED
        return False
    raise RuntimeError(f"cuDNN call failed: finalize {what} (status {status})")


def _make_tensor_desc(lib, uid: int, dtype: int, dims: tuple, strides: tuple) -> ctypes.c_void_p:
    desc = _be_create(lib, _DESC_TENSOR)
    _be_set(lib, desc, _ATTR_TENSOR_UNIQUE_ID, _TYPE_INT64, [uid], "uid")
    _be_set(lib, desc, _ATTR_TENSOR_DATA_TYPE, _TYPE_DATA_TYPE, [dtype], "dtype")
    _be_set(lib, desc, _ATTR_TENSOR_DIMENSIONS, _TYPE_INT64, list(dims), "dims")
    _be_set(lib, desc, _ATTR_TENSOR_STRIDES, _TYPE_INT64, list(strides), "strides")
    # Required by the v9 backend; torch's allocator guarantees 256 B.
    _be_set(lib, desc, _ATTR_TENSOR_BYTE_ALIGNMENT, _TYPE_INT64, [16], "align")
    _check(lib.cudnnBackendFinalize(desc), "finalize tensor")
    return desc


def _build_plan(lib, handle, op_graph):
    """Ask the heuristics for engine configs; build the first viable plan.

    On cuDNN 9.20 the A/INSTANT modes report a nonzero config count but hand
    back zero descriptors, so modes are tried in order and FALLBACK (the
    generic catch-all engine list) is what actually yields configs.
    """
    for mode in (_HEUR_MODE_A, 0, 2):  # A, INSTANT, FALLBACK
        heur = _be_create(lib, _DESC_ENGINEHEUR)
        _be_set(
            lib, heur, _ATTR_ENGINEHEUR_OPGRAPH, _TYPE_BACKEND_DESCRIPTOR, [op_graph], "heur op"
        )
        _be_set(lib, heur, _ATTR_ENGINEHEUR_MODE, _TYPE_HEUR_MODE, [mode], "heur mode")
        _check(lib.cudnnBackendFinalize(heur), "finalize heur")

        cfgs = [_be_create(lib, _DESC_ENGINECFG) for _ in range(16)]
        arr = (ctypes.c_void_p * len(cfgs))(*[c.value for c in cfgs])
        got = ctypes.c_int64(0)
        _check(
            lib.cudnnBackendGetAttribute(
                heur,
                _ATTR_ENGINEHEUR_RESULTS,
                _TYPE_BACKEND_DESCRIPTOR,
                len(cfgs),
                ctypes.byref(got),
                arr,
            ),
            "heur results",
        )
        kept = cfgs[: got.value]
        for cfg in cfgs[got.value :]:
            lib.cudnnBackendDestroyDescriptor(cfg)
        cfgs = kept

        for cfg in cfgs:
            # Heuristic-fetched configs are ready-made; finalizing them again
            # fails with BAD_PARAM. Go straight to the execution plan.
            plan = _be_create(lib, _DESC_PLAN)
            _be_set(lib, plan, _ATTR_PLAN_HANDLE, _TYPE_HANDLE, [handle.value], "plan handle")
            _be_set(lib, plan, _ATTR_PLAN_ENGINECFG, _TYPE_BACKEND_DESCRIPTOR, [cfg], "plan cfg")
            if not _be_finalize(lib, plan, "plan"):
                lib.cudnnBackendDestroyDescriptor(plan)
                lib.cudnnBackendDestroyDescriptor(cfg)
                continue
            workspace = _be_get_int64(lib, plan, _ATTR_PLAN_WORKSPACE_SIZE, "workspace")
            # Keep the descriptors the plan depends on alive with the plan.
            return plan, workspace, (cfg, heur)
        for cfg in cfgs:
            lib.cudnnBackendDestroyDescriptor(cfg)
        lib.cudnnBackendDestroyDescriptor(heur)
    raise RuntimeError("no cuDNN engine supports this resample graph")


def cudnn_pool_fn(
    kind: str,
    kernel_size: tuple,
    stride: tuple,
    padding: tuple,
    ceil_mode: bool,
    count_include_pad: bool = True,
    dilation: tuple = (1, 1),
    divisor_override: Optional[int] = None,
    return_indices: bool = False,
) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Return a direct-cuDNN pooling callable, or None if unsupported.

    Built on the v9 graph API Resample node. ``kernel_size``/``stride``/
    ``padding`` are per-spatial-dim tuples (2D or 3D). Resample has no dilation, no divisor_override, and
    its index output is a backward mask rather than torch's indices; those
    cases return None.
    """
    if return_indices:
        return None
    if any(d != 1 for d in dilation):
        return None
    if kind == "avg" and divisor_override is not None:
        return None
    if kind not in ("avg", "max"):
        return None

    lib = _cudnn_lib()
    ctx = _CudnnContext.get()
    ndim = len(kernel_size)
    if kind == "max":
        mode, pad_mode = _RESAMPLE_MAXPOOL, _NEG_INF_PAD
    elif count_include_pad:
        mode, pad_mode = _RESAMPLE_AVGPOOL_INCLUDE, _ZERO_PAD
    else:
        mode, pad_mode = _RESAMPLE_AVGPOOL_EXCLUDE, _ZERO_PAD

    def _build(x: torch.Tensor, out_spatial: tuple, cudnn_dtype: int):
        """Assemble and finalize the one-node graph for these shapes."""
        in_spatial = x.shape[2:]
        out_shape = tuple(x.shape[:2]) + tuple(out_spatial)
        y_proto = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        # Tensor descriptors must be created before the resample descriptor:
        # with the reverse creation order the operation fails to finalize
        # with CUDNN_STATUS_BAD_PARAM (observed on cuDNN 9.20).
        x_desc = _make_tensor_desc(lib, 0, cudnn_dtype, tuple(x.shape), tuple(x.stride()))
        y_desc = _make_tensor_desc(lib, 1, cudnn_dtype, out_shape, tuple(y_proto.stride()))

        # post padding sized so every claimed output window is defined;
        # ceil_mode needs more than the symmetric pre padding.
        post = [
            max(p, (o - 1) * s + w - n - p)
            for n, o, s, p, w in zip(
                in_spatial, out_spatial, stride, padding, kernel_size, strict=True
            )
        ]
        resample = _be_create(lib, _DESC_RESAMPLE)
        _be_set(lib, resample, _ATTR_RESAMPLE_MODE, _TYPE_RESAMPLE_MODE, [mode], "mode")
        _be_set(
            lib, resample, _ATTR_RESAMPLE_COMP_TYPE, _TYPE_DATA_TYPE, [_CUDNN_DATA_FLOAT], "comp"
        )
        _be_set(lib, resample, _ATTR_RESAMPLE_NAN, _TYPE_NAN, [_PROPAGATE_NAN], "nan")
        _be_set(
            lib, resample, _ATTR_RESAMPLE_PADDING_MODE, _TYPE_PADDING_MODE, [pad_mode], "padmode"
        )
        _be_set(lib, resample, _ATTR_RESAMPLE_SPATIAL_DIMS, _TYPE_INT64, [ndim], "spatial")
        _be_set(lib, resample, _ATTR_RESAMPLE_WINDOW, _TYPE_FRACTION, list(kernel_size), "window")
        _be_set(lib, resample, _ATTR_RESAMPLE_STRIDES, _TYPE_FRACTION, list(stride), "strides")
        _be_set(lib, resample, _ATTR_RESAMPLE_PRE_PAD, _TYPE_FRACTION, list(padding), "pre pad")
        _be_set(lib, resample, _ATTR_RESAMPLE_POST_PAD, _TYPE_FRACTION, post, "post pad")
        _check(lib.cudnnBackendFinalize(resample), "finalize resample")

        operation = _be_create(lib, _DESC_OP_RESAMPLE_FWD)
        _be_set(lib, operation, _ATTR_OP_RESAMPLE_XDESC, _TYPE_BACKEND_DESCRIPTOR, [x_desc], "x")
        _be_set(lib, operation, _ATTR_OP_RESAMPLE_YDESC, _TYPE_BACKEND_DESCRIPTOR, [y_desc], "y")
        _be_set(
            lib, operation, _ATTR_OP_RESAMPLE_DESC, _TYPE_BACKEND_DESCRIPTOR, [resample], "rdesc"
        )
        _check(lib.cudnnBackendFinalize(operation), "finalize operation")

        op_graph = _be_create(lib, _DESC_OPGRAPH)
        _be_set(lib, op_graph, _ATTR_OPGRAPH_HANDLE, _TYPE_HANDLE, [ctx.handle.value], "g handle")
        _be_set(lib, op_graph, _ATTR_OPGRAPH_OPS, _TYPE_BACKEND_DESCRIPTOR, [operation], "ops")
        _check(lib.cudnnBackendFinalize(op_graph), "finalize opgraph")
        plan, workspace_bytes, keep = _build_plan(lib, ctx.handle, op_graph)
        workspace = (
            torch.empty(workspace_bytes, device=x.device, dtype=torch.int8)
            if workspace_bytes > 0
            else None
        )
        return out_shape, plan, workspace, (resample, x_desc, y_desc, operation, op_graph, *keep)

    state: dict = {}  # per-(shape, dtype) plan cache

    def run(x: torch.Tensor) -> torch.Tensor:
        key = (tuple(x.shape), x.dtype)
        entry = state.get(key)
        if entry is None:
            out_spatial = tuple(
                pool_output_dim(n, k, s, p, ceil_mode)
                for n, k, s, p in zip(x.shape[2:], kernel_size, stride, padding, strict=True)
            )
            try:
                built = _build(x, out_spatial, _CUDNN_DTYPES[x.dtype])
                entry = (False, *built)
            except RuntimeError:
                # No engine for this dtype: pool in float32 and cast back.
                # The math matches torch's fp32-accumulated half kernels, so
                # results stay within the reference tolerance.
                built = _build(x.float(), out_spatial, _CUDNN_DATA_FLOAT)
                entry = (True, *built)
            state[key] = entry
        cast, out_shape, plan, workspace, _keep = entry
        x_in = x.float() if cast else x
        y = torch.empty(out_shape, device=x.device, dtype=torch.float32 if cast else x.dtype)
        varpack = _be_create(lib, _DESC_VARPACK)
        _be_set(lib, varpack, _ATTR_VARPACK_UIDS, _TYPE_INT64, [0, 1], "uids")
        _be_set(
            lib,
            varpack,
            _ATTR_VARPACK_PTRS,
            _TYPE_VOID_PTR,
            [x_in.data_ptr(), y.data_ptr()],
            "ptrs",
        )
        if workspace is not None:
            _be_set(
                lib, varpack, _ATTR_VARPACK_WORKSPACE, _TYPE_VOID_PTR, [workspace.data_ptr()], "ws"
            )
        _check(lib.cudnnBackendFinalize(varpack), "finalize varpack")
        _check(lib.cudnnBackendExecute(ctx.handle, plan, varpack), "execute")
        lib.cudnnBackendDestroyDescriptor(varpack)
        return y.to(x.dtype) if cast else y

    return run


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

    FlagGems 5.0.2's LibEntry kernel cache only handles triton 3.3-3.6; under
    triton 3.7 its repeat-call fast path misaligns kernel arguments and
    segfaults on the second launch. The adapter therefore launches FlagGems'
    triton kernels through the standard triton Autotuner path — the measured
    kernel and its autotuned config are identical, only the broken host-side
    cache is bypassed.
    """
    if len(kernel_size) != 2:
        return None
    try:
        import importlib

        import triton
        from flag_gems.utils.libentry import LibEntry

        avg_mod = importlib.import_module("flag_gems.ops.avg_pool2d")
        max_mod = importlib.import_module("flag_gems.ops.max_pool2d_with_indices")
    except ImportError:
        return None

    def _raw_kernel(libentry_kernel):
        if isinstance(libentry_kernel, LibEntry):
            return libentry_kernel.fn
        return libentry_kernel

    avg_kernel = _raw_kernel(avg_mod.avg_pool2d_forward_kernel)
    max_kernel = _raw_kernel(max_mod.max_pool2d_forward_kernel)
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


# Baseline choice per op dimension (user decision, 2026-08-20; measured
# numbers in output/change_pooling_baseline/selection.md): 1D -> torch (the
# cuDNN/FlagGems 2D-emulated paths lose to the native 1D kernels), 2D ->
# flaggems, 3D -> cudnn. "torch" keeps the torch reference as the op's
# baseline. Cases a library cannot run (dilation, indices, adaptive, ...)
# fall back per workload.
_SELECTED_BASELINE: dict[str, str] = {
    "AvgPool1dFwdOp": "torch",
    "AvgPool2dFwdOp": "flaggems",
    "AvgPool3dFwdOp": "cudnn",
    "MaxPool1dFwdOp": "torch",
    "MaxPool1dIndicesFwdOp": "torch",
    "MaxPool2dFwdOp": "flaggems",
    "MaxPool2dIndicesFwdOp": "flaggems",
    "MaxPool3dFwdOp": "cudnn",
    "MaxPool3dIndicesFwdOp": "torch",
    "AdaptiveAvgPool2dFwdOp": "torch",
    "AdaptiveMaxPool2dFwdOp": "torch",
    "AdaptiveMaxPool2dIndicesFwdOp": "torch",
}

# (op_name, kind, kernel_size, stride, padding, ceil_mode) cases where the
# selected library failed the numeric gate during the baseline study.
_EXCLUDED_CASES: frozenset = frozenset()

# 1D ops are intentionally absent: the selected baseline for 1D is torch
# (native 1D kernels beat the 2D-emulated paths), so they take the torch-ref
# early return in pool_baseline regardless of _SELECTED_BASELINE.
_OP_KIND_NDIM = {
    "AvgPool2dFwdOp": ("avg", 2),
    "AvgPool3dFwdOp": ("avg", 3),
    "MaxPool2dFwdOp": ("max", 2),
    "MaxPool2dIndicesFwdOp": ("max", 2),
    "MaxPool3dFwdOp": ("max", 3),
    "MaxPool3dIndicesFwdOp": ("max", 3),
}


def _as_tuple(value, ndim: int) -> tuple:
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,) * ndim


def pool_baseline(op_name: str, test) -> tuple:
    """Return (tag, callable) for op_name's selected baseline.

    Any miss — library not installed, parameters unsupported, case excluded
    by the numeric gate — falls back to ("torch-ref", test.ref_program).
    """
    choice = _SELECTED_BASELINE.get(op_name, "torch")
    if op_name not in _OP_KIND_NDIM or choice == "torch":
        return "torch-ref", test.ref_program

    kind, ndim = _OP_KIND_NDIM[op_name]
    kernel = _as_tuple(test.kernel_size, ndim)
    stride = kernel if test.stride is None else _as_tuple(test.stride, ndim)
    padding = _as_tuple(test.padding, ndim)
    dilation = _as_tuple(getattr(test, "dilation", 1), ndim)
    case_key = (op_name, kind, kernel, stride, padding, test.ceil_mode)
    if case_key in _EXCLUDED_CASES:
        return "torch-ref", test.ref_program

    kwargs = dict(
        count_include_pad=getattr(test, "count_include_pad", True),
        dilation=dilation,
        divisor_override=getattr(test, "divisor_override", None),
    )
    if kind == "max":
        kwargs["return_indices"] = getattr(test, "return_indices", False)
    factory = cudnn_pool_fn if choice == "cudnn" else flaggems_pool_fn

    try:
        fn = factory(kind, kernel, stride, padding, test.ceil_mode, **kwargs)
    except Exception:
        fn = None
    if fn is None:
        return "torch-ref", test.ref_program
    return choice, fn
