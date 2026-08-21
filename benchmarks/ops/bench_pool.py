"""Pooling benchmarks.

Workloads are loaded from ``src/tileops/manifest/pool.yaml``. The 2D cases model
vision-backbone downsampling patterns such as ResNet/ConvNeXt feature stages.
The 3D cases model video CNN spatiotemporal pooling patterns such as
I3D/SlowFast-style feature stages.
"""

import atexit
import ctypes
import os
from typing import Callable, Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.kernels.pool.common import pool_output_dim
from tileops.manifest import load_workloads
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
from workloads.pool import (
    AdaptivePool2dWorkload,
    AvgPool1dBenchCase,
    AvgPool2dBenchCase,
    AvgPool3dBenchCase,
    MaxPool1dBenchCase,
    MaxPool2dBenchCase,
    MaxPool3dBenchCase,
)

# ---------------------------------------------------------------------------
# Baselines. cuDNN is reached through the v9 backend's Resample node in ctypes, not a binding:
# nvidia-cudnn-frontend 1.27 exposes 141 graph ops and none of them is resample, and the
# legacy cudnnPoolingForward rejects bfloat16. Measured on an H200, the legacy entry point
# reaches the same kernel as the graph API (0.0867 vs 0.0865 ms device-busy).
# ---------------------------------------------------------------------------

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
_ATTR_ENGINEHEUR_MODE = 200
_ATTR_ENGINEHEUR_OPGRAPH = 201
_ATTR_ENGINEHEUR_RESULTS = 202
_ATTR_PLAN_HANDLE = 400
_ATTR_PLAN_ENGINECFG = 401
_ATTR_PLAN_WORKSPACE_SIZE = 402
_ATTR_OPGRAPH_HANDLE = 800
_ATTR_OPGRAPH_OPS = 801
_ATTR_TENSOR_DATA_TYPE = 901
_ATTR_TENSOR_DIMENSIONS = 902
_ATTR_TENSOR_STRIDES = 903
_ATTR_TENSOR_UNIQUE_ID = 906
_ATTR_VARPACK_UIDS = 1000
_ATTR_VARPACK_PTRS = 1001
_ATTR_VARPACK_WORKSPACE = 1003
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
    _fields_ = [("numerator", ctypes.c_int64), ("denominator", ctypes.c_int64)]


class _Cudnn:
    """The v9 backend, and the descriptor calls this file makes against it.

    One handle per process, on torch's current stream, with every descriptor it
    hands out destroyed at exit.
    """

    _instance: Optional["_Cudnn"] = None

    def __init__(self) -> None:
        import nvidia.cudnn

        path = os.path.join(next(iter(nvidia.cudnn.__path__)), "lib", "libcudnn.so.9")
        self._lib = ctypes.CDLL(path)
        self.version = self._lib.cudnnGetVersion()
        self.handle = ctypes.c_void_p()
        self._check(self._lib.cudnnCreate(ctypes.byref(self.handle)), "cudnnCreate")
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self._check(self._lib.cudnnSetStream(self.handle, stream), "cudnnSetStream")
        self._owned: list = []

    @classmethod
    def get(cls) -> "_Cudnn":
        if cls._instance is None:
            cls._instance = cls()
            atexit.register(cls._release)
        return cls._instance

    @classmethod
    def _release(cls) -> None:
        self = cls._instance
        if self is None:
            return
        for desc in self._owned:
            self._lib.cudnnBackendDestroyDescriptor(desc)
        self._lib.cudnnDestroy(self.handle)
        cls._instance = None

    @staticmethod
    def _check(status: int, what: str) -> None:
        if status != 0:  # CUDNN_STATUS_SUCCESS
            raise RuntimeError(f"cuDNN call failed: {what} (status {status})")

    def create(self, desc_type: int, keep: bool = True) -> ctypes.c_void_p:
        desc = ctypes.c_void_p()
        self._check(
            self._lib.cudnnBackendCreateDescriptor(desc_type, ctypes.byref(desc)),
            f"create({desc_type})",
        )
        if keep:
            self._owned.append(desc)
        return desc

    def destroy(self, desc: ctypes.c_void_p) -> None:
        self._lib.cudnnBackendDestroyDescriptor(desc)

    def set(self, desc, attr: int, atype: int, values: list, what: str) -> None:
        """SetAttribute; a scalar attribute goes in as a one-element list."""
        if atype == _TYPE_INT64:
            arr = (ctypes.c_int64 * len(values))(*values)
        elif atype == _TYPE_FRACTION:
            arr = (_Fraction * len(values))(*[(v, 1) for v in values])
        elif atype in (_TYPE_VOID_PTR, _TYPE_BACKEND_DESCRIPTOR, _TYPE_HANDLE):
            arr = (ctypes.c_void_p * len(values))(*values)
        else:  # enum-typed scalars are C ints
            arr = (ctypes.c_int * len(values))(*values)
        self._check(self._lib.cudnnBackendSetAttribute(desc, attr, atype, len(values), arr), what)

    def execute(self, plan, varpack) -> None:
        self._check(self._lib.cudnnBackendExecute(self.handle, plan, varpack), "execute")

    def get_int64(self, desc, attr: int, what: str) -> int:
        value = ctypes.c_int64(0)
        got = ctypes.c_int64(0)
        self._check(
            self._lib.cudnnBackendGetAttribute(
                desc, attr, _TYPE_INT64, 1, ctypes.byref(got), ctypes.byref(value)
            ),
            what,
        )
        return value.value

    def finalize(self, desc, what: str) -> bool:
        """Finalize; False (not raise) when the descriptor is simply unsupported."""
        status = self._lib.cudnnBackendFinalize(desc)
        if status == 0:
            return True
        if status == 3000:  # CUDNN_STATUS_NOT_SUPPORTED
            return False
        raise RuntimeError(f"cuDNN call failed: finalize {what} (status {status})")

    def finalize_or_raise(self, desc, what: str) -> None:
        self._check(self._lib.cudnnBackendFinalize(desc), f"finalize {what}")

    def tensor_desc(self, uid: int, dtype: int, dims: tuple, strides: tuple) -> ctypes.c_void_p:
        desc = self.create(_DESC_TENSOR)
        self.set(desc, _ATTR_TENSOR_UNIQUE_ID, _TYPE_INT64, [uid], "uid")
        self.set(desc, _ATTR_TENSOR_DATA_TYPE, _TYPE_DATA_TYPE, [dtype], "dtype")
        self.set(desc, _ATTR_TENSOR_DIMENSIONS, _TYPE_INT64, list(dims), "dims")
        self.set(desc, _ATTR_TENSOR_STRIDES, _TYPE_INT64, list(strides), "strides")
        # Required by the v9 backend; torch's allocator guarantees 256 B.
        self.set(desc, _ATTR_TENSOR_BYTE_ALIGNMENT, _TYPE_INT64, [16], "align")
        self.finalize_or_raise(desc, "tensor")
        return desc

    def build_plan(self, op_graph):
        """Ask the heuristics for engine configs; build the first viable plan.

        On cuDNN 9.20 the A/INSTANT modes report a nonzero config count but hand
        back zero descriptors, so modes are tried in order and FALLBACK (the
        generic catch-all engine list) is what actually yields configs.
        """
        for mode in (_HEUR_MODE_A, 0, 2):  # A, INSTANT, FALLBACK
            heur = self.create(_DESC_ENGINEHEUR)
            self.set(
                heur, _ATTR_ENGINEHEUR_OPGRAPH, _TYPE_BACKEND_DESCRIPTOR, [op_graph], "heur op"
            )
            self.set(heur, _ATTR_ENGINEHEUR_MODE, _TYPE_HEUR_MODE, [mode], "heur mode")
            self.finalize_or_raise(heur, "heur")

            cfgs = [self.create(_DESC_ENGINECFG, keep=False) for _ in range(16)]
            arr = (ctypes.c_void_p * len(cfgs))(*[c.value for c in cfgs])
            got = ctypes.c_int64(0)
            self._check(
                self._lib.cudnnBackendGetAttribute(
                    heur,
                    _ATTR_ENGINEHEUR_RESULTS,
                    _TYPE_BACKEND_DESCRIPTOR,
                    len(cfgs),
                    ctypes.byref(got),
                    arr,
                ),
                "heur results",
            )
            for cfg in cfgs[got.value :]:
                self.destroy(cfg)
            for cfg in cfgs[: got.value]:
                # Heuristic-fetched configs are ready-made; finalizing them again
                # fails with BAD_PARAM. Go straight to the execution plan.
                plan = self.create(_DESC_PLAN)
                self.set(plan, _ATTR_PLAN_HANDLE, _TYPE_HANDLE, [self.handle.value], "plan handle")
                self.set(plan, _ATTR_PLAN_ENGINECFG, _TYPE_BACKEND_DESCRIPTOR, [cfg], "plan cfg")
                if not self.finalize(plan, "plan"):
                    self.destroy(plan)
                    self.destroy(cfg)
                    continue
                self._owned.append(cfg)
                return plan, self.get_int64(plan, _ATTR_PLAN_WORKSPACE_SIZE, "workspace")
            for cfg in cfgs[: got.value]:
                self.destroy(cfg)
        raise RuntimeError(f"no cuDNN engine supports this resample graph (cuDNN {self.version})")


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
    """Return a direct-cuDNN pooling callable, or None where Resample cannot express it.

    ``kernel_size`` / ``stride`` / ``padding`` are per-spatial-dim tuples, 2D or 3D.
    Resample has no dilation and no divisor_override, and its index output is a backward
    mask rather than torch's indices.
    """
    if return_indices:
        return None
    if any(d != 1 for d in dilation):
        return None
    if kind == "avg" and divisor_override is not None:
        return None
    if kind not in ("avg", "max"):
        return None

    c = _Cudnn.get()
    ndim = len(kernel_size)
    if kind == "max":
        mode, pad_mode = _RESAMPLE_MAXPOOL, _NEG_INF_PAD
    elif count_include_pad:
        mode, pad_mode = _RESAMPLE_AVGPOOL_INCLUDE, _ZERO_PAD
    else:
        mode, pad_mode = _RESAMPLE_AVGPOOL_EXCLUDE, _ZERO_PAD

    def _build(x: torch.Tensor, out_spatial: tuple, cudnn_dtype: int):
        in_spatial = x.shape[2:]
        out_shape = tuple(x.shape[:2]) + tuple(out_spatial)
        y_proto = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        # Tensor descriptors must be created before the resample descriptor:
        # with the reverse creation order the operation fails to finalize
        # with CUDNN_STATUS_BAD_PARAM (observed on cuDNN 9.20).
        x_desc = c.tensor_desc(0, cudnn_dtype, tuple(x.shape), tuple(x.stride()))
        y_desc = c.tensor_desc(1, cudnn_dtype, out_shape, tuple(y_proto.stride()))

        # post padding sized so every claimed output window is defined;
        # ceil_mode needs more than the symmetric pre padding.
        post = [
            max(p, (o - 1) * s + w - n - p)
            for n, o, s, p, w in zip(
                in_spatial, out_spatial, stride, padding, kernel_size, strict=True
            )
        ]
        resample = c.create(_DESC_RESAMPLE)
        c.set(resample, _ATTR_RESAMPLE_MODE, _TYPE_RESAMPLE_MODE, [mode], "mode")
        c.set(resample, _ATTR_RESAMPLE_COMP_TYPE, _TYPE_DATA_TYPE, [_CUDNN_DATA_FLOAT], "comp")
        # Propagating costs ~19% of the kernel on an H200 max-pool, and torch propagates:
        # the cheaper mode returns a number where torch returns NaN.
        c.set(resample, _ATTR_RESAMPLE_NAN, _TYPE_NAN, [_PROPAGATE_NAN], "nan")
        c.set(resample, _ATTR_RESAMPLE_PADDING_MODE, _TYPE_PADDING_MODE, [pad_mode], "padmode")
        c.set(resample, _ATTR_RESAMPLE_SPATIAL_DIMS, _TYPE_INT64, [ndim], "spatial")
        c.set(resample, _ATTR_RESAMPLE_WINDOW, _TYPE_FRACTION, list(kernel_size), "window")
        c.set(resample, _ATTR_RESAMPLE_STRIDES, _TYPE_FRACTION, list(stride), "strides")
        c.set(resample, _ATTR_RESAMPLE_PRE_PAD, _TYPE_FRACTION, list(padding), "pre pad")
        c.set(resample, _ATTR_RESAMPLE_POST_PAD, _TYPE_FRACTION, post, "post pad")
        c.finalize_or_raise(resample, "resample")

        operation = c.create(_DESC_OP_RESAMPLE_FWD)
        c.set(operation, _ATTR_OP_RESAMPLE_XDESC, _TYPE_BACKEND_DESCRIPTOR, [x_desc], "x")
        c.set(operation, _ATTR_OP_RESAMPLE_YDESC, _TYPE_BACKEND_DESCRIPTOR, [y_desc], "y")
        c.set(operation, _ATTR_OP_RESAMPLE_DESC, _TYPE_BACKEND_DESCRIPTOR, [resample], "rdesc")
        c.finalize_or_raise(operation, "operation")

        op_graph = c.create(_DESC_OPGRAPH)
        c.set(op_graph, _ATTR_OPGRAPH_HANDLE, _TYPE_HANDLE, [c.handle.value], "g handle")
        c.set(op_graph, _ATTR_OPGRAPH_OPS, _TYPE_BACKEND_DESCRIPTOR, [operation], "ops")
        c.finalize_or_raise(op_graph, "opgraph")
        plan, workspace_bytes = c.build_plan(op_graph)
        workspace = (
            torch.empty(workspace_bytes, device=x.device, dtype=torch.int8)
            if workspace_bytes > 0
            else None
        )
        return out_shape, plan, workspace

    state: dict = {}  # per-(shape, dtype) plan cache

    def run(x: torch.Tensor) -> torch.Tensor:
        key = (tuple(x.shape), x.dtype)
        entry = state.get(key)
        if entry is None:
            out_spatial = tuple(
                pool_output_dim(n, k, s, p, ceil_mode)
                for n, k, s, p in zip(x.shape[2:], kernel_size, stride, padding, strict=True)
            )
            entry = _build(x, out_spatial, _CUDNN_DTYPES[x.dtype])
            state[key] = entry
        out_shape, plan, workspace = entry
        y = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        varpack = c.create(_DESC_VARPACK, keep=False)
        c.set(varpack, _ATTR_VARPACK_UIDS, _TYPE_INT64, [0, 1], "uids")
        c.set(
            varpack,
            _ATTR_VARPACK_PTRS,
            _TYPE_VOID_PTR,
            [x.data_ptr(), y.data_ptr()],
            "ptrs",
        )
        if workspace is not None:
            c.set(varpack, _ATTR_VARPACK_WORKSPACE, _TYPE_VOID_PTR, [workspace.data_ptr()], "ws")
        c.finalize_or_raise(varpack, "varpack")
        c.execute(plan, varpack)
        c.destroy(varpack)
        return y

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


# Which library serves an op, and the pooling kind and rank its adapter needs. An op absent
# here has none: no library covers 1D, adaptive pooling, or 3D max-pool indices. Every row
# is also timed against torch, eager and compiled, so this table is not the whole baseline.
_BASELINE: dict[str, tuple[str, str, int]] = {
    "AvgPool2dFwdOp": ("flaggems", "avg", 2),
    "AvgPool3dFwdOp": ("cudnn", "avg", 3),
    "MaxPool2dFwdOp": ("flaggems", "max", 2),
    "MaxPool2dIndicesFwdOp": ("flaggems", "max", 2),
    "MaxPool3dFwdOp": ("cudnn", "max", 3),
}


def _as_tuple(value, ndim: int) -> tuple:
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,) * ndim


def _assert_matches_reference(fn, test, inputs: tuple) -> None:
    """A baseline that computes something else is worse than no baseline."""
    got, expected = fn(*inputs), test.ref_program(*inputs)
    if isinstance(got, tuple):
        got, expected = got[0], expected[0]
    tol = 1e-5 if got.dtype is torch.float32 else 2e-2
    torch.testing.assert_close(got.float(), expected.float(), atol=tol, rtol=tol)


def compiled_reference(test):
    """torch's own kernels through inductor; on 1D pooling it beats eager by 2-6x.

    dynamo caches eight graphs per code object and all cases share one ``ref_program``,
    so without the reset the later ones would time eager under a tag that says compiled.
    """
    torch._dynamo.reset()
    return torch.compile(test.ref_program, dynamic=False)


def pool_baseline(op_name: str, test, *inputs) -> tuple:
    """Return (tag, callable) for op_name's baseline.

    An op this table does not name, and a case the selected library cannot express,
    take the torch reference; the tag says so in the report. A selected library that
    is missing raises instead: silently reporting torch under a case that claims a
    library baseline is how a benchmark ends up measuring nothing it says it does.
    """
    selected = _BASELINE.get(op_name)
    if selected is None:
        return "torch-ref", test.ref_program

    choice, kind, ndim = selected
    kernel = _as_tuple(test.kernel_size, ndim)
    stride = kernel if test.stride is None else _as_tuple(test.stride, ndim)
    kwargs = dict(
        count_include_pad=getattr(test, "count_include_pad", True),
        dilation=_as_tuple(getattr(test, "dilation", 1), ndim),
        divisor_override=getattr(test, "divisor_override", None),
    )
    if kind == "max":
        kwargs["return_indices"] = getattr(test, "return_indices", False)

    factory = cudnn_pool_fn if choice == "cudnn" else flaggems_pool_fn
    fn = factory(kind, kernel, stride, _as_tuple(test.padding, ndim), test.ceil_mode, **kwargs)
    if fn is None:
        return "torch-ref", test.ref_program
    _assert_matches_reference(fn, test, inputs)
    return choice, fn


_ADAPTIVE_AVG_POOL2D_OP_NAME = "AdaptiveAvgPool2dFwdOp"
_ADAPTIVE_MAX_POOL2D_OP_NAME = "AdaptiveMaxPool2dFwdOp"
_ADAPTIVE_MAX_POOL2D_INDICES_OP_NAME = "AdaptiveMaxPool2dIndicesFwdOp"
_AVG_POOL1D_OP_NAME = "AvgPool1dFwdOp"
_AVG_POOL2D_OP_NAME = "AvgPool2dFwdOp"
_AVG_POOL3D_OP_NAME = "AvgPool3dFwdOp"
_MAX_POOL1D_OP_NAME = "MaxPool1dFwdOp"
_MAX_POOL1D_INDICES_OP_NAME = "MaxPool1dIndicesFwdOp"
_MAX_POOL2D_OP_NAME = "MaxPool2dFwdOp"
_MAX_POOL2D_INDICES_OP_NAME = "MaxPool2dIndicesFwdOp"
_MAX_POOL3D_OP_NAME = "MaxPool3dFwdOp"
_MAX_POOL3D_INDICES_OP_NAME = "MaxPool3dIndicesFwdOp"


def _avg_pool1d_bench_params() -> list:
    params = []
    for workload in load_workloads(_AVG_POOL1D_OP_NAME):
        n, c_in, l_in = workload["input_shape"]
        kernel_size = workload["kernel_size"]
        stride = workload.get("stride")
        padding = workload.get("padding", 0)
        ceil_mode = workload.get("ceil_mode", False)
        count_include_pad = workload.get("count_include_pad", True)
        label = workload.get("label", f"{n}x{c_in}x{l_in}")
        for dtype_str in workload["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    n,
                    c_in,
                    l_in,
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad,
                    dtype,
                    True,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


class AvgPool1dBenchmarkWorkload(AvgPool1dBenchCase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool1d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
        )


class AvgPool2dBenchmarkWorkload(AvgPool2dBenchCase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )


class AvgPool3dBenchmarkWorkload(AvgPool3dBenchCase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )


class MaxPool2dBenchmarkWorkload(MaxPool2dBenchCase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class MaxPool1dBenchmarkWorkload(MaxPool1dBenchCase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return F.max_pool1d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class MaxPool3dBenchmarkWorkload(MaxPool3dBenchCase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return F.max_pool3d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


@pytest.mark.parametrize(
    "n, c_in, l_in, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype, tune",
    _avg_pool1d_bench_params(),
)
def test_avg_pool1d_bench(
    n: int,
    c_in: int,
    l_in: int,
    kernel_size: int,
    stride: Optional[int],
    padding: int,
    ceil_mode: bool,
    count_include_pad: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AvgPool1dBenchmarkWorkload(
        n, c_in, l_in, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype
    )
    inputs = test.gen_inputs()

    op = AvgPool1dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        tune=tune,
    )
    bm = ManifestBenchmark(_AVG_POOL1D_OP_NAME, op, test)

    _tag, _baseline_fn = pool_baseline(type(op).__name__, test, *inputs)
    # torch stays alongside the library baseline: it is what the nightly's ratio alert and
    # its history were measured against, and both numbers belong in the same row.
    bm.compare(
        {
            "tileops": op,
            _tag: _baseline_fn,
            "torch-ref": test.ref_program,
            "torch-compile": compiled_reference(test),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )


def _avg_pool2d_bench_params() -> list:
    params = []
    for workload in load_workloads(_AVG_POOL2D_OP_NAME):
        n, c_in, h_in, w_in = workload["input_shape"]
        kernel_size = tuple(workload["kernel_size"])
        stride = workload.get("stride")
        if stride is not None:
            stride = tuple(stride)
        padding = tuple(workload.get("padding", (0, 0)))
        ceil_mode = workload.get("ceil_mode", False)
        count_include_pad = workload.get("count_include_pad", True)
        divisor_override = workload.get("divisor_override")
        label = workload.get("label", f"{n}x{c_in}x{h_in}x{w_in}")
        for dtype_str in workload["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    n,
                    c_in,
                    h_in,
                    w_in,
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad,
                    divisor_override,
                    dtype,
                    True,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


@pytest.mark.parametrize(
    "n, c_in, h_in, w_in, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dtype, tune",
    _avg_pool2d_bench_params(),
)
def test_avg_pool2d_bench(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int],
    stride: Optional[tuple[int, int]],
    padding: tuple[int, int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AvgPool2dBenchmarkWorkload(
        n,
        c_in,
        h_in,
        w_in,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dtype,
    )
    inputs = test.gen_inputs()

    op = AvgPool2dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        tune=tune,
    )
    bm = ManifestBenchmark(_AVG_POOL2D_OP_NAME, op, test)

    _tag, _baseline_fn = pool_baseline(type(op).__name__, test, *inputs)
    bm.compare(
        {
            "tileops": op,
            _tag: _baseline_fn,
            "torch-ref": test.ref_program,
            "torch-compile": compiled_reference(test),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )


def _avg_pool3d_bench_params() -> list:
    params = []
    for workload in load_workloads(_AVG_POOL3D_OP_NAME):
        n, c_in, d_in, h_in, w_in = workload["input_shape"]
        kernel_size = tuple(workload["kernel_size"])
        stride = workload.get("stride")
        if stride is not None:
            stride = tuple(stride)
        padding = tuple(workload.get("padding", (0, 0, 0)))
        ceil_mode = workload.get("ceil_mode", False)
        count_include_pad = workload.get("count_include_pad", True)
        divisor_override = workload.get("divisor_override")
        label = workload.get("label", f"{n}x{c_in}x{d_in}x{h_in}x{w_in}")
        for dtype_str in workload["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    n,
                    c_in,
                    d_in,
                    h_in,
                    w_in,
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad,
                    divisor_override,
                    dtype,
                    True,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


@pytest.mark.parametrize(
    "n, c_in, d_in, h_in, w_in, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dtype, tune",
    _avg_pool3d_bench_params(),
)
def test_avg_pool3d_bench(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int, int],
    stride: Optional[tuple[int, int, int]],
    padding: tuple[int, int, int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AvgPool3dBenchmarkWorkload(
        n,
        c_in,
        d_in,
        h_in,
        w_in,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dtype,
    )
    inputs = test.gen_inputs()

    op = AvgPool3dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        tune=tune,
    )
    bm = ManifestBenchmark(_AVG_POOL3D_OP_NAME, op, test)

    _tag, _baseline_fn = pool_baseline(type(op).__name__, test, *inputs)
    bm.compare(
        {
            "tileops": op,
            _tag: _baseline_fn,
            "torch-ref": test.ref_program,
            "torch-compile": compiled_reference(test),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )


def _max_pool2d_bench_params_from_workloads(workloads: list[dict]) -> list:
    params = []
    for workload in workloads:
        n, c_in, h_in, w_in = workload["input_shape"]
        kernel_size = tuple(workload["kernel_size"])
        stride = workload.get("stride")
        if stride is not None:
            stride = tuple(stride)
        padding = tuple(workload.get("padding", (0, 0)))
        dilation = tuple(workload.get("dilation", (1, 1)))
        ceil_mode = workload.get("ceil_mode", False)
        label = workload.get("label", f"{n}x{c_in}x{h_in}x{w_in}")
        for dtype_str in workload["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    n,
                    c_in,
                    h_in,
                    w_in,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    dtype,
                    True,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


def _max_pool2d_bench_params() -> list:
    return _max_pool2d_bench_params_from_workloads(load_workloads(_MAX_POOL2D_OP_NAME))


def _max_pool2d_indices_bench_params() -> list:
    return _max_pool2d_bench_params_from_workloads(load_workloads(_MAX_POOL2D_INDICES_OP_NAME))


@pytest.mark.parametrize(
    "n, c_in, h_in, w_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune",
    _max_pool2d_bench_params(),
)
def test_max_pool2d_bench(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int],
    stride: Optional[tuple[int, int]],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MaxPool2dBenchmarkWorkload(
        n,
        c_in,
        h_in,
        w_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
    )
    inputs = test.gen_inputs()

    op = MaxPool2dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        tune=tune,
    )
    bm = ManifestBenchmark(_MAX_POOL2D_OP_NAME, op, test)

    _tag, _baseline_fn = pool_baseline(type(op).__name__, test, *inputs)
    bm.compare(
        {
            "tileops": op,
            _tag: _baseline_fn,
            "torch-ref": test.ref_program,
            "torch-compile": compiled_reference(test),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )


@pytest.mark.parametrize(
    "n, c_in, h_in, w_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune",
    _max_pool2d_indices_bench_params(),
)
def test_max_pool2d_indices_bench(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int],
    stride: Optional[tuple[int, int]],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MaxPool2dBenchmarkWorkload(
        n,
        c_in,
        h_in,
        w_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
        return_indices=True,
    )
    inputs = test.gen_inputs()

    op = MaxPool2dIndicesFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        tune=tune,
    )
    bm = ManifestBenchmark(_MAX_POOL2D_INDICES_OP_NAME, op, test)

    _tag, _baseline_fn = pool_baseline(type(op).__name__, test, *inputs)
    bm.compare(
        {
            "tileops": op,
            _tag: _baseline_fn,
            "torch-ref": test.ref_program,
            "torch-compile": compiled_reference(test),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )


def _max_pool1d_bench_params_from_workloads(workloads: list[dict]) -> list:
    params = []
    for workload in workloads:
        n, c_in, l_in = workload["input_shape"]
        kernel_size = workload["kernel_size"]
        kernel_size = tuple(kernel_size) if isinstance(kernel_size, list) else (kernel_size,)
        stride = workload.get("stride")
        if stride is not None:
            stride = tuple(stride) if isinstance(stride, list) else (stride,)
        padding = workload.get("padding", 0)
        padding = tuple(padding) if isinstance(padding, list) else (padding,)
        dilation = workload.get("dilation", 1)
        dilation = tuple(dilation) if isinstance(dilation, list) else (dilation,)
        ceil_mode = workload.get("ceil_mode", False)
        label = workload.get("label", f"{n}x{c_in}x{l_in}")
        for dtype_str in workload["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    n,
                    c_in,
                    l_in,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    dtype,
                    True,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


def _max_pool1d_bench_params() -> list:
    return _max_pool1d_bench_params_from_workloads(load_workloads(_MAX_POOL1D_OP_NAME))


def _max_pool1d_indices_bench_params() -> list:
    return _max_pool1d_bench_params_from_workloads(load_workloads(_MAX_POOL1D_INDICES_OP_NAME))


@pytest.mark.parametrize(
    "n, c_in, l_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune",
    _max_pool1d_bench_params(),
)
def test_max_pool1d_bench(
    n: int,
    c_in: int,
    l_in: int,
    kernel_size: tuple[int],
    stride: Optional[tuple[int]],
    padding: tuple[int],
    dilation: tuple[int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MaxPool1dBenchmarkWorkload(
        n,
        c_in,
        l_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
    )
    inputs = test.gen_inputs()

    op = MaxPool1dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        tune=tune,
    )
    bm = ManifestBenchmark(_MAX_POOL1D_OP_NAME, op, test)

    _tag, _baseline_fn = pool_baseline(type(op).__name__, test, *inputs)
    bm.compare(
        {
            "tileops": op,
            _tag: _baseline_fn,
            "torch-ref": test.ref_program,
            "torch-compile": compiled_reference(test),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )


@pytest.mark.parametrize(
    "n, c_in, l_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune",
    _max_pool1d_indices_bench_params(),
)
def test_max_pool1d_indices_bench(
    n: int,
    c_in: int,
    l_in: int,
    kernel_size: tuple[int],
    stride: Optional[tuple[int]],
    padding: tuple[int],
    dilation: tuple[int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MaxPool1dBenchmarkWorkload(
        n,
        c_in,
        l_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
        return_indices=True,
    )
    inputs = test.gen_inputs()

    op = MaxPool1dIndicesFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        tune=tune,
    )
    bm = ManifestBenchmark(_MAX_POOL1D_INDICES_OP_NAME, op, test)

    _tag, _baseline_fn = pool_baseline(type(op).__name__, test, *inputs)
    bm.compare(
        {
            "tileops": op,
            _tag: _baseline_fn,
            "torch-ref": test.ref_program,
            "torch-compile": compiled_reference(test),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )


def _max_pool3d_bench_params_from_workloads(workloads: list[dict]) -> list:
    params = []
    for workload in workloads:
        n, c_in, d_in, h_in, w_in = workload["input_shape"]
        kernel_size = workload["kernel_size"]
        kernel_size = tuple(kernel_size) if isinstance(kernel_size, list) else (kernel_size,) * 3
        stride = workload.get("stride")
        if stride is not None:
            stride = tuple(stride) if isinstance(stride, list) else (stride,) * 3
        padding = workload.get("padding", 0)
        padding = tuple(padding) if isinstance(padding, list) else (padding,) * 3
        dilation = workload.get("dilation", 1)
        dilation = tuple(dilation) if isinstance(dilation, list) else (dilation,) * 3
        ceil_mode = workload.get("ceil_mode", False)
        label = workload.get("label", f"{n}x{c_in}x{d_in}x{h_in}x{w_in}")
        for dtype_str in workload["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    n,
                    c_in,
                    d_in,
                    h_in,
                    w_in,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    dtype,
                    True,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


def _max_pool3d_bench_params() -> list:
    return _max_pool3d_bench_params_from_workloads(load_workloads(_MAX_POOL3D_OP_NAME))


def _max_pool3d_indices_bench_params() -> list:
    return _max_pool3d_bench_params_from_workloads(load_workloads(_MAX_POOL3D_INDICES_OP_NAME))


@pytest.mark.parametrize(
    "n, c_in, d_in, h_in, w_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune",
    _max_pool3d_bench_params(),
)
def test_max_pool3d_bench(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int, int],
    stride: Optional[tuple[int, int, int]],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MaxPool3dBenchmarkWorkload(
        n,
        c_in,
        d_in,
        h_in,
        w_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
    )
    inputs = test.gen_inputs()

    op = MaxPool3dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        tune=tune,
    )
    bm = ManifestBenchmark(_MAX_POOL3D_OP_NAME, op, test)

    _tag, _baseline_fn = pool_baseline(type(op).__name__, test, *inputs)
    bm.compare(
        {
            "tileops": op,
            _tag: _baseline_fn,
            "torch-ref": test.ref_program,
            "torch-compile": compiled_reference(test),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )


@pytest.mark.parametrize(
    "n, c_in, d_in, h_in, w_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune",
    _max_pool3d_indices_bench_params(),
)
def test_max_pool3d_indices_bench(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int, int],
    stride: Optional[tuple[int, int, int]],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MaxPool3dBenchmarkWorkload(
        n,
        c_in,
        d_in,
        h_in,
        w_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
        return_indices=True,
    )
    inputs = test.gen_inputs()

    op = MaxPool3dIndicesFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        tune=tune,
    )
    bm = ManifestBenchmark(_MAX_POOL3D_INDICES_OP_NAME, op, test)

    _tag, _baseline_fn = pool_baseline(type(op).__name__, test, *inputs)
    bm.compare(
        {
            "tileops": op,
            _tag: _baseline_fn,
            "torch-ref": test.ref_program,
            "torch-compile": compiled_reference(test),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )


class AdaptiveAvgPool2dBenchmarkWorkload(AdaptivePool2dWorkload):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        # torch rejects a scalar None here; (None, None) means the same.
        size = (None, None) if self.output_size is None else self.output_size
        return F.adaptive_avg_pool2d(x, size)


class AdaptiveMaxPool2dBenchmarkWorkload(AdaptivePool2dWorkload):
    def __init__(self, *args, return_indices: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.return_indices = return_indices

    def ref_program(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # torch rejects a scalar None here; (None, None) means the same.
        size = (None, None) if self.output_size is None else self.output_size
        return F.adaptive_max_pool2d(x, size, return_indices=self.return_indices)


def _adaptive_pool2d_bench_params_from_workloads(workloads) -> list:
    params = []
    for workload in workloads:
        n, c_in, h_in, w_in = workload["input_shape"]
        output_size = tuple(workload["output_size"])
        label = workload.get("label", f"{n}x{c_in}x{h_in}x{w_in}")
        for dtype_str in workload["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    n,
                    c_in,
                    h_in,
                    w_in,
                    output_size,
                    dtype,
                    True,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


@pytest.mark.parametrize(
    "n, c_in, h_in, w_in, output_size, dtype, tune",
    _adaptive_pool2d_bench_params_from_workloads(load_workloads(_ADAPTIVE_AVG_POOL2D_OP_NAME)),
)
def test_adaptive_avg_pool2d_bench(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    output_size: tuple[int, int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AdaptiveAvgPool2dBenchmarkWorkload(n, c_in, h_in, w_in, output_size, dtype)
    inputs = test.gen_inputs()

    op = AdaptiveAvgPool2dFwdOp(output_size=output_size, tune=tune)
    bm = ManifestBenchmark(_ADAPTIVE_AVG_POOL2D_OP_NAME, op, test)

    _tag, _baseline_fn = pool_baseline(type(op).__name__, test, *inputs)
    bm.compare(
        {
            "tileops": op,
            _tag: _baseline_fn,
            "torch-ref": test.ref_program,
            "torch-compile": compiled_reference(test),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )


@pytest.mark.parametrize(
    "n, c_in, h_in, w_in, output_size, dtype, tune",
    _adaptive_pool2d_bench_params_from_workloads(load_workloads(_ADAPTIVE_MAX_POOL2D_OP_NAME)),
)
def test_adaptive_max_pool2d_bench(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    output_size: tuple[int, int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AdaptiveMaxPool2dBenchmarkWorkload(n, c_in, h_in, w_in, output_size, dtype)
    inputs = test.gen_inputs()

    op = AdaptiveMaxPool2dFwdOp(output_size=output_size, tune=tune)
    bm = ManifestBenchmark(_ADAPTIVE_MAX_POOL2D_OP_NAME, op, test)

    _tag, _baseline_fn = pool_baseline(type(op).__name__, test, *inputs)
    bm.compare(
        {
            "tileops": op,
            _tag: _baseline_fn,
            "torch-ref": test.ref_program,
            "torch-compile": compiled_reference(test),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )


@pytest.mark.parametrize(
    "n, c_in, h_in, w_in, output_size, dtype, tune",
    _adaptive_pool2d_bench_params_from_workloads(
        load_workloads(_ADAPTIVE_MAX_POOL2D_INDICES_OP_NAME)
    ),
)
def test_adaptive_max_pool2d_indices_bench(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    output_size: tuple[int, int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AdaptiveMaxPool2dBenchmarkWorkload(
        n, c_in, h_in, w_in, output_size, dtype, return_indices=True
    )
    inputs = test.gen_inputs()

    op = AdaptiveMaxPool2dIndicesFwdOp(output_size=output_size, tune=tune)
    bm = ManifestBenchmark(_ADAPTIVE_MAX_POOL2D_INDICES_OP_NAME, op, test)

    _tag, _baseline_fn = pool_baseline(type(op).__name__, test, *inputs)
    bm.compare(
        {
            "tileops": op,
            _tag: _baseline_fn,
            "torch-ref": test.ref_program,
            "torch-compile": compiled_reference(test),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )
