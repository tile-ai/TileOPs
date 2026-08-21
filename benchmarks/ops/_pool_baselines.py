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
        """SetAttribute; *values* is a list of python ints/floats/pointers."""
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
        cast, out_shape, plan, workspace = entry
        x_in = x.float() if cast else x
        y = torch.empty(out_shape, device=x.device, dtype=torch.float32 if cast else x.dtype)
        varpack = c.create(_DESC_VARPACK, keep=False)
        c.set(varpack, _ATTR_VARPACK_UIDS, _TYPE_INT64, [0, 1], "uids")
        c.set(
            varpack,
            _ATTR_VARPACK_PTRS,
            _TYPE_VOID_PTR,
            [x_in.data_ptr(), y.data_ptr()],
            "ptrs",
        )
        if workspace is not None:
            c.set(varpack, _ATTR_VARPACK_WORKSPACE, _TYPE_VOID_PTR, [workspace.data_ptr()], "ws")
        c.finalize_or_raise(varpack, "varpack")
        c.execute(plan, varpack)
        c.destroy(varpack)
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


# Per-op baseline: the strongest implementation measured on this hardware, with the
# pooling kind and rank the adapters need. An op absent here keeps the torch reference —
# 1D native kernels beat both libraries' 2D-emulated paths, and neither library covers
# adaptive pooling or 3D max-pool indices.
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


def pool_baseline(op_name: str, test) -> tuple:
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
    return choice, fn
