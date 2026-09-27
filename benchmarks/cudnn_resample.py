"""Pooling through cuDNN's v9 backend Resample node, called with ctypes.

nvidia-cudnn-frontend 1.27 exposes 141 graph ops and none of them is resample, and the
legacy cudnnPoolingForward rejects bfloat16. The legacy entry point reaches the same
kernel as the graph API, so neither is faster than the other.
"""

import atexit
import ctypes
import os
from enum import IntEnum
from typing import Callable, Optional

import torch

from tileops.kernels.pool.common import pool_output_dim

# Each enum below mirrors the cudnn_graph.h type named beside it, value for value.


class _Status(IntEnum):  # cudnnStatus_t
    SUCCESS = 0
    NOT_SUPPORTED = 3000


class _DataType(IntEnum):  # cudnnDataType_t; HALF=2/BF16=9, unlike the legacy API
    FLOAT = 0
    HALF = 2
    BFLOAT16 = 9


_DATA_TYPES = {
    torch.float32: _DataType.FLOAT,
    torch.float16: _DataType.HALF,
    torch.bfloat16: _DataType.BFLOAT16,
}


class _ResampleMode(IntEnum):  # cudnnResampleMode_t
    AVGPOOL_INCLUDE_PADDING = 2
    MAXPOOL = 3
    AVGPOOL_EXCLUDE_PADDING = 4


class _PaddingMode(IntEnum):  # cudnnPaddingMode_t
    ZERO_PAD = 0
    NEG_INF_PAD = 1


class _NanPropagation(IntEnum):  # cudnnNanPropagation_t
    PROPAGATE_NAN = 1


class _HeurMode(IntEnum):  # cudnnBackendHeurMode_t
    INSTANT = 0
    FALLBACK = 2
    A = 3


class _AttrType(IntEnum):  # cudnnBackendAttributeType_t
    HANDLE = 0
    DATA_TYPE = 1
    INT64 = 3
    VOID_PTR = 6
    HEUR_MODE = 8
    NAN = 10
    BACKEND_DESCRIPTOR = 15
    RESAMPLE_MODE = 21
    PADDING_MODE = 22
    FRACTION = 26


class _DescType(IntEnum):  # cudnnBackendDescriptorType_t
    ENGINECFG = 3
    ENGINEHEUR = 4
    PLAN = 5
    OPGRAPH = 15
    VARPACK = 16
    TENSOR = 17
    RESAMPLE = 24
    OP_RESAMPLE_FWD = 25


class _Attr(IntEnum):  # cudnnBackendAttributeName_t
    ENGINEHEUR_MODE = 200
    ENGINEHEUR_OPGRAPH = 201
    ENGINEHEUR_RESULTS = 202
    PLAN_HANDLE = 400
    PLAN_ENGINECFG = 401
    PLAN_WORKSPACE_SIZE = 402
    OPGRAPH_HANDLE = 800
    OPGRAPH_OPS = 801
    TENSOR_BYTE_ALIGNMENT = 900
    TENSOR_DATA_TYPE = 901
    TENSOR_DIMENSIONS = 902
    TENSOR_STRIDES = 903
    TENSOR_UNIQUE_ID = 906
    VARPACK_UIDS = 1000
    VARPACK_PTRS = 1001
    VARPACK_WORKSPACE = 1003
    RESAMPLE_MODE = 1700
    RESAMPLE_COMP_TYPE = 1701
    RESAMPLE_SPATIAL_DIMS = 1702
    RESAMPLE_POST_PAD = 1703
    RESAMPLE_PRE_PAD = 1704
    RESAMPLE_STRIDES = 1705
    RESAMPLE_WINDOW = 1706
    RESAMPLE_NAN = 1707
    RESAMPLE_PADDING_MODE = 1708
    OP_RESAMPLE_XDESC = 1710
    OP_RESAMPLE_YDESC = 1711
    OP_RESAMPLE_DESC = 1716


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
        if status != _Status.SUCCESS:
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
        if atype == _AttrType.INT64:
            arr = (ctypes.c_int64 * len(values))(*values)
        elif atype == _AttrType.FRACTION:
            arr = (_Fraction * len(values))(*[(v, 1) for v in values])
        elif atype in (_AttrType.VOID_PTR, _AttrType.BACKEND_DESCRIPTOR, _AttrType.HANDLE):
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
                desc, attr, _AttrType.INT64, 1, ctypes.byref(got), ctypes.byref(value)
            ),
            what,
        )
        return value.value

    def finalize(self, desc, what: str) -> bool:
        """Finalize; False (not raise) when the descriptor is simply unsupported."""
        status = self._lib.cudnnBackendFinalize(desc)
        if status == _Status.SUCCESS:
            return True
        if status == _Status.NOT_SUPPORTED:
            return False
        raise RuntimeError(f"cuDNN call failed: finalize {what} (status {status})")

    def finalize_or_raise(self, desc, what: str) -> None:
        self._check(self._lib.cudnnBackendFinalize(desc), f"finalize {what}")

    def tensor_desc(self, uid: int, dtype: int, dims: tuple, strides: tuple) -> ctypes.c_void_p:
        desc = self.create(_DescType.TENSOR)
        self.set(desc, _Attr.TENSOR_UNIQUE_ID, _AttrType.INT64, [uid], "uid")
        self.set(desc, _Attr.TENSOR_DATA_TYPE, _AttrType.DATA_TYPE, [dtype], "dtype")
        self.set(desc, _Attr.TENSOR_DIMENSIONS, _AttrType.INT64, list(dims), "dims")
        self.set(desc, _Attr.TENSOR_STRIDES, _AttrType.INT64, list(strides), "strides")
        # Required by the v9 backend; torch's allocator guarantees 256 B.
        self.set(desc, _Attr.TENSOR_BYTE_ALIGNMENT, _AttrType.INT64, [16], "align")
        self.finalize_or_raise(desc, "tensor")
        return desc

    def build_plan(self, op_graph):
        """Ask the heuristics for engine configs; build the first viable plan.

        On cuDNN 9.20 the A/INSTANT modes report a nonzero config count but hand
        back zero descriptors, so modes are tried in order and FALLBACK (the
        generic catch-all engine list) is what actually yields configs.
        """
        for mode in (_HeurMode.A, _HeurMode.INSTANT, _HeurMode.FALLBACK):
            heur = self.create(_DescType.ENGINEHEUR)
            self.set(
                heur, _Attr.ENGINEHEUR_OPGRAPH, _AttrType.BACKEND_DESCRIPTOR, [op_graph], "heur op"
            )
            self.set(heur, _Attr.ENGINEHEUR_MODE, _AttrType.HEUR_MODE, [mode], "heur mode")
            self.finalize_or_raise(heur, "heur")

            cfgs = [self.create(_DescType.ENGINECFG, keep=False) for _ in range(16)]
            arr = (ctypes.c_void_p * len(cfgs))(*[c.value for c in cfgs])
            got = ctypes.c_int64(0)
            self._check(
                self._lib.cudnnBackendGetAttribute(
                    heur,
                    _Attr.ENGINEHEUR_RESULTS,
                    _AttrType.BACKEND_DESCRIPTOR,
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
                plan = self.create(_DescType.PLAN)
                self.set(
                    plan, _Attr.PLAN_HANDLE, _AttrType.HANDLE, [self.handle.value], "plan handle"
                )
                self.set(
                    plan, _Attr.PLAN_ENGINECFG, _AttrType.BACKEND_DESCRIPTOR, [cfg], "plan cfg"
                )
                if not self.finalize(plan, "plan"):
                    self.destroy(plan)
                    self.destroy(cfg)
                    continue
                self._owned.append(cfg)
                return plan, self.get_int64(plan, _Attr.PLAN_WORKSPACE_SIZE, "workspace")
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
        mode, pad_mode = _ResampleMode.MAXPOOL, _PaddingMode.NEG_INF_PAD
    elif count_include_pad:
        mode, pad_mode = _ResampleMode.AVGPOOL_INCLUDE_PADDING, _PaddingMode.ZERO_PAD
    else:
        mode, pad_mode = _ResampleMode.AVGPOOL_EXCLUDE_PADDING, _PaddingMode.ZERO_PAD

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
        resample = c.create(_DescType.RESAMPLE)
        c.set(resample, _Attr.RESAMPLE_MODE, _AttrType.RESAMPLE_MODE, [mode], "mode")
        c.set(resample, _Attr.RESAMPLE_COMP_TYPE, _AttrType.DATA_TYPE, [_DataType.FLOAT], "comp")
        # Propagating costs a fifth of the kernel, and torch propagates: the cheaper
        # mode returns a number where torch returns NaN.
        c.set(resample, _Attr.RESAMPLE_NAN, _AttrType.NAN, [_NanPropagation.PROPAGATE_NAN], "nan")
        c.set(resample, _Attr.RESAMPLE_PADDING_MODE, _AttrType.PADDING_MODE, [pad_mode], "padmode")
        c.set(resample, _Attr.RESAMPLE_SPATIAL_DIMS, _AttrType.INT64, [ndim], "spatial")
        c.set(resample, _Attr.RESAMPLE_WINDOW, _AttrType.FRACTION, list(kernel_size), "window")
        c.set(resample, _Attr.RESAMPLE_STRIDES, _AttrType.FRACTION, list(stride), "strides")
        c.set(resample, _Attr.RESAMPLE_PRE_PAD, _AttrType.FRACTION, list(padding), "pre pad")
        c.set(resample, _Attr.RESAMPLE_POST_PAD, _AttrType.FRACTION, post, "post pad")
        c.finalize_or_raise(resample, "resample")

        operation = c.create(_DescType.OP_RESAMPLE_FWD)
        c.set(operation, _Attr.OP_RESAMPLE_XDESC, _AttrType.BACKEND_DESCRIPTOR, [x_desc], "x")
        c.set(operation, _Attr.OP_RESAMPLE_YDESC, _AttrType.BACKEND_DESCRIPTOR, [y_desc], "y")
        c.set(operation, _Attr.OP_RESAMPLE_DESC, _AttrType.BACKEND_DESCRIPTOR, [resample], "rdesc")
        c.finalize_or_raise(operation, "operation")

        op_graph = c.create(_DescType.OPGRAPH)
        c.set(op_graph, _Attr.OPGRAPH_HANDLE, _AttrType.HANDLE, [c.handle.value], "g handle")
        c.set(op_graph, _Attr.OPGRAPH_OPS, _AttrType.BACKEND_DESCRIPTOR, [operation], "ops")
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
            entry = _build(x, out_spatial, _DATA_TYPES[x.dtype])
            state[key] = entry
        out_shape, plan, workspace = entry
        y = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        varpack = c.create(_DescType.VARPACK, keep=False)
        c.set(varpack, _Attr.VARPACK_UIDS, _AttrType.INT64, [0, 1], "uids")
        c.set(
            varpack,
            _Attr.VARPACK_PTRS,
            _AttrType.VOID_PTR,
            [x.data_ptr(), y.data_ptr()],
            "ptrs",
        )
        if workspace is not None:
            c.set(
                varpack, _Attr.VARPACK_WORKSPACE, _AttrType.VOID_PTR, [workspace.data_ptr()], "ws"
            )
        c.finalize_or_raise(varpack, "varpack")
        c.execute(plan, varpack)
        c.destroy(varpack)
        return y

    return run
