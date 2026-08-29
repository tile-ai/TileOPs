import functools
from typing import ClassVar, Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.linear_attention.deltanet_call import DeltaNetDecodeCall
from tileops.kernels.linear_attention.gated_deltanet import (
    GatedDeltaNetBwdKernel,
    GatedDeltaNetFwdKernel,
    GatedDeltaNetFwdProductionKernel,
    GatedDeltaNetPrefillFwdKernel,
)
from tileops.kernels.linear_attention.gated_deltanet_recurrence import (
    GatedDeltaNetDecodeFP32Kernel,
    GatedDeltaNetDecodeKernel,
    GatedDeltaNetDecodeRawCudaFlaStyleKernel,
)
from tileops.perf.profile import tensor_core_roof

from .._validation import check_tensor_shape
from ..op_base import Op, UnmanifestedOp

__all__ = [
    "GatedDeltaNetBHTDFwdOp",
    "GatedDeltaNetBTHDFwdOp",
    "GatedDeltaNetBwdOp",
    "GatedDeltaNetDecodeFwdOp",
    "GatedDeltaNetOp",
    "GatedDeltaNetPrefillBHTDFwdOp",
    "GatedDeltaNetPrefillBTHDFwdOp",
]

# Implementations of the gated DeltaNet decode slot.
GATED_DELTANET_DECODE_KEYS = (
    "GatedDeltaNetDecodeFP32Kernel",
    "GatedDeltaNetDecodeRawCudaFlaStyleKernel",
    "GatedDeltaNetDecodeKernel",
)


def _resolve_gated_bhsd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    do: Optional[torch.Tensor] = None,
) -> tuple[int, int, int, int, int, torch.dtype]:
    if not all(tensor.is_cuda for tensor in (q, k, v, g, beta)):
        raise ValueError("q, k, v, g, and beta must be CUDA tensors")
    if q.ndim != 4:
        raise ValueError("q must have shape [batch, heads, seq_len, dim_k]")
    batch, heads, seq_len, dim_k = q.shape
    if k.shape != (batch, heads, seq_len, dim_k):
        raise ValueError("k must match q shape")
    if v.ndim != 4 or v.shape[:3] != (batch, heads, seq_len):
        raise ValueError("v must have shape [batch, heads, seq_len, dim_v]")
    dim_v = v.shape[-1]
    if g.shape != (batch, heads, seq_len):
        raise ValueError("g must have shape [batch, heads, seq_len]")
    if beta.shape != (batch, heads, seq_len):
        raise ValueError("beta must have shape [batch, heads, seq_len]")
    if do is not None and do.shape != (batch, heads, seq_len, dim_v):
        raise ValueError("do must have shape [batch, heads, seq_len, dim_v]")
    dtype = q.dtype
    for name, tensor in (("k", k), ("v", v), ("g", g), ("beta", beta)):
        if tensor.dtype != dtype:
            raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")
    if do is not None and do.dtype != dtype:
        raise ValueError(f"do.dtype must be {dtype}, got {do.dtype}")
    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})")
    return batch, heads, seq_len, dim_k, dim_v, dtype


def _resolve_gated_bthd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int, int, int, int, torch.dtype]:
    if not all(tensor.is_cuda for tensor in (q, k, v, g, beta)):
        raise ValueError("q, k, v, g, and beta must be CUDA tensors")
    if q.ndim != 4:
        raise ValueError("q must have shape [batch, seq_len, heads, dim_k]")
    batch, seq_len, heads, dim_k = q.shape
    if k.shape != (batch, seq_len, heads, dim_k):
        raise ValueError("k must match q shape")
    if v.ndim != 4 or v.shape[:3] != (batch, seq_len, heads):
        raise ValueError("v must have shape [batch, seq_len, heads, dim_v]")
    dim_v = v.shape[-1]
    if g.shape != (batch, seq_len, heads):
        raise ValueError("g must have shape [batch, seq_len, heads]")
    if beta.shape != (batch, seq_len, heads):
        raise ValueError("beta must have shape [batch, seq_len, heads]")
    dtype = q.dtype
    for name, tensor in (("k", k), ("v", v), ("g", g), ("beta", beta)):
        if tensor.dtype != dtype:
            raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")
    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})")
    return batch, heads, seq_len, dim_k, dim_v, dtype


@functools.lru_cache(maxsize=8)
def _device_capability(device_index: "int | None") -> tuple[int, int]:
    """Compute capability of *device_index*. A device property, so read it once."""
    return torch.cuda.get_device_capability(device_index)


def _bthd_production_gaps(
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    device_index: "int | None",
) -> list[str]:
    """Requirements of the BTHD production pipeline this call does not meet.

    Each entry names one requirement and the value that failed it, so a refusal
    says which one to change.
    """
    gaps = []
    if chunk_size != 64:
        gaps.append(f"chunk_size must be 64, got {chunk_size}")
    if dim_k != dim_v:
        gaps.append(f"dim_k and dim_v must be equal, got {dim_k} and {dim_v}")
    if dim_k not in (64, 128):
        gaps.append(f"dim_k must be 64 or 128, got {dim_k}")
    if dtype not in (torch.float16, torch.bfloat16):
        gaps.append(f"dtype must be float16 or bfloat16, got {dtype}")
    major, minor = _device_capability(device_index)
    if major != 9:
        gaps.append(
            f"the warp-specialized kernel needs Hopper (compute capability 9.x), "
            f"got {major}.{minor}"
        )
    return gaps


class GatedDeltaNetBHTDFwdOp(Op):
    """Gated DeltaNet forward operator.

    Pipeline: prepare_wy_repr(k, g, beta) -> (Aw, Au) -> gated_deltanet_fwd(q, k, v, g, beta, Aw, Au) -> o.

    Head-major (BHTD) inputs: ``q/k [B, H, S, DK]``, ``v [B, H, S, DV]``,
    ``g/beta [B, H, S]``. ``GatedDeltaNetBwdOp`` consumes the ``S`` this returns,
    in the same layout. Token-major callers want ``GatedDeltaNetBTHDFwdOp``.

    """

    def __init__(
        self,
        chunk_size: int = 64,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            chunk_size: Chunk size for chunked linear attention.
            kernel_map: Optional kernel overrides.
            tune: Whether to autotune kernels.
        """
        self.batch = None
        self.heads = None
        self.seq_len = None
        self.dim_k = None
        self.dim_v = None
        self.chunk_size = chunk_size
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"GatedDeltaNetFwdKernel": GatedDeltaNetFwdKernel}

    def eval_roofline(self) -> tuple[int, int]:
        from tileops.perf.formulas import gated_deltanet_fwd_roofline

        return gated_deltanet_fwd_roofline(self)

    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        kernel_name = "GatedDeltaNetFwdKernel"
        key = (
            batch,
            heads,
            seq_len,
            self.chunk_size,
            dim_k,
            dim_v,
            dtype,
            device_index,
            self.tune,
        )
        return self.get_or_build_kernel(
            kernel_name,
            inputs,
            key=key,
            build=lambda: self.kernel_map[kernel_name](
                batch,
                heads,
                seq_len,
                self.chunk_size,
                dim_k,
                dim_v,
                dtype=Kernel.dtype_to_str(dtype),
                tune=self.tune,
            ),
        )

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        g_shape: tuple[int, ...],
        beta_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: the output, the per-chunk state, and the two chunk buffers."""
        b, h, s, dk = q_shape
        dv = v_shape[3]
        return {
            "o": (b, h, s, dv),
            "S": (b, h, s // self.chunk_size + 1, dk, dv),
            "Aw": (b, h, s, self.chunk_size),
            "Au": (b, h, s, self.chunk_size),
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Run gated deltanet forward.

        Args:
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            g: Gate tensor [B, H, S].
            beta: Beta tensor [B, H, S].

        Returns:
            Tuple of (o, S, Aw, Au).
        """
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        batch, heads, seq_len, dim_k, dim_v, dtype = _resolve_gated_bhsd(
            q, k, v, g, beta, self.chunk_size
        )
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.kernel = self._get_kernel(
            (q, k, v, g, beta), batch, heads, seq_len, dim_k, dim_v, dtype, q.device.index
        )
        o, S, Aw, Au = self.kernel(q, k, v, g, beta)
        return o, S, Aw, Au

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


class GatedDeltaNetBTHDFwdOp(Op):
    """Gated DeltaNet forward over token-major (BTHD) inputs.

    Same operator as ``GatedDeltaNetBHTDFwdOp`` and the same four outputs, over the
    token-major memory order the FLA reference uses: ``q/k [B, S, H, DK]``,
    ``v [B, S, H, DV]``, ``g/beta [B, S, H]``. A separate entry because the
    memory order is part of the signature, not a mode of one signature.

    It runs the warp-specialized production pipeline, so it serves Hopper with
    ``chunk_size=64``, equal K/V dimensions in {64, 128}, and float16 or
    bfloat16. Any other call is refused, naming what it failed.

    """

    def __init__(
        self,
        chunk_size: int = 64,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            chunk_size: Chunk size for chunked linear attention.
            kernel_map: Optional kernel overrides.
            tune: Whether to autotune kernels.
        """
        self.batch = None
        self.heads = None
        self.seq_len = None
        self.dim_k = None
        self.dim_v = None
        self.chunk_size = chunk_size
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"GatedDeltaNetFwdProductionKernel": GatedDeltaNetFwdProductionKernel}

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        """Manifest ``dtype``: every input carries q's dtype, which must be half."""
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"Expected q.dtype float16 or bfloat16, got {q.dtype}")
        for name, tensor in (("k", k), ("v", v), ("g", g), ("beta", beta)):
            if tensor.dtype != q.dtype:
                raise ValueError(f"Expected {name}.dtype {q.dtype}, got {tensor.dtype}")
        self.dtype = q.dtype

    def _infer_output_shapes(
        self,
        q_shape: Tuple[int, ...],
        k_shape: Tuple[int, ...],
        v_shape: Tuple[int, ...],
        g_shape: Tuple[int, ...],
        beta_shape: Tuple[int, ...],
    ) -> Dict[str, Tuple[int, ...]]:
        """Manifest ``shape_rules`` for the four outputs, in token-major order."""
        batch, seq_len, heads, _ = q_shape
        dim_k, dim_v = q_shape[3], v_shape[3]
        num_chunks = seq_len // self.chunk_size
        return {
            "o": (batch, seq_len, heads, dim_v),
            "S": (batch, heads, num_chunks + 1, dim_k, dim_v),
            "Aw": (batch, seq_len, heads, self.chunk_size),
            "Au": (batch, seq_len, heads, self.chunk_size),
        }

    def eval_roofline(self) -> tuple[int, int]:
        from tileops.perf.formulas import gated_deltanet_fwd_roofline

        return gated_deltanet_fwd_roofline(self)

    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        gaps = _bthd_production_gaps(self.chunk_size, dim_k, dim_v, dtype, device_index)
        if gaps:
            raise ValueError(
                "BTHD GatedDeltaNet forward has no kernel for this call: " + "; ".join(gaps)
            )
        key = (batch, heads, seq_len, self.chunk_size, dim_k, dim_v, dtype, device_index, self.tune)
        return self.get_or_build_kernel(
            "GatedDeltaNetFwdProductionKernel",
            inputs,
            key=key,
            build=lambda: self.kernel_map["GatedDeltaNetFwdProductionKernel"](
                batch,
                heads,
                seq_len,
                self.chunk_size,
                dim_k,
                dim_v,
                dtype=Kernel.dtype_to_str(dtype),
                tune=self.tune,
            ),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Run the token-major forward.

        Args:
            q: Query tensor [B, S, H, DK].
            k: Key tensor [B, S, H, DK].
            v: Value tensor [B, S, H, DV].
            g: Gate tensor [B, S, H].
            beta: Beta tensor [B, S, H].

        Returns:
            Tuple of (o, S, Aw, Au).
        """
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        batch, heads, seq_len, dim_k, dim_v, dtype = _resolve_gated_bthd(
            q, k, v, g, beta, self.chunk_size
        )
        self._validate_dtypes(q, k, v, g, beta)
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.kernel = self._get_kernel(
            (q, k, v, g, beta), batch, heads, seq_len, dim_k, dim_v, dtype, q.device.index
        )
        o, S, Aw, Au = self.kernel(q, k, v, g, beta)
        return o, S, Aw, Au

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


class GatedDeltaNetPrefillBTHDFwdOp(Op):
    """Gated DeltaNet inference prefill operator.

    This is the serving-oriented zero-state prefill interface:
    ``(q, k, v, g, beta) -> (o, final_state)``. It intentionally does not
    expose backward-only training artifacts such as ``Aw`` and ``Au``.
    Token-major (BTHD) inputs, the FLA/Qwen convention: ``q/k/v/o [B, T, H, D]``,
    ``g/beta [B, T, H]``. Head-major callers want ``GatedDeltaNetPrefillBHTDFwdOp``.
    When ``chunk_size`` is not specified, the op uses a small-stream serving
    default: 128 for ``batch * heads <= 8`` when the sequence length allows it,
    otherwise 64.
    """

    # The memory order this op takes. One entry declares one order: the order changes
    # what an axis means, so an op serving two layouts is two entries.
    LAYOUT: ClassVar[str] = "bthd"

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            chunk_size: Manifest ``params.chunk_size``, ``int | None``, default ``None``.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.batch = None
        self.heads = None
        self.seq_len = None
        self.dim_k = None
        self.dim_v = None
        self._requested_chunk_size = chunk_size
        self.chunk_size = chunk_size
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._active_sig: Optional[tuple] = None
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GatedDeltaNetPrefillFwdKernel": GatedDeltaNetPrefillFwdKernel,
        }

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        g_shape: tuple[int, ...],
        beta_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        del k_shape, g_shape, beta_shape
        layout = self.LAYOUT
        if layout == "bthd":
            return {
                "o": (q_shape[0], q_shape[1], q_shape[2], v_shape[-1]),
                "final_state": (
                    q_shape[0],
                    q_shape[2],
                    q_shape[-1],
                    v_shape[-1],
                ),
            }
        return {
            "o": tuple(q_shape[:-1]) + (v_shape[-1],),
            "final_state": (
                q_shape[0],
                q_shape[1],
                q_shape[-1],
                v_shape[-1],
            ),
        }

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        dtype = q.dtype
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported dtype: {dtype}")
        for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
            if tensor.dtype != dtype:
                raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")

    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch: int,
        heads: int,
        seq_len: int,
        chunk_size: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (
            batch,
            heads,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype,
            self.LAYOUT,
            device_index,
            self.tune,
        )
        return self.get_or_build_kernel(
            "GatedDeltaNetPrefillFwdKernel",
            inputs,
            key=key,
            build=lambda: self.kernel_map["GatedDeltaNetPrefillFwdKernel"](
                batch,
                heads,
                seq_len,
                chunk_size,
                dim_k,
                dim_v,
                dtype=Kernel.dtype_to_str(dtype),
                layout=self.LAYOUT,
                tune=self.tune,
            ),
        )

    def _validate_shapes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        if self.LAYOUT == "bthd":
            if q.ndim != 4:
                raise ValueError("q must have shape [batch, seq_len, heads, dim_k]")
            batch, seq_len, heads, dim_k = q.shape
            q_shape = (batch, seq_len, heads, dim_k)
            v_shape = (batch, seq_len, heads, v.shape[-1])
            gate_shape = (batch, seq_len, heads)
        else:
            if q.ndim != 4:
                raise ValueError("q must have shape [batch, heads, seq_len, dim_k]")
            batch, heads, seq_len, dim_k = q.shape
            q_shape = (batch, heads, seq_len, dim_k)
            v_shape = (batch, heads, seq_len, v.shape[-1])
            gate_shape = (batch, heads, seq_len)
        if tuple(q.shape) != q_shape:
            raise ValueError(f"q must have shape {q_shape}, got {tuple(q.shape)}")
        if tuple(k.shape) != q_shape:
            raise ValueError(f"k must have shape {q_shape}, got {tuple(k.shape)}")
        if tuple(v.shape) != v_shape:
            raise ValueError(f"v must have shape {v_shape}, got {tuple(v.shape)}")
        if tuple(g.shape) != gate_shape:
            raise ValueError(f"g must have shape {gate_shape}, got {tuple(g.shape)}")
        if tuple(beta.shape) != gate_shape:
            raise ValueError(f"beta must have shape {gate_shape}, got {tuple(beta.shape)}")
        if not all(tensor.is_cuda for tensor in (q, k, v, g, beta)):
            raise ValueError("q, k, v, g, and beta must be CUDA tensors")
        chunk_size = self._requested_chunk_size
        if chunk_size is None:
            streams = batch * heads
            chunk_size = 128 if streams <= 8 and seq_len % 128 == 0 else 64
        if seq_len % chunk_size != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})")
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = v.shape[-1]
        self.chunk_size = chunk_size
        self.dtype = q.dtype
        self.kernel = self._get_kernel(
            (q, k, v, g, beta),
            batch,
            heads,
            seq_len,
            chunk_size,
            dim_k,
            self.dim_v,
            q.dtype,
            q.device.index,
        )

    def eval_roofline(self) -> tuple[int, int]:
        from tileops.perf.formulas import gated_deltanet_prefill_fwd_roofline

        return gated_deltanet_prefill_fwd_roofline(self)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            q: Input tensor, dtype ``float16 | bfloat16 | float32``.
            k: Input tensor, dtype ``same_as(q)``.
            v: Input tensor, dtype ``same_as(q)``.
            g: Input tensor, dtype ``same_as(q)``.
            beta: Input tensor, dtype ``same_as(q)``.

        Returns:
            ``o``, ``final_state``, as the manifest declares. Shape rules: ``final_state.shape == (B, H, DK, DV)``.
        """
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        sig = (
            q.shape,
            k.shape,
            v.shape,
            g.shape,
            beta.shape,
            q.dtype,
            k.dtype,
            v.dtype,
            g.dtype,
            beta.dtype,
            q.device,
            k.device,
            v.device,
            g.device,
            beta.device,
            self.LAYOUT,
            self._requested_chunk_size,
            getattr(self, "tune", None),
        )
        if sig != getattr(self, "_active_sig", None):
            self._validate_dtypes(q, k, v, g, beta)
            self._validate_shapes(q, k, v, g, beta)
            self._active_sig = sig
        return self.kernel(q, k, v, g, beta)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


class GatedDeltaNetPrefillBHTDFwdOp(GatedDeltaNetPrefillBTHDFwdOp):
    """Gated DeltaNet inference prefill over head-major (BHTD) inputs.

    ``q/k/v/o [B, H, T, D]``, ``g/beta [B, H, T]`` — the TileOps convention.
    Same kernel and same arithmetic as ``GatedDeltaNetPrefillBTHDFwdOp``; only the
    memory order the tensors arrive in differs, and memory order is part of the
    signature, so it is its own entry.
    """

    LAYOUT: ClassVar[str] = "bhtd"


class GatedDeltaNetBwdOp(Op):
    """Gated DeltaNet backward operator.

    Pipeline: prepare_wy_repr -> fwd (to get Aw, Au) -> bwd kernel -> (dq, dk, dv, dg, dbeta).

    """

    def __init__(
        self,
        chunk_size: int = 64,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            chunk_size: Chunk size for chunked linear attention.
            kernel_map: Optional kernel overrides.
            tune: Whether to autotune kernels.
        """
        self.batch = None
        self.heads = None
        self.seq_len = None
        self.dim_k = None
        self.dim_v = None
        self.chunk_size = chunk_size
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GatedDeltaNetBwdKernel": GatedDeltaNetBwdKernel,
        }

    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (batch, heads, seq_len, self.chunk_size, dim_k, dim_v, dtype, device_index, self.tune)
        return self.get_or_build_kernel(
            "GatedDeltaNetBwdKernel",
            inputs,
            key=key,
            build=lambda: self.kernel_map["GatedDeltaNetBwdKernel"](
                batch,
                heads,
                seq_len,
                self.chunk_size,
                dim_k,
                dim_v,
                dtype=Kernel.dtype_to_str(dtype),
                tune=self.tune,
            ),
        )

    def _infer_output_shapes(
        self,
        do_shape: tuple[int, ...],
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        g_shape: tuple[int, ...],
        beta_shape: tuple[int, ...],
        S_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: each gradient has the shape of what it is for."""
        return {
            "dq": tuple(q_shape),
            "dk": tuple(k_shape),
            "dv": tuple(v_shape),
            "dg": tuple(g_shape),
            "dbeta": tuple(beta_shape),
        }

    def forward(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        S: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run gated deltanet backward.

        Args:
            do: Gradient of output [B, H, S, DV].
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            g: Gate tensor [B, H, S].
            beta: Beta tensor [B, H, S].
            S: Per-chunk boundary states from forward [B, H, NC+1, DK, DV].

        Returns:
            Tuple of (dq, dk, dv, dg, dbeta).
        """
        batch, heads, seq_len, dim_k, dim_v, dtype = _resolve_gated_bhsd(
            q, k, v, g, beta, self.chunk_size, do=do
        )
        self._validate_dtypes(do, q, k, v, g, beta, S)
        check_tensor_shape("S", S, (batch, heads, seq_len // self.chunk_size + 1, dim_k, dim_v))
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.kernel = self._get_kernel(
            (do, q, k, v, g, beta, S), batch, heads, seq_len, dim_k, dim_v, dtype, q.device.index
        )
        dq, dk, dv, dg, dbeta = self.kernel(do, q, k, v, g, beta, S)
        return dq, dk, dv, dg, dbeta

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


class _GatedDeltaNetFunction(torch.autograd.Function):
    """Autograd function wrapping TileOPs fwd + bwd kernels."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, fwd_kernel, bwd_kernel):
        """Run the op on ``q``, ``k``, ``v``, ``g``, ``beta``, ``fwd_kernel`` and ``bwd_kernel``."""
        o, S, Aw, Au = fwd_kernel(q, k, v, g, beta)
        ctx.save_for_backward(q, k, v, g, beta, S)
        ctx.bwd_kernel = bwd_kernel
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, S = ctx.saved_tensors
        dq, dk, dv, dg, dbeta = ctx.bwd_kernel(do, q, k, v, g, beta, S)
        return dq, dk, dv, dg, dbeta, None, None


class GatedDeltaNetOp(UnmanifestedOp):
    """Combined Gated DeltaNet fwd+bwd operator with autograd support.

    Wraps ``GatedDeltaNetFwdKernel`` and ``GatedDeltaNetBwdKernel`` in a
    ``torch.autograd.Function`` so that ``output.backward(do)`` automatically
    invokes the TileOPs backward kernels.

    This makes end-to-end benchmarking against FLA straightforward::

        op = GatedDeltaNetOp(chunk_size=chunk_size)
        o = op(q, k, v, g, beta)   # forward
        o.backward(do)              # backward via TileOPs kernels

    Layout: BHSD (batch, head, seq_len, dim).

    """

    def __init__(
        self,
        chunk_size: int = 64,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            chunk_size: Chunk size for chunked linear attention.
            kernel_map: Optional kernel overrides.
            tune: Whether to autotune kernels.
        """
        self.batch = None
        self.heads = None
        self.seq_len = None
        self.dim_k = None
        self.dim_v = None
        self.chunk_size = chunk_size
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GatedDeltaNetFwdKernel": GatedDeltaNetFwdKernel,
            "GatedDeltaNetBwdKernel": GatedDeltaNetBwdKernel,
        }

    def _bind_from_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[Kernel, Kernel]:
        batch, heads, seq_len, dim_k, dim_v, dtype = _resolve_gated_bhsd(
            q, k, v, g, beta, self.chunk_size
        )
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        key = (
            batch,
            heads,
            seq_len,
            self.chunk_size,
            dim_k,
            dim_v,
            dtype,
            q.device.index,
            self.tune,
        )
        return self.get_or_build_kernel(
            "GatedDeltaNetFwdKernel",
            (q, k, v, g, beta),
            key=key,
            build=lambda: (
                self.kernel_map["GatedDeltaNetFwdKernel"](
                    batch,
                    heads,
                    seq_len,
                    self.chunk_size,
                    dim_k,
                    dim_v,
                    dtype=Kernel.dtype_to_str(dtype),
                    tune=self.tune,
                ),
                self.kernel_map["GatedDeltaNetBwdKernel"](
                    batch,
                    heads,
                    seq_len,
                    self.chunk_size,
                    dim_k,
                    dim_v,
                    dtype=Kernel.dtype_to_str(dtype),
                    tune=self.tune,
                ),
            ),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Run gated deltanet forward with autograd backward support.

        Args:
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            g: Gate tensor [B, H, S].
            beta: Beta tensor [B, H, S].

        Returns:
            Output tensor o [B, H, S, DV] (supports .backward()).
        """
        fwd_kernel, bwd_kernel = self._bind_from_inputs(q, k, v, g, beta)
        return _GatedDeltaNetFunction.apply(q, k, v, g, beta, fwd_kernel, bwd_kernel)


class GatedDeltaNetDecodeFwdOp(Op):
    """Gated DeltaNet decode (single-step recurrence).

    Computes one step of the gated delta rule:
        S_t = S_{t-1} (alpha_t (I - beta_t k_t k_t^T)) + beta_t v_t k_t^T
        o_t = S_t q_t

    Layout: BHD (batch, head, dim).
    Supports float32, float16, and bfloat16 with fp32 accumulation.

    For fp32 dtype, dispatches to a dedicated FP32 kernel that uses
    element-wise matvec instead of T.gemm to avoid TF32 mantissa truncation.
    """

    def __init__(
        self,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.batch = None
        self.heads = None
        self.dim_k = None
        self.dim_v = None
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._active_sig: Optional[tuple] = None
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GatedDeltaNetDecodeKernel": GatedDeltaNetDecodeKernel,
            "GatedDeltaNetDecodeFP32Kernel": GatedDeltaNetDecodeFP32Kernel,
            "GatedDeltaNetDecodeRawCudaFlaStyleKernel": GatedDeltaNetDecodeRawCudaFlaStyleKernel,
        }

    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (batch, heads, dim_k, dim_v, dtype, device_index, self.tune)
        call = DeltaNetDecodeCall(
            batch=batch, heads=heads, dim_k=dim_k, dim_v=dim_v, dtype=dtype, tune=self.tune
        )
        chosen = self.select_kernel_key(GATED_DELTANET_DECODE_KEYS, call)

        def build() -> Kernel:
            return self.kernel_map[chosen](
                batch,
                heads,
                dim_k,
                dim_v,
                dtype=Kernel.dtype_to_str(dtype),
                tune=self.tune,
            )

        return self.get_or_build_kernel(chosen, inputs, key=key, build=build)

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        g_shape: tuple[int, ...],
        beta_shape: tuple[int, ...],
        state_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        del k_shape, g_shape, beta_shape
        return {
            "o": (q_shape[0], q_shape[1], v_shape[-1]),
            "new_state": state_shape,
        }

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> None:
        dtype = q.dtype
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported dtype: {dtype}")
        for name, tensor in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("g", g),
            ("beta", beta),
            ("state", state),
        ):
            if tensor.dtype != dtype:
                raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")

    def _validate_shapes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> None:
        if q.ndim != 3:
            raise ValueError("q must have shape [batch, heads, dim_k]")
        batch, heads, dim_k = q.shape
        if v.ndim != 3 or v.shape[:2] != (batch, heads):
            raise ValueError("v must have shape [batch, heads, dim_v]")
        dim_v = v.shape[2]
        q_shape = (batch, heads, dim_k)
        v_shape = (batch, heads, dim_v)
        gate_shape = (batch, heads)
        state_shape = (batch, heads, dim_k, dim_v)
        expected_shapes = (
            ("q", q, q_shape),
            ("k", k, q_shape),
            ("v", v, v_shape),
            ("g", g, gate_shape),
            ("beta", beta, gate_shape),
            ("state", state, state_shape),
        )
        for name, tensor, expected in expected_shapes:
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        if not all(tensor.is_cuda for tensor in (q, k, v, g, beta, state)):
            raise ValueError("q, k, v, g, beta, and state must be CUDA tensors")
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = q.dtype
        self.kernel = self._get_kernel(
            (q, k, v, g, beta, state), batch, heads, dim_k, dim_v, q.dtype, q.device.index
        )

    def _validate_output_shapes(
        self,
        o: torch.Tensor,
        new_state: torch.Tensor,
    ) -> None:
        o_shape = (self.batch, self.heads, self.dim_v)
        state_shape = (self.batch, self.heads, self.dim_k, self.dim_v)
        if tuple(o.shape) != o_shape:
            raise ValueError(f"o must have shape {o_shape}, got {tuple(o.shape)}")
        if tuple(new_state.shape) != state_shape:
            raise ValueError(
                f"new_state must have shape {state_shape}, got {tuple(new_state.shape)}"
            )

    def eval_roofline(self) -> tuple[int, int]:
        from tileops.perf.formulas import gated_deltanet_decode_roofline

        return gated_deltanet_decode_roofline(self)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            q: Input tensor, dtype ``float16 | bfloat16 | float32``.
            k: Input tensor, dtype ``same_as(q)``.
            v: Input tensor, dtype ``same_as(q)``.
            g: Input tensor, dtype ``same_as(q)``.
            beta: Input tensor, dtype ``same_as(q)``.
            state: Input tensor, dtype ``same_as(q)``.

        Returns:
            ``o``, ``new_state``, as the manifest declares. Shape rules: ``o.shape == (B, H, DV)``; ``new_state.shape == (B, H, DK, DV)``.
        """
        sig = (
            q.shape,
            k.shape,
            v.shape,
            g.shape,
            beta.shape,
            state.shape,
            q.dtype,
            k.dtype,
            v.dtype,
            g.dtype,
            beta.dtype,
            state.dtype,
            q.device,
            k.device,
            v.device,
            g.device,
            beta.device,
            state.device,
            getattr(self, "tune", None),
        )
        if sig != getattr(self, "_active_sig", None):
            self._validate_dtypes(q, k, v, g, beta, state)
            self._validate_shapes(q, k, v, g, beta, state)
            self._active_sig = sig
        o, new_state = self.kernel(q, k, v, g, beta, state)
        self._validate_output_shapes(o, new_state)
        return o, new_state
