from typing import ClassVar, Dict, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel, adapt_entry
from tileops.kernels.linear_attention.deltanet import (
    DeltaNetBwdKernel,
    DeltaNetFwdKernel,
)
from tileops.perf.profile import tensor_core_roof

from .._compile_boundary_codegen import OperatorSpec
from .._validation import check_tensor_shape
from ..op_base import Op

__all__ = ["DeltaNetBwdOp", "DeltaNetFwdOp", "DeltaNetAutogradOp"]


class DeltaNetFwdOp(Op):
    """DeltaNet forward operator (ungated).

    Pipeline: prepare_wy_repr(k, beta) -> (Aw, Au) -> deltanet_fwd(q, k, v, beta, Aw, Au) -> o.

    Layout: BHSD (batch, head, seq_len, dim).

    !!! note "Layout convention difference with FLA"

        TileOPs uses **BHSD** layout: ``q/k [B, H, S, DK]``, ``v [B, H, S, DV]``,
        ``beta [B, H, S]``.

        FLA (``fla.ops.delta_rule.chunk_delta_rule``) uses **BTHN**
        layout: ``q/k [B, T, H, K]``, ``v [B, T, H, V]``, ``beta [B, T, H]``.

    """

    compile_boundary: ClassVar[tuple[OperatorSpec, ...]] = (OperatorSpec(),)

    def __init__(
        self,
        chunk_size: int = 64,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            chunk_size: Chunk size for chunked linear attention.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
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

        self.target = target
        self.dispatch_kernel(kernel_map)
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "DeltaNetFwdKernel": DeltaNetFwdKernel,
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
        return self.kernel_for("DeltaNetFwdKernel", inputs, key)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, chunk length, dtype and device."""
        batch, heads, seq_len, chunk_size, dim_k, dim_v, dtype, _device, tune = call
        return call, lambda: self.kernel_map["DeltaNetFwdKernel"](
            batch,
            heads,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype=Kernel.dtype_to_str(dtype),
            tune=tune,
        )

    def _bind_from_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> None:
        if not all(tensor.is_cuda for tensor in (q, k, v, beta)):
            raise ValueError("q, k, v, and beta must be CUDA tensors")
        if q.ndim != 4:
            raise ValueError("q must have shape [batch, heads, seq_len, dim_k]")
        batch, heads, seq_len, dim_k = q.shape
        if k.shape != (batch, heads, seq_len, dim_k):
            raise ValueError("k must match q shape")
        if v.ndim != 4 or v.shape[:3] != (batch, heads, seq_len):
            raise ValueError("v must have shape [batch, heads, seq_len, dim_v]")
        if beta.shape != (batch, heads, seq_len):
            raise ValueError("beta must have shape [batch, heads, seq_len]")
        self._validate_dtypes(q, k, v, beta)
        dtype = q.dtype
        if seq_len % self.chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by chunk_size ({self.chunk_size})"
            )

        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = v.shape[-1]
        self.dtype = dtype
        self.kernel = self._get_kernel(
            (q, k, v, beta), batch, heads, seq_len, dim_k, self.dim_v, dtype, q.device.index
        )

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        beta_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: the output, the per-chunk state, and the four chunk buffers."""
        b, h, s, dk = q_shape
        dv = v_shape[3]
        chunks = s // self.chunk_size
        return {
            "o": (b, h, s, dv),
            "S": (b, h, chunks + 1, dk, dv),
            "Aw": (b, h, s, self.chunk_size),
            "Au": (b, h, s, self.chunk_size),
            "w": (b, h, s, dk),
            "u": (b, h, s, dv),
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Run deltanet forward.

        Args:
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            beta: Beta tensor [B, H, S].

        Returns:
            Tuple of (o, S, Aw, Au, w, u).
        """
        return self._wrapped(q, k, v, beta, self._instance_key)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        self._bind_from_inputs(q, k, v, beta)
        o, S, Aw, Au, w, u = self.kernel(q, k, v, beta)
        return o, S, Aw, Au, w, u

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


class DeltaNetBwdOp(Op):
    """DeltaNet backward operator (ungated).

    Pipeline: prepare_wy_repr -> fwd (to get Aw, Au) -> bwd kernel -> (dq, dk, dv, dbeta).

    """

    compile_boundary: ClassVar[tuple[OperatorSpec, ...]] = (OperatorSpec(),)

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
            "DeltaNetBwdKernel": DeltaNetBwdKernel,
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
        return self.kernel_for("DeltaNetBwdKernel", inputs, key)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, chunk length, dtype and device."""
        batch, heads, seq_len, chunk_size, dim_k, dim_v, dtype, _device, tune = call
        return call, lambda: self.kernel_map["DeltaNetBwdKernel"](
            batch,
            heads,
            seq_len,
            chunk_size,
            dim_k,
            dim_v,
            dtype=Kernel.dtype_to_str(dtype),
            tune=tune,
        )

    def _bind_from_inputs(self, inputs: "tuple[torch.Tensor, ...]") -> None:
        """Validate the ten declared inputs, then bind the kernel they select."""
        do, q, k, v, beta, S, Aw, Au, w, u = inputs
        if not all(tensor.is_cuda for tensor in (do, q, k, v, beta)):
            raise ValueError("do, q, k, v, and beta must be CUDA tensors")
        if q.ndim != 4:
            raise ValueError("q must have shape [batch, heads, seq_len, dim_k]")
        batch, heads, seq_len, dim_k = q.shape
        if k.shape != (batch, heads, seq_len, dim_k):
            raise ValueError("k must match q shape")
        if v.ndim != 4 or v.shape[:3] != (batch, heads, seq_len):
            raise ValueError("v must have shape [batch, heads, seq_len, dim_v]")
        dim_v = v.shape[-1]
        if do.shape != (batch, heads, seq_len, dim_v):
            raise ValueError("do must have shape [batch, heads, seq_len, dim_v]")
        if beta.shape != (batch, heads, seq_len):
            raise ValueError("beta must have shape [batch, heads, seq_len]")
        self._validate_dtypes(*inputs)
        dtype = q.dtype
        if seq_len % self.chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by chunk_size ({self.chunk_size})"
            )
        chunk = self.chunk_size
        check_tensor_shape("S", S, (batch, heads, seq_len // chunk + 1, dim_k, dim_v))
        check_tensor_shape("Aw", Aw, (batch, heads, seq_len, chunk))
        check_tensor_shape("Au", Au, (batch, heads, seq_len, chunk))
        check_tensor_shape("w", w, (batch, heads, seq_len, dim_k))
        check_tensor_shape("u", u, (batch, heads, seq_len, dim_v))

        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.kernel = self._get_kernel(
            inputs, batch, heads, seq_len, dim_k, dim_v, dtype, q.device.index
        )

    def _infer_output_shapes(
        self,
        do_shape: tuple[int, ...],
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        beta_shape: tuple[int, ...],
        S_shape: tuple[int, ...],
        Aw_shape: tuple[int, ...],
        Au_shape: tuple[int, ...],
        w_shape: tuple[int, ...],
        u_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: each gradient has the shape of what it is for."""
        return {
            "dq": tuple(q_shape),
            "dk": tuple(k_shape),
            "dv": tuple(v_shape),
            "dbeta": tuple(beta_shape),
        }

    def forward(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        S: torch.Tensor,
        Aw: torch.Tensor,
        Au: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run deltanet backward.

        Args:
            do: Gradient of output [B, H, S, DV].
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            beta: Beta tensor [B, H, S].
            S: Per-chunk boundary states from forward [B, H, NC+1, DK, DV].
            Aw: A_inv matrix from forward [B, H, S, BC].
            Au: A_inv matrix from forward [B, H, S, BC].
            w: WY w vectors from forward [B, H, S, DK].
            u: WY u vectors from forward [B, H, S, DV].

        Returns:
            Tuple of (dq, dk, dv, dbeta).
        """
        return self._wrapped(do, q, k, v, beta, S, Aw, Au, w, u, self._instance_key)

    def _eager_forward(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        S: torch.Tensor,
        Aw: torch.Tensor,
        Au: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        self._bind_from_inputs((do, q, k, v, beta, S, Aw, Au, w, u))
        dq, dk, dv, dbeta = self.kernel(do, q, k, v, beta, S, Aw, Au, w, u)
        return dq, dk, dv, dbeta

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


class _DeltaNetFunction(torch.autograd.Function):
    """Autograd function wrapping TileOPs fwd + bwd kernels."""

    @staticmethod
    def forward(ctx, q, k, v, beta, fwd_kernel, bwd_kernel):
        """Run the op on ``q``, ``k``, ``v``, ``beta``, ``fwd_kernel`` and ``bwd_kernel``."""
        o, S, Aw, Au, w, u = fwd_kernel(q, k, v, beta)
        ctx.save_for_backward(q, k, v, beta, S, Aw, Au, w, u)
        ctx.bwd_kernel = bwd_kernel
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, beta, S, Aw, Au, w, u = ctx.saved_tensors
        dq, dk, dv, dbeta = ctx.bwd_kernel(do, q, k, v, beta, S, Aw, Au, w, u)
        return dq, dk, dv, dbeta, None, None


class DeltaNetAutogradOp(Op):
    """Combined DeltaNet fwd+bwd operator with autograd support (ungated).

    Wraps ``DeltaNetFwdKernel`` and ``DeltaNetBwdKernel`` in a
    ``torch.autograd.Function`` so that ``output.backward(do)`` automatically
    invokes the TileOPs backward kernels.

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
            "DeltaNetFwdKernel": DeltaNetFwdKernel,
            "DeltaNetBwdKernel": DeltaNetBwdKernel,
        }

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        beta_shape: tuple[int, ...],
    ) -> Dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: only ``o``; the chunk buffers stay in the autograd context."""
        return {"o": tuple(v_shape)}

    def _bind_from_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[Kernel, Kernel]:
        if not all(tensor.is_cuda for tensor in (q, k, v, beta)):
            raise ValueError("q, k, v, and beta must be CUDA tensors")
        batch, heads, seq_len, dim_k = q.shape
        if k.shape != (batch, heads, seq_len, dim_k):
            raise ValueError("k must match q shape")
        if v.ndim != 4 or v.shape[:3] != (batch, heads, seq_len):
            raise ValueError("v must have shape [batch, heads, seq_len, dim_v]")
        if beta.shape != (batch, heads, seq_len):
            raise ValueError("beta must have shape [batch, heads, seq_len]")
        dtype = q.dtype
        for name, tensor in (("k", k), ("v", v), ("beta", beta)):
            if tensor.dtype != dtype:
                raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")
        if seq_len % self.chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by chunk_size ({self.chunk_size})"
            )
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = v.shape[-1]
        self.dtype = dtype

        key = (
            batch,
            heads,
            seq_len,
            self.chunk_size,
            dim_k,
            self.dim_v,
            dtype,
            q.device.index,
            self.tune,
        )
        return self.kernel_for("deltanet", (q, k, v, beta), key)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """Forward and backward are built together, so they are one entry."""
        batch, heads, seq_len, chunk_size, dim_k, dim_v, dtype, _device, tune = call

        def build() -> tuple:
            args = (batch, heads, seq_len, chunk_size, dim_k, dim_v)
            kwargs = {"dtype": Kernel.dtype_to_str(dtype), "tune": tune}
            return (
                self.kernel_map["DeltaNetFwdKernel"](*args, **kwargs),
                self.kernel_map["DeltaNetBwdKernel"](*args, **kwargs),
            )

        return adapt_entry((call, build), _with_in_tree_backward)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Run deltanet forward with autograd backward support.

        Args:
            q: Query tensor [B, H, S, DK].
            k: Key tensor [B, H, S, DK].
            v: Value tensor [B, H, S, DV].
            beta: Beta tensor [B, H, S].

        Returns:
            Output tensor o [B, H, S, DV] (supports .backward()).
        """
        self._validate_dtypes(q, k, v, beta)
        # A target's kernel returns an output that carries its own backward; the in-tree
        # entry attaches the in-tree one.
        return self._bind_from_inputs(q, k, v, beta)(q, k, v, beta)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


def _with_in_tree_backward(kernels, q, k, v, beta):
    """Run the in-tree forward kernel under an autograd function whose backward is the in-tree one."""
    fwd_kernel, bwd_kernel = kernels
    return _DeltaNetFunction.apply(q, k, v, beta, fwd_kernel, bwd_kernel)
