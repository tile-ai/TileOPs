import math
from typing import Dict, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention import GatedDeltaNetDensePrefillFwdKernel
from tileops.perf.formulas import gated_deltanet_fwd_roofline
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op

__all__ = ["GatedDeltaNetFwdOp"]


class GatedDeltaNetFwdOp(Op):
    """Inference forward for the gated delta rule.

    ``q`` and ``k`` use ``[B, T, H, K]``. ``v``, ``g``, ``beta`` and the
    output use ``HV`` recurrent heads, where ``HV`` is a multiple of ``H``.
    The recurrent state is always FP32. It is key-major ``[N, HV, K, V]`` by
    default and value-major ``[N, HV, V, K]`` when ``state_v_first=True``.
    For equal-length inputs, ``N == B``. Passing ``cu_seqlens`` selects packed
    varlen mode: the token tensors have ``B == 1`` and ``N`` is the number of
    packed sequences.

    By default, ``g`` contains the precomputed log-space decay and ``beta``
    contains the post-sigmoid update strength. ``use_gate_in_kernel=True``
    instead treats ``g`` as the raw gate and requires ``A_log`` and
    ``dt_bias``; ``use_beta_sigmoid_in_kernel=True`` treats ``beta`` as raw
    logits. With ``allow_neg_eigval=True``, the beta transform is
    ``2 * sigmoid(beta)`` rather than ``sigmoid(beta)``.

    One interface covers prefill and decode, and always returns
    ``(o, final_state)``. This intentionally pins FLA's
    ``output_final_state=True`` because inference callers need the state to
    continue recurrence. The target callable selects the execution path from
    the current inputs; in particular, ``T == 1`` is decode rather than a
    separate public Op.

    The in-tree implementation currently covers equal-length Hopper prefill
    with zero initial state, matching recurrent head counts, 128-wide state,
    and precomputed gate and beta values. Other regions still require an
    external target implementation while their retained kernels are migrated.
    """

    def __init__(
        self,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        use_gate_in_kernel: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        *,
        target: Target = None,
    ) -> None:
        """Fix recurrence semantics; tensor metadata comes from each call.

        Args:
            scale: Query scale, or ``None`` for ``K**-0.5``.
            use_qk_l2norm_in_kernel: Normalize Q and K internally.
            use_beta_sigmoid_in_kernel: Treat ``beta`` as raw logits and apply
                sigmoid internally.
            allow_neg_eigval: Compute ``2 * sigmoid(beta)`` instead of
                ``sigmoid(beta)``. Valid only with
                ``use_beta_sigmoid_in_kernel=True``.
            state_v_first: Use value-major recurrent state layout
                ``[N, HV, V, K]`` instead of key-major ``[N, HV, K, V]``.
            use_gate_in_kernel: Treat ``g`` as a raw gate and internally
                compute ``-exp(A_log) * softplus(g + dt_bias)``. Otherwise,
                ``g`` must already contain the log-space decay.
            kernel_map: Optional in-tree kernel overrides.
            target: Backend target, or ``None`` to resolve from the input
                device.
        """
        if scale is not None and not math.isfinite(scale):
            raise ValueError(f"scale must be finite, got {scale}")
        if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
            raise ValueError("allow_neg_eigval requires use_beta_sigmoid_in_kernel=True")

        self.scale = scale
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        self.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        self.allow_neg_eigval = allow_neg_eigval
        self.state_v_first = state_v_first
        self.use_gate_in_kernel = use_gate_in_kernel
        self.target = target
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gated_deltanet_dense_prefill": GatedDeltaNetDensePrefillFwdKernel}

    def entry_for(self, role: str, call: tuple) -> Entry:
        """Build the one migrated in-tree specialization for this call."""
        del role
        (
            batch,
            seq_len,
            heads,
            value_heads,
            dim_k,
            dim_v,
            dtype,
            device_index,
            scale,
            has_initial_state,
            has_cu_seqlens,
        ) = call
        unsupported = []
        if has_initial_state:
            unsupported.append("initial_state")
        if has_cu_seqlens:
            unsupported.append("packed varlen")
        if self.state_v_first:
            unsupported.append("state_v_first=True")
        if self.use_qk_l2norm_in_kernel:
            unsupported.append("use_qk_l2norm_in_kernel=True")
        if self.use_gate_in_kernel:
            unsupported.append("use_gate_in_kernel=True")
        if self.use_beta_sigmoid_in_kernel:
            unsupported.append("use_beta_sigmoid_in_kernel=True")
        if value_heads != heads:
            unsupported.append("HV != H")
        if dim_k != 128 or dim_v != 128:
            unsupported.append("K or V != 128")
        if seq_len < 64 or seq_len % 64 != 0:
            unsupported.append("T is not a positive multiple of 64")
        if unsupported:
            raise ValueError(
                "the in-tree GatedDeltaNet dense-prefill kernel does not yet support "
                + ", ".join(unsupported)
            )
        return call, lambda: self.kernel_map["gated_deltanet_dense_prefill"](
            batch=batch,
            heads=heads,
            seq_len=seq_len,
            dim=dim_k,
            scale=scale,
            dtype=dtype,
            device_index=device_index,
        )

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        g_shape: tuple[int, ...],
        beta_shape: tuple[int, ...],
        initial_state_shape: Optional[tuple[int, ...]] = None,
        cu_seqlens_shape: Optional[tuple[int, ...]] = None,
        cu_seqlens_cpu_shape: Optional[tuple[int, ...]] = None,
        A_log_shape: Optional[tuple[int, ...]] = None,
        dt_bias_shape: Optional[tuple[int, ...]] = None,
    ) -> dict[str, tuple[int, ...]]:
        del (
            k_shape,
            g_shape,
            beta_shape,
            initial_state_shape,
            cu_seqlens_cpu_shape,
            A_log_shape,
            dt_bias_shape,
        )
        batch, seq_len, _heads, dim_k = q_shape
        _batch, _seq_len, value_heads, dim_v = v_shape
        state_batch = cu_seqlens_shape[0] - 1 if cu_seqlens_shape is not None else batch
        state_tail = (dim_v, dim_k) if self.state_v_first else (dim_k, dim_v)
        return {
            "o": (batch, seq_len, value_heads, dim_v),
            "final_state": (state_batch, value_heads, *state_tail),
        }

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> None:
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("q must have float16 or bfloat16 dtype")
        for name, tensor in (("k", k), ("v", v), ("g", g), ("beta", beta)):
            if tensor.dtype != q.dtype:
                raise ValueError(f"{name} must have the same dtype as q")
        for name, tensor in (
            ("initial_state", initial_state),
            ("A_log", A_log),
            ("dt_bias", dt_bias),
        ):
            if tensor is not None and tensor.dtype != torch.float32:
                raise ValueError(f"{name} must have float32 dtype")
        for name, tensor in (("cu_seqlens", cu_seqlens), ("cu_seqlens_cpu", cu_seqlens_cpu)):
            if tensor is not None and tensor.dtype != torch.int64:
                raise ValueError(f"{name} must have int64 dtype")

    def eval_roofline(self) -> tuple[int, int]:
        if not hasattr(self, "q_shape"):
            raise RuntimeError("eval_roofline() requires one completed forward call")
        return gated_deltanet_fwd_roofline(
            q_shape=self.q_shape,
            v_shape=self.v_shape,
            dtype=self.dtype,
            initial_state=self.initial_state,
            cu_seqlens_shape=self.cu_seqlens_shape,
        )

    def compute_roof(self) -> str:
        if not hasattr(self, "dtype"):
            raise RuntimeError("compute_roof() requires one completed forward call")
        return tensor_core_roof(self.dtype)

    def _validate_forward_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        cu_seqlens_cpu: Optional[torch.Tensor],
        A_log: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
    ) -> None:
        if q.ndim != 4 or k.shape != q.shape:
            raise ValueError("q and k must have the same [B, T, H, K] shape")
        if v.ndim != 4 or v.shape[:2] != q.shape[:2]:
            raise ValueError("v must have shape [B, T, HV, V]")

        batch, seq_len, heads, dim_k = q.shape
        value_heads, dim_v = v.shape[2:]
        if heads <= 0:
            raise ValueError("H must be positive")
        if value_heads % heads != 0:
            raise ValueError("HV must be divisible by H")
        if g.shape != (batch, seq_len, value_heads):
            raise ValueError("g must have shape [B, T, HV]")
        if beta.shape != g.shape:
            raise ValueError("beta must have the same shape as g")

        state_batch = batch
        if cu_seqlens is not None:
            if batch != 1:
                raise ValueError("packed varlen inputs require B == 1")
            if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
                raise ValueError("cu_seqlens must have shape [N + 1]")
            state_batch = cu_seqlens.shape[0] - 1
        if cu_seqlens_cpu is not None:
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_cpu requires cu_seqlens")
            if cu_seqlens_cpu.shape != cu_seqlens.shape:
                raise ValueError("cu_seqlens_cpu must have the same shape as cu_seqlens")
            if cu_seqlens_cpu.device.type != "cpu":
                raise ValueError("cu_seqlens_cpu must be on CPU")

        state_tail = (dim_v, dim_k) if self.state_v_first else (dim_k, dim_v)
        expected_state_shape = (state_batch, value_heads, *state_tail)
        if initial_state is not None and initial_state.shape != expected_state_shape:
            layout = "[N, HV, V, K]" if self.state_v_first else "[N, HV, K, V]"
            raise ValueError(f"initial_state must have shape {layout}")

        if self.use_gate_in_kernel:
            if A_log is None or dt_bias is None:
                raise ValueError("use_gate_in_kernel=True requires A_log and dt_bias")
            if A_log.shape != (value_heads,) or dt_bias.shape != (value_heads,):
                raise ValueError("A_log and dt_bias must have shape [HV]")
        elif A_log is not None or dt_bias is not None:
            raise ValueError("A_log and dt_bias require use_gate_in_kernel=True")

        self._validate_dtypes(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            cu_seqlens,
            cu_seqlens_cpu,
            A_log,
            dt_bias,
        )
        for name, tensor in (
            ("k", k),
            ("v", v),
            ("g", g),
            ("beta", beta),
            ("initial_state", initial_state),
            ("cu_seqlens", cu_seqlens),
            ("A_log", A_log),
            ("dt_bias", dt_bias),
        ):
            if tensor is not None and tensor.device != q.device:
                raise ValueError(f"{name} must be on the same device as q")

    @staticmethod
    def _canonicalize_inputs(
        *inputs: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], ...]:
        return tuple(tensor.contiguous() if tensor is not None else None for tensor in inputs)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run prefill or decode and return ``(o, final_state)``."""
        self._validate_forward_inputs(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            cu_seqlens,
            cu_seqlens_cpu,
            A_log,
            dt_bias,
        )
        inputs = self._canonicalize_inputs(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            cu_seqlens,
            cu_seqlens_cpu,
            A_log,
            dt_bias,
        )
        self.q_shape = tuple(q.shape)
        self.v_shape = tuple(v.shape)
        self.dtype = q.dtype
        self.initial_state = True if initial_state is not None else None
        self.cu_seqlens_shape = None if cu_seqlens is None else tuple(cu_seqlens.shape)
        batch, seq_len, heads, dim_k = q.shape
        value_heads, dim_v = v.shape[2:]
        scale = self.scale if self.scale is not None else dim_k**-0.5
        call = (
            batch,
            seq_len,
            heads,
            value_heads,
            dim_k,
            dim_v,
            q.dtype,
            q.device.index,
            scale,
            initial_state is not None,
            cu_seqlens is not None,
        )
        kernel = self.kernel_for("gated_deltanet", inputs, call)
        return kernel(*inputs)
