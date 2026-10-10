"""Kimi Delta Attention (KDA) operator (L2 Op layer).

Provides:
  - KDAFwdOp: the gated delta rule whose decay is one log-space
    value per key channel, returning the output and the FP32 final state.
"""

from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.linear_attention import (
    KDACall,
    KDAChunkPrefillFwdKernel,
    KDAFusedPrefillFwdKernel,
    KDAFwdInterface,
    KDARecurrentDecodeFwdKernel,
)
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["KDAFwdOp"]


class KDAFwdOp(Op):
    """Kimi Delta Attention (KDA): the gated delta rule with a per-key-channel decay.

    ``q`` and ``k`` use ``[B, T, H, K]``. ``v``, ``beta`` and the output use
    ``HV`` recurrent heads, where ``HV`` is a multiple of ``H``; ``g`` carries
    one log-space decay per key channel, ``[B, T, HV, K]``, which is what
    separates this operator from the scalar per-head gate of
    ``GDNFwdOp``. The recurrent state is always FP32 and key-major
    ``[N, HV, K, V]``; ``state_v_first=True`` asks for the value-major layout.
    For equal-length inputs ``N == B``. Passing ``cu_seqlens`` selects packed
    varlen mode: the token tensors have ``B == 1`` and ``N`` is the number of
    packed sequences.

    By default ``g`` contains the precomputed log-space decay and ``beta`` the
    post-sigmoid update strength. ``use_gate_in_kernel=True`` instead treats
    ``g`` as the raw gate and requires ``A_log``;
    ``use_beta_sigmoid_in_kernel=True`` treats ``beta`` as raw logits, and with
    ``allow_neg_eigval=True`` the beta transform is ``2 * sigmoid(beta)``.

    One interface covers equal-length prefill, packed varlen and single-token
    decode, and always returns ``(o, final_state)``: an inference caller needs
    the state to continue the recurrence. The target callable selects the
    execution path from the current inputs, so ``T == 1`` is decode rather than
    a separate public Op.

    The in-tree implementations cover SM90 prefill and decode over a 64- or
    128-wide square state with precomputed gate and beta values. The raw-gate
    and raw-beta variants, the value-major state and other state widths still
    require an external target implementation.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "kda_chunk_prefill": KDAChunkPrefillFwdKernel,
        "kda_fused_prefill": KDAFusedPrefillFwdKernel,
        "kda_recurrent_decode": KDARecurrentDecodeFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"kda": KDAFwdInterface}

    def __init__(
        self,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        use_gate_in_kernel: bool = False,
        lower_bound: Optional[float] = None,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
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
            use_gate_in_kernel: Treat ``g`` as a raw gate and internally compute
                the log-space decay from ``A_log`` and ``dt_bias``. Otherwise
                ``g`` must already contain it.
            lower_bound: Lower bound of the forget gate in log space, or
                ``None``. Only used with ``use_gate_in_kernel=True``.
            target: Backend target, or ``None`` to resolve from the input device.
            kernel_map: Optional in-tree kernel overrides.
            tune: Autotune a kernel when it is first built.
        """
        self.scale = scale
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        self.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        self.allow_neg_eigval = allow_neg_eigval
        self.state_v_first = state_v_first
        self.use_gate_in_kernel = use_gate_in_kernel
        self.lower_bound = lower_bound
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def compute_roof(self) -> str:
        """The state contractions are priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])

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
        return self._call_boundary(
            q, k, v, g, beta, initial_state, cu_seqlens, cu_seqlens_cpu, A_log, dt_bias
        )

    def _eager_forward(
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
        """Resolve the kernel and launch, inside the operator."""
        inputs = tuple(
            tensor.contiguous() if tensor is not None else None
            for tensor in (
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
        )
        batch, seq_len, heads, dim_k = q.shape
        value_heads, dim_v = v.shape[2:]
        call = KDACall(
            batch=batch,
            seq_len=seq_len,
            sequences=batch if cu_seqlens is None else cu_seqlens.shape[0] - 1,
            heads=heads,
            value_heads=value_heads,
            dim_k=dim_k,
            dim_v=dim_v,
            dtype=q.dtype,
            scale=self.scale if self.scale is not None else dim_k**-0.5,
            has_initial_state=initial_state is not None,
            varlen=cu_seqlens is not None,
            state_v_first=self.state_v_first,
            l2norm=self.use_qk_l2norm_in_kernel,
            gate_in_kernel=self.use_gate_in_kernel,
            beta_sigmoid=self.use_beta_sigmoid_in_kernel,
            allow_neg_eigval=self.allow_neg_eigval,
            bounded_gate=self.lower_bound is not None,
            device=q.device,
        )
        return self.kernel_for("kda", call)(*inputs)
