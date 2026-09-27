import math
from typing import Dict, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention import (
    GatedDeltaNetDenseDecodeFwdKernel,
    GatedDeltaNetDensePrefillFwdKernel,
)
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

    The in-tree implementation currently covers equal-length SM90 prefill
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
            use_gate_in_kernel: Treat ``g`` as a raw gate and internally
                compute ``-exp(A_log) * softplus(g + dt_bias)``. Otherwise,
                ``g`` must already contain the log-space decay.
            target: Backend target, or ``None`` to resolve from the input
                device.
            kernel_map: Optional in-tree kernel overrides.
            tune: Autotune a kernel when it is first built.
        """
        if scale is not None and not math.isfinite(scale):
            raise ValueError(f"scale must be finite, got {scale}")

        self.scale = scale
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        self.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        self.allow_neg_eigval = allow_neg_eigval
        self.state_v_first = state_v_first
        self.use_gate_in_kernel = use_gate_in_kernel
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gated_deltanet_dense_decode": GatedDeltaNetDenseDecodeFwdKernel,
            "gated_deltanet_dense_prefill": GatedDeltaNetDensePrefillFwdKernel,
        }

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
        is_decode = seq_len == 1
        if is_decode:
            if not has_initial_state:
                unsupported.append("decode without initial_state")
        else:
            if has_initial_state:
                unsupported.append("prefill with initial_state")
            if seq_len < 64 or seq_len % 64 != 0:
                unsupported.append("prefill T is not a positive multiple of 64")
        if unsupported:
            raise ValueError(
                "the in-tree GatedDeltaNet kernel does not yet support " + ", ".join(unsupported)
            )
        role = "gated_deltanet_dense_decode" if is_decode else "gated_deltanet_dense_prefill"
        return (role, call), lambda: self.kernel_map[role](
            batch=batch,
            heads=heads,
            **({} if is_decode else {"seq_len": seq_len}),
            dim=dim_k,
            scale=scale,
            dtype=dtype,
            device_index=device_index,
        )

    def compute_roof(self) -> str:
        """The state contractions are priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])

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
