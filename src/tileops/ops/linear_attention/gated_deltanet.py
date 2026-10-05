from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.linear_attention import (
    GatedDeltaNetCall,
    GatedDeltaNetDenseDecodeFwdKernel,
    GatedDeltaNetDensePrefillFwdKernel,
    GatedDeltaNetFwdInterface,
)
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

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

    The in-tree implementations currently cover SM90 prefill over a 64- or
    128-wide square state in either layout -- equal-length or packed, with a
    sequence that is not a whole number of 64-token chunks, with ``HV`` a
    multiple of ``H``, and under any combination of the three input
    transforms -- and single-token SM90 decode over a key-major 128-wide one
    with matching recurrent head counts and the gate, the step size and the
    Q/K normalization settled before the call. Other regions still require an
    external target implementation while their retained kernels are migrated.
    """

    compile_boundary: ClassVar[bool] = True

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gated_deltanet_dense_decode": GatedDeltaNetDenseDecodeFwdKernel,
        "gated_deltanet_dense_prefill": GatedDeltaNetDensePrefillFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "gated_deltanet": GatedDeltaNetFwdInterface
    }

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
        self.scale = scale
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        self.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        self.allow_neg_eigval = allow_neg_eigval
        self.state_v_first = state_v_first
        self.use_gate_in_kernel = use_gate_in_kernel
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
        call = GatedDeltaNetCall(
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            value_heads=value_heads,
            dim_k=dim_k,
            dim_v=dim_v,
            dtype=q.dtype,
            scale=self.scale if self.scale is not None else dim_k**-0.5,
            has_initial_state=initial_state is not None,
            varlen=cu_seqlens is not None,
            num_sequences=batch if cu_seqlens is None else cu_seqlens.numel() - 1,
            state_v_first=self.state_v_first,
            l2norm=self.use_qk_l2norm_in_kernel,
            gate_in_kernel=self.use_gate_in_kernel,
            beta_sigmoid=self.use_beta_sigmoid_in_kernel,
            allow_neg_eigval=self.allow_neg_eigval,
            device=q.device,
        )
        return self.kernel_for("gated_deltanet", call)(*inputs)
