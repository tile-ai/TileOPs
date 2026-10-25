from typing import ClassVar, Mapping, Optional, Tuple

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.linear_attention import (
    GDNCall,
    GDNChunkPrefillFwdKernel,
    GDNDenseDecodeFwdKernel,
    GDNDensePrefillFwdKernel,
    GDNFwdInterface,
)
from tileops._ops.op_base import Op
from tileops._perf.profile import tensor_core_roof
from tileops.backend import Target

__all__ = ["GDNFwdOp"]


class GDNFwdOp(Op):
    """Gated DeltaNet (GDN): inference forward for the gated delta rule.

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

    The in-tree kernels serve a 64- or 128-wide square state in either
    layout, with ``HV`` a multiple of ``H``, an even or single head count, and
    any combination of the three input transforms. Prefill runs on SM89 and
    SM90, equal-length or packed, with a sequence that is not a whole number
    of 64-token chunks. On SM89, shared memory holds the offsets of at most
    703 packed sequences over a 128-wide state. Decode runs one token per
    sequence on SM80, SM89 and SM90, and does not take ``cu_seqlens``.
    Other calls need an external target implementation.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gdn_dense_decode": GDNDenseDecodeFwdKernel,
        "gdn_dense_prefill": GDNDensePrefillFwdKernel,
        "gdn_chunk_prefill": GDNChunkPrefillFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"gdn": GDNFwdInterface}

    def __init__(
        self,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        use_gate_in_kernel: bool = False,
        *,
        validate_inputs: bool = False,
        target: Target = None,
    ) -> None:
        """Fix recurrence semantics; tensor metadata comes from each call.

        Args:
            scale: Query scale, or ``None`` for ``K**-0.5`` (K the key head dim).
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
            validate_inputs: Check on the CPU that ``cu_seqlens`` (and
                ``cu_seqlens_cpu``) start at 0, never decrease and end at ``S``.
                It synchronizes the device; leave it off under CUDA Graph capture.
            target: Backend target, or ``None`` to resolve from the input
                device.
        """
        self.scale = scale
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        self.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        self.allow_neg_eigval = allow_neg_eigval
        self.state_v_first = state_v_first
        self.use_gate_in_kernel = use_gate_in_kernel
        self.validate_inputs = validate_inputs
        super().__init__(target=target)

    def roof_key(self) -> str:
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
        """Run prefill or decode and return ``(o, final_state)``.

        Args:
            q: Queries, $[B \\times S \\times H \\times K]$, ``float16`` or ``bfloat16``.
            k: Keys, $[B \\times S \\times H \\times K]$, same dtype as ``q``.
            v: Values, $[B \\times S \\times \\mathit{HV} \\times V]$, same dtype as ``q``.
            g: Log-space decay, or the raw gate under ``use_gate_in_kernel``,
                $[B \\times S \\times \\mathit{HV}]$, same dtype as ``q``.
            beta: Post-sigmoid update strength, or raw logits under
                ``use_beta_sigmoid_in_kernel``, $[B \\times S \\times \\mathit{HV}]$, same dtype
                as ``q``.
            initial_state: ``float32`` starting state, or ``None``:
                $[N \\times \\mathit{HV} \\times K \\times V]$, or
                $[N \\times \\mathit{HV} \\times V \\times K]$ under ``state_v_first``.
            cu_seqlens: ``int64`` offsets of the packed sequences, $[N + 1]$, or ``None`` for
                equal-length input. Requires $B = 1$.
            cu_seqlens_cpu: The same offsets on the CPU, or ``None``. Requires ``cu_seqlens``.
            A_log: ``float32`` decay parameter, $[\\mathit{HV}]$. Required under
                ``use_gate_in_kernel``.
            dt_bias: ``float32`` gate bias, $[\\mathit{HV}]$. Required under
                ``use_gate_in_kernel``.

        Returns:
            ``(o, final_state)``. ``o`` is the output,
            $[B \\times S \\times \\mathit{HV} \\times V]$, in the dtype of ``q``.
            ``final_state`` is the ``float32`` state after the last token, in the layout
            ``initial_state`` takes. $N = B$ unless ``cu_seqlens`` is given.
        """
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
        return self.kernel_for(
            "gdn",
            lambda: GDNCall(
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
            ),
        )(*inputs)
