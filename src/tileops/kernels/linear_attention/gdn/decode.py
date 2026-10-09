"""Single-token Gated DeltaNet (GDN) inference decode."""

from typing import Optional, Tuple

import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    GDNCall,
    GDNFwdInterface,
    head_count_refusal,
)
from tileops.kernels.linear_attention.delta_decode import decode_launch, delta_decode_tl

__all__ = ["GDNDenseDecodeFwdKernel"]


class GDNDenseDecodeFwdKernel(Kernel, GDNFwdInterface):
    """FP16/BF16 decode with FP32 recurrent state.

    The decayed state slice stays in registers, so the state update and output
    projection reuse the same load. The gate transform, the beta sigmoid, the Q/K
    L2 normalization and the starting state are build flags of the program, so a
    call that asks for none of them runs a program that contains none of them.
    """

    supported_archs = [80, 89, 90]

    @classmethod
    def applies(cls, call: GDNCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: GDNCall) -> Optional[str]:
        """Why this kernel does not serve *call*, or ``None`` when it does.

        One token continuing a 64- or 128-wide square state, in either layout, under
        any combination of the recurrence flags the operator fixes.
        """
        heads = head_count_refusal(call.heads, call.value_heads)
        if heads is not None:
            return heads
        unsupported = [
            name
            for name, present in (
                ("packed varlen", call.varlen),
                (
                    "K and V other than matching 64 or 128",
                    call.dim_k != call.dim_v or call.dim_k not in (64, 128),
                ),
                ("T other than 1", call.seq_len != 1),
            )
            if present
        ]
        return "does not support " + ", ".join(unsupported) if unsupported else None

    @classmethod
    def entry_for(cls, call: GDNCall) -> Entry:
        index = call.device.index if call.device is not None else None
        identity = (
            call.batch,
            call.heads,
            call.value_heads,
            call.dim_k,
            call.scale,
            call.dtype,
            call.has_initial_state,
            call.state_v_first,
            call.l2norm,
            call.gate_in_kernel,
            call.beta_sigmoid,
            call.allow_neg_eigval,
            index,
        )
        return identity, lambda: cls(
            batch=call.batch,
            heads=call.heads,
            value_heads=call.value_heads,
            dim=call.dim_k,
            scale=call.scale,
            dtype=call.dtype,
            has_initial_state=call.has_initial_state,
            state_v_first=call.state_v_first,
            l2norm=call.l2norm,
            gate_in_kernel=call.gate_in_kernel,
            beta_sigmoid=call.beta_sigmoid,
            allow_neg_eigval=call.allow_neg_eigval,
            device_index=index,
        )

    def __init__(
        self,
        batch: int,
        heads: int,
        value_heads: int,
        dim: int,
        scale: float,
        dtype: torch.dtype,
        has_initial_state: bool,
        state_v_first: bool,
        l2norm: bool,
        gate_in_kernel: bool,
        beta_sigmoid: bool,
        allow_neg_eigval: bool,
        *,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.value_heads = value_heads
        self.dim = dim
        self.scale = scale
        self.dtype = dtype
        self.has_initial_state = has_initial_state
        self.state_v_first = state_v_first
        self.gate_in_kernel = gate_in_kernel
        self.init_config()
        self._kernel_fn = delta_decode_tl(
            batch,
            heads,
            value_heads,
            dim,
            scale,
            self.dtype_str,
            gated=True,
            gate_in_kernel=gate_in_kernel,
            beta_sigmoid=beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            l2norm=l2norm,
            has_initial_state=has_initial_state,
            state_v_first=state_v_first,
            threads=self.config["threads"],
            lane_group=self.config["lane_group"],
        )()
        device = (
            torch.device("cuda", device_index) if device_index is not None else torch.device("cuda")
        )
        # The float32 parameters this build leaves unread are declared one element wide.
        self._unread = torch.empty(1, dtype=torch.float32, device=device)

    @property
    def default_config(self) -> dict:
        """The block shape this call's grid and state layout call for."""
        return decode_launch(
            self.batch, self.value_heads, self.dim, self.state_v_first, self.device_index
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del cu_seqlens, cu_seqlens_cpu
        self._require_cuda(q=q, k=k, v=v, g=g, beta=beta)
        if self.has_initial_state and initial_state is None:
            raise ValueError("the build reads initial_state, but the call passed none")
        state = initial_state if self.has_initial_state else self._unread
        gate_params = (A_log, dt_bias) if self.gate_in_kernel else (self._unread, self._unread)
        return self._kernel_fn(q, k, v, g, beta, *gate_params, state)
