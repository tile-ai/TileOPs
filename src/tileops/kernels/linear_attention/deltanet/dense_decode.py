"""SM90 single-token DeltaNet inference decode."""

from typing import Optional, Tuple

import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    DeltaNetInferenceCall,
    DeltaNetInferenceFwdInterface,
)
from tileops.kernels.linear_attention.delta_decode import decode_launch, delta_decode_sm90_tl

__all__ = ["DeltaNetDenseDecodeFwdKernel"]


class DeltaNetDenseDecodeFwdKernel(Kernel, DeltaNetInferenceFwdInterface):
    """SM90 FP16/BF16 ungated decode with FP32 recurrent state.

    The shared delta-rule decode program is built with its gate left out, so no
    decay is read, exponentiated or multiplied into the state slice. The Q/K L2
    normalization and the starting state are build flags of the same program.
    """

    supported_archs = [90]

    @classmethod
    def applies(cls, call: DeltaNetInferenceCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: DeltaNetInferenceCall) -> Optional[str]:
        """Why this kernel does not serve *call*, or ``None`` when it does.

        One 16-bit token over a 64- or 128-wide square float32 state.
        """
        unsupported = [
            name
            for name, present in (
                ("packed varlen", call.varlen),
                ("T other than 1", call.seq_len != 1),
                (
                    "K and V other than matching 64 or 128",
                    call.dim_k != call.dim_v or call.dim_k not in (64, 128),
                ),
                (
                    "dtype other than float16 or bfloat16",
                    call.dtype not in (torch.float16, torch.bfloat16),
                ),
            )
            if present
        ]
        return "does not support " + ", ".join(unsupported) if unsupported else None

    @classmethod
    def entry_for(cls, call: DeltaNetInferenceCall) -> Entry:
        index = call.device.index if call.device is not None else None
        identity = (
            call.batch,
            call.heads,
            call.dim_k,
            call.scale,
            call.dtype,
            call.l2norm,
            call.has_initial_state,
            index,
        )
        return identity, lambda: cls(
            batch=call.batch,
            heads=call.heads,
            dim=call.dim_k,
            scale=call.scale,
            dtype=call.dtype,
            l2norm=call.l2norm,
            has_initial_state=call.has_initial_state,
            device_index=index,
        )

    def __init__(
        self,
        batch: int,
        heads: int,
        dim: int,
        scale: float,
        dtype: torch.dtype,
        l2norm: bool,
        has_initial_state: bool,
        *,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.dim = dim
        self.scale = scale
        self.dtype = dtype
        self.has_initial_state = has_initial_state
        self.init_config()
        self._kernel_fn = delta_decode_sm90_tl(
            batch,
            heads,
            heads,
            dim,
            scale,
            self.dtype_str,
            gated=False,
            gate_in_kernel=False,
            beta_sigmoid=False,
            allow_neg_eigval=False,
            l2norm=l2norm,
            has_initial_state=has_initial_state,
            state_v_first=False,
            threads=self.config["threads"],
            lane_group=self.config["lane_group"],
        )()
        device = (
            torch.device("cuda", device_index) if device_index is not None else torch.device("cuda")
        )
        # The parameters an ungated build leaves unread are declared one element wide.
        self._unread_gate = torch.empty(1, dtype=dtype, device=device)
        self._unread = torch.empty(1, dtype=torch.float32, device=device)

    @property
    def default_config(self) -> dict:
        """The block shape this call's grid calls for, over a key-major state."""
        return decode_launch(self.batch, self.heads, self.dim, False, self.device_index)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del cu_seqlens, cu_seqlens_cpu
        self._require_cuda(q=q, k=k, v=v, beta=beta)
        if self.has_initial_state and initial_state is None:
            raise ValueError("the build reads initial_state, but the call passed none")
        state = initial_state if self.has_initial_state else self._unread
        return self._kernel_fn(q, k, v, self._unread_gate, beta, self._unread, self._unread, state)
