"""SM90 single-token DeltaNet inference decode."""

from typing import Optional, Tuple

import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    DeltaNetInferenceCall,
    DeltaNetInferenceFwdInterface,
)
from tileops.kernels.linear_attention.gated_deltanet.decode import (
    DENSE_DECODE_SM90_CONFIG,
    gated_deltanet_dense_decode_sm90_tl,
)

__all__ = ["DeltaNetDenseDecodeFwdKernel"]


class DeltaNetDenseDecodeFwdKernel(Kernel, DeltaNetInferenceFwdInterface):
    """Ungated delta rule: the gated decode program with a zero log decay.

    A zero gate makes the gated step's decay ``exp(0) = 1``, which is the
    ungated recurrence. The program already takes the token tensors in the
    call's dtype over a float32 state, so the inference contract needs no
    layout or precision adapter beyond the gate.
    """

    supported_archs = [90]

    @classmethod
    def applies(cls, call: DeltaNetInferenceCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: DeltaNetInferenceCall) -> Optional[str]:
        """Why this kernel does not serve *call*, or ``None`` when it does.

        One 16-bit token over a 128-wide square float32 state, with Q and K already
        normalized.
        """
        unsupported = [
            name
            for name, present in (
                ("Q/K L2 normalization", call.l2norm),
                ("packed varlen", call.varlen),
                ("T other than 1", call.seq_len != 1),
                ("K/V dimensions other than 128", call.dim_k != 128 or call.dim_v != 128),
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
        identity = (call.batch, call.heads, call.dim_k, call.scale, call.dtype, index)
        return identity, lambda: cls(
            batch=call.batch,
            heads=call.heads,
            dim=call.dim_k,
            scale=call.scale,
            dtype=call.dtype,
            device_index=index,
        )

    def __init__(
        self,
        batch: int,
        heads: int,
        dim: int,
        scale: float,
        dtype: torch.dtype,
        *,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.dim = dim
        self.scale = scale
        self.dtype = dtype
        self.init_config()
        self._kernel_fn = gated_deltanet_dense_decode_sm90_tl(
            batch,
            heads,
            dim,
            scale,
            self.dtype_str,
            self.config["v_tile"],
            self.config["lane_group"],
            self.config["maxrregcount"],
        )(self.config["threads"])
        # A zero gate is the ungated delta rule. Keep it across calls so the
        # measured path does not include an extra GPU memset per invocation.
        device = (
            torch.device("cuda", device_index) if device_index is not None else torch.device("cuda")
        )
        self.zero_gate = torch.zeros((batch, 1, heads), dtype=dtype, device=device)

    @property
    def default_config(self) -> dict:
        return dict(DENSE_DECODE_SM90_CONFIG)

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
        if initial_state is None:
            initial_state = torch.zeros(
                (self.batch, self.heads, self.dim, self.dim),
                dtype=torch.float32,
                device=q.device,
            )
        return self._kernel_fn(q, k, v, self.zero_gate, beta, initial_state)
