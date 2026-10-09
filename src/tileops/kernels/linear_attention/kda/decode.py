"""Single-token Kimi Delta Attention (KDA) decode over a caller-owned state."""

from typing import Optional, Tuple

import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    KDACall,
    KDAFwdInterface,
)
from tileops.kernels.linear_attention.kda.decode_program import decode_program
from tileops.utils import get_sm_count

__all__ = ["KDARecurrentDecodeFwdKernel"]


class KDARecurrentDecodeFwdKernel(Kernel, KDAFwdInterface):
    """One recurrence step per sequence, one value channel per thread."""

    supported_archs = [80, 89, 90]

    @classmethod
    def applies(cls, call: KDACall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: KDACall) -> Optional[str]:
        """Why this kernel does not serve *call*, or ``None`` when it does."""
        chunked = call.chunk_refusal
        if chunked is not None:
            return chunked
        if call.seq_len != 1:
            return "serves a single-token step"
        return None

    @classmethod
    def entry_for(cls, call: KDACall) -> Entry:
        index = call.device.index if call.device is not None else None
        identity = (
            call.batch,
            call.heads,
            call.value_heads,
            call.dim_k,
            call.dim_v,
            call.scale,
            call.l2norm,
            call.dtype,
            index,
        )
        return identity, lambda: cls(
            batch=call.batch,
            heads=call.heads,
            value_heads=call.value_heads,
            dim_k=call.dim_k,
            dim_v=call.dim_v,
            scale=call.scale,
            l2norm=call.l2norm,
            dtype=call.dtype,
            device_index=index,
        )

    def __init__(
        self,
        batch: int,
        heads: int,
        value_heads: int,
        dim_k: int,
        dim_v: int,
        scale: float,
        l2norm: bool,
        dtype: torch.dtype,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        """Hold the call facts the decode program is built from."""
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.value_heads = value_heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.scale = scale
        self.l2norm = l2norm
        self.dtype = dtype
        self.value_tile = self._value_tile(batch * value_heads, dim_v, device_index)
        self.threads = 128 if self.value_tile <= 32 else 256
        self._program = decode_program(
            heads,
            value_heads,
            dim_k,
            dim_v,
            self.dtype_to_str(dtype),
            scale,
            l2norm,
            batch,
            self.value_tile,
            self.threads,
        )

    @staticmethod
    def _value_tile(blocks: int, dim_v: int, device_index: Optional[int]) -> int:
        """How many value channels one block owns.

        A decode step reads the state once and writes it once, so the only thing
        that decides its latency is how much of the machine is reading. One
        (sequence, head) pair is one block at the full width; slicing the value
        axis is what gives a small batch more of them.
        """
        wanted = get_sm_count(device_index) * 3 // 4
        for tile in (dim_v, 64, 32, 16):
            if tile <= dim_v and blocks * (dim_v // tile) >= wanted:
                return tile
        return 16

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
        """Advance every sequence by one token; see the interface for the tensors."""
        del cu_seqlens, cu_seqlens_cpu, A_log, dt_bias
        self._require_cuda(q=q, k=k, v=v, g=g, beta=beta)
        batch = q.shape[0]
        HV, V = v.shape[2], v.shape[3]
        K = q.shape[3]
        packed = lambda t: t.transpose(0, 1).contiguous()  # noqa: E731
        state = (
            torch.zeros(batch, HV, K, V, dtype=torch.float32, device=q.device)
            if initial_state is None
            else initial_state
        )
        o, final_state = self._program(
            packed(q), packed(k), packed(v), packed(g), packed(beta), state
        )
        return o.transpose(0, 1).contiguous(), final_state
