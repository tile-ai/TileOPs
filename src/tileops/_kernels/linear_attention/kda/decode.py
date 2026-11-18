"""Single-token Kimi Delta Attention (KDA) decode over a caller-owned state."""

import torch

from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.linear_attention.call_spec import (
    KDACall,
    KDAFwdInterface,
)
from tileops._kernels.linear_attention.delta_decode import decode_launch
from tileops._kernels.linear_attention.kda.decode_program import decode_program
from tileops._kernels.linear_attention.kda.gate import raw_gate_arguments

__all__ = ["KDARecurrentDecodeFwdKernel"]


class KDARecurrentDecodeFwdKernel(Kernel, KDAFwdInterface):
    """One recurrence step per sequence, in either state layout, in batch order or in the
    slots of a state pool."""

    supported_archs = [80, 89, 90]

    @classmethod
    def refusal(cls, call: KDACall) -> str | None:
        """Why this kernel does not serve *call*, or ``None`` when it does."""
        chunked = call.chunk_refusal
        if chunked is not None:
            return chunked
        if call.seq_len != 1:
            return "serves a single-token step"
        if call.sequences != call.batch:
            # The step reads one token and one state a row; packed offsets that leave a
            # sequence empty would pair the states with the wrong tokens.
            return "serves one token a sequence"
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
            call.gate_in_kernel,
            call.has_dt_bias,
            call.lower_bound,
            call.state_v_first,
            call.pooled,
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
            gate_in_kernel=call.gate_in_kernel,
            has_dt_bias=call.has_dt_bias,
            lower_bound=call.lower_bound,
            state_v_first=call.state_v_first,
            pooled=call.pooled,
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
        device_index: int | None = None,
        gate_in_kernel: bool = False,
        has_dt_bias: bool = False,
        lower_bound: float | None = None,
        state_v_first: bool = False,
        pooled: bool = False,
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
        self.gate_in_kernel = gate_in_kernel
        self.has_dt_bias = has_dt_bias
        self.lower_bound = lower_bound
        self.state_v_first = state_v_first
        self.pooled = pooled
        self.dtype = dtype
        if state_v_first:
            # The delta-rule decode's value-major block: whole rows, a lane group a row.
            launch = decode_launch(batch, value_heads, dim_k, True, self.build_target.sm_count)
            self.threads = launch["threads"]
            self.value_tile = self.threads // launch["lane_group"]
        else:
            self.value_tile = self._value_tile(
                batch * value_heads, dim_v, self.build_target.sm_count
            )
            self.threads = 128 if self.value_tile <= 32 else 256
        self.program(self._compile)

    def _compile(self, cfg: dict):
        """The decode program for the call facts this kernel holds."""
        return decode_program(
            self.heads,
            self.value_heads,
            self.dim_k,
            self.dim_v,
            self.dtype_to_str(self.dtype),
            self.scale,
            self.l2norm,
            self.batch,
            self.value_tile,
            self.threads,
            gate_in_kernel=self.gate_in_kernel,
            has_dt_bias=self.has_dt_bias,
            lower_bound=self.lower_bound,
            state_v_first=self.state_v_first,
            pooled=self.pooled,
        )

    @staticmethod
    def _value_tile(blocks: int, dim_v: int, sm_count: int) -> int:
        """How many value channels one block owns.

        A decode step reads the state once and writes it once, so the only thing
        that decides its latency is how much of the machine is reading. One
        (sequence, head) pair is one block at the full width; slicing the value
        axis is what gives a small batch more of them.
        """
        wanted = sm_count * 3 // 4
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
        initial_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance every sequence by one token; see the interface for the tensors."""
        del cu_seqlens, cu_seqlens_cpu
        self._require_devices(q=q, k=k, v=v, g=g, beta=beta)
        batch = q.shape[0]
        HV, V = v.shape[2], v.shape[3]
        K = q.shape[3]
        packed = lambda t: t.transpose(0, 1).contiguous()  # noqa: E731
        # A square state: the zero state is the same in either layout.
        state = (
            torch.zeros(batch, HV, K, V, dtype=torch.float32, device=q.device)
            if initial_state is None
            else initial_state
        )
        # A build without the raw gate reads neither gate parameter; a one-element view of
        # the state fills both slots.
        gate_params = (
            raw_gate_arguments(self.gate_in_kernel, A_log, dt_bias) or (state.view(-1)[:1],) * 2
        )
        tokens = packed(q), packed(k), packed(v), packed(g), packed(beta)
        program = self.program(self._compile)
        if self.pooled:
            # The pool is advanced in place and returned as the final state.
            o = program(*tokens, state, state_indices, *gate_params)
            return o.transpose(0, 1).contiguous(), state
        o, final_state = program(*tokens, state, *gate_params)
        return o.transpose(0, 1).contiguous(), final_state
