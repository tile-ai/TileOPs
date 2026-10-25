from typing import Optional

import torch

from tileops._kernels.attention.call_spec import AttentionCall, GQADenseFwdInterface
from tileops._kernels.attention.gqa.dense import make_dense_qk_rope_preprocessor
from tileops._kernels.attention.gqa.prefill_varlen import (
    _gqa_prefill_varlen_fwd_kernel,
    prefill_default_config,
)
from tileops._kernels.attention.gqa.varlen_fp8 import (
    _gqa_varlen_fp8_fwd_kernel,
    fp8_prefill_shared_bytes,
)
from tileops._kernels.kernel_base import CompiledProgram, Entry, Kernel

__all__ = ["GQADensePrefillFwdKernel"]


def _offsets(batch: int, length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(batch + 1, device=device, dtype=torch.int32) * length


def _plan(call: AttentionCall):
    if call.is_fp8:
        config = {"block_m": 64, "block_n": 64, "num_stages": 1, "threads": 128}
        needed = fp8_prefill_shared_bytes(
            config, batch=call.batch, dim=call.dim, element=call.dtype.itemsize
        )
        return config if needed <= call.smem_budget else None
    return prefill_default_config(
        batch=call.batch, dim=call.dim, element=call.dtype.itemsize, budget=call.smem_budget
    )


class GQADensePrefillFwdKernel(Kernel, GQADenseFwdInterface):
    """Dense SM89 prefill using the existing packed attention program."""

    supported_archs = [89]

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        if call.has_sinks:
            return "attention sinks require SM90"
        if call.max_seqlen_q == 1:
            return "serves prefill"
        if call.is_fp8 and call.dim != 128:
            return "native FP8 prefill requires head dimension 128"
        if call.tensor_core_dim_refusal is not None:
            return call.tensor_core_dim_refusal
        plan = _plan(call)
        if plan is None:
            return "no prefill tile fits the device shared-memory budget"
        return None

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        identity = (
            call.batch,
            call.heads,
            call.heads_kv,
            call.dim,
            call.dtype,
            call.is_fp8,
            call.max_seqlen_q,
            call.seqlen_kv,
            call.is_causal,
            call.sm_scale,
            call.softcap,
            call.window_size_left,
            call.window_size_right,
            call.fuse_rope,
            call.max_position,
            call.rotary_dim,
            call.rope_layout,
            call.device,
        )
        return identity, lambda: cls(call)

    def __init__(self, call: AttentionCall) -> None:
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        builder = _gqa_varlen_fp8_fwd_kernel if call.is_fp8 else _gqa_prefill_varlen_fwd_kernel
        self.kernel = builder(
            call.batch,
            call.heads,
            call.heads_kv,
            call.dim,
            call.is_causal,
            call.sm_scale,
            call.softcap,
            self.dtype_str,
            call.window_size_left,
            call.window_size_right,
        )
        self.rope = make_dense_qk_rope_preprocessor(
            batch=call.batch,
            heads=call.heads,
            heads_kv=call.heads_kv,
            seq_len_q=call.max_seqlen_q,
            seq_len_kv=call.seqlen_kv,
            dim=call.dim,
            dtype="float8_e4m3fn" if call.is_fp8 else self.dtype_str,
            rope_dtype=self.dtype_str,
            **call.rope_args,
        )
        self.init_config(None)

    @property
    def default_config(self) -> dict:
        call = self.call
        plan = _plan(call)
        if plan is None:
            raise ValueError("no prefill tile fits the device shared-memory budget")
        return plan

    @property
    def autotune_configs(self) -> list[dict]:
        # Keep the admitted one-stage plan until other configurations are measured.
        return [self.default_config]

    @property
    def autotune_supply_prog(self):
        call = self.call

        def supply(params):
            if len(params) != (8 if call.is_fp8 else 5):
                raise ValueError("unexpected dense prefill program input count")
            q_shape = (call.batch * call.max_seqlen_q, call.heads, call.dim)
            kv_shape = (call.batch * call.seqlen_kv, call.heads_kv, call.dim)
            input_dtype = torch.float8_e4m3fn if call.is_fp8 else call.dtype
            tensors = [
                torch.randn(shape, dtype=call.dtype, device=call.device).to(input_dtype)
                for shape in (q_shape, kv_shape, kv_shape)
            ]
            tensors += [
                _offsets(call.batch, call.max_seqlen_q, call.device),
                _offsets(call.batch, call.seqlen_kv, call.device),
            ]
            if call.is_fp8:
                tensors += [
                    torch.ones((call.batch, call.heads_kv), device=call.device, dtype=torch.float32)
                    for _ in range(3)
                ]
            return tensors

        return supply

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention, rope = self.program(self._compile)
        if rope is not None:
            q, k = rope(q, k, rope_cos, rope_sin)
        cu_q = _offsets(q.shape[0], q.shape[1], q.device)
        cu_kv = _offsets(k.shape[0], k.shape[1], k.device)
        inputs = [
            q.flatten(0, 1),
            k.flatten(0, 1),
            v.flatten(0, 1),
            cu_q,
            cu_kv,
        ]
        if self.call.is_fp8:
            inputs += [q_scale, k_scale, v_scale]
        output = attention(*inputs)
        return output.reshape_as(q)

    def _compile(self, cfg: dict) -> tuple:
        """The attention program for *cfg*, and the Q/K rotation ahead of it or ``None``."""
        rope = self.rope.build() if self.rope is not None else None
        return CompiledProgram(self.kernel(**cfg)), rope
