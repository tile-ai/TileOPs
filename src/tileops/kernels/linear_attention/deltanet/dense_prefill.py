"""Hopper DeltaNet prefill with its own dense partition orchestration."""

import functools
import math
import os
from typing import Tuple

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.linear_attention.gated_deltanet.prefill.forward import fused_gdr_fwd
from tileops.kernels.linear_attention.gated_deltanet.prefill.prepare import (
    _prefill_blocksolve_A_bthd,
    correct_initial_states,
    fused_gdr_h,
    get_warmup_chunks,
)

__all__ = ["DeltaNetDensePrefillFwdKernel"]


@functools.lru_cache(maxsize=32)
def _dense_partition_metadata(
    batch: int,
    seq_len: int,
    heads: int,
    max_local_chunks: int,
    use_partition: bool,
    device_index: int | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Cache only shape-derived offsets and maps, never recurrent state."""
    device = (
        torch.device("cuda", device_index) if device_index is not None else torch.device("cuda")
    )
    raw_offsets = [i * seq_len for i in range(batch + 1)]
    raw_cu = torch.tensor(raw_offsets, dtype=torch.int32, device=device)
    if not use_partition:
        return raw_cu, None, None, None, None

    cp_offsets = []
    cp_to_raw = []
    raw_to_cp = [0]
    final_partition_mask = []
    partition_tokens = max_local_chunks * 64
    for raw_idx, (start, end) in enumerate(zip(raw_offsets, raw_offsets[1:], strict=False)):
        for offset in range(start, end, partition_tokens):
            cp_offsets.append(offset)
            cp_to_raw.append(raw_idx)
            final_partition_mask.append(False)
        final_partition_mask[-1] = True
        raw_to_cp.append(len(cp_offsets))
    cp_offsets.append(raw_offsets[-1])
    return (
        raw_cu,
        torch.tensor(cp_offsets, dtype=torch.int32, device=device),
        torch.tensor(cp_to_raw, dtype=torch.int32, device=device),
        torch.tensor(raw_to_cp, dtype=torch.int32, device=device),
        torch.tensor(final_partition_mask, dtype=torch.bool, device=device),
    )


def _dense_local_chunks(num_chunks: int, heads: int, device_index: int | None) -> int:
    override = os.environ.get("TILEOPS_DELTANET_PREFILL_MAX_LOCAL_CHUNKS")
    if override is not None:
        return max(int(override), 4)
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    local_chunks = 2 ** round(math.log2(math.sqrt(heads * num_chunks / sm_count) * 3))
    if heads >= 64 and num_chunks >= 512:
        local_chunks = max(local_chunks, 256)
    return max(local_chunks, 4)


def _dense_partition_initial_state(
    k: torch.Tensor,
    v: torch.Tensor,
    inverse: torch.Tensor,
    zero_gate: torch.Tensor,
    beta: torch.Tensor,
    batch: int,
    seq_len: int,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Prepare the correct start state for every independent prefill partition."""
    heads = k.shape[2]
    num_chunks = batch * seq_len // 64
    max_local_chunks = _dense_local_chunks(num_chunks, heads, k.device.index)
    use_partition = num_chunks > max_local_chunks and (
        heads <= 40 or (heads <= 64 and num_chunks >= 128)
    )
    raw_cu, cp_cu, cp_to_raw, raw_to_cp, final_mask = _dense_partition_metadata(
        batch, seq_len, heads, max_local_chunks, use_partition, k.device.index
    )
    if not use_partition:
        return initial_state, raw_cu, None, raw_cu
    assert cp_cu is not None
    assert cp_to_raw is not None
    assert raw_to_cp is not None
    assert final_mask is not None

    warmup_chunks, fallback_mask = get_warmup_chunks(
        g=zero_gate,
        cu_seqlens=cp_cu,
        ht_mask=final_mask,
        chunk_size=64,
        threshold=-10.0,
    )
    _, partition_h, partition_m = fused_gdr_h(
        k=k,
        v=v,
        a=inverse,
        g=zero_gate,
        b=beta,
        initial_state=None,
        output_final_state=True,
        output_h=False,
        cu_seqlens=cp_cu,
        num_warmup_chunks=warmup_chunks,
    )
    partition_h0 = correct_initial_states(
        raw_h0=initial_state,
        ht_buffer=partition_h,
        mt_buffer=partition_m,
        fallback_mask=fallback_mask,
        seq_map_r2c=raw_to_cp,
    )
    return partition_h0, cp_cu, cp_to_raw, raw_cu


class DeltaNetDensePrefillFwdKernel(Kernel):
    """Ungated delta rule: GDN's block solve and partitioned recurrence with g=0."""

    supported_archs = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim: int,
        scale: float,
        dtype: torch.dtype,
        *,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.scale = scale
        # A zero gate is the ungated delta rule. Keep it across calls so the
        # measured path does not include an extra GPU memset per invocation.
        device = (
            torch.device("cuda", device_index) if device_index is not None else torch.device("cuda")
        )
        self.zero_gate = torch.zeros((batch, seq_len, heads), dtype=dtype, device=device)

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
        batch, seq_len, heads, dim = q.shape
        inverse = _prefill_blocksolve_A_bthd(k, self.zero_gate, beta, 64, use_gate=False)
        flattened = (
            tensor.reshape(1, batch * seq_len, *tensor.shape[2:])
            for tensor in (q, k, v, self.zero_gate, beta, inverse)
        )
        q_flat, k_flat, v_flat, g_flat, beta_flat, inverse_flat = flattened
        partition_h0, cu, seq_map, raw_cu = _dense_partition_initial_state(
            k_flat,
            v_flat,
            inverse_flat,
            g_flat,
            beta_flat,
            batch,
            seq_len,
            initial_state,
        )
        o, _, final_state = fused_gdr_fwd(
            q_flat,
            k_flat,
            v_flat,
            inverse_flat,
            g_flat,
            beta_flat,
            scale=self.scale,
            initial_state=partition_h0,
            output_final_state=True,
            output_h=False,
            cu_seqlens=cu,
            cp_seq_map=seq_map,
            raw_cu_seqlens=raw_cu,
            chunk_size=64,
        )
        return o.reshape(batch, seq_len, heads, dim), final_state
