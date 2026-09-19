"""Dense equal-length Gated DeltaNet inference prefill."""

import functools
import math
import os
from typing import Tuple

import tilelang
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["GatedDeltaNetDensePrefillFwdKernel"]


@functools.lru_cache(maxsize=32)
def _prefill_partition_metadata(
    raw_sequence_lengths: tuple[int, ...],
    num_heads: int,
    chunk_size: int,
    max_local_chunks: int,
    use_partition: bool,
    device_idx: int | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Build shape-only partition metadata once per static workload."""
    device = "cuda" if device_idx is None else f"cuda:{device_idx}"
    raw_offsets = [0]
    for length in raw_sequence_lengths:
        raw_offsets.append(raw_offsets[-1] + length)
    raw_cu_seqlens = torch.tensor(
        raw_offsets, dtype=torch.int32, device=device, requires_grad=False
    )
    if not use_partition:
        return raw_cu_seqlens, None, None, None, None

    cp_cu_seqlens = []
    ht_mask = []
    seq_map_c2r = []
    seq_map_r2c = [0]
    split_tokens = max_local_chunks * chunk_size
    for raw_idx, (raw_start, raw_end) in enumerate(zip(raw_offsets, raw_offsets[1:], strict=False)):
        start = raw_start
        while start < raw_end:
            cp_cu_seqlens.append(start)
            ht_mask.append(False)
            seq_map_c2r.append(raw_idx)
            start += split_tokens
        ht_mask[-1] = True
        seq_map_r2c.append(len(cp_cu_seqlens))
    cp_cu_seqlens.append(raw_offsets[-1])
    return (
        raw_cu_seqlens,
        torch.tensor(cp_cu_seqlens, dtype=torch.int32, device=device),
        torch.tensor(seq_map_c2r, dtype=torch.int32, device=device),
        torch.tensor(seq_map_r2c, dtype=torch.int32, device=device),
        torch.tensor(ht_mask, dtype=torch.bool, device=device),
    )


def _prefill_auto_cp_local_chunks(num_chunks: int, num_heads: int, device_index: int) -> int:
    env_max_local_chunks = os.environ.get(
        "TILEOPS_GDN_PREFILL_MAX_LOCAL_CHUNKS",
        os.environ.get("TILEOPS_GDN_PREFILL_CP_MAX_LOCAL_CHUNKS"),
    )
    if env_max_local_chunks:
        max_local_chunks = int(env_max_local_chunks)
    else:
        # Without an argument this reads the current device, not the tensors'.
        sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
        max_local_chunks = 2 ** round(math.log2(math.sqrt(num_heads * num_chunks / sm_count) * 3))
        if num_heads >= 64 and num_chunks >= 512:
            max_local_chunks = max(max_local_chunks, 256)
    return max(max_local_chunks, 4)


def _prefill_partitioned_initial_state_bthd(
    k: torch.Tensor,
    v: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    raw_sequence_lengths: tuple[int, ...] | None = None,
    min_partition_chunks: int = 0,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    from . import _dense_prefill_kernels as impl

    batch, num_tokens, num_heads, _ = k.shape
    assert batch == 1
    if raw_sequence_lengths is None:
        raw_sequence_lengths = (num_tokens,)
    if any(length <= 0 or length % chunk_size != 0 for length in raw_sequence_lengths):
        raise ValueError("raw sequence lengths must be positive multiples of chunk_size")
    if sum(raw_sequence_lengths) != num_tokens:
        raise ValueError("raw sequence lengths must sum to the flattened token count")
    num_chunks = tilelang.cdiv(num_tokens, chunk_size)
    max_local_chunks = _prefill_auto_cp_local_chunks(num_chunks, num_heads, k.device.index)

    has_partition = num_chunks > max_local_chunks
    force_partition = (
        os.environ.get(
            "TILEOPS_GDN_PREFILL_FORCE_PARTITION",
            os.environ.get("TILEOPS_GDN_PREFILL_CP_FORCE", "0"),
        )
        == "1"
    )
    partition_large_enough = (
        force_partition or max(raw_sequence_lengths) // chunk_size >= min_partition_chunks
    )
    use_partition = (
        has_partition
        and partition_large_enough
        and (force_partition or num_heads <= 40 or (num_heads <= 64 and num_chunks >= 128))
    )
    (
        raw_cu_seqlens,
        cp_cu_seqlens_t,
        seq_map_c2r_t,
        seq_map_r2c_t,
        ht_mask_t,
    ) = _prefill_partition_metadata(
        raw_sequence_lengths,
        num_heads,
        chunk_size,
        max_local_chunks,
        use_partition,
        k.device.index,
    )
    if not use_partition:
        return None, raw_cu_seqlens, None, raw_cu_seqlens
    assert cp_cu_seqlens_t is not None
    assert seq_map_c2r_t is not None
    assert seq_map_r2c_t is not None
    assert ht_mask_t is not None

    num_warmup_chunks, fallback_mask = impl.get_warmup_chunks(
        g=g,
        cu_seqlens=cp_cu_seqlens_t,
        ht_mask=ht_mask_t,
        chunk_size=chunk_size,
        threshold=-10.0,
    )
    _, ht, mt = impl.fused_gdr_h(
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        initial_state=None,
        output_final_state=True,
        output_h=False,
        cu_seqlens=cp_cu_seqlens_t,
        num_warmup_chunks=num_warmup_chunks,
    )
    cp_h0 = impl.correct_initial_states(
        raw_h0=None,
        ht_buffer=ht,
        mt_buffer=mt,
        fallback_mask=fallback_mask,
        seq_map_r2c=seq_map_r2c_t,
    )
    return cp_h0, cp_cu_seqlens_t, seq_map_c2r_t, raw_cu_seqlens


def _gated_deltanet_production_bthd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    *,
    scale: float = 1.0,
    output_states: bool = False,
    min_partition_chunks: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Run the production recurrence shared by BTHD prefill and forward."""
    from . import _dense_prefill_kernels as impl

    batch, seq_len, head = q.shape[:3]
    g_cum = impl._prefill_chunk_local_cumsum_bthd_tl(
        batch, head, seq_len, chunk_size, str(q.dtype).split(".")[-1]
    )(g)
    inverse = impl._prefill_blocksolve_A_bthd(k, g_cum, beta, chunk_size, use_gate=False)
    g = g_cum
    if batch > 1:
        q, k, v, g, beta, inverse = (
            tensor.reshape(1, batch * seq_len, *tensor.shape[2:])
            for tensor in (q, k, v, g, beta, inverse)
        )
    initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens = _prefill_partitioned_initial_state_bthd(
        k,
        v,
        inverse,
        g,
        beta,
        chunk_size,
        raw_sequence_lengths=None if batch == 1 else (seq_len,) * batch,
        min_partition_chunks=min_partition_chunks,
    )
    o, states, final_state = impl.fused_gdr_fwd(
        q,
        k,
        v,
        inverse,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_h=output_states,
        cu_seqlens=cu_seqlens,
        cp_seq_map=cp_seq_map,
        raw_cu_seqlens=raw_cu_seqlens,
        chunk_size=chunk_size,
        state_head_first=output_states,
        chunks_per_sequence=seq_len // chunk_size if output_states else 0,
    )
    return o.reshape(batch, seq_len, head, v.shape[-1]), states, final_state, g_cum


class GatedDeltaNetDensePrefillFwdKernel(Kernel):
    """Hopper equal-length BTHD inference prefill.

    This is the inference owner of the retained partitioned prefill pipeline.
    The Op layer admits only the currently supported region; the kernel keeps
    the unified public call signature so no layout or ABI adapter is needed.
    """

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
        self.dtype = dtype

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
        del initial_state, cu_seqlens, cu_seqlens_cpu, A_log, dt_bias
        self._require_cuda(q=q, k=k, v=v, g=g, beta=beta)
        o, _states, final_state, _g_cum = _gated_deltanet_production_bthd(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=64,
            scale=self.scale,
        )
        return o, final_state
