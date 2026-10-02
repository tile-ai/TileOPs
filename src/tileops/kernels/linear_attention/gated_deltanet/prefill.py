"""Dense equal-length Gated DeltaNet inference prefill."""

import functools
import math
from typing import Any, Dict, Optional, Tuple

import tilelang
import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    GatedDeltaNetCall,
    GatedDeltaNetFwdInterface,
)
from tileops.kernels.linear_attention.gated_deltanet.prefill_forward import fused_gdr_fwd
from tileops.kernels.linear_attention.gated_deltanet.prefill_prepare import (
    correct_initial_states,
    fused_gdr_h,
    get_warmup_chunks,
    prefill_blocksolve_A_bthd,
    prefill_chunk_local_cumsum_bthd_tl,
)
from tileops.utils import get_sm_count

__all__ = ["GatedDeltaNetDensePrefillFwdKernel"]


class GatedDeltaNetDensePrefillFwdKernel(Kernel, GatedDeltaNetFwdInterface):
    """SM90 equal-length BTHD inference prefill.

    This is the inference owner of the retained partitioned prefill pipeline.
    The Op layer admits only the currently supported region; the kernel keeps
    the unified public call signature so no layout or ABI adapter is needed.
    """

    supported_archs = [90]

    @classmethod
    def applies(cls, call: GatedDeltaNetCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: GatedDeltaNetCall) -> Optional[str]:
        """Why this kernel does not serve *call*, or ``None`` when it does.

        Prefill in chunks of 64 tokens over a 64- or 128-wide square state, equal-length or
        packed, with a row that is not a whole chunk, with grouped value heads, with the
        state key-major or value-major, and with the Q/K normalization, the gate and the
        beta transform taken in kernel.
        """
        if call.dim_k != call.dim_v or call.dim_k not in (64, 128):
            return "does not support K and V other than matching 64 or 128"
        if call.seq_len < 1 or (call.seq_len == 1 and not call.varlen):
            return "serves prefill, which is more than one token per sequence"
        return None

    @classmethod
    def entry_for(cls, call: GatedDeltaNetCall) -> Entry:
        index = call.device.index if call.device is not None else None
        arguments = dict(
            batch=call.batch,
            heads=call.heads,
            value_heads=call.value_heads,
            seq_len=call.seq_len,
            num_sequences=call.num_sequences,
            varlen=call.varlen,
            dim=call.dim_k,
            scale=call.scale,
            dtype=call.dtype,
            state_v_first=call.state_v_first,
            l2norm=call.l2norm,
            gate_in_kernel=call.gate_in_kernel,
            beta_sigmoid=call.beta_sigmoid,
            allow_neg_eigval=call.allow_neg_eigval,
            device_index=index,
        )
        return tuple(sorted(arguments.items(), key=lambda item: item[0])), lambda: cls(**arguments)

    def __init__(
        self,
        batch: int,
        heads: int,
        value_heads: int,
        seq_len: int,
        num_sequences: int,
        varlen: bool,
        dim: int,
        scale: float,
        dtype: torch.dtype,
        state_v_first: bool = False,
        l2norm: bool = False,
        gate_in_kernel: bool = False,
        beta_sigmoid: bool = False,
        allow_neg_eigval: bool = False,
        config: Optional[Dict[str, Any]] = None,
        *,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.value_heads = value_heads
        self.seq_len = seq_len
        self.num_sequences = num_sequences
        # A packed call reads the caller's int64 offsets and an equal-length one the
        # int32 offsets this kernel builds, so the two compile different programs.
        self.varlen = varlen
        self.dim = dim
        self.scale = scale
        self.dtype = dtype
        # The layout decides which axis the recurrence accumulates along, so it belongs to
        # the build identity.
        self.state_v_first = state_v_first
        # Each transform the op left to the kernel changes what every stage is built to
        # read, so all four belong to the build identity rather than to a launch argument.
        self.l2norm = l2norm
        self.gate_in_kernel = gate_in_kernel
        self.beta_sigmoid = beta_sigmoid
        self.allow_neg_eigval = allow_neg_eigval
        self.init_config(config)
        if self.config["max_local_chunks"] < 4:
            raise ValueError(
                f"max_local_chunks must be at least 4, got {self.config['max_local_chunks']}"
            )

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def _partition_metadata(
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
        for raw_idx, (raw_start, raw_end) in enumerate(
            zip(raw_offsets, raw_offsets[1:], strict=False)
        ):
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

    @staticmethod
    def _auto_local_chunks(num_chunks: int, num_heads: int, device_index: int | None) -> int:
        """The partition width the device's SM count fits."""
        sm_count = get_sm_count(device_index)
        max_local_chunks = 2 ** round(math.log2(math.sqrt(num_heads * num_chunks / sm_count) * 3))
        if num_heads >= 64 and num_chunks >= 512:
            max_local_chunks = max(max_local_chunks, 256)
        return max(max_local_chunks, 4)

    def _partitioned_initial_state(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        A: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        k_rnorm: torch.Tensor,
        chunk_size: int,
        max_local_chunks: int,
        initial_state: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        sequence_lengths: tuple[int, ...] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """The state each partition starts from, and the offsets the recurrence walks.

        Partitioning splits a long sequence at a chunk boundary and replays each piece from
        a corrected state, which asks for every length on the host. A packed call that
        passes no host copy of the offsets keeps its sequences whole instead.

        A sequence that is not a whole number of chunks still partitions: every split lands
        on a chunk boundary by construction, so only a sequence's last partition is short,
        and the warmup pass that floors a partition's chunk count is the one pass that
        skips a last partition.
        """
        num_tokens = k.shape[1]
        if sequence_lengths is None or any(length <= 0 for length in sequence_lengths):
            return initial_state, cu_seqlens, None, cu_seqlens
        num_heads = v.shape[2]
        num_chunks = tilelang.cdiv(num_tokens, chunk_size)
        use_partition = num_chunks > max_local_chunks and (
            num_heads <= 40 or (num_heads <= 64 and num_chunks >= 128)
        )
        if not use_partition:
            return initial_state, cu_seqlens, None, cu_seqlens
        (
            raw_cu_seqlens,
            cp_cu_seqlens_t,
            seq_map_c2r_t,
            seq_map_r2c_t,
            ht_mask_t,
        ) = self._partition_metadata(
            sequence_lengths,
            num_heads,
            chunk_size,
            max_local_chunks,
            True,
            k.device.index,
        )
        num_warmup_chunks, fallback_mask = get_warmup_chunks(
            g=g,
            cu_seqlens=cp_cu_seqlens_t,
            ht_mask=ht_mask_t,
            chunk_size=chunk_size,
            threshold=-10.0,
        )
        _, ht, mt = fused_gdr_h(
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
            k_rnorm=k_rnorm,
            l2norm=self.l2norm,
            beta_sigmoid=self.beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
        )
        cp_h0 = correct_initial_states(
            raw_h0=initial_state,
            ht_buffer=ht,
            mt_buffer=mt,
            fallback_mask=fallback_mask,
            seq_map_r2c=seq_map_r2c_t,
            state_v_first=self.state_v_first,
        )
        return cp_h0, cp_cu_seqlens_t, seq_map_c2r_t, raw_cu_seqlens

    @property
    def default_config(self) -> Dict[str, Any]:
        # A partition holds at most this many 64-token chunks; a longer sequence is split.
        num_chunks = tilelang.cdiv(self.batch * self.seq_len, 64)
        return {
            "max_local_chunks": self._auto_local_chunks(num_chunks, self.heads, self.device_index)
        }

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
        self._require_cuda(q=q, k=k, v=v, g=g, beta=beta)
        chunk_size = 64
        batch, seq_len = q.shape[:2]
        value_heads, dim_v = v.shape[2:]
        q, k, v, g, beta = (
            tensor.reshape(1, batch * seq_len, *tensor.shape[2:]) for tensor in (q, k, v, g, beta)
        )
        lengths: tuple[int, ...] | None
        if cu_seqlens is None:
            # An equal-length call is a packed call whose offsets step by the row length:
            # the bytes are the same, so one set of kernels serves both.
            cu_seqlens = torch.arange(
                0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=q.device
            )
            lengths = (seq_len,) * batch
        elif cu_seqlens_cpu is not None:
            lengths = tuple(int(length) for length in (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]))
        else:
            lengths = None
        cumsum = prefill_chunk_local_cumsum_bthd_tl(
            batch * seq_len,
            self.num_sequences,
            value_heads,
            chunk_size,
            str(q.dtype).split(".")[-1],
            str(cu_seqlens.dtype).split(".")[-1],
            self.gate_in_kernel,
        )
        g = cumsum(g, cu_seqlens, A_log, dt_bias) if self.gate_in_kernel else cumsum(g, cu_seqlens)
        inverse, k_rnorm = prefill_blocksolve_A_bthd(
            k,
            g,
            beta,
            cu_seqlens,
            chunk_size,
            use_gate=False,
            l2norm=self.l2norm,
            beta_sigmoid=self.beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
        )
        initial, offsets, cp_seq_map, raw_offsets = self._partitioned_initial_state(
            k,
            v,
            inverse,
            g,
            beta,
            k_rnorm,
            chunk_size,
            self.config["max_local_chunks"],
            initial_state,
            cu_seqlens,
            lengths,
        )
        o, _states, final_state = fused_gdr_fwd(
            q,
            k,
            v,
            inverse,
            g,
            beta,
            scale=self.scale,
            initial_state=initial,
            output_h=False,
            cu_seqlens=offsets,
            cp_seq_map=cp_seq_map,
            raw_cu_seqlens=raw_offsets,
            chunk_size=chunk_size,
            state_head_first=False,
            chunks_per_sequence=0,
            state_v_first=self.state_v_first,
            k_rnorm=k_rnorm,
            l2norm=self.l2norm,
            beta_sigmoid=self.beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
        )
        return o.reshape(batch, seq_len, value_heads, dim_v), final_state
