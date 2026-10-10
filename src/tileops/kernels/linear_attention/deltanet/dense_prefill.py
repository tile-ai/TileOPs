"""SM90 DeltaNet prefill with its own dense partition orchestration."""

import functools
import math
from typing import Any, Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    DeltaNetCall,
    DeltaNetFwdInterface,
    head_count_refusal,
)
from tileops.kernels.linear_attention.deltanet.partition_scan import partition_scan
from tileops.kernels.linear_attention.deltanet.prefill_forward import deltanet_prefill_fwd
from tileops.kernels.linear_attention.deltanet.prefill_prepare import deltanet_partition_states
from tileops.kernels.linear_attention.gdn.prefill_forward import fused_gdr_fwd
from tileops.kernels.linear_attention.gdn.prefill_prepare import (
    fused_gdr_h,
    prefill_blocksolve_A_bthd,
)
from tileops.utils import get_sm_count

__all__ = ["DeltaNetDensePrefillFwdKernel"]


class DeltaNetDensePrefillFwdKernel(Kernel, DeltaNetFwdInterface):
    """Ungated delta rule: the Gated DeltaNet (GDN) block solve and partitioned recurrence, g=0."""

    supported_archs = [90]

    @classmethod
    def refusal(cls, call: DeltaNetCall) -> Optional[str]:
        """Why this kernel does not serve *call*, or ``None`` when it does.

        The pipeline runs chunks of 64 tokens over a square 16-bit state, equal-length or
        packed, with a row that is not a whole chunk, and with the Q/K L2 normalization
        taken in kernel.
        """
        heads = head_count_refusal(call.heads)
        if heads is not None:
            return heads
        unsupported = [
            name
            for name, present in (
                ("a single token outside a packed call", call.seq_len == 1 and not call.varlen),
                (
                    "K/V dimensions other than matching 64 or 128",
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
    def entry_for(cls, call: DeltaNetCall) -> Entry:
        index = call.device.index if call.device is not None else None
        arguments = dict(
            batch=call.batch,
            heads=call.heads,
            seq_len=call.seq_len,
            num_sequences=call.num_sequences,
            varlen=call.varlen,
            dim=call.dim_k,
            scale=call.scale,
            dtype=call.dtype,
            l2norm=call.l2norm,
            device_index=index,
        )
        return tuple(sorted(arguments.items(), key=lambda item: item[0])), lambda: cls(**arguments)

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        num_sequences: int,
        varlen: bool,
        dim: int,
        scale: float,
        dtype: torch.dtype,
        l2norm: bool = False,
        config: Optional[Dict[str, Any]] = None,
        *,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.num_sequences = num_sequences
        # A packed call reads the caller's int64 offsets and an equal-length one the
        # int32 offsets this kernel builds, so the two compile different programs.
        self.varlen = varlen
        self.dim = dim
        self.scale = scale
        # Normalizing Q and K changes what every stage is built to read, so it belongs to
        # the build identity rather than to a launch argument.
        self.l2norm = l2norm
        self.init_config(config)
        if self.config["max_local_chunks"] < 4:
            raise ValueError(
                f"max_local_chunks must be at least 4, got {self.config['max_local_chunks']}"
            )
        # A zero gate is the ungated delta rule. Keep it across calls so the
        # measured path does not include an extra GPU memset per invocation.
        device = (
            torch.device("cuda", device_index) if device_index is not None else torch.device("cuda")
        )
        self.zero_gate = torch.zeros(
            (1, batch * seq_len, heads), dtype=torch.float32, device=device
        )

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def _partition_metadata(
        sequence_lengths: tuple[int, ...],
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
        torch.Tensor | None,
    ]:
        """Cache only shape-derived offsets, maps and replay counts, never recurrent state.

        Without a gate no decay cuts a partition's replay short, so the warmup pass replays
        every chunk of a partition that is not its sequence's last, and none of the last. The
        count, per partition and per partition and head, is shape-derived and is cached with
        the offsets.
        """
        device = (
            torch.device("cuda", device_index) if device_index is not None else torch.device("cuda")
        )
        raw_offsets = [0]
        for length in sequence_lengths:
            raw_offsets.append(raw_offsets[-1] + length)
        raw_cu = torch.tensor(raw_offsets, dtype=torch.int32, device=device)
        if not use_partition:
            return raw_cu, None, None, None, None, None

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
        replay = [
            0 if final else (end - start) // 64
            for start, end, final in zip(
                cp_offsets, cp_offsets[1:], final_partition_mask, strict=False
            )
        ]
        return (
            raw_cu,
            torch.tensor(cp_offsets, dtype=torch.int32, device=device),
            torch.tensor(cp_to_raw, dtype=torch.int32, device=device),
            torch.tensor(raw_to_cp, dtype=torch.int32, device=device),
            torch.tensor(replay, dtype=torch.int32, device=device),
            torch.tensor([[count] * heads for count in replay], dtype=torch.int32, device=device),
        )

    @staticmethod
    def _local_chunks(num_chunks: int, heads: int, device_index: int | None) -> int:
        """The partition width the device's SM count fits."""
        sm_count = get_sm_count(device_index)
        local_chunks = 2 ** round(math.log2(math.sqrt(heads * num_chunks / sm_count) * 3))
        if heads >= 64 and num_chunks >= 512:
            local_chunks = max(local_chunks, 256)
        return max(local_chunks, 4)

    def _partitions(
        self,
        num_tokens: int,
        heads: int,
        sequence_lengths: tuple[int, ...] | None,
        device_index: int | None,
    ) -> tuple[torch.Tensor, ...] | None:
        """Where the sequences split, or ``None`` when they are walked whole.

        Partitioning splits a long sequence at a chunk boundary and replays each piece from
        a corrected state, which asks for every length on the host. A packed call that
        passes no host copy of the offsets keeps its sequences whole instead.

        A sequence that is not a whole number of chunks still partitions: every split lands
        on a chunk boundary by construction, so only a sequence's last partition is short,
        and it is the one partition the warmup pass skips.
        """
        if sequence_lengths is None or any(length <= 0 for length in sequence_lengths):
            return None
        max_local_chunks = self.config["max_local_chunks"]
        num_chunks = -(-num_tokens // 64)
        # Splitting pays once the longest sequence spans two partitions; below that the
        # partition pass costs more than the chunks it takes off the walk.
        longest = -(-max(sequence_lengths) // 64)
        use_partition = longest >= 2 * max_local_chunks and (
            heads <= 40 or (heads <= 64 and num_chunks >= 128)
        )
        if not use_partition:
            return None
        return self._partition_metadata(
            sequence_lengths, heads, max_local_chunks, True, device_index
        )

    def _gdn_partitioned(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        inverse: torch.Tensor,
        beta: torch.Tensor,
        k_rnorm: torch.Tensor,
        initial_state: torch.Tensor | None,
        split: tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Partitioned prefill on the GDN programs with a zero gate."""
        raw_cu, offsets, sequence_of, first_of, _, replay_per_head = split
        partition_h, partition_m = fused_gdr_h(
            k=k,
            v=v,
            a=inverse,
            g=self.zero_gate,
            b=beta,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=offsets,
            num_warmup_chunks=replay_per_head,
            k_rnorm=k_rnorm,
            l2norm=self.l2norm,
        )
        start_state = partition_scan(initial_state, partition_h, partition_m, first_of)
        return fused_gdr_fwd(
            q,
            k,
            v,
            inverse,
            self.zero_gate,
            beta,
            scale=self.scale,
            initial_state=start_state,
            output_final_state=True,
            cu_seqlens=offsets,
            cp_seq_map=sequence_of,
            raw_cu_seqlens=raw_cu,
            chunk_size=64,
            k_rnorm=k_rnorm,
            l2norm=self.l2norm,
        )

    @property
    def default_config(self) -> Dict[str, Any]:
        # A partition holds at most this many 64-token chunks; a longer sequence is split.
        num_chunks = -(-self.batch * self.seq_len // 64)
        return {"max_local_chunks": self._local_chunks(num_chunks, self.heads, self.device_index)}

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
        self._require_cuda(q=q, k=k, v=v, beta=beta)
        batch, seq_len, heads, dim = q.shape
        q_flat, k_flat, v_flat, beta_flat = (
            tensor.reshape(1, batch * seq_len, *tensor.shape[2:]) for tensor in (q, k, v, beta)
        )
        lengths: tuple[int, ...] | None
        if cu_seqlens is None:
            # An equal-length call is a packed call whose offsets step by the row length:
            # the bytes are the same, so one set of kernels serves both. The offsets are
            # shape-derived and come from the cache rather than a per-call launch.
            lengths = (seq_len,) * batch
            cu_seqlens = self._partition_metadata(
                lengths, heads, self.config["max_local_chunks"], False, q.device.index
            )[0]
        elif cu_seqlens_cpu is not None:
            lengths = tuple(int(length) for length in (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]))
        else:
            lengths = None
        inverse, k_rnorm = prefill_blocksolve_A_bthd(
            k_flat, self.zero_gate, beta_flat, cu_seqlens, 64, use_gate=False, l2norm=self.l2norm
        )
        split = self._partitions(k_flat.shape[1], heads, lengths, q.device.index)
        if dim == 128 and split is not None:
            # Partitions of a 128-wide state would take this file's programs to 64 state
            # columns per CTA and two waves, so they keep the GDN programs at 128.
            o, final_state = self._gdn_partitioned(
                q_flat, k_flat, v_flat, inverse, beta_flat, k_rnorm, initial_state, split
            )
        else:
            partitions = None
            if split is not None:
                raw_cu, offsets, sequence_of, first_of, replay, _ = split
                partition_h, partition_m = deltanet_partition_states(
                    k_flat, v_flat, inverse, beta_flat, offsets, replay, k_rnorm, self.l2norm
                )
                partitions = (offsets, sequence_of, first_of, partition_h, partition_m)
                cu_seqlens = raw_cu
            o, final_state = deltanet_prefill_fwd(
                q_flat,
                k_flat,
                v_flat,
                inverse,
                beta_flat,
                self.scale,
                initial_state,
                cu_seqlens,
                partitions,
                k_rnorm,
                self.l2norm,
            )
        return o.reshape(batch, seq_len, heads, dim), final_state
