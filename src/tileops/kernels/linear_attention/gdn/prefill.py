"""Dense Gated DeltaNet (GDN) inference prefill, equal-length or packed."""

import functools
import math
from typing import Any, Dict, Optional, Tuple

import tilelang
import torch

from tileops.kernels.kernel_base import CompiledProgram, Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    GDNCall,
    GDNFwdInterface,
    head_count_refusal,
)
from tileops.kernels.linear_attention.gdn.prefill_forward import (
    fused_gdr_fwd,
    fused_gdr_fwd_program,
)
from tileops.kernels.linear_attention.gdn.prefill_fused import (
    FUSED_PREFILL_DIM,
    fused_gdr_prefill,
    fused_gdr_prefill_program,
)
from tileops.kernels.linear_attention.gdn.prefill_prepare import (
    correct_initial_states,
    correct_initial_states_program,
    fused_gdr_h,
    fused_gdr_h_program,
    get_warmup_chunks,
    prefill_blocksolve_A_bthd,
    prefill_blocksolve_A_bthd_tl,
    prefill_chunk_local_cumsum_bthd_tl,
    prefill_solve_warpgroup,
    prefill_solve_warpgroup_tl,
    warmup_chunks_program,
)
from tileops.utils import get_sm_count

__all__ = ["GDNDensePrefillFwdKernel"]


class GDNDensePrefillFwdKernel(Kernel, GDNFwdInterface):
    """SM90 equal-length BTHD inference prefill.

    This is the inference owner of the retained partitioned prefill pipeline.
    The Op layer admits only the currently supported region; the kernel keeps
    the unified public call signature so no layout or ABI adapter is needed.
    """

    supported_archs = [90]

    @classmethod
    def refusal(cls, call: GDNCall) -> Optional[str]:
        """Why this kernel does not serve *call*, or ``None`` when it does.

        Prefill in chunks of 64 tokens over a 64- or 128-wide square state, equal-length or
        packed, with a row that is not a whole chunk, with grouped value heads, with the
        state key-major or value-major, and with the Q/K normalization, the gate and the
        beta transform taken in kernel.
        """
        heads = head_count_refusal(call.heads, call.value_heads)
        if heads is not None:
            return heads
        if call.dim_k != call.dim_v or call.dim_k not in (64, 128):
            return "does not support K and V other than matching 64 or 128"
        if call.seq_len < 1 or (call.seq_len == 1 and not call.varlen):
            return "serves prefill, which is more than one token per sequence"
        return None

    @classmethod
    def entry_for(cls, call: GDNCall) -> Entry:
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

    @staticmethod
    def _uniform_length(sequence_lengths: tuple[int, ...] | None) -> int:
        """The one length every sequence has, or 0 when they differ or are unknown.

        Packed sequences of one length are the same bytes as an equal-length batch, so the
        block solve addresses their chunks by arithmetic instead of searching the offsets.
        """
        if not sequence_lengths or len(set(sequence_lengths)) != 1 or sequence_lengths[0] <= 0:
            return 0
        return sequence_lengths[0]

    @staticmethod
    def _partition_pays(
        sequence_lengths: tuple[int, ...],
        num_heads: int,
        chunk_size: int,
        device_index: int | None,
    ) -> bool:
        """Whether splitting the sequences shortens the recurrence enough to repay its passes.

        One CTA walks one (sequence, value head) and holds an SM for the walk, so an
        unsplit call lasts about as long as its long walks: those over half the longest.
        Splitting helps only by spreading the long walks over SMs they leave idle, and
        it adds warmup and correction passes. When the long walks already hold more than
        half the SMs, or more than a quarter over walks of at most 32 chunks, those passes
        cost more than the spread saves and the sequences stay whole.
        """
        sm_count = get_sm_count(device_index)
        chunks = [tilelang.cdiv(length, chunk_size) for length in sequence_lengths]
        longest = max(chunks)
        long_walks = num_heads * sum(1 for count in chunks if 2 * count > longest)
        # Walks this short gain too little from spreading to repay the passes at a quarter
        # of the SMs busy; measured on H200 rows of 16 to 1024 chunks.
        short_walk_chunks = 32
        if 2 * long_walks > sm_count:
            return False
        return not (4 * long_walks > sm_count and longest <= short_walk_chunks)

    def _fused_serves(
        self,
        num_tokens: int,
        num_heads: int,
        sequence_lengths: tuple[int, ...] | None,
        device_index: int | None,
    ) -> bool:
        """Whether the one-kernel prefill outruns the solve pass and the partitioned walk.

        The one-kernel prefill forms each chunk's inverse on the SM that walks the chunk, so
        it needs walks for most SMs. It serves the call when the walks of more than half the
        longest length hold more than half the SMs and the sequences stay whole, or when all
        walks together hold more than three quarters of the SMs, where it also beats
        splitting the long walks (measured on H200 rows of 64 to 128 walks).
        """
        if not sequence_lengths or any(length <= 0 for length in sequence_lengths):
            return False
        sm_count = get_sm_count(device_index)
        longest = max(sequence_lengths)
        long_walks = num_heads * sum(1 for length in sequence_lengths if 2 * length > longest)
        walks = num_heads * len(sequence_lengths)
        if 4 * walks > 3 * sm_count:
            return True
        return 2 * long_walks > sm_count and not self._splits(
            num_tokens, num_heads, sequence_lengths, device_index
        )

    def _splits(
        self,
        num_tokens: int,
        num_heads: int,
        sequence_lengths: tuple[int, ...] | None,
        device_index: int | None,
    ) -> bool:
        """Whether the recurrence walks the sequences in partitions."""
        chunk_size = 64
        if sequence_lengths is None or any(length <= 0 for length in sequence_lengths):
            return False
        num_chunks = tilelang.cdiv(num_tokens, chunk_size)
        return (
            num_chunks > self.config["max_local_chunks"]
            and (num_heads <= 40 or (num_heads <= 64 and num_chunks >= 128))
            and self._partition_pays(sequence_lengths, num_heads, chunk_size, device_index)
        )

    def _split(
        self,
        num_tokens: int,
        num_heads: int,
        sequence_lengths: tuple[int, ...] | None,
        device_index: int | None,
    ) -> tuple[torch.Tensor, ...] | None:
        """Where the sequences split, or ``None`` when they are walked whole.

        Partitioning splits a long sequence at a chunk boundary and replays each piece from
        a corrected state, which asks for every length on the host. A packed call that
        passes no host copy of the offsets keeps its sequences whole instead.

        A sequence that is not a whole number of chunks still partitions: every split lands
        on a chunk boundary by construction, so only a sequence's last partition is short,
        and the warmup pass that floors a partition's chunk count is the one pass that
        skips a last partition.
        """
        if not self._splits(num_tokens, num_heads, sequence_lengths, device_index):
            return None
        return self._partition_metadata(
            sequence_lengths,
            num_heads,
            64,
            self.config["max_local_chunks"],
            True,
            device_index,
        )

    def _partitioned_initial_state(
        self,
        programs: tuple,
        k: torch.Tensor,
        v: torch.Tensor,
        A: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        k_rnorm: torch.Tensor | None,
        initial_state: torch.Tensor | None,
        split: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The state each partition starts from, and the offsets the recurrence walks."""
        _, _, warmup_program, states_program, correct_program, _ = programs
        (
            raw_cu_seqlens,
            cp_cu_seqlens_t,
            seq_map_c2r_t,
            seq_map_r2c_t,
            ht_mask_t,
        ) = split
        num_warmup_chunks, fallback_mask = get_warmup_chunks(
            warmup_program,
            g=g,
            cu_seqlens=cp_cu_seqlens_t,
            ht_mask=ht_mask_t,
        )
        ht, mt = fused_gdr_h(
            states_program,
            k=k,
            v=v,
            a=A,
            g=g,
            b=beta,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=cp_cu_seqlens_t,
            num_warmup_chunks=num_warmup_chunks,
            k_rnorm=k_rnorm,
        )
        cp_h0 = correct_initial_states(
            correct_program,
            raw_h0=initial_state,
            ht_buffer=ht,
            mt_buffer=mt,
            fallback_mask=fallback_mask,
            seq_map_r2c=seq_map_r2c_t,
            state_v_first=self.state_v_first,
        )
        return cp_h0, cp_cu_seqlens_t, seq_map_c2r_t, raw_cu_seqlens

    def _programs(
        self,
        cfg: Dict[str, Any],
        offsets_dtype: torch.dtype,
        beta_dtype: torch.dtype,
        state_dtype: torch.dtype | None,
        solve: str,
        uniform: int,
        partitions: int,
        device_index: int | None,
    ) -> tuple:
        """The programs one launch runs, ``None`` for a pass it skips: the gate cumsum, the
        one-kernel prefill or the block solve, the warmup, partition and correction passes,
        and the forward.

        *solve* is ``"fused"`` for the one-kernel prefill, else ``"warpgroup"`` or
        ``"blocksolve"`` for the solve pass over rows of length *uniform* (0 when they
        differ); *partitions* is how many partitions the sequences are split into, 0 when
        they are walked whole; *state_dtype* is the initial state's dtype, ``None`` without
        one.
        """
        total = self.batch * self.seq_len
        heads, value_heads, dim, dtype = self.heads, self.value_heads, self.dim, self.dtype
        dtype_name, offsets_name = self.dtype_to_str(dtype), self.dtype_to_str(offsets_dtype)
        # The cumsum writes the gate in float32, which every later pass reads.
        gate_dtype = torch.float32
        cumsum = prefill_chunk_local_cumsum_bthd_tl(
            total,
            self.num_sequences,
            value_heads,
            64,
            dtype_name,
            offsets_name,
            self.gate_in_kernel,
        )
        if solve == "fused":
            prefill = fused_gdr_prefill_program(
                value_heads,
                heads,
                self.scale,
                dtype,
                gate_dtype,
                beta_dtype,
                offsets_dtype,
                state_dtype is not None,
                False,
                state_v_first=self.state_v_first,
                beta_sigmoid=self.beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
            )
            return CompiledProgram(cumsum), CompiledProgram(prefill), None, None, None, None
        if solve == "warpgroup":
            inverse = prefill_solve_warpgroup_tl(
                total,
                total // uniform,
                value_heads,
                heads,
                dim,
                dtype_name,
                self.beta_sigmoid,
                self.allow_neg_eigval,
                uniform,
            )
        else:
            inverse = prefill_blocksolve_A_bthd_tl(
                total,
                self.num_sequences,
                value_heads,
                heads,
                64,
                dim,
                dtype_name,
                offsets_name,
                use_gate=False,
                l2norm=self.l2norm,
                beta_sigmoid=self.beta_sigmoid,
                allow_neg_eigval=self.allow_neg_eigval,
                uniform_seq_len=uniform,
            )
        warmup = states = correct = None
        start_dtype = state_dtype
        if partitions:
            start_dtype = torch.float32 if state_dtype is None else state_dtype
            warmup = CompiledProgram(
                warmup_chunks_program(value_heads, gate_dtype, torch.bool, torch.int32)
            )
            states = CompiledProgram(
                fused_gdr_h_program(
                    value_heads,
                    heads,
                    dim,
                    dtype,
                    gate_dtype,
                    beta_dtype,
                    torch.int32,
                    is_varlen=True,
                    is_cp=True,
                    l2norm=self.l2norm,
                    beta_sigmoid=self.beta_sigmoid,
                    allow_neg_eigval=self.allow_neg_eigval,
                )
            )
            correct = CompiledProgram(
                correct_initial_states_program(
                    value_heads,
                    dim,
                    start_dtype,
                    dtype,
                    torch.int32,
                    torch.bool,
                    state_dtype is not None,
                    self.state_v_first,
                )
            )
        forward = fused_gdr_fwd_program(
            value_heads,
            heads,
            dim,
            partitions or self.num_sequences,
            self.scale,
            dtype,
            gate_dtype,
            beta_dtype,
            torch.int32 if partitions else offsets_dtype,
            h0_dtype=start_dtype,
            is_varlen=True,
            is_cp=bool(partitions),
            state_v_first=self.state_v_first,
            l2norm=self.l2norm,
            beta_sigmoid=self.beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            device_index=device_index,
        )
        return (
            CompiledProgram(cumsum),
            CompiledProgram(inverse),
            warmup,
            states,
            correct,
            CompiledProgram(forward),
        )

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
        self._require_devices(q=q, k=k, v=v, g=g, beta=beta)
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
        device_index = q.device.index
        if (
            dim_v == FUSED_PREFILL_DIM
            and not self.l2norm
            and self._fused_serves(batch * seq_len, value_heads, lengths, device_index)
        ):
            # Whole sequences of 128-wide heads that keep most SMs busy: one kernel forms
            # each chunk's inverse beside the walk instead of a separate solve pass.
            solve, uniform, split = "fused", 0, None
        else:
            uniform = self._uniform_length(lengths)
            # An SM holds about seven warp-group solves at once; below about one such wave
            # of (chunk, head) pairs each pair's step chain decides and the one-warp solve
            # is as fast.
            warp_group_pairs = 8 * get_sm_count(device_index)
            pairs = (
                len(lengths) * tilelang.cdiv(uniform, chunk_size) * value_heads if uniform else 0
            )
            # The warp-group solve beats the one-warp solve on bfloat16 rows of one length,
            # whose chunks it addresses by arithmetic; in float16 the one-warp solve is the
            # faster (measured on H200), and a normalized key takes it for the norms the
            # later passes read.
            warp_group = (
                pairs >= warp_group_pairs and not self.l2norm and self.dtype == torch.bfloat16
            )
            solve = "warpgroup" if warp_group else "blocksolve"
            split = self._split(k.shape[1], value_heads, lengths, device_index)
        programs = self.program(
            self._programs,
            cu_seqlens.dtype,
            beta.dtype,
            None if initial_state is None else initial_state.dtype,
            solve,
            uniform,
            0 if split is None else split[1].shape[0] - 1,
            device_index,
        )
        cumsum, solve_program = programs[:2]
        g = cumsum(g, cu_seqlens, A_log, dt_bias) if self.gate_in_kernel else cumsum(g, cu_seqlens)
        if solve == "fused":
            o, final_state = fused_gdr_prefill(
                solve_program,
                q,
                k,
                v,
                g,
                beta,
                cu_seqlens,
                initial_state=initial_state,
            )
            return o.reshape(batch, seq_len, value_heads, dim_v), final_state
        if solve == "warpgroup":
            inverse = prefill_solve_warpgroup(solve_program, k, beta)
            k_rnorm = None
        else:
            inverse, k_rnorm = prefill_blocksolve_A_bthd(
                solve_program, k, g, beta, cu_seqlens, chunk_size, l2norm=self.l2norm
            )
        if split is None:
            initial, offsets, cp_seq_map, raw_offsets = (
                initial_state,
                cu_seqlens,
                None,
                cu_seqlens,
            )
        else:
            initial, offsets, cp_seq_map, raw_offsets = self._partitioned_initial_state(
                programs, k, v, inverse, g, beta, k_rnorm, initial_state, split
            )
        o, final_state = fused_gdr_fwd(
            programs[-1],
            q,
            k,
            v,
            inverse,
            g,
            beta,
            initial_state=initial,
            cu_seqlens=offsets,
            cp_seq_map=cp_seq_map,
            raw_cu_seqlens=raw_offsets,
            state_v_first=self.state_v_first,
            k_rnorm=k_rnorm,
        )
        return o.reshape(batch, seq_len, value_heads, dim_v), final_state
