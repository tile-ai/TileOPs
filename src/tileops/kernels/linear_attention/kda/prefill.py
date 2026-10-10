"""Chunked Kimi Delta Attention (KDA) prefill: chunk-local work, then one scan."""

from typing import Optional, Tuple

import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    KDACall,
    KDAFwdInterface,
)
from tileops.kernels.linear_attention.kda.chunk_programs import (
    chunk_prepare_program,
    chunk_scan_program,
)
from tileops.kernels.tiling import align_up
from tileops.utils import get_shared_memory_optin, get_sm_version

__all__ = ["KDAChunkPrefillFwdKernel"]


def packed_offsets(
    batch: int, seq_len: int, cu_seqlens: Optional[torch.Tensor], device: torch.device
) -> torch.Tensor:
    """The offsets both programs read their sequence boundaries from.

    They take one packed token axis, so an equal-length call states the same
    boundaries its shapes already state. Writing them out keeps one path through
    the kernels rather than a second schedule for the equal-length case.
    """
    if cu_seqlens is not None:
        return cu_seqlens
    return torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int64, device=device)


CHUNK_SIZE = 64


def _prepare_bytes(dim: int, element: int, sequences: int, lean: bool) -> int:
    """Shared memory of the chunk-local program, for a K = V = *dim* state.

    The default form holds every buffer at once. In the lean one TileLang main puts
    the inverse's two tiles in the gated query's space once it is written out, and
    what does not fit there past the other buffers.
    """
    tiles = 4 * CHUNK_SIZE * dim * element + 4 * CHUNK_SIZE * dim + 4 * CHUNK_SIZE
    inverse = 2 * CHUNK_SIZE * CHUNK_SIZE * element
    if lean:
        inverse = max(0, inverse - CHUNK_SIZE * dim * element)
    return tiles + inverse + align_up(4 * (sequences + 1), 16)


def _scan_bytes(dim: int, value_tile: int, element: int, sequences: int) -> int:
    """Shared memory of the scan, every buffer, with *value_tile* state columns a CTA."""
    tiles = dim * value_tile + 3 * CHUNK_SIZE * dim + CHUNK_SIZE * (CHUNK_SIZE + value_tile)
    return tiles * element + align_up(4 * (sequences + 1), 16)


def _plan(
    dim: int, element: int, sequences: int, budget: int, arch: int
) -> Optional[Tuple[bool, int]]:
    """Whether the chunk-local program takes its lean form, and the scan's value tile.

    The widest programs that fit *budget*, or ``None`` when none does. On SM90
    TileLang rounds a program's shared memory up to 1 KB.
    """

    def fits(size: int) -> bool:
        return (align_up(size, 1024) if arch == 90 else size) <= budget

    lean = not fits(_prepare_bytes(dim, element, sequences, lean=False))
    if lean and not fits(_prepare_bytes(dim, element, sequences, lean=True)):
        return None
    for value_tile in (dim, dim // 2):
        if fits(_scan_bytes(dim, value_tile, element, sequences)):
            return lean, value_tile
    return None


class KDAChunkPrefillFwdKernel(Kernel, KDAFwdInterface):
    """Prefill over a 64-token chunk, equal-length or packed varlen.

    The chunk-local half runs one CTA per (chunk, value head) and the scan one
    CTA per (sequence, value head); the two meet in a workspace holding the WY
    vectors, the gated query and key, the intra-chunk attention and the chunk
    decay. Where shared memory holds less, the chunk-local half takes its lean
    form and the scan splits the state's value columns across CTAs.
    """

    supported_archs = [80, 89, 90]

    @classmethod
    def refusal(cls, call: KDACall) -> Optional[str]:
        """Why this kernel does not serve *call*, or ``None`` when it does."""
        chunked = call.chunk_refusal
        if chunked is not None:
            return chunked
        if call.seq_len < 2:
            return "serves a sequence of at least two tokens"
        plan = _plan(call.dim_k, call.dtype.itemsize, call.sequences, call.smem_budget, call.arch)
        if plan is None:
            return (
                f"needs more than {call.smem_budget} bytes of shared memory for "
                f"{call.sequences} sequences"
            )
        return None

    @classmethod
    def entry_for(cls, call: KDACall) -> Entry:
        index = call.device.index if call.device is not None else None
        identity = (
            call.batch,
            call.seq_len,
            call.heads,
            call.value_heads,
            call.dim_k,
            call.dim_v,
            call.scale,
            call.l2norm,
            call.varlen,
            call.dtype,
            index,
        )
        return identity, lambda: cls(
            batch=call.batch,
            seq_len=call.seq_len,
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
        seq_len: int,
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
        """Hold the call facts both programs are built from."""
        super().__init__(device_index=device_index)
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.value_heads = value_heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.scale = scale
        self.l2norm = l2norm
        self.dtype = dtype

    def _scan_value_tile(self, value_tile: int, arch: int, *, total: int, num_seqs: int) -> int:
        """Trade repeated key loads for more CTAs on sparse, long SM89 scans."""
        minimum_tokens = 4096
        sparse_heads = 8
        narrow_value_tile = 16
        if (
            num_seqs == 1
            and total >= minimum_tokens
            and self.value_heads <= sparse_heads
            and self.dim_k == 128
            and arch == 89
        ):
            return min(value_tile, narrow_value_tile)
        return value_tile

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
        """Run the chunk-local half, then the scan; see the interface for the tensors."""
        del A_log, dt_bias, cu_seqlens_cpu
        self._require_cuda(q=q, k=k, v=v, g=g, beta=beta)
        batch, seq_len = q.shape[:2]
        H, K = q.shape[2], q.shape[3]
        HV, V = v.shape[2], v.shape[3]
        total = batch * seq_len
        offsets = packed_offsets(batch, seq_len, cu_seqlens, q.device)
        num_seqs = offsets.numel() - 1

        flat = lambda tensor, width: tensor.reshape(1, total, tensor.shape[2], width)  # noqa: E731
        qf, kf = flat(q, K), flat(k, K)
        vf, gf = flat(v, V), flat(g, K)
        bf = beta.reshape(1, total, HV)
        state = (
            torch.zeros(num_seqs, HV, K, V, dtype=torch.float32, device=q.device)
            if initial_state is None
            else initial_state
        )

        name = self.dtype_to_str(self.dtype)
        index = q.device.index
        arch = get_sm_version(index)
        plan = _plan(K, self.dtype.itemsize, num_seqs, get_shared_memory_optin(index), arch)
        if plan is None:
            raise ValueError(f"{num_seqs} sequences need more shared memory than the device has")
        lean, value_tile = plan
        value_tile = self._scan_value_tile(value_tile, arch, total=total, num_seqs=num_seqs)
        prepare = chunk_prepare_program(
            H, HV, K, V, CHUNK_SIZE, name, self.scale, self.l2norm, total, num_seqs, lean=lean
        )
        scan = chunk_scan_program(
            HV, K, V, CHUNK_SIZE, name, total, num_seqs, value_tile=value_tile
        )
        w, u, qg, kg, aqk, dec = prepare(qf, kf, vf, gf, bf, offsets)
        o, final_state = scan(w, u, qg, kg, aqk, dec, state, offsets)
        return o.reshape(batch, seq_len, HV, V), final_state
