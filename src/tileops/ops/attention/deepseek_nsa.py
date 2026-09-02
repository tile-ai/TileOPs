from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.attention import (
    NSACmpFwdVarlenKernel,
    NSAFwdVarlenKernel,
    NSATopkVarlenKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.perf.profile import tensor_core_roof

from .._validation import check_tensor_shape
from ..op_base import Op

__all__ = [
    "NSACmpFwdVarlenOp",
    "NSAFwdVarlenOp",
    "NSATopkVarlenOp",
]


def _packed_query_dims(q: torch.Tensor) -> tuple[int, int, int]:
    """``(c_seq_len, heads, dim)`` off a packed query. Every other input is sized from it."""
    if not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    if q.ndim != 3:
        raise ValueError(f"q must have shape [c_seq_len, heads, dim]; got {tuple(q.shape)}")
    return q.shape[0], q.shape[1], q.shape[2]


def _packed_request_count(offsets: torch.Tensor) -> int:
    """Requests in the packed batch. ``offsets`` sizes every other metadata tensor."""
    if not offsets.is_cuda:
        raise ValueError("offsets must be a CUDA tensor")
    if offsets.ndim != 1 or offsets.shape[0] < 2:
        raise ValueError(
            f"offsets must be a 1D tensor of at least two bounds; got {tuple(offsets.shape)}"
        )
    return offsets.shape[0] - 1


def _require_rank(name: str, tensor: torch.Tensor, ndim: int) -> None:
    """Gate a rank before a dimension is read off it."""
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D; got shape {tuple(tensor.shape)}")


def _validate_tiling(**values: int) -> None:
    """Tiling widths and kept-block counts reach the kernel builder as loop bounds."""
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive; got {value}")


def _resolve_group(heads: int, head_kv: int) -> int:
    """Query heads per KV head. One program serves one group, and its MMA needs 16 rows."""
    if head_kv <= 0 or heads % head_kv:
        raise ValueError(f"heads ({heads}) must be a positive multiple of head_kv ({head_kv})")
    group = heads // head_kv
    if group % 16:
        raise ValueError(f"heads // head_kv must be a multiple of 16; got {group}")
    return group


class NSATopkVarlenOp(Op):
    """Native Sparse Attention (NSA) block selection over a ragged batch.

    Scores each compressed chunk against the query and returns, per token and per KV
    head, the ``selected_block_num`` block ids the sparse forward will attend to.

    Sequence layout is packed: ``q`` holds every request's tokens back to back and
    ``offsets`` marks the boundaries, so the batch size and the chunk count come from
    the call rather than from construction.
    """

    def __init__(
        self,
        scale: float,
        selected_block_num: int,
        bc: int,
        bs: int,
        accum_dtype: torch.dtype,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            scale: Softmax scale applied to the QK product.
            selected_block_num: Blocks to keep per token and KV head.
            bc: Chunk tile width.
            bs: Compression block size.
            accum_dtype: Accumulator dtype.
            tune: Whether to autotune, applied when a kernel is first built.
            kernel_map: Optional kernel override dict.
        """
        _validate_tiling(selected_block_num=selected_block_num, bc=bc, bs=bs)
        self.scale = scale
        self.selected_block_num = selected_block_num
        self.bc = bc
        self.bs = bs
        self.accum_dtype = accum_dtype
        self.tune = tune

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"nsa_topk_varlen_kernel": NSATopkVarlenKernel}

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_cmp_shape: tuple[int, ...],
        lse_in_shape: tuple[int, ...],
        offsets_shape: tuple[int, ...],
        chunk_offsets_shape: tuple[int, ...],
        token_indices_shape: tuple[int, ...],
    ) -> Dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: one block-id row per token and KV head."""
        return {"block_indices": (q_shape[0], k_cmp_shape[1], self.selected_block_num)}

    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        seq_num: int,
        c_seq_len: int,
        heads: int,
        dim: int,
        chunk_num: int,
        group: int,
        dtype: torch.dtype,
        device_index: "int | None",
    ) -> Kernel:
        key = (
            seq_num,
            c_seq_len,
            heads,
            dim,
            chunk_num,
            group,
            self.scale,
            self.selected_block_num,
            self.bc,
            self.bs,
            dtype,
            self.accum_dtype,
            device_index,
            self.tune,
        )
        return self.get_or_build_kernel(
            "nsa_topk_varlen_kernel",
            inputs,
            key=key,
            build=lambda: self.kernel_map["nsa_topk_varlen_kernel"](
                seq_num=seq_num,
                c_seq_len=c_seq_len,
                heads=heads,
                dim=dim,
                chunk_num=chunk_num,
                group=group,
                scale=self.scale,
                selected_block_num=self.selected_block_num,
                bc=self.bc,
                bs=self.bs,
                dtype=dtype,
                accum_dtype=self.accum_dtype,
                tune=self.tune,
            ),
        )

    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        lse_in: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Select the blocks each token attends to.

        Args:
            q: Queries, packed over the batch [c_seq_len, heads, dim].
            k_cmp: Compressed keys [chunk_num, head_kv, dim].
            lse_in: Log-sum-exp from the compression forward [c_seq_len, heads].
            offsets: Request boundaries into the packed sequence [seq_num + 1].
            chunk_offsets: Per-request chunk boundaries [seq_num + 1].
            token_indices: Request id and in-request position per token [c_seq_len, 2].

        Returns:
            Selected block ids [c_seq_len, head_kv, selected_block_num].
        """
        self._validate_dtypes(q, k_cmp, lse_in, offsets, chunk_offsets, token_indices)
        c_seq_len, heads, dim = _packed_query_dims(q)
        seq_num = _packed_request_count(offsets)
        _require_rank("k_cmp", k_cmp, 3)
        chunk_num, head_kv = k_cmp.shape[0], k_cmp.shape[1]
        check_tensor_shape("k_cmp", k_cmp, (chunk_num, head_kv, dim))
        check_tensor_shape("lse_in", lse_in, (c_seq_len, heads))
        check_tensor_shape("chunk_offsets", chunk_offsets, (seq_num + 1,))
        check_tensor_shape("token_indices", token_indices, (c_seq_len, 2))
        self.c_seq_len, self.heads, self.dim = c_seq_len, heads, dim
        self.chunk_num, self.head_kv = chunk_num, head_kv
        self.seq_num = seq_num
        self.offsets = offsets
        self.group = _resolve_group(heads, head_kv)
        self.dtype = q.dtype

        tensors = (q, k_cmp, lse_in, offsets, chunk_offsets, token_indices)
        kernel = self._get_kernel(
            tensors,
            self.seq_num,
            c_seq_len,
            heads,
            dim,
            chunk_num,
            self.group,
            q.dtype,
            q.device.index,
        )
        return kernel(*tensors)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


class NSAFwdVarlenOp(Op):
    """Native Sparse Attention (NSA) sparse forward over a ragged batch.

    Attends each token to the blocks ``NSATopkVarlenOp`` selected for it. Sequence
    layout is packed: ``offsets`` marks the request boundaries, so the batch size and
    the block count come from the call rather than from construction.
    """

    def __init__(
        self,
        is_causal: bool,
        scale: float,
        block_size: int,
        accum_dtype: torch.dtype,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            is_causal: Whether a token may attend past its own position.
            scale: Softmax scale applied to the QK product.
            block_size: Tokens per selected block.
            accum_dtype: Accumulator dtype.
            tune: Whether to autotune, applied when a kernel is first built.
            kernel_map: Optional kernel override dict.
        """
        _validate_tiling(block_size=block_size)
        self.is_causal = is_causal
        self.scale = scale
        self.block_size = block_size
        self.accum_dtype = accum_dtype
        self.tune = tune

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"nsa_fwd_varlen_kernel": NSAFwdVarlenKernel}

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        block_indices_shape: tuple[int, ...],
        block_counts_shape: tuple[int, ...],
        offsets_shape: tuple[int, ...],
        token_indices_shape: tuple[int, ...],
    ) -> Dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: the attention output, shaped like the queries."""
        return {"o_slc": tuple(q_shape)}

    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch: int,
        heads: int,
        c_seq_len: int,
        dim: int,
        groups: int,
        selected_blocks: int,
        dtype: torch.dtype,
        device_index: "int | None",
    ) -> Kernel:
        key = (
            batch,
            heads,
            c_seq_len,
            dim,
            self.is_causal,
            self.scale,
            self.block_size,
            groups,
            selected_blocks,
            dtype,
            self.accum_dtype,
            device_index,
            self.tune,
        )
        return self.get_or_build_kernel(
            "nsa_fwd_varlen_kernel",
            inputs,
            key=key,
            build=lambda: self.kernel_map["nsa_fwd_varlen_kernel"](
                batch=batch,
                heads=heads,
                c_seq_len=c_seq_len,
                dim=dim,
                is_causal=self.is_causal,
                scale=self.scale,
                block_size=self.block_size,
                groups=groups,
                selected_blocks=selected_blocks,
                dtype=dtype,
                accum_dtype=self.accum_dtype,
                tune=self.tune,
            ),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices: torch.Tensor,
        block_counts: torch.Tensor,
        offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Attend each token to its selected blocks.

        Args:
            q: Queries, packed over the batch [c_seq_len, heads, dim].
            k: Keys [c_seq_len, head_kv, dim].
            v: Values [c_seq_len, head_kv, dim].
            block_indices: Selected block ids [c_seq_len, head_kv, selected_blocks].
            block_counts: Valid block count per token and KV head [c_seq_len, head_kv].
            offsets: Request boundaries into the packed sequence [batch + 1].
            token_indices: Request id and in-request position per token [c_seq_len, 2].

        Returns:
            Attention output [c_seq_len, heads, dim].
        """
        self._validate_dtypes(q, k, v, block_indices, block_counts, offsets, token_indices)
        c_seq_len, heads, dim = _packed_query_dims(q)
        batch = _packed_request_count(offsets)
        _require_rank("k", k, 3)
        _require_rank("block_indices", block_indices, 3)
        head_kv, selected = k.shape[1], block_indices.shape[2]
        # The kernel tiles the head dimension whole; a wider one silently truncates.
        if dim > 128:
            raise ValueError(f"dim must be at most 128; got {dim}")
        check_tensor_shape("k", k, (c_seq_len, head_kv, dim))
        check_tensor_shape("v", v, (c_seq_len, head_kv, dim))
        check_tensor_shape("block_indices", block_indices, (c_seq_len, head_kv, selected))
        check_tensor_shape("block_counts", block_counts, (c_seq_len, head_kv))
        check_tensor_shape("token_indices", token_indices, (c_seq_len, 2))
        self.c_seq_len, self.heads, self.dim = c_seq_len, heads, dim
        self.head_kv = head_kv
        self.groups = _resolve_group(heads, head_kv)
        self.selected_blocks = selected
        self.batch = batch
        self.offsets = offsets
        # The roofline counts the blocks this call actually keeps, which only these say.
        self.block_indices, self.block_counts = block_indices, block_counts
        self.token_indices = token_indices
        self.dtype = q.dtype

        tensors = (q, k, v, block_indices, block_counts, offsets, token_indices)
        kernel = self._get_kernel(
            tensors,
            self.batch,
            heads,
            c_seq_len,
            dim,
            self.groups,
            self.selected_blocks,
            q.dtype,
            q.device.index,
        )
        return kernel(*tensors)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


class NSACmpFwdVarlenOp(Op):
    """Native Sparse Attention (NSA) compression forward over a ragged batch.

    Attends each token to the compressed chunk summaries of its own request and
    returns both the output and the log-sum-exp ``NSATopkVarlenOp`` scores against.

    Sequence layout is packed: ``offsets`` marks the request boundaries, so the batch
    size and the chunk count come from the call rather than from construction.
    """

    def __init__(
        self,
        scale: float,
        bc: int,
        bs: int,
        accum_dtype: torch.dtype,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            scale: Softmax scale applied to the QK product.
            bc: Chunk tile width.
            bs: Compression block size.
            accum_dtype: Accumulator dtype.
            tune: Whether to autotune, applied when a kernel is first built.
            kernel_map: Optional kernel override dict.
        """
        _validate_tiling(bc=bc, bs=bs)
        self.scale = scale
        self.bc = bc
        self.bs = bs
        self.accum_dtype = accum_dtype
        self.tune = tune

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"nsa_cmp_fwd_varlen_kernel": NSACmpFwdVarlenKernel}

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_cmp_shape: tuple[int, ...],
        v_cmp_shape: tuple[int, ...],
        offsets_shape: tuple[int, ...],
        chunk_offsets_shape: tuple[int, ...],
        token_indices_shape: tuple[int, ...],
    ) -> Dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: the attention output and the lse the top-k pass reads."""
        c_seq_len, heads, _ = q_shape
        return {"o": (c_seq_len, heads, v_cmp_shape[2]), "lse": (c_seq_len, heads)}

    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        seq_num: int,
        c_seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_num: int,
        group: int,
        dtype: torch.dtype,
        device_index: "int | None",
    ) -> Kernel:
        key = (
            seq_num,
            c_seq_len,
            heads,
            dim_k,
            dim_v,
            chunk_num,
            group,
            self.scale,
            self.bc,
            self.bs,
            dtype,
            self.accum_dtype,
            device_index,
            self.tune,
        )
        return self.get_or_build_kernel(
            "nsa_cmp_fwd_varlen_kernel",
            inputs,
            key=key,
            build=lambda: self.kernel_map["nsa_cmp_fwd_varlen_kernel"](
                seq_num=seq_num,
                c_seq_len=c_seq_len,
                heads=heads,
                dim_k=dim_k,
                dim_v=dim_v,
                chunk_num=chunk_num,
                group=group,
                scale=self.scale,
                bc=self.bc,
                bs=self.bs,
                dtype=dtype,
                accum_dtype=self.accum_dtype,
                tune=self.tune,
            ),
        )

    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        v_cmp: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attend each token to its request's compressed chunks.

        Args:
            q: Queries, packed over the batch [c_seq_len, heads, dim_k].
            k_cmp: Compressed keys [chunk_num, head_kv, dim_k].
            v_cmp: Compressed values [chunk_num, head_kv, dim_v].
            offsets: Request boundaries into the packed sequence [seq_num + 1].
            chunk_offsets: Per-request chunk boundaries [seq_num + 1].
            token_indices: Request id and in-request position per token [c_seq_len, 2].

        Returns:
            Tuple of (o, lse).
        """
        self._validate_dtypes(q, k_cmp, v_cmp, offsets, chunk_offsets, token_indices)
        c_seq_len, heads, dim_k = _packed_query_dims(q)
        seq_num = _packed_request_count(offsets)
        _require_rank("k_cmp", k_cmp, 3)
        _require_rank("v_cmp", v_cmp, 3)
        chunk_num, head_kv = k_cmp.shape[0], k_cmp.shape[1]
        dim_v = v_cmp.shape[2]
        check_tensor_shape("k_cmp", k_cmp, (chunk_num, head_kv, dim_k))
        check_tensor_shape("v_cmp", v_cmp, (chunk_num, head_kv, dim_v))
        check_tensor_shape("chunk_offsets", chunk_offsets, (seq_num + 1,))
        check_tensor_shape("token_indices", token_indices, (c_seq_len, 2))
        self.c_seq_len, self.heads, self.dim_k = c_seq_len, heads, dim_k
        self.dim_v = dim_v
        self.chunk_num, self.head_kv = chunk_num, head_kv
        self.seq_num = seq_num
        self.offsets = offsets
        self.group = _resolve_group(heads, head_kv)
        self.dtype = q.dtype

        tensors = (q, k_cmp, v_cmp, offsets, chunk_offsets, token_indices)
        kernel = self._get_kernel(
            tensors,
            self.seq_num,
            c_seq_len,
            heads,
            dim_k,
            self.dim_v,
            chunk_num,
            self.group,
            q.dtype,
            q.device.index,
        )
        return kernel(*tensors)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)
