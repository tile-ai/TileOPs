from typing import ClassVar, Mapping, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.attention import (
    NSACompressedFwdVarlenKernel,
    NSAFwdVarlenKernel,
    NSAFwdVarlenTMAKernel,
    NSATopKVarlenKernel,
)
from tileops.kernels.attention.call_spec import (
    NSACall,
    NSACompressedFwdInterface,
    NSAFwdInterface,
    NSATopKFwdInterface,
)
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = [
    "NSACompressedVarlenFwdOp",
    "NSATopKVarlenFwdOp",
    "NSAVarlenFwdOp",
]


class NSATopKVarlenFwdOp(Op):
    """Native Sparse Attention (NSA) block selection over a ragged batch.

    Scores each compressed chunk against the query and returns, per token and per KV
    head, the ``selected_block_num`` block ids the sparse forward will attend to.
    Follows FLA: the first, previous and current blocks score one per query head;
    other blocks use normalized attention summed across the GQA group. Rank raw
    scores descending, with unspecified tie selection/order and trailing -1 padding.

    Sequence layout is packed: ``q`` holds every request's tokens back to back and
    ``offsets`` marks the boundaries, so the batch size and the chunk count come from
    the call rather than from construction.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "nsa_topk_varlen_kernel": NSATopKVarlenKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "nsa_topk_varlen_kernel": NSATopKFwdInterface
    }

    def __init__(
        self,
        scale: float,
        selected_block_num: int,
        bs: int,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            scale: Multiplies Q in its input dtype before the QK product, as in FLA.
            selected_block_num: Blocks to keep per token and KV head.
            bs: Compression block size.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.scale = scale
        self.selected_block_num = selected_block_num
        self.bs = bs

        super().__init__(target=target)

    def roofline_data_terms(self) -> "dict[str, int]":
        """The (token, chunk) pairs this call's request lengths make it score."""
        from tileops.perf.formulas import nsa_topk_scored_pairs

        return {"scored_pairs": nsa_topk_scored_pairs(self.last_call)}

    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Select the blocks each token attends to.

        Args:
            q: Queries, packed over the batch [c_seq_len, heads, dim].
            k_cmp: Compressed keys [chunk_num, head_kv, dim].
            offsets: Request boundaries into the packed sequence [seq_num + 1].
            chunk_offsets: Per-request chunk boundaries [seq_num + 1].
            token_indices: Request id and in-request position per token [c_seq_len, 2].

        Returns:
            Selected block ids [c_seq_len, head_kv, selected_block_num].
        """
        tensors = (q, k_cmp, offsets, chunk_offsets, token_indices)
        c_seq_len, heads, dim = q.shape
        call = NSACall(
            batch=offsets.shape[0] - 1,
            c_seq_len=c_seq_len,
            heads=heads,
            heads_kv=k_cmp.shape[1],
            dim=dim,
            chunk_num=k_cmp.shape[0],
            selected_blocks=self.selected_block_num,
            block_size=self.bs,
            scale=self.scale,
            dtype=q.dtype,
            device=q.device,
        )
        return self.kernel_for("nsa_topk_varlen_kernel", call)(*tensors)

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])


class NSAVarlenFwdOp(Op):
    """Native Sparse Attention (NSA) sparse forward over a ragged batch.

    Attends each token to the blocks ``NSATopKVarlenFwdOp`` selected for it. Sequence
    layout is packed: ``offsets`` marks the request boundaries, so the batch size and
    the block count come from the call rather than from construction.

    A causal token sees the keys of its selected blocks up to its own position, a
    non-causal one up to the end of its sequence; a block starting past that bound
    contributes nothing, and a token no block gives a key outputs zeros.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "nsa_fwd_varlen_kernel": NSAFwdVarlenKernel,
        "nsa_fwd_varlen_tma_kernel": NSAFwdVarlenTMAKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "nsa_fwd_varlen_kernel": NSAFwdInterface
    }

    def __init__(
        self,
        is_causal: bool,
        scale: float,
        block_size: int,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            is_causal: Whether a token sees keys only up to its own position.
            scale: Softmax scale applied to the QK product.
            block_size: Tokens per selected block.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.is_causal = is_causal
        self.scale = scale
        self.block_size = block_size

        super().__init__(target=target)

    def roofline_data_terms(self) -> "dict[str, int]":
        """The key rows this call's selection scores and the distinct rows it reads."""
        from tileops.perf.formulas import nsa_selected_rows

        scored, distinct = nsa_selected_rows(self.last_call)
        return {"scored_rows": scored, "distinct_rows": distinct}

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
        tensors = (q, k, v, block_indices, block_counts, offsets, token_indices)
        c_seq_len, heads, dim = q.shape
        call = NSACall(
            batch=offsets.shape[0] - 1,
            c_seq_len=c_seq_len,
            heads=heads,
            heads_kv=k.shape[1],
            dim=dim,
            selected_blocks=block_indices.shape[2],
            block_size=self.block_size,
            scale=self.scale,
            is_causal=self.is_causal,
            dtype=q.dtype,
            device=q.device,
        )
        return self.kernel_for("nsa_fwd_varlen_kernel", call)(*tensors)

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])


class NSACompressedVarlenFwdOp(Op):
    """Native Sparse Attention (NSA) compression forward over a ragged batch.

    Attends each token to the compressed chunk summaries of its own request and
    returns both the output and the log-sum-exp ``NSATopKVarlenFwdOp`` scores against.

    Sequence layout is packed: ``offsets`` marks the request boundaries, so the batch
    size and the chunk count come from the call rather than from construction.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "nsa_compressed_fwd_varlen_kernel": NSACompressedFwdVarlenKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "nsa_compressed_fwd_varlen_kernel": NSACompressedFwdInterface
    }

    def __init__(
        self,
        scale: float,
        bs: int,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            scale: Softmax scale applied to the QK product.
            bs: Compression block size.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.scale = scale
        self.bs = bs

        super().__init__(target=target)

    def roofline_data_terms(self) -> "dict[str, int]":
        """The (token, chunk) pairs this call's request lengths make it score."""
        from tileops.perf.formulas import nsa_closed_chunk_pairs

        return {"scored_pairs": nsa_closed_chunk_pairs(self.last_call)}

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
        tensors = (q, k_cmp, v_cmp, offsets, chunk_offsets, token_indices)
        c_seq_len, heads, dim_k = q.shape
        chunk_num, head_kv, dim_v = v_cmp.shape
        call = NSACall(
            batch=offsets.shape[0] - 1,
            c_seq_len=c_seq_len,
            heads=heads,
            heads_kv=head_kv,
            dim=dim_k,
            dim_v=dim_v,
            chunk_num=chunk_num,
            block_size=self.bs,
            scale=self.scale,
            dtype=q.dtype,
            device=q.device,
        )
        return self.kernel_for("nsa_compressed_fwd_varlen_kernel", call)(*tensors)

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
