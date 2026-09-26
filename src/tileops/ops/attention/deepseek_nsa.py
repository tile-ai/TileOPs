from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.attention import (
    NSACmpFwdVarlenKernel,
    NSAFwdVarlenKernel,
    NSATopkVarlenKernel,
)
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op

__all__ = [
    "NSACmpVarlenFwdOp",
    "NSAVarlenFwdOp",
    "NSATopkVarlenFwdOp",
]


class NSATopkVarlenFwdOp(Op):
    """Native Sparse Attention (NSA) block selection over a ragged batch.

    Scores each compressed chunk against the query and returns, per token and per KV
    head, the ``selected_block_num`` block ids the sparse forward will attend to.

    Sequence layout is packed: ``q`` holds every request's tokens back to back and
    ``offsets`` marks the boundaries, so the batch size and the chunk count come from
    the call rather than from construction.
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "nsa_topk_varlen_kernel": NSATopkVarlenKernel
    }

    # Kernel configuration, not contract: the accumulator dtype. The chunk tile width is the
    # block size, because the kernel's candidate pool keeps the best tile-width chunks.
    accum_dtype: ClassVar[torch.dtype] = torch.float32

    def __init__(
        self,
        scale: float,
        selected_block_num: int,
        bs: int,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            scale: Softmax scale applied to the QK product.
            selected_block_num: Blocks to keep per token and KV head.
            bs: Compression block size.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.scale = scale
        self.selected_block_num = selected_block_num
        self.bs = bs
        self.tune = tune

        self.dispatch_kernel(kernel_map)

    def roofline_inputs(self) -> "dict[str, int]":
        """The (token, chunk) pairs this call's request lengths make it score."""
        from tileops.perf.formulas import nsa_topk_scored_pairs

        return {"scored_pairs": nsa_topk_scored_pairs(self.last_call)}

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, dtype and device."""
        seq_num, c_seq_len, heads, dim, chunk_num, head_kv, dtype, _device = call
        return call, lambda: self.kernel_map["nsa_topk_varlen_kernel"](
            seq_num=seq_num,
            c_seq_len=c_seq_len,
            heads=heads,
            dim=dim,
            chunk_num=chunk_num,
            group=heads // head_kv,
            scale=self.scale,
            selected_block_num=self.selected_block_num,
            bc=self.bs,
            bs=self.bs,
            dtype=dtype,
            accum_dtype=self.accum_dtype,
            tune=self.tune,
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
        return self._call_boundary(q, k_cmp, lse_in, offsets, chunk_offsets, token_indices)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        lse_in: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        tensors = (q, k_cmp, lse_in, offsets, chunk_offsets, token_indices)
        c_seq_len, heads, dim = q.shape
        chunk_num, head_kv = k_cmp.shape[0], k_cmp.shape[1]
        call = (
            offsets.shape[0] - 1,
            c_seq_len,
            heads,
            dim,
            chunk_num,
            head_kv,
            q.dtype,
            q.device.index,
        )
        return self.kernel_for("nsa_topk_varlen_kernel", tensors, call)(*tensors)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])


class NSAVarlenFwdOp(Op):
    """Native Sparse Attention (NSA) sparse forward over a ragged batch.

    Attends each token to the blocks ``NSATopkVarlenFwdOp`` selected for it. Sequence
    layout is packed: ``offsets`` marks the request boundaries, so the batch size and
    the block count come from the call rather than from construction.
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "nsa_fwd_varlen_kernel": NSAFwdVarlenKernel
    }

    # Kernel configuration, not contract: the accumulator dtype.
    accum_dtype: ClassVar[torch.dtype] = torch.float32

    def __init__(
        self,
        is_causal: bool,
        scale: float,
        block_size: int,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            is_causal: Whether a token may attend past its own position.
            scale: Softmax scale applied to the QK product.
            block_size: Tokens per selected block.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.is_causal = is_causal
        self.scale = scale
        self.block_size = block_size
        self.tune = tune

        self.dispatch_kernel(kernel_map)

    def roofline_inputs(self) -> "dict[str, int]":
        """The block tiles this call's selection kept, which its key and value reads follow."""
        from tileops.perf.formulas import nsa_selected_block_loads

        return {"selected_block_loads": nsa_selected_block_loads(self.last_call)}

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, dtype and device."""
        batch, heads, c_seq_len, dim, head_kv, selected_blocks, dtype, _device = call
        return call, lambda: self.kernel_map["nsa_fwd_varlen_kernel"](
            batch=batch,
            heads=heads,
            c_seq_len=c_seq_len,
            dim=dim,
            is_causal=self.is_causal,
            scale=self.scale,
            block_size=self.block_size,
            groups=heads // head_kv,
            selected_blocks=selected_blocks,
            dtype=dtype,
            accum_dtype=self.accum_dtype,
            tune=self.tune,
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
        return self._call_boundary(q, k, v, block_indices, block_counts, offsets, token_indices)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices: torch.Tensor,
        block_counts: torch.Tensor,
        offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        tensors = (q, k, v, block_indices, block_counts, offsets, token_indices)
        c_seq_len, heads, dim = q.shape
        call = (
            offsets.shape[0] - 1,
            heads,
            c_seq_len,
            dim,
            k.shape[1],
            block_indices.shape[2],
            q.dtype,
            q.device.index,
        )
        return self.kernel_for("nsa_fwd_varlen_kernel", tensors, call)(*tensors)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])


class NSACmpVarlenFwdOp(Op):
    """Native Sparse Attention (NSA) compression forward over a ragged batch.

    Attends each token to the compressed chunk summaries of its own request and
    returns both the output and the log-sum-exp ``NSATopkVarlenFwdOp`` scores against.

    Sequence layout is packed: ``offsets`` marks the request boundaries, so the batch
    size and the chunk count come from the call rather than from construction.
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "nsa_cmp_fwd_varlen_kernel": NSACmpFwdVarlenKernel
    }

    # Kernel configuration, not contract: the chunk tile width and the accumulator dtype.
    bc: ClassVar[int] = 32
    accum_dtype: ClassVar[torch.dtype] = torch.float32

    def __init__(
        self,
        scale: float,
        bs: int,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            scale: Softmax scale applied to the QK product.
            bs: Compression block size.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.scale = scale
        self.bs = bs
        self.tune = tune

        self.dispatch_kernel(kernel_map)

    def roofline_inputs(self) -> "dict[str, int]":
        """The (token, chunk) pairs this call's request lengths make it score."""
        from tileops.perf.formulas import nsa_closed_chunk_pairs

        return {"scored_pairs": nsa_closed_chunk_pairs(self.last_call)}

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, dtype and device."""
        seq_num, c_seq_len, heads, dim_k, dim_v, chunk_num, head_kv, dtype, _device = call
        return call, lambda: self.kernel_map["nsa_cmp_fwd_varlen_kernel"](
            seq_num=seq_num,
            c_seq_len=c_seq_len,
            heads=heads,
            dim_k=dim_k,
            dim_v=dim_v,
            chunk_num=chunk_num,
            group=heads // head_kv,
            scale=self.scale,
            bc=self.bc,
            bs=self.bs,
            dtype=dtype,
            accum_dtype=self.accum_dtype,
            tune=self.tune,
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
        return self._call_boundary(q, k_cmp, v_cmp, offsets, chunk_offsets, token_indices)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        v_cmp: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        tensors = (q, k_cmp, v_cmp, offsets, chunk_offsets, token_indices)
        c_seq_len, heads, dim_k = q.shape
        chunk_num, head_kv, dim_v = v_cmp.shape
        call = (
            offsets.shape[0] - 1,
            c_seq_len,
            heads,
            dim_k,
            dim_v,
            chunk_num,
            head_kv,
            q.dtype,
            q.device.index,
        )
        return self.kernel_for("nsa_cmp_fwd_varlen_kernel", tensors, call)(*tensors)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
