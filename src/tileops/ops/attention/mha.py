from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention import (
    MHADecodePagedKernel,
    MHADecodePagedWsKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op
from .gqa import GroupedQueryAttentionBwdOp
from .selection import AttentionCall

__all__ = [
    "MultiHeadAttentionBwdOp",
    "MultiHeadAttentionDecodePagedWithKVCacheFwdOp",
]


class MultiHeadAttentionBwdOp(Op):
    """Layout: BSHD.

    MHA backward is the ``heads_kv == heads`` specialization of GQA backward,
    matching the forward path's dispatch through GQA.
    """

    compile_boundary = True
    # Every kernel this op runs is built by GQA backward.
    delegate_types: ClassVar[Mapping[str, type[Op]]] = {"gqa_backward": GroupedQueryAttentionBwdOp}

    _LEGACY_KERNEL_MAP_KEYS = frozenset(
        {
            "mha_bwd_preprocess_kernel",
            "mha_bwd_kernel",
            "mha_bwd_postprocess_kernel",
        }
    )

    def __init__(
        self,
        is_causal: bool = True,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            is_causal: Manifest ``params.is_causal``, ``bool``, default ``True``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.is_causal = is_causal

        self.tune = tune
        self.dispatch_kernel(self._gqa_kernel_map(kernel_map))
        self._gqa_op = self.delegate_for(
            "gqa_backward",
            None,
            is_causal=is_causal,
        )
        self.kernel_map = self._gqa_op.kernel_map

    @staticmethod
    def _gqa_kernel_map(kernel_map: Optional[Dict[str, Kernel]]) -> Optional[Dict[str, Kernel]]:
        if kernel_map is None:
            return None
        legacy_keys = MultiHeadAttentionBwdOp._LEGACY_KERNEL_MAP_KEYS.intersection(kernel_map)
        if legacy_keys:
            keys = ", ".join(sorted(legacy_keys))
            raise ValueError(
                "MultiHeadAttentionBwdOp delegates to GroupedQueryAttentionBwdOp; "
                f"legacy MHA backward kernel_map keys are not compatible: {keys}. "
                "Use gqa_bwd_* keys with kernels that implement the GQA backward ABI."
            )
        return dict(kernel_map)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        do: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            q: Input tensor, dtype ``float16 | bfloat16``.
            k: Input tensor, dtype ``same_as(q)``.
            v: Input tensor, dtype ``same_as(q)``.
            o: Input tensor, dtype ``same_as(q)``.
            do: Input tensor, dtype ``same_as(q)``.
            lse: Input tensor, dtype ``float32``.

        Returns:
            ``dq``, ``dk``, ``dv``, as the manifest declares. Shape rules: ``dq.shape == (B, S, H, D)``; ``dk.shape == (B, S, H, D)``; ``dv.shape == (B, S, H, D)``.
        """
        return self._call_boundary(q, k, v, o, do, lse)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        do: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        return self._gqa_op(q, k, v, o, do, lse)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])


class MultiHeadAttentionDecodePagedWithKVCacheFwdOp(Op):
    """Paged MHA decode with dynamic KV cache. Layout: ``Q`` $[batch \\times seqlen\\_q \\times heads \\times dim]$ (BSHD);
    K, V physical cache [seqlen_kv, heads, dim]; real_seqlen_kv [batch]; block_table [batch, num_pages].
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "mha_decode_paged_kernel": MHADecodePagedKernel,
        "mha_decode_paged_ws_kernel": MHADecodePagedWsKernel,
    }

    def __init__(
        self,
        page_size: int,
        is_causal: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            page_size: Manifest ``params.page_size``, ``int``.
            is_causal: Manifest ``params.is_causal``, ``bool``, default ``False``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.page_size = page_size
        self.is_causal = is_causal
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _attention_call(self, q: torch.Tensor, k: torch.Tensor) -> AttentionCall:
        """State what one paged decode call is, for selection to filter against.

        Every extent and the element type arrive with the inputs, so one instance serves
        every shape and dtype it is handed.
        """
        batch, seqlen_q, heads, dim = q.shape
        return AttentionCall(
            dtype=q.dtype,
            batch=batch,
            heads=heads,
            heads_kv=heads,
            dim=dim,
            max_seqlen_q=seqlen_q,
            seqlen_kv=k.shape[0],
            page_size=self.page_size,
            is_causal=self.is_causal,
            tune=self.tune,
            device=q.device,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            q: Input tensor, dtype ``float16 | bfloat16``.
            k: Input tensor, dtype ``same_as(q)``.
            v: Input tensor, dtype ``same_as(q)``.
            real_seqlen_kv: Input tensor, dtype ``int32``.
            block_table: Input tensor, dtype ``int32``.

        Returns:
            ``o``, as the manifest declares. Shape rules: ``o.shape == (B, S_q, H, D)``.
        """
        return self._call_boundary(q, k, v, real_seqlen_kv, block_table)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        inputs = (q, k, v, real_seqlen_kv, block_table)
        kernel = self.kernel_for("mha_decode_paged", inputs, self._attention_call(q, k))
        return kernel(*inputs)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
