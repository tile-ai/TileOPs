from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention.fp8_lightning_indexer import (
    FP8LightningIndexerCall,
    FP8LightningIndexerFwdInterface,
    FP8LightningIndexerKernel,
)
from tileops.kernels.constants import FP8_E4M3_MAX
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.op_base import Op

__all__ = ["FP8LightningIndexerFwdOp"]


class FP8LightningIndexerFwdOp(Op):
    """Lightning indexer logits over FP8 index keys.

    For query ``s`` and key ``t`` of group ``g``, the logit sums ``weights[s, h]`` times
    ``relu(q[s, h] . k[t, g])`` over the heads of group ``g``, for keys inside the query's
    window ``[cu_seqlen_ks[s], cu_seqlen_ke[s])``. A bf16 call is quantized to FP8 by the
    op; an FP8 call passes the per-key scales in ``index_k_scale``.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "fp8_lightning_indexer_kernel": FP8LightningIndexerKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "fp8_lightning_indexer": FP8LightningIndexerFwdInterface
    }

    def roofline_inputs(self) -> "dict[str, int]":
        """The keys this call's windows make each batch row score, which its flops follow."""
        from tileops.perf.formulas import lightning_indexer_scored_keys

        return {"scored_keys": lightning_indexer_scored_keys(self.last_call)}

    def __init__(
        self,
        clean_logits: bool = True,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            clean_logits: Manifest ``params.clean_logits``, ``bool``, default ``True``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.clean_logits = clean_logits
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self.kernel = None

    def forward(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        index_k_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            index_q: Input tensor, dtype ``bfloat16 | float8_e4m3fn``.
            index_k: Input tensor, dtype ``bfloat16 | float8_e4m3fn``.
            weights: Input tensor, dtype ``float32``.
            cu_seqlen_ks: Input tensor, dtype ``int32``.
            cu_seqlen_ke: Input tensor, dtype ``int32``.
            index_k_scale: Input tensor, dtype ``float32``. Optional.

        Returns:
            ``logits``, as the manifest declares.
        """
        return self._call_boundary(
            index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, index_k_scale
        )

    def _eager_forward(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        index_k_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        if index_k_scale is None:
            # A bf16 call is quantized here; the kernel indexes FP8 keys and their scales.
            index_q = index_q.to(torch.float8_e4m3fn)
            key_f32 = index_k.float()
            scale = key_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4) / FP8_E4M3_MAX
            index_k = (key_f32 * scale.reciprocal()).to(torch.float8_e4m3fn)
            index_k_scale = scale.squeeze(-1)
        batch, seq_len, heads, index_dim = index_q.shape
        _, seq_len_kv, kv_group, _ = index_k.shape
        call = FP8LightningIndexerCall(
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            index_dim=index_dim,
            seq_len_kv=seq_len_kv,
            kv_group=kv_group,
            clean_logits=self.clean_logits,
            device=index_q.device,
        )
        inputs = tuple(
            t.contiguous()
            for t in (index_q, index_k, index_k_scale, weights, cu_seqlen_ks, cu_seqlen_ke)
        )
        self.kernel = self.kernel_for("fp8_lightning_indexer", call)
        return self.kernel(*inputs)

    def compute_roof(self) -> str:
        """Index scores contract at fp8 regardless of the input form."""
        return "tensor_core.fp8"
