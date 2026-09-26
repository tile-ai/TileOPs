from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.constants import FP8_E4M3_MAX
from tileops.kernels.fp8_lightning_indexer import FP8LightningIndexerKernel
from tileops.kernels.kernel_base import Entry, Kernel

from .op_base import Op

__all__ = ["FP8LightningIndexerFwdOp"]


class FP8LightningIndexerFwdOp(Op):
    """Lightning indexer logits over FP8 index keys.

    For query ``s`` and key ``t`` of group ``g``, the logit sums ``weights[s, h]`` times
    ``relu(q[s, h] . k[t, g])`` over the heads of group ``g``, for keys inside the query's
    window ``[cu_seqlen_ks[s], cu_seqlen_ke[s])``. A bf16 call is quantized to FP8 by the
    op; an FP8 call passes the per-key scales in ``index_k_scale``.
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "fp8_lightning_indexer_kernel": FP8LightningIndexerKernel
    }

    def __init__(
        self,
        clean_logits: bool = True,
        *,
        config: Optional[dict] = None,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            clean_logits: Manifest ``params.clean_logits``, ``bool``, default ``True``.
            config: Kernel configuration, passed only to the kernel.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.clean_logits = clean_logits
        self.config = config
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self.kernel = None

    @property
    def _config_cache_key(self) -> tuple:
        if not self.config:
            return ()
        return tuple(sorted((key, repr(value)) for key, value in self.config.items()))

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape and device; the config is the op's."""
        batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits, _config, _dev = call
        return call, lambda: self.kernel_map["fp8_lightning_indexer_kernel"](
            batch,
            seq_len,
            heads,
            index_dim,
            seq_len_kv,
            kv_group,
            clean_logits,
            config=self.config,
            tune=self.tune,
        )

    def _bind_kernel(self, index_q: torch.Tensor, index_k: torch.Tensor, inputs: tuple) -> None:
        batch, seq_len, heads, index_dim = index_q.shape
        _, seq_len_kv, kv_group, _ = index_k.shape
        self.kernel = self.kernel_for(
            "fp8_lightning_indexer_kernel",
            inputs,
            (
                batch,
                seq_len,
                heads,
                index_dim,
                seq_len_kv,
                kv_group,
                self.clean_logits,
                self._config_cache_key,
                index_q.device.index,
            ),
        )

    def torch_quant_forward(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
    ) -> torch.Tensor:
        index_q = index_q.to(torch.float8_e4m3fn)
        index_k, index_k_scale = self.per_custom_dims_cast_to_fp8(index_k, (0,), False)

        return self.kernel(index_q, index_k, index_k_scale, weights, cu_seqlen_ks, cu_seqlen_ke)

    def tl_quant_forward(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
    ) -> torch.Tensor:
        return self.kernel(index_q, index_k, index_k_scale, weights, cu_seqlen_ks, cu_seqlen_ke)

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
        self._bind_kernel(
            index_q,
            index_k,
            (index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, index_k_scale),
        )
        if index_k_scale is None:
            return self.torch_quant_forward(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke)
        return self.tl_quant_forward(
            index_q, index_k, index_k_scale, weights, cu_seqlen_ks, cu_seqlen_ke
        )

    def per_custom_dims_cast_to_fp8(
        self, x: torch.Tensor, dims: Tuple[int], use_ue8m0: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_absmax = x.to(torch.float32).abs().amax(dim=-1, keepdim=True).clamp(1e-4)
        sf = x_absmax / FP8_E4M3_MAX
        if use_ue8m0:
            assert sf.view(-1).amax().item() > 0
            sf = torch.pow(2.0, torch.ceil(torch.log2(x_absmax)))
        x_scaled = (x.to(torch.float32) * (1.0 / sf)).to(torch.float8_e4m3fn)
        return x_scaled, sf.squeeze(-1)

    def compute_roof(self) -> str:
        """Index scores contract at fp8 regardless of the input form."""
        return "tensor_core.fp8"
