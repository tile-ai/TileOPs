"""Communication-independent expert MLP for tight expert-major batches."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from torch import Tensor

from tileops.kernels.grouped_gemm import GroupedGemmPersistent3WGKernel
from tileops.kernels.grouped_gemm.grouped_gemm_persistent_3wg import (
    _DEFAULT_CONFIG as _3WG_DEFAULT_CONFIG,
)
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadKernel
from tileops.kernels.moe.moe_grouped_gemm_persistent_3wg_fused_act import (
    _DEFAULT_CONFIG as _FUSED_ACT_DEFAULT_CONFIG,
)
from tileops.ops.moe._activation import build_activation_op
from tileops.ops.op_base import Op

from .abc import ExpertBatch, ExpertBatchOutput
from .moe_grouped_gemm_nopad import MoeGroupedGemmNopadFwdOp

__all__ = ["DispatchedExpertMLPFwdOp"]

_logger = logging.getLogger(__name__)


class DispatchedExpertMLPFwdOp(Op):
    """Run an expert MLP over a pre-dispatched tight expert-major batch.

    ``expert_input`` is partitioned into contiguous expert segments described
    by ``true_offsets`` and ``true_sizes``. Empty experts are represented by a
    zero size. Output row ``i`` always corresponds to input row ``i``.

    This computation boundary deliberately has no routing, communication, or
    weighted-reduction inputs. In particular, routing weights are not applied.
    """

    def __init__(
        self,
        num_pairs: int,
        num_experts: int,
        hidden_size: int,
        ffn_size: int,
        dtype: torch.dtype = torch.bfloat16,
        gemm_kernel: Optional[type] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        *,
        activation: str = "silu_and_mul",
        use_fused_activation: bool = False,
    ):
        self.dispatch_kernel(kernel_map)
        self.num_pairs = num_pairs
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dtype = dtype
        self.activation = activation

        kernel_cls = gemm_kernel or GroupedGemmPersistent3WGKernel
        block_n = _3WG_DEFAULT_CONFIG["block_n"]
        block_k = _3WG_DEFAULT_CONFIG["block_k"]
        gate_up_n = ffn_size * 2
        if kernel_cls is GroupedGemmPersistent3WGKernel:
            gate_up_ok = gate_up_n % block_n == 0 and hidden_size % block_k == 0
            down_ok = hidden_size % block_n == 0 and ffn_size % block_k == 0
            if not (gate_up_ok and down_ok):
                _logger.warning(
                    "DispatchedExpertMLPFwdOp: dims not aligned to 3WG block "
                    "(gate_up_n=%d, hidden_size=%d, ffn_size=%d; block_n=%d, "
                    "block_k=%d) — falling back to MoeGroupedGemmNopadKernel.",
                    gate_up_n,
                    hidden_size,
                    ffn_size,
                    block_n,
                    block_k,
                )
                kernel_cls = MoeGroupedGemmNopadKernel

        gemm_override = (kernel_map or {}).get("moe_grouped_gemm_kernel")
        self.use_fused_activation = use_fused_activation
        if use_fused_activation:
            fused_block_n = _FUSED_ACT_DEFAULT_CONFIG["block_n"]
            eligible = (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 9
                and kernel_cls is GroupedGemmPersistent3WGKernel
                and (
                    gemm_override is None
                    or gemm_override is GroupedGemmPersistent3WGKernel
                )
                and activation in ("silu_and_mul", "gelu_and_mul")
                and ffn_size % fused_block_n == 0
            )
            if not eligible:
                _logger.warning(
                    "use_fused_activation=True not eligible (requires CUDA + SM90 + "
                    "GroupedGemmPersistent3WGKernel gate_up GEMM with no conflicting "
                    "moe_grouped_gemm_kernel override + activation in {silu_and_mul, "
                    "gelu_and_mul} + ffn_size %% %d == 0); falling back to unfused "
                    "activation. ffn_size=%d, activation=%s.",
                    fused_block_n,
                    ffn_size,
                    activation,
                )
                self.use_fused_activation = False

        if self.use_fused_activation:
            from .moe_grouped_gemm_nopad_fused_act import (
                MoeGroupedGemmNopad3WGFusedActFwdOp,
            )

            self._gemm_gate_up = MoeGroupedGemmNopad3WGFusedActFwdOp(
                numel=num_pairs,
                num_experts=num_experts,
                ffn=ffn_size,
                k=hidden_size,
                dtype=dtype,
                activation=activation,
                kernel_map=kernel_map,
            )
            self._activation_op = None
        else:
            self._gemm_gate_up = MoeGroupedGemmNopadFwdOp(
                numel=num_pairs,
                num_experts=num_experts,
                n=ffn_size * 2,
                k=hidden_size,
                dtype=dtype,
                kernel_map={"moe_grouped_gemm_kernel": kernel_cls, **(kernel_map or {})},
            )
            self._activation_op = build_activation_op(
                activation,
                M=num_pairs,
                N=ffn_size,
                dtype=dtype,
                kernel_map=kernel_map,
            )
        self._gemm_down = MoeGroupedGemmNopadFwdOp(
            numel=num_pairs,
            num_experts=num_experts,
            n=hidden_size,
            k=ffn_size,
            dtype=dtype,
            kernel_map={"moe_grouped_gemm_kernel": kernel_cls, **(kernel_map or {})},
        )

    @property
    def default_kernel_map(self) -> dict:
        return {}

    def forward(
        self,
        expert_input: Tensor,
        w_gate_up: Tensor,
        w_down: Tensor,
        true_sizes: Tensor,
        true_offsets: Tensor,
    ) -> Tensor:
        """Return ``[num_pairs, hidden_size]`` without changing row order."""
        return self._forward(
            expert_input,
            w_gate_up,
            w_down,
            true_sizes,
            true_offsets,
            valid_rows=None,
        )

    def _forward(
        self,
        expert_input: Tensor,
        w_gate_up: Tensor,
        w_down: Tensor,
        true_sizes: Tensor,
        true_offsets: Tensor,
        valid_rows: Tensor | None,
    ) -> Tensor:
        gate_up = self._gemm_gate_up(
            expert_input, w_gate_up, true_sizes, true_offsets
        )
        act = (
            gate_up
            if self.use_fused_activation
            else (
                self._activation_op(gate_up)
                if valid_rows is None
                else self._activation_op.kernel.forward_rows(
                    gate_up, valid_rows
                )
            )
        )
        return self._gemm_down(act, w_down, true_sizes, true_offsets)

    def forward_batch(
        self,
        batch: ExpertBatch,
        w_gate_up: Tensor,
        w_down: Tensor,
    ) -> ExpertBatchOutput:
        """Run the MLP from canonical ``expert_offsets[E_local + 1]``.

        Offset differencing and the received row count stay on the input
        device. Grouped GEMMs schedule only expert tiles described by the
        offsets; the standalone activation is bounded by ``valid_rows``.
        """
        if batch.capacity != self.num_pairs:
            raise ValueError(
                f"batch capacity must equal num_pairs={self.num_pairs}, "
                f"got {batch.capacity}"
            )
        if batch.hidden.shape[1] != self.hidden_size:
            raise ValueError(
                f"batch hidden size must equal {self.hidden_size}, "
                f"got {batch.hidden.shape[1]}"
            )
        if batch.expert_offsets.numel() != self.num_experts + 1:
            raise ValueError(
                "expert_offsets must contain num_experts + 1 entries; "
                f"expected {self.num_experts + 1}, "
                f"got {batch.expert_offsets.numel()}"
            )
        true_offsets = batch.expert_offsets[:-1]
        true_sizes = batch.expert_offsets[1:] - true_offsets
        valid_rows = batch.valid_rows
        hidden = self._forward(
            batch.hidden,
            w_gate_up,
            w_down,
            true_sizes,
            true_offsets,
            valid_rows,
        )
        return ExpertBatchOutput(hidden=hidden, valid_rows=valid_rows)
