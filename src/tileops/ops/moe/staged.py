"""Public staged Mixture-of-Experts operator boundaries."""

from __future__ import annotations

from typing import ClassVar, Mapping

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe import (
    MoeGroupedGemmKernel,
    MoePrePermuteContiguousKernel,
    MoeUnpermuteKernel,
)
from tileops.kernels.moe.call_spec import MGroupedGemmCall, PostPermuteCall, PrePermuteCall
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof
from tileops.utils import get_sm_version, is_h200

from .contracts import MaskedLayoutSpec, MGroupedLayoutSpec, RoutingEpilogueSpec

__all__ = [
    "MoeExpertMLPFwdOp",
    "MoeGroupedGemmFwdOp",
    "MoePostPermuteFwdOp",
    "MoePrePermuteFwdOp",
]


class _ContiguousPostPermuteKernel(Kernel):
    """Adapt contiguous staged indices to the shipped weighted unpermute kernel."""

    supported_archs = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: PostPermuteCall) -> bool:
        return (
            call.layout_key in ("tight_physical_psum", "aligned_per_row")
            and call.input_dtype in (torch.bfloat16, torch.float16)
            and call.output_dtype == call.input_dtype
        )

    def __init__(self, call: PostPermuteCall) -> None:
        """Build the weighted no-pad inverse specialization selected by ``call``."""
        super().__init__()
        self.inner = MoeUnpermuteKernel(
            call.num_tokens,
            call.top_k,
            call.hidden_size,
            call.materialized_rows,
            scaling=call.epilogue.routed_scaling_factor,
            dtype=call.input_dtype,
            sm_count=call.sm_count,
            tune=call.tune,
        )

    def forward(
        self,
        expert_output: torch.Tensor,
        inverse_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Restore token order and apply the configured routing epilogue."""
        return self.inner(expert_output, inverse_indices, topk_weights, out=out)


class MoePrePermuteFwdOp(Op):
    """Materialize rank-grouped activations into a local expert layout.

    ``local_expert_ids`` must already be in ``[0, num_local_experts)``.
    Global placement and communication belong to EPDispatch.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "contiguous": MoePrePermuteContiguousKernel
    }

    def __init__(
        self,
        layout: MGroupedLayoutSpec,
        num_local_experts: int,
        *,
        target: object = None,
        kernel_map: dict[str, Kernel] | None = None,
        tune: bool = False,
    ) -> None:
        """Configure a pre-permute boundary for one layout and expert domain.

        Args:
            layout: How the expert rows are materialized.
            num_local_experts: Number of local experts the ids index.
            target: Which backend serves this instance; detected from the tensors when
                ``None``.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune the kernel.
        """
        self.layout = layout
        self.num_local_experts = num_local_experts
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(
        self,
        hidden_states: torch.Tensor,
        local_expert_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(expert_input, layout_metadata, inverse_indices)``."""
        return self._call_boundary(hidden_states, local_expert_ids)

    def _eager_forward(
        self,
        hidden_states: torch.Tensor,
        local_expert_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = hidden_states.device
        call = PrePermuteCall(
            arch=get_sm_version(device.index),
            h200=is_h200(device.index),
            layout=self.layout,
            device_type=device.type,
            input_dtype=hidden_states.dtype,
            num_experts=self.num_local_experts,
            num_tokens=hidden_states.shape[0],
            hidden_size=hidden_states.shape[1],
            top_k=local_expert_ids.shape[1],
            routing_input_kind="local_expert_ids",
            tune=self.tune,
        )
        kernel = self.kernel_for("pre_permute", (hidden_states, local_expert_ids), call)
        return kernel(hidden_states, local_expert_ids)


class MoeGroupedGemmFwdOp(Op):
    """M-grouped GEMM over expert-materialized rows: ``out[rows of g] = a[rows of g] @ b[g]^T``.

    ``layout`` fixes how the rows of ``a`` are organised by expert and what the
    ``layout_metadata`` tensor means; ``E`` is read off ``b``, ``M``/``N``/``K`` off
    the operands. Accumulation is fp32; the output is written in the operand dtype
    unless ``out_dtype`` asks for fp32.

    With ``activation`` set, ``b`` stacks the gate and up projections along ``N``
    (``[E, 2 * ffn, K]``) and ``out`` is ``act(gate) * up`` with ``ffn`` columns: the
    gated activation is fused into the GEMM's epilogue and the ``[.., 2 * ffn]``
    intermediate is never written.

    Per layout the operands are (``trans``-free, ``b`` is ``[E, N, K]``):

    | layout                      | ``a``             | ``layout_metadata``          | ``out``           |
    | --------------------------- | ----------------- | ---------------------------- | ----------------- |
    | contiguous · physical_psum  | ``[M, K]``        | ``[E]`` segment end rows     | ``[M, N]``        |
    | contiguous · per_row        | ``[M, K]``        | ``[M]`` expert id per row    | ``[M, N]``        |
    | masked                      | ``[E, max_m, K]`` | ``[E]`` valid rows per expert| ``[E, max_m, N]`` |

    Rows an aligned layout pads with, and a masked slab's rows past its valid
    count, hold unspecified values in ``out``. The metadata's values must satisfy
    the layout (ordering, segment ends, ranges); the op does not check them, since
    doing so would synchronise.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"grouped_gemm": MoeGroupedGemmKernel}

    def __init__(
        self,
        layout: MGroupedLayoutSpec,
        *,
        activation: str | None = None,
        out_dtype: torch.dtype | None = None,
        target: object = None,
        kernel_map: dict[str, Kernel] | None = None,
        tune: bool = False,
    ) -> None:
        """Fix the expert layout, the fused activation and the output dtype policy.

        Args:
            layout: How ``a``'s rows are grouped by expert.
            activation: ``None`` for a plain GEMM, or a gated activation
                (``"silu_and_mul"``, ``"gelu_and_mul"``) fused into the epilogue over a
                gate||up ``b``, which halves the output width.
            out_dtype: ``None`` writes the operand dtype, ``torch.float32`` keeps the
                fp32 accumulator.
            target: Which backend serves this instance; detected from the tensors when
                ``None``.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune the kernel.
        """
        self.layout = layout
        self.activation = activation
        self.out_dtype = out_dtype
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.ix["D"])

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one GEMM per expert over the rows the layout assigns it.

        Args:
            a: Expert-materialized activations, ``[M, K]`` or ``[E, max_m, K]`` per the layout.
            b: Per-expert weights, ``[E, N, K]``.
            layout_metadata: ``int32`` metadata whose shape and meaning the layout fixes.
            out: Optional preallocated output in the resolved output dtype.

        Returns:
            ``[M, N]`` or ``[E, max_m, N]`` in the operand dtype, or fp32 when asked for.
        """
        return self._call_boundary(a, b, layout_metadata, out)

    def _eager_forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layout = self.layout
        masked = isinstance(layout, MaskedLayoutSpec)
        num_experts, n, k = b.shape
        device = a.device
        call = MGroupedGemmCall(
            arch=get_sm_version(device.index),
            h200=is_h200(device.index),
            kind=layout.kind,
            packing=None if masked else layout.packing.value,
            metadata_kind=None if masked else layout.metadata_kind.value,
            alignment=getattr(layout, "alignment", 1),
            max_m=layout.max_m,
            activation=self.activation,
            ab_dtype=a.dtype,
            cd_dtype=a.dtype if self.out_dtype is None else self.out_dtype,
            num_groups=num_experts,
            m=a.numel() // k,
            n=n,
            k=k,
            tune=self.tune,
        )
        kernel = self.kernel_for("grouped_gemm", (a, b, layout_metadata), call)
        return kernel(a, b, layout_metadata, out=out)


class MoeExpertMLPFwdOp(Op):
    """Two grouped GEMMs on one expert layout, the gated activation fused into the first.

    ``out = (act(expert_input @ w_gate_up[g]^T)) @ w_down[g]^T`` per expert ``g``, where
    ``act`` is the gated activation: ``w_gate_up`` stacks the gate and up projections
    along ``N`` and the gate_up GEMM's epilogue applies the activation and halves it,
    so the ``[.., 2 * ffn]`` intermediate is never written. A composite: it registers
    no operator of its own, its graph is its two leaves'.
    """

    delegate_types: ClassVar[Mapping[str, type[Op]]] = {
        "gate_up": MoeGroupedGemmFwdOp,
        "down": MoeGroupedGemmFwdOp,
    }

    def __init__(
        self,
        layout: MGroupedLayoutSpec,
        activation: str = "silu_and_mul",
        *,
        target: object = None,
        kernel_map: dict[str, Kernel] | None = None,
        tune: bool = False,
    ) -> None:
        """Configure two grouped GEMMs on ``layout``, the first fusing the gated activation.

        Args:
            layout: Shared by both GEMMs and the metadata.
            activation: ``"silu_and_mul"`` or ``"gelu_and_mul"``, fused into the gate_up
                GEMM's epilogue.
            target: Which backend serves the delegates.
            kernel_map: Optional overrides, forwarded to both GEMMs.
            tune: Whether to autotune the GEMM kernels.
        """
        self.layout = layout
        self.activation = activation
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self.gate_up = self.delegate_for("gate_up", None, layout=layout, activation=activation)
        self.down = self.delegate_for("down", None, layout=layout)

    def compute_roof(self) -> str:
        """The two GEMMs dominate the FLOPs; priced on tensor cores."""
        return tensor_core_roof(self.last_call.ix["D"])

    def forward(
        self,
        expert_input: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the gate/up GEMM with its fused activation, then the down GEMM.

        Args:
            expert_input: ``[M, hidden]`` or ``[E, max_m, hidden]`` per the layout.
            w_gate_up: ``[E, 2 * ffn, hidden]``, gate and up stacked along the output axis.
            w_down: ``[E, hidden, ffn]``.
            layout_metadata: ``int32`` metadata the layout fixes; shared by both GEMMs.
            out: Optional preallocated ``[.., hidden]`` output in the operand dtype.

        Returns:
            ``[M, hidden]`` or ``[E, max_m, hidden]`` in the operand dtype.
        """
        activated = self.gate_up(expert_input, w_gate_up, layout_metadata)
        return self.down(activated, w_down, layout_metadata, out=out)


class MoePostPermuteFwdOp(Op):
    """Restore token order and apply the declared local routing epilogue."""

    compile_boundary: ClassVar[bool] = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "contiguous": _ContiguousPostPermuteKernel
    }

    def __init__(
        self,
        layout: MGroupedLayoutSpec,
        epilogue: RoutingEpilogueSpec | None = None,
        out_dtype: torch.dtype | None = None,
        *,
        target: object = None,
        kernel_map: dict[str, Kernel] | None = None,
        tune: bool = False,
    ) -> None:
        """Configure inverse permutation and the exactly-once routing epilogue.

        Args:
            layout: How the expert rows are materialized.
            epilogue: The routing epilogue; ``None`` takes the default one.
            out_dtype: The dtype the reduced result is written in; ``None`` keeps the
                expert output's own.
            target: Which backend serves this instance; detected from the tensors when
                ``None``.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune the kernel.
        """
        self.layout = layout
        self.epilogue = epilogue
        self.out_dtype = out_dtype
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(
        self,
        expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        inverse_indices: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Restore token order, apply routing weights, reduce top-k, and cast."""
        return self._call_boundary(expert_output, topk_weights, inverse_indices, out)

    def _eager_forward(
        self,
        expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        inverse_indices: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        masked = isinstance(self.layout, MaskedLayoutSpec)
        device = expert_output.device
        call = PostPermuteCall(
            arch=get_sm_version(device.index),
            layout_key=self.layout.selection_key,
            max_m=self.layout.max_m,
            epilogue=RoutingEpilogueSpec() if self.epilogue is None else self.epilogue,
            device_type=device.type,
            input_dtype=expert_output.dtype,
            routing_weight_dtype=topk_weights.dtype,
            output_dtype=expert_output.dtype if self.out_dtype is None else self.out_dtype,
            num_experts=expert_output.shape[0] if masked else 0,
            materialized_rows=expert_output.numel() // expert_output.shape[-1],
            num_tokens=topk_weights.shape[0],
            hidden_size=expert_output.shape[-1],
            top_k=topk_weights.shape[1],
            tune=self.tune,
        )
        kernel = self.kernel_for(
            "post_permute", (expert_output, topk_weights, inverse_indices), call
        )
        return kernel(expert_output, inverse_indices, topk_weights, out=out)
