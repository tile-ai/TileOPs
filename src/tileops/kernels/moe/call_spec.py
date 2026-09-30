"""The facts of one Mixture-of-Experts call that its in-tree kernels select and build on, and
the kernel interfaces their implementations inherit."""

import dataclasses
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface

if TYPE_CHECKING:
    from tileops.ops.moe.contracts import (
        MGroupedLayoutSpec,
        RoutingEpilogueSpec,
    )

__all__ = [
    "FusedTopKCall",
    "FusedTopKFwdInterface",
    "IndexedExpertCall",
    "IndexedExpertDownFwdInterface",
    "IndexedExpertGateUpFwdInterface",
    "IndexedRouteStatsFwdInterface",
    "IndexedWeightedReduceFwdInterface",
    "MGroupedGemmCall",
    "MGroupedGemmFwdInterface",
    "PermuteAlignCall",
    "PermuteAlignFwdInterface",
    "PostPermuteCall",
    "PostPermuteFwdInterface",
    "PrePermuteCall",
    "PrePermuteFwdInterface",
    "SharedExpertMLPCall",
    "SharedExpertMLPFwdInterface",
]


@dataclasses.dataclass(frozen=True)
class FusedTopKCall(CallSpec):
    """One top-k routing call, with the scoring the op fixed at construction."""

    num_tokens: int = 0
    num_experts: int = 0
    top_k: int = 0
    scoring_func: str = "softmax"
    renormalize: bool = False
    with_correction_bias: bool = False
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class PermuteAlignCall(CallSpec):
    """One permute-align call; ``num_routes`` is ``T * K``, the routed assignment count."""

    num_routes: int = 0
    num_experts: int = 0
    block_size: int = 0


@dataclasses.dataclass(frozen=True)
class SharedExpertMLPCall(CallSpec):
    """One dense shared-expert call, as the op knows it after reading its inputs."""

    num_tokens: int = 0
    hidden_size: int = 0
    ffn_size: int = 0
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class IndexedExpertCall(CallSpec):
    """One route-major expert MLP call, shared by its four kernel interfaces."""

    num_tokens: int = 0
    top_k: int = 0
    num_experts: int = 0
    hidden_size: int = 0
    ffn_size: int = 0
    dtype: Optional[torch.dtype] = None
    routed_scaling_factor: float = 1.0

    @property
    def grouped_dispatch(self) -> bool:
        """Whether routes that share an expert are computed as one group.

        Several tokens make groups worth describing: the route-statistics pass runs and both
        GEMMs are built to read its metadata. One token has one route per expert at most, so
        each GEMM computes its own route and no metadata is produced.
        """
        return self.num_tokens > 1


@dataclasses.dataclass(frozen=True)
class PrePermuteCall(CallSpec):
    """Complete selection facts for one pre-permute invocation."""

    layout: "MGroupedLayoutSpec | None" = None
    input_dtype: torch.dtype | None = None
    num_experts: int = 0
    num_tokens: int = 0
    hidden_size: int = 0
    top_k: int = 0
    routing_input_kind: str = "topk_ids"


@dataclasses.dataclass(frozen=True)
class MGroupedGemmCall(CallSpec):
    """Complete selection facts for one M-grouped GEMM invocation.

    The layout arrives structured — ``kind`` first, then the contiguous
    sub-axes — so a candidate can claim a region such as "every contiguous
    layout with psum metadata" without enumerating keys. ``m`` is the
    materialized row count (``num_groups * max_m`` for masked layouts). It is a
    fact of the call and not of the built kernel, so it is excluded from this
    record's equality: a candidate selected on it is still cached without it.
    """

    kind: str = ""  # "contiguous" | "masked"
    packing: str | None = None  # "tight" | "aligned"; None for masked
    metadata_kind: str | None = None  # "physical_psum" | "per_row"; None for masked
    alignment: int = 1  # 1 unless packing == "aligned"
    max_m: int | None = None  # masked only
    # Gated activation fused into the epilogue ("silu_and_mul", ...); None for a
    # plain GEMM. With one, ``n`` is B's stacked gate||up width and C has n / 2.
    activation: str | None = None
    ab_dtype: torch.dtype | None = None
    cd_dtype: torch.dtype | None = None
    num_groups: int = 0
    m: int = dataclasses.field(default=0, compare=False)
    n: int = 0
    k: int = 0


@dataclasses.dataclass(frozen=True)
class PostPermuteCall(CallSpec):
    """Complete selection facts for one post-permute invocation."""

    layout_key: str = ""
    max_m: int | None = None
    epilogue: "RoutingEpilogueSpec | None" = None
    input_dtype: torch.dtype | None = None
    routing_weight_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    num_experts: int = 0
    materialized_rows: int = 0
    num_tokens: int = 0
    hidden_size: int = 0
    top_k: int = 0


class FusedTopKFwdInterface(KernelInterface):
    """The ``call.top_k`` experts each token routes to, and the weight of each."""

    request = FusedTopKCall

    @abstractmethod
    def forward(
        self, gating_output: torch.Tensor, correction_bias: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score the router logits, then select each token's ``call.top_k`` experts.

        Scoring is ``call.scoring_func`` over each row, in float32 whatever ``call.dtype`` is.
        Ties break towards the lower expert index. Nothing is written in place.

        Args:
            gating_output: Contiguous ``(call.num_tokens, call.num_experts)`` in
                ``call.dtype`` on ``call.device``.
            correction_bias: Contiguous ``float32`` ``(call.num_experts,)``, passed exactly
                when ``call.with_correction_bias``. It is added to the sigmoid scores for the
                selection only; the returned weights stay the unbiased scores.

        Returns:
            New ``(topk_weights, topk_ids)``: ``float32`` and ``int32``, both
            ``(call.num_tokens, call.top_k)``. The weights are the selected experts' scores,
            divided by their own sum when ``call.renormalize``.
        """


class PermuteAlignFwdInterface(KernelInterface):
    """The index arrays a block-aligned MoE grouped GEMM reads, built from the routing ids."""

    request = PermuteAlignCall

    @abstractmethod
    def forward(self, topk_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sort the routes by expert and pad each expert's run to ``call.block_size``.

        Nothing is written in place. A padding slot holds ``call.num_routes`` in
        *sorted_token_ids*, an index no route has, and a block past the padded length holds
        ``call.num_experts`` in *expert_ids*.

        Args:
            topk_ids: Contiguous ``int32`` with ``call.num_routes`` elements, each in
                ``[0, call.num_experts)``, on ``call.device``.

        Returns:
            New ``(sorted_token_ids, expert_ids, num_tokens_post_pad)``, all ``int32``:
            ``(call.num_routes + (call.num_experts + 1) * (call.block_size - 1),)`` route
            indices grouped by expert, one expert id per block of that array, and the one
            element holding the padded route count.
        """


class SharedExpertMLPFwdInterface(KernelInterface):
    """The dense gated MLP of an MoE layer's shared expert, over every token."""

    request = SharedExpertMLPCall

    @abstractmethod
    def forward(
        self, hidden: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor
    ) -> torch.Tensor:
        """Apply ``down(silu(hidden @ gate.T) * (hidden @ up.T))``; nothing is written in place.

        Both matmuls accumulate in float32; the gated activation is rounded to ``call.dtype``
        before the down projection. Every tensor is contiguous on ``call.device`` in
        ``call.dtype``.

        Args:
            hidden: ``(call.num_tokens, call.hidden_size)``.
            w_gate_up: ``(2 * call.ffn_size, call.hidden_size)``, the gate rows then the up rows.
            w_down: ``(call.hidden_size, call.ffn_size)``.

        Returns:
            A new ``(call.num_tokens, call.hidden_size)`` tensor in ``call.dtype``.
        """


class PrePermuteFwdInterface(KernelInterface):
    """Rank-grouped activations materialized into the expert layout ``call.layout`` names."""

    request = PrePermuteCall

    @abstractmethod
    def forward(
        self, hidden_states: torch.Tensor, local_expert_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather each route's token row into its expert's segment; nothing is written in place.

        The row count ``P`` is ``call.num_tokens * call.top_k`` for tight packing, and that
        count padded so every expert's segment starts on a multiple of the layout's alignment
        otherwise. Rows an aligned layout pads with are zero.

        Args:
            hidden_states: Contiguous ``(call.num_tokens, call.hidden_size)`` in
                ``call.input_dtype`` on ``call.device``.
            local_expert_ids: Contiguous ``int32`` ``(call.num_tokens, call.top_k)``, each in
                ``[0, call.num_experts)``.

        Returns:
            New ``(expert_input, layout_metadata, inverse_indices)``: ``(P, call.hidden_size)``
            in ``call.input_dtype``, ``int32`` metadata the layout fixes — ``(call.num_experts,)``
            segment end rows for ``physical_psum``, one expert id per row for ``per_row`` — and
            ``int32`` ``(call.num_tokens * call.top_k,)`` mapping each route to its row.
        """


class MGroupedGemmFwdInterface(KernelInterface):
    """One GEMM per expert over the rows ``call``'s layout assigns it."""

    request = MGroupedGemmCall

    @abstractmethod
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Contract each expert's rows of *a* with its weights: ``out[rows of g] = a @ b[g].T``.

        Accumulation is float32. With ``call.activation``, *b* stacks the gate and up
        projections along ``call.n`` and the epilogue writes ``act(gate) * up``, half as wide.
        Rows an aligned layout pads with, and a masked slab's rows past its valid count, hold
        unspecified values. The metadata's values are not checked, since checking would
        synchronise.

        Args:
            a: Contiguous ``(call.m, call.k)``, or ``(call.num_groups, call.max_m, call.k)``
                for a masked layout, in ``call.ab_dtype`` on ``call.device``.
            b: Contiguous ``(call.num_groups, call.n, call.k)`` in ``call.ab_dtype``.
            layout_metadata: Contiguous ``int32`` whose shape the layout fixes.
            out: An optional buffer in ``call.cd_dtype``, shaped like the return value,
                written in place and returned; ``None`` allocates one.

        Returns:
            ``(call.m, N)`` or ``(call.num_groups, call.max_m, N)`` in ``call.cd_dtype``, where
            ``N`` is ``call.n`` halved when ``call.activation`` is set.
        """


class PostPermuteFwdInterface(KernelInterface):
    """Token order restored from the expert rows, with the routing epilogue applied once."""

    request = PostPermuteCall

    @abstractmethod
    def forward(
        self,
        expert_output: torch.Tensor,
        inverse_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Weight each route's row, sum a token's routes, and store in ``call.output_dtype``.

        The reduction accumulates in float32 and is scaled by
        ``call.epilogue.routed_scaling_factor`` before the cast. *out* must not overlap
        *expert_output*, which is read concurrently.

        Args:
            expert_output: Contiguous ``(call.materialized_rows, call.hidden_size)`` in
                ``call.input_dtype`` on ``call.device``.
            inverse_indices: Contiguous ``int32`` ``(call.num_tokens * call.top_k,)`` mapping
                each route to its row of *expert_output*.
            topk_weights: Contiguous ``float32`` ``(call.num_tokens, call.top_k)``.
            out: An optional contiguous ``(call.num_tokens, call.hidden_size)`` buffer in
                ``call.output_dtype``, written in place and returned; ``None`` allocates one.

        Returns:
            ``(call.num_tokens, call.hidden_size)`` in ``call.output_dtype``.
        """


class IndexedRouteStatsFwdInterface(KernelInterface):
    """The same-expert route groups of one multi-token call, described on device."""

    request = IndexedExpertCall

    @abstractmethod
    def forward(self, expert_ids: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Count each expert's routes and list them, so a GEMM block computes a whole group.

        Reached only when ``call.grouped_dispatch``. *out* is written in place and returned;
        its length is the implementation's ``output_size``, which the caller reads off the
        built kernel rather than computing.

        Args:
            expert_ids: Contiguous ``int32`` ``(call.num_tokens, call.top_k)``, each in
                ``[0, call.num_experts)``, on ``call.device``.
            out: A contiguous ``int32`` buffer of ``output_size`` elements on ``call.device``.

        Returns:
            *out*, holding the per-expert route counts, each route's rank within its expert,
            and each expert's route list, in the layout the two GEMM interfaces read.
        """


class IndexedExpertGateUpFwdInterface(KernelInterface):
    """The gate/up projection of each route, with the gated activation fused."""

    request = IndexedExpertCall

    @abstractmethod
    def forward(
        self,
        a: torch.Tensor,
        weights: torch.Tensor,
        expert_ids: torch.Tensor,
        route_metadata: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project each token onto its routed experts and apply ``silu(gate) * up``.

        One route is one row of its expert's GEMM. Accumulation is float32.

        Args:
            a: Contiguous ``(call.num_tokens, call.hidden_size)`` in ``call.dtype`` on
                ``call.device``.
            weights: Contiguous ``(call.num_experts, 2 * call.ffn_size, call.hidden_size)`` in
                ``call.dtype``, the gate rows then the up rows.
            expert_ids: Contiguous ``int32`` ``(call.num_tokens, call.top_k)``.
            route_metadata: What :class:`IndexedRouteStatsFwdInterface` wrote, passed exactly
                when ``call.grouped_dispatch``.
            out: An optional ``(call.num_tokens, call.top_k, call.ffn_size)`` buffer in
                ``call.dtype``, written in place and returned; ``None`` allocates one.

        Returns:
            ``(call.num_tokens, call.top_k, call.ffn_size)`` in ``call.dtype``.
        """


class IndexedExpertDownFwdInterface(KernelInterface):
    """The down projection of each route's activated hidden row."""

    request = IndexedExpertCall

    @abstractmethod
    def forward(
        self,
        a: torch.Tensor,
        weights: torch.Tensor,
        expert_ids: torch.Tensor,
        route_metadata: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Contract each route's row with its expert's down projection.

        The route axis is already materialized, so *a* carries one row per route rather than
        one per token. Accumulation is float32.

        Args:
            a: Contiguous ``(call.num_tokens, call.top_k, call.ffn_size)`` in ``call.dtype`` on
                ``call.device``.
            weights: Contiguous ``(call.num_experts, call.hidden_size, call.ffn_size)`` in
                ``call.dtype``.
            expert_ids: Contiguous ``int32`` ``(call.num_tokens, call.top_k)``.
            route_metadata: What :class:`IndexedRouteStatsFwdInterface` wrote, passed exactly
                when ``call.grouped_dispatch``.
            out: An optional ``(call.num_tokens, call.top_k, call.hidden_size)`` buffer in
                ``call.dtype``, written in place and returned; ``None`` allocates one.

        Returns:
            ``(call.num_tokens, call.top_k, call.hidden_size)`` in ``call.dtype``.
        """


class IndexedWeightedReduceFwdInterface(KernelInterface):
    """Each token's routes weighted and summed into one row."""

    request = IndexedExpertCall

    @abstractmethod
    def forward(
        self, expert_output: torch.Tensor, topk_weights: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        """Reduce the route axis, scaled by ``call.routed_scaling_factor``.

        The sum accumulates in float32 and is cast to ``call.dtype`` on the store. *output* is
        written in place and returned; it must not overlap *expert_output*.

        Args:
            expert_output: Contiguous ``(call.num_tokens, call.top_k, call.hidden_size)`` in
                ``call.dtype`` on ``call.device``.
            topk_weights: Contiguous ``float32`` ``(call.num_tokens, call.top_k)``.
            output: A contiguous ``(call.num_tokens, call.hidden_size)`` buffer in
                ``call.dtype``.

        Returns:
            *output*.
        """
