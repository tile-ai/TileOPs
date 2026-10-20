import itertools
import math
from collections.abc import Iterable

import torch

from workloads.workload_base import CallWorkload


class FusedTopKWorkload(CallWorkload):
    """Router logits, and the optional bias, of one ``FusedTopKFwdOp`` call."""

    def ref_program(
        self, gating_output: torch.Tensor, correction_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score, select ``top_k`` (on the biased scores when a bias is passed) inside the best
        expert groups, renormalize, scale."""
        p = self.call.params
        return ref_fused_topk(
            gating_output,
            correction_bias,
            p["top_k"],
            p["scoring_func"],
            p["renormalize"],
            p["num_expert_group"],
            p["topk_group"],
            p["routed_scaling_factor"],
        )

    def verification(self, *inputs):
        from workloads.numerics import Custom

        gating_output, *bias = inputs
        correction_bias = bias[0] if bias else None
        p = self.call.params
        scale = p["routed_scaling_factor"]
        router = _Router(
            gating_output,
            correction_bias,
            p["top_k"],
            p["scoring_func"],
            p["renormalize"],
            p["num_expert_group"],
            p["topk_group"],
        )

        def validate(got, expected):
            weights, ids = got[:2]
            ref_weights, ref_ids = expected
            assert weights.shape == ref_weights.shape and weights.dtype == ref_weights.dtype
            assert ids.shape == ref_ids.shape and ids.dtype == ref_ids.dtype
            assert ((ids >= 0) & (ids < gating_output.shape[-1])).all()
            ordered = ids.sort(-1).values
            assert (ordered[:, 1:] != ordered[:, :-1]).all(), "duplicate expert"
            router.assert_admissible(ids)
            selected = router.scores.gather(1, ids.long())
            if router.renormalize:
                selected = selected / selected.sum(-1, keepdim=True)
            torch.testing.assert_close(weights, selected * scale, rtol=1e-4, atol=0)

        return Custom(
            validate,
            "an admissible expert selection and its weights, independent of tie order",
        )


class MoEPermuteAlignWorkload(CallWorkload):
    """The routing ids of one ``MoEPermuteAlignFwdOp`` call."""

    def ref_program(self, topk_ids):
        """Routes in ascending index order within each expert's range, padded with ``numel``.

        Blocks past the padded length hold ``num_experts``, one of the values that name no
        expert.
        """
        p = self.call.params
        ids, experts, count = ref_permute_align(topk_ids, p["block_size"], p["num_experts"])
        capacity = topk_ids.numel() + (p["num_experts"] + 1) * (p["block_size"] - 1)
        blocks = math.ceil(capacity / p["block_size"])
        return (
            torch.nn.functional.pad(ids, (0, capacity - ids.numel()), value=topk_ids.numel()),
            torch.nn.functional.pad(experts, (0, blocks - experts.numel()), value=p["num_experts"]),
            count,
        )

    def verification(self, *inputs):
        """Exact up to route order within each expert and the value past the padded length."""
        from workloads.numerics import Custom

        num_experts = self.call.params["num_experts"]
        block_size = self.call.params["block_size"]
        flat = inputs[0].flatten().long()
        counts = torch.bincount(flat, minlength=num_experts)
        padded = (counts + block_size - 1) // block_size * block_size
        starts = torch.cumsum(padded, 0) - padded
        # Each route's slot when every expert's routes are in ascending index order.
        rank = torch.arange(flat.numel(), device=flat.device) - (
            torch.cumsum(counts, 0) - counts
        ).repeat_interleave(counts)
        group_start = starts.repeat_interleave(counts)
        routed_slots = group_start + rank

        def validate(got, expected):
            # compare_outputs has already matched every output's shape, dtype and device.
            tokens, experts, count = got
            ref_tokens, ref_experts, ref_count = expected
            assert torch.equal(count, ref_count), "num_tokens_post_pad differs"
            blocks = int(ref_count.item()) // block_size
            assert torch.equal(experts[:blocks], ref_experts[:blocks]), "a block's owner differs"
            tail = experts[blocks:]
            assert ((tail < 0) | (tail >= num_experts)).all(), (
                "a block past the end names an expert"
            )
            # Sort each expert's routed slots, which may hold its routes in any order.
            key = torch.arange(tokens.numel(), device=tokens.device)
            key[routed_slots] = group_start
            order = torch.sort(tokens, stable=True).indices
            order = order[torch.sort(key[order], stable=True).indices]
            assert torch.equal(tokens[order], ref_tokens), "sorted_token_ids differs"

        return Custom(validate, "exact up to route order within each expert's routed slots")


class MoEPrePermuteWorkload(CallWorkload):
    """Hidden states and local expert ids of one ``MoEPrePermuteFwdOp`` call."""

    def ref_program(
        self, hidden_states: torch.Tensor, local_expert_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tight physical-psum layout: rows grouped by expert in route order, exclusive ends.

        Raises:
            NotImplementedError: The call's layout is not tight physical-psum.
        """
        layout = self.call.params["layout"]
        if layout.packing != "tight" or layout.metadata_kind != "physical_psum":
            raise NotImplementedError(f"no pre-permute reference for {layout!r}")
        top_k = local_expert_ids.shape[1]
        flat = local_expert_ids.flatten().long()
        order = torch.argsort(flat, stable=True)
        expert_input = hidden_states[order // top_k]
        counts = torch.bincount(flat, minlength=self.call.params["num_local_experts"])
        ends = torch.cumsum(counts, dim=0).to(torch.int32)
        inverse = torch.empty_like(order, dtype=torch.int32)
        inverse[order] = torch.arange(order.numel(), dtype=torch.int32, device=order.device)
        return expert_input, ends, inverse

    def verification(self, *inputs):
        from workloads.numerics import Custom

        hidden_states, local_ids = inputs

        def validate(got, expected):
            rows, ends, inverse = got
            ref_rows, ref_ends, ref_inverse = expected
            assert rows.shape == ref_rows.shape and rows.dtype == ref_rows.dtype
            assert inverse.shape == ref_inverse.shape and inverse.dtype == ref_inverse.dtype
            torch.testing.assert_close(ends, ref_ends, rtol=0, atol=0)
            torch.testing.assert_close(
                inverse.long().sort().values,
                torch.arange(inverse.numel(), device=inverse.device),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                rows[inverse.long()], ref_rows[ref_inverse.long()], rtol=0, atol=0
            )
            owners = torch.searchsorted(ends, inverse, right=True)
            torch.testing.assert_close(owners, local_ids.flatten().to(owners.dtype), rtol=0, atol=0)

        return Custom(validate, "route permutation and expert segment ownership")


def valid_rows(layout, layout_metadata: torch.Tensor, rows: int, num_experts: int) -> torch.Tensor:
    """Bool mask over the flat materialized rows a grouped GEMM defines an output for."""
    meta = layout_metadata.tolist()
    if layout.kind == "contiguous" and layout.metadata_kind == "per_row":
        return layout_metadata < num_experts
    valid = torch.zeros(rows, dtype=torch.bool)
    if layout.kind == "masked":
        for g, count in enumerate(meta):
            valid[g * layout.max_m : g * layout.max_m + count] = True
    else:
        start = 0
        for end in meta:
            start = -(-start // layout.alignment) * layout.alignment
            valid[start:end] = True
            start = end
    return valid.to(layout_metadata.device)


def ref_moe_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    layout_metadata: torch.Tensor,
    layout,
    out_dtype: torch.dtype | None = None,
    activation: str | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``out[rows of g] = a[rows of g] @ b[g]^T + bias[g]`` in fp32 per expert; other rows are
    zero.

    With ``activation``, ``b`` and ``bias`` hold gate and up along ``N`` in the order the
    activation reads (:func:`gated_activation`) and the result is the gated activation, half
    as wide, as the fused epilogue writes it. The bias is added before the activation.
    """
    out_dtype = a.dtype if out_dtype is None else out_dtype
    num_experts, n, _ = b.shape
    out = torch.zeros(*a.shape[:-1], n, dtype=torch.float32, device=a.device)

    def product(rows: torch.Tensor, g: int) -> torch.Tensor:
        result = rows.float() @ b[g].float().T
        return result if bias is None else result + bias[g].float()

    meta = layout_metadata.tolist()
    if layout.kind == "masked":
        for g, count in enumerate(meta):
            out[g, :count] = product(a[g, :count], g)
    elif layout.metadata_kind == "per_row":
        ids = layout_metadata.long()
        for g in range(num_experts):
            selected = ids == g
            out[selected] = product(a[selected], g)
    else:
        start = 0
        for g, end in enumerate(meta):
            start = -(-start // layout.alignment) * layout.alignment
            out[start:end] = product(a[start:end], g)
            start = end
    if activation is not None:
        out = gated_activation(out, activation)
    return out.to(out_dtype)


class MoEGroupedGemmWorkload(CallWorkload):
    """Expert-materialized ``a``, per-expert ``b`` and the layout's generated metadata."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        a, b, metadata, bias = super().gen_inputs()
        # A small scale keeps fp16 accumulation over K = 7168 finite and within tolerance.
        a.mul_(0.02)
        b.mul_(0.02)
        activation = self.call.params["activation"]
        if bias is not None and activation != "swigluoai":
            # A product at std 1 stands above the 1e-1 atol beside the standard-normal bias.
            _spread(a, b, 1.0)
        _spread_gate_up(a, b, activation)
        return a, b, metadata, bias

    def ref_program(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        layout_metadata: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p = self.call.params
        out_dtype = None if p["out_dtype"] is None else getattr(torch, p["out_dtype"])
        return ref_moe_grouped_gemm(
            a, b, layout_metadata, p["layout"], out_dtype, activation=p["activation"], bias=bias
        )

    def verification(self, *inputs):
        from workloads.numerics import Custom

        x, b, metadata = inputs[:3]
        mask = valid_rows(
            self.call.params["layout"], metadata, x.numel() // x.shape[-1], b.shape[0]
        )

        def validate(got, expected):
            torch.testing.assert_close(
                got.reshape(-1, got.shape[-1])[mask].float(),
                expected.reshape(-1, expected.shape[-1])[mask].float(),
                rtol=1e-3 if got.dtype == torch.float32 else 2e-2,
                atol=1e-2 if got.dtype == torch.float32 else 1e-1,
            )

        return Custom(validate, "defined rows of grouped expert layout")


class MoEGroupedGemmFP8Workload(CallWorkload):
    """FP8 masked slabs and weights, their block scales, and the layout's generated counts."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        a, a_scale, b, b_scale, metadata = super().gen_inputs()
        # Values well inside e4m3's range, and positive scales of order one, as a
        # quantizer would produce them.
        return (
            (a.float() * 0.25).to(a.dtype),
            torch.rand_like(a_scale).add_(0.5),
            (b.float() * 0.25).to(b.dtype),
            torch.rand_like(b_scale).add_(0.5),
            metadata,
        )

    def ref_program(
        self,
        a: torch.Tensor,
        a_scale: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        layout_metadata: torch.Tensor,
    ) -> torch.Tensor:
        """Each operand dequantized in fp32 under its block scales, then the masked GEMM."""
        a_f = a.float() * a_scale.repeat_interleave(128, dim=-1)
        b_f = b.float() * b_scale.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
        return ref_moe_grouped_gemm(
            a_f, b_f, layout_metadata, self.call.params["layout"], torch.bfloat16
        )

    def verification(self, *inputs):
        from workloads.gemm import fp8_matmul_verification
        from workloads.numerics import Custom

        a, metadata = inputs[0], inputs[-1]
        tolerance = fp8_matmul_verification(a.shape[-1])
        mask = valid_rows(
            self.call.params["layout"], metadata, a.numel() // a.shape[-1], inputs[2].shape[0]
        )

        def validate(got, expected):
            torch.testing.assert_close(
                got.reshape(-1, got.shape[-1])[mask].float(),
                expected.reshape(-1, expected.shape[-1])[mask].float(),
                rtol=tolerance.rtol,
                atol=tolerance.atol,
            )

        return Custom(validate, "valid rows of each masked expert slab, fp8 matmul bound")


class MoEExpertMLPWorkload(CallWorkload):
    """Expert-materialized input, gate/up and down weights, generated metadata, optional
    biases."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x, w_gate_up, w_down, metadata, b_gate_up, b_down = super().gen_inputs()
        x.mul_(0.02)
        w_gate_up.mul_(0.02)
        w_down.mul_(0.02)
        _spread_gate_up(x, w_gate_up, self.call.params["activation"])
        return x, w_gate_up, w_down, metadata, b_gate_up, b_down

    def ref_program(
        self,
        expert_input: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        layout_metadata: torch.Tensor,
        b_gate_up: torch.Tensor | None = None,
        b_down: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p = self.call.params
        activated = ref_moe_grouped_gemm(
            expert_input,
            w_gate_up,
            layout_metadata,
            p["layout"],
            activation=p["activation"],
            bias=b_gate_up,
        )
        return ref_moe_grouped_gemm(activated, w_down, layout_metadata, p["layout"], bias=b_down)

    def verification(self, *inputs):
        from workloads.numerics import Custom

        x, _, w_down, metadata = inputs[:4]
        mask = valid_rows(
            self.call.params["layout"], metadata, x.numel() // x.shape[-1], w_down.shape[0]
        )

        def validate(got, expected):
            torch.testing.assert_close(
                got.reshape(-1, got.shape[-1])[mask].float(),
                expected.reshape(-1, expected.shape[-1])[mask].float(),
                rtol=2e-2,
                atol=1e-1,
            )

        return Custom(validate, "defined rows of grouped expert layout")


class MoEPostPermuteWorkload(CallWorkload):
    """Expert outputs, routing weights and inverse indices of one ``MoEPostPermuteFwdOp`` call."""

    def ref_program(
        self,
        expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        """``out[t] = scale * sum_k w[t, k] * rows[inverse[t * K + k]]`` in fp32, then cast."""
        p = self.call.params
        tokens, top_k = topk_weights.shape
        rows = expert_output.reshape(-1, expert_output.shape[-1])
        gathered = rows[inverse_indices.long()].float().view(tokens, top_k, -1)
        out = (gathered * topk_weights.unsqueeze(-1)).sum(dim=1)
        if p["epilogue"] is not None:
            out = out * p["epilogue"].routed_scaling_factor
        dtype = expert_output.dtype if p["out_dtype"] is None else getattr(torch, p["out_dtype"])
        return out.to(dtype)

    def verification(self, *inputs):
        epilogue = self.call.params["epilogue"]
        return post_permute_verification(
            1.0 if epilogue is None else epilogue.routed_scaling_factor
        )


def post_permute_verification(routed_scaling_factor=1.0):
    """Routing reduction error scales with the public epilogue multiplier."""
    from workloads.numerics import Exact

    return Exact(atol=2e-2 * routed_scaling_factor, rtol=2e-2)


def gated_activation(gate_up: torch.Tensor, activation: str) -> torch.Tensor:
    """The gated activation over the last axis, half as wide, in the input's dtype:
    ``MoEGroupedGemmFwdOp``'s docstring defines each one and the order of its gate and up
    columns. ``swigluoai`` is computed in fp32."""
    if activation == "swigluoai":
        # gpt-oss's fixed alpha and limit, as MoEGroupedGemmFwdOp's docstring states them.
        alpha, limit = 1.702, 7.0
        gate = gate_up[..., ::2].float().clamp(max=limit)
        up = gate_up[..., 1::2].float().clamp(min=-limit, max=limit)
        return ((up + 1) * gate * torch.sigmoid(alpha * gate)).to(gate_up.dtype)
    gate, up = gate_up.chunk(2, dim=-1)
    if activation == "silu_and_mul":
        return torch.nn.functional.silu(gate) * up
    if activation == "gelu_and_mul":
        return torch.nn.functional.gelu(gate) * up
    raise ValueError(f"unknown gated activation: {activation}")


def _sample(rows: torch.Tensor) -> torch.Tensor:
    """The rows a workload measures a scale on: the first 64 of *rows*, in fp32."""
    return rows.reshape(-1, rows.shape[-1])[:64].float()


def _spread(rows: torch.Tensor, weights: torch.Tensor, std: float) -> None:
    """Scale *weights* in place so the product of :func:`_sample` of *rows* with expert 0's
    weights has std *std*."""
    weights.mul_(std / (_sample(rows) @ weights[0].float().T).std().item())


def _spread_gate_up(rows: torch.Tensor, w_gate_up: torch.Tensor, activation: str | None) -> None:
    """Scale ``w_gate_up`` in place so a ``swigluoai`` gate/up result, measured on expert 0 and
    :func:`_sample` of *rows*, has std 4: swigluoai's clamp at 7 then lies 1.75 std out, so the
    gate's upper tail and both of up's tails pass it. Any other activation keeps its scale."""
    if activation == "swigluoai":
        _spread(rows, w_gate_up, 4.0)


def _scale_routed(call, rows, w_gate_up, w_down, b_gate_up=None, b_down=None) -> None:
    """Scale a routed call's weights and standard-normal biases in place, measured on expert 0
    and :func:`_sample` of *rows*.

    Under ``swigluoai`` the gate/up result is spread as :func:`_spread_gate_up` states, and the
    down product taken to std 0.1, where the routed verification's 1e-2 elementwise bound lies
    above the operand-dtype rounding of the output. Each bias is scaled to half the std of the
    product it joins, so the normalized-error bound sees a dropped or misplaced one.
    """
    activation = call.params.get("activation", "silu_and_mul")
    _spread_gate_up(rows, w_gate_up, activation)
    gate_up = _sample(rows) @ w_gate_up[0].float().T
    if b_gate_up is not None:
        b_gate_up.mul_(0.5 * gate_up.std().item())
        gate_up = gate_up + b_gate_up[0].float()
    activated = gated_activation(gate_up, activation)
    if activation == "swigluoai":
        _spread(activated, w_down, 0.1)
    if b_down is not None:
        b_down.mul_(0.5 * (activated @ w_down[0].float().T).std().item())


def ref_routed_experts(
    hidden: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu_and_mul",
    scale: float = 1.0,
    *,
    b_gate_up: torch.Tensor | None = None,
    b_down: torch.Tensor | None = None,
) -> torch.Tensor:
    """``scale * sum_k w[t, k] * (down_k(act(gate_up_k(h[t]) + b_gate_up[k])) + b_down[k])``
    over each token's routed experts, in fp32 per expert, cast to the hidden dtype; each bias
    joins its GEMM's fp32 result, ``b_down`` before the routing weight scales it."""
    output = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
    ids = topk_ids.to(torch.int64)
    for e in range(w_gate_up.shape[0]):
        t_idx, k_idx = (ids == e).nonzero(as_tuple=True)
        if t_idx.numel() == 0:
            continue
        gate_up = hidden[t_idx].float() @ w_gate_up[e].float().T
        if b_gate_up is not None:
            gate_up = gate_up + b_gate_up[e].float()
        down = gated_activation(gate_up, activation) @ w_down[e].float().T
        if b_down is not None:
            down = down + b_down[e].float()
        output.index_add_(0, t_idx, down * topk_weights[t_idx, k_idx].float().unsqueeze(-1))
    return (output * scale).to(hidden.dtype)


class MoEExpertsWorkload(CallWorkload):
    """Tokens, expert weights and their routing for one ``FusedMoEExpertsFwdOp`` call."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        output, hidden, w_gate_up, w_down, topk_weights, topk_ids, *biases = super().gen_inputs()
        # Small scales keep fp16 accumulation over H = 7168 finite; routing weights sum to one.
        hidden.mul_(0.1)
        w_gate_up.mul_(0.02)
        w_down.mul_(0.02)
        _scale_routed(self.call, hidden, w_gate_up, w_down, *biases)
        return (
            output,
            hidden,
            w_gate_up,
            w_down,
            topk_weights.softmax(dim=-1),
            topk_ids,
            *biases,
        )

    def ref_program(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        b_gate_up: torch.Tensor | None = None,
        b_down: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p = self.call.params
        return ref_routed_experts(
            hidden_states,
            w_gate_up,
            w_down,
            topk_weights,
            topk_ids,
            p.get("activation", "silu_and_mul"),
            p["routed_scaling_factor"],
            b_gate_up=b_gate_up,
            b_down=b_down,
        )

    def verification(self, *inputs):
        return moe_verification(2)


class IndexedExpertMLPWorkload(MoEExpertsWorkload):
    """One ``IndexedExpertMLPFwdOp`` call: the same inputs and reference as the expert MLP."""


def ref_group_scores(select: torch.Tensor, num_groups: int, biased: bool) -> torch.Tensor:
    """``[T, num_groups]`` scores of consecutive expert groups: the sum of each group's two best
    selection scores when the selection carries a bias, else its best one (vLLM
    ``grouped_topk``)."""
    grouped = select.view(select.shape[0], num_groups, -1)
    if biased:
        return grouped.topk(2, dim=-1).values.sum(dim=-1)
    return grouped.max(dim=-1).values


def ref_fused_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    top_k: int,
    scoring_func: str,
    renormalize: bool,
    num_expert_group: int = 1,
    topk_group: int = 1,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score, select ``top_k`` (on the biased scores when a bias is passed), renormalize, scale.

    With ``num_expert_group > 1`` the selection is restricted to the ``topk_group`` best groups
    of :func:`ref_group_scores`, ties to the lower group index, as vLLM's ``grouped_topk``
    torch fallback computes it in float32.
    """
    logits = gating_output.float()
    scores = torch.softmax(logits, dim=-1) if scoring_func == "softmax" else logits.sigmoid()
    select = scores if correction_bias is None else scores + correction_bias
    if num_expert_group > 1:
        group_scores = ref_group_scores(select, num_expert_group, correction_bias is not None)
        # A stable descending sort keeps equal groups in index order: the lower one is kept.
        kept = group_scores.sort(dim=-1, descending=True, stable=True).indices[:, :topk_group]
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, kept, True)
        expert_mask = group_mask.repeat_interleave(select.shape[-1] // num_expert_group, dim=-1)
        select = select.masked_fill(~expert_mask, float("-inf"))
    topk_ids = torch.topk(select, top_k, dim=-1, sorted=False).indices
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids.int()


class _Router:
    """The admissible routings of each token, from the float32 reference scores.

    A token's routing is a set of ``topk_group`` groups and ``top_k`` experts inside them.
    It is admissible when every group above the last kept group score by more than the
    routing tolerance is kept and none below it by more, and when the experts' sorted
    selection scores match, within the tolerance, the top-k inside that group set. A token
    whose canonical routing has no score within the tolerance of its boundary admits only
    the canonical routing.
    """

    def __init__(
        self,
        gating_output,
        correction_bias,
        top_k,
        scoring_func,
        renormalize,
        num_groups,
        topk_group,
    ):
        logits = gating_output.float()
        self.scores = logits.softmax(-1) if scoring_func == "softmax" else logits.sigmoid()
        self.selection = (
            self.scores if correction_bias is None else self.scores + correction_bias.float()
        )
        self.biased = correction_bias is not None
        self.top_k, self.renormalize = top_k, renormalize
        self.num_groups, self.topk_group = num_groups, topk_group
        self.canonical = ref_fused_topk(
            gating_output,
            correction_bias,
            top_k,
            scoring_func,
            renormalize,
            num_groups,
            topk_group,
        )[1].long()
        self.ambiguous = self._ambiguous_rows()

    @staticmethod
    def _slack(score: torch.Tensor) -> torch.Tensor:
        """The routing tolerance the op docstrings state, around *score*."""
        # Why: the absolute term admits a float32 sigmoid's rounding where a correction bias
        # cancels the score to near zero.
        return 1e-5 * score.abs() + 1e-6

    def _group_scores(self) -> torch.Tensor:
        return ref_group_scores(self.selection, self.num_groups, self.biased)

    def _ambiguous_rows(self) -> torch.Tensor:
        selection = self.selection
        rows = torch.zeros(selection.shape[0], dtype=torch.bool, device=selection.device)
        if self.num_groups > 1 and self.topk_group < self.num_groups:
            order = self._group_scores().sort(dim=-1, descending=True, stable=True)
            last = order.values[:, self.topk_group - 1 : self.topk_group]
            next_ = order.values[:, self.topk_group : self.topk_group + 1]
            # Why twice the tolerance: this only picks the tokens to enumerate, and rounding at
            # the band edge must not drop one the enumeration admits.
            rows |= (next_ >= last - 2 * self._slack(last)).squeeze(-1)
            # Expert ties count only inside the kept groups.
            kept = torch.zeros_like(order.values, dtype=torch.bool)
            kept.scatter_(1, order.indices[:, : self.topk_group], True)
            mask = kept.repeat_interleave(selection.shape[-1] // self.num_groups, dim=-1)
            selection = selection.masked_fill(~mask, float("-inf"))
        kth = selection.gather(1, self.canonical).amin(-1, keepdim=True)
        outside = selection.scatter(1, self.canonical, float("-inf"))
        near = (outside >= kth - 2 * self._slack(kth)).any(-1)
        return rows | near

    def _group_sets(self, row: int) -> Iterable[tuple[int, ...]]:
        if self.num_groups == 1 or self.topk_group >= self.num_groups:
            return [tuple(range(self.num_groups))]
        scores = self._group_scores()[row]
        last = scores.sort(descending=True).values[self.topk_group - 1]
        slack = self._slack(last)
        always = {g for g in range(self.num_groups) if scores[g] > last + slack}
        allowed = [g for g in range(self.num_groups) if scores[g] >= last - slack]
        return (
            groups
            for groups in itertools.combinations(allowed, self.topk_group)
            if always <= set(groups)
        )

    def _expert_sets(self, row: int, groups: tuple[int, ...], limit: int) -> list[frozenset[int]]:
        selection = self.selection[row]
        size = selection.numel() // self.num_groups
        experts = torch.cat(
            [torch.arange(g * size, (g + 1) * size, device=selection.device) for g in groups]
        )
        scores = selection[experts]
        top = scores.topk(self.top_k).values.sort().values
        kth = top[0]
        reach = 2 * self._slack(kth)
        forced = experts[scores > kth + reach].tolist()
        pool = [e for e in experts[scores >= kth - reach].tolist() if e not in forced]
        need = self.top_k - len(forced)
        candidates = math.comb(len(pool), need)
        if candidates > limit * self.top_k:
            raise ValueError(f"token {row} has {candidates} candidate routings to enumerate")
        sets = []
        for rest in itertools.combinations(pool, need):
            chosen = forced + list(rest)
            if ((selection[chosen].sort().values - top).abs() <= self._slack(top)).all():
                sets.append(frozenset(chosen))
        return sets

    def routings(self, row: int) -> list[torch.Tensor]:
        """The admissible expert sets of token *row*, the canonical one first.

        Raises:
            ValueError: The token admits more than 64 routings, or has more candidates
                within the tolerance than can be enumerated.
        """
        # Stop once the distinct-route limit is exceeded; later groups cannot shrink the set.
        limit = 64
        canonical = self.canonical[row]
        if not self.ambiguous[row]:
            return [canonical]
        found = {frozenset(canonical.tolist())}
        for groups in self._group_sets(row):
            found.update(self._expert_sets(row, groups, limit))
            if len(found) > limit:
                raise ValueError(f"token {row} admits more than {limit} routings")
        found.discard(frozenset(canonical.tolist()))
        others = [torch.tensor(sorted(ids), device=canonical.device) for ids in found]
        return [canonical, *others]

    def weights(self, row: int, ids: torch.Tensor) -> torch.Tensor:
        """The unbiased scores of *ids*, renormalized when the call renormalizes."""
        weights = self.scores[row, ids]
        return weights / weights.sum() if self.renormalize else weights

    def assert_admissible(self, ids: torch.Tensor) -> None:
        returned, canonical = ids.long().sort(-1).values, self.canonical.sort(-1).values
        # Why: enumerating every near-tied token refuses one admitting too many routings even
        # when it returns the canonical routing.
        rows = (returned != canonical).any(-1) | self.ambiguous
        for row in rows.nonzero().flatten().tolist():
            admitted = {frozenset(r.tolist()) for r in self.routings(row)}
            assert frozenset(returned[row].tolist()) in admitted, (
                f"token {row} returns experts no admissible routing selects"
            )


def _routed_reference_rows(router, rows, inputs, activation, scale):
    """For each token in *rows*, the routed output under each of its admissible routings."""
    hidden, w_gate_up, w_down, b_gate_up, b_down = inputs
    candidates = {}
    for row in rows:
        routings = router.routings(row)
        experts = sorted({e for ids in routings for e in ids.tolist()})
        token = hidden[row : row + 1].float()
        outputs = {}
        for e in experts:
            gate_up = token @ w_gate_up[e].float().T
            if b_gate_up is not None:
                gate_up = gate_up + b_gate_up[e].float()
            down = gated_activation(gate_up, activation) @ w_down[e].float().T
            if b_down is not None:
                down = down + b_down[e].float()
            outputs[e] = down[0]
        refs = []
        for ids in routings:
            total = torch.zeros_like(outputs[experts[0]])
            # Expert order matches ref_routed_experts' accumulation.
            for e, weight in sorted(
                zip(ids.tolist(), router.weights(row, ids).tolist(), strict=True)
            ):
                total = total + outputs[e] * weight
            refs.append((total * scale).to(hidden.dtype))
        candidates[row] = refs
    return candidates


def _moe_close(got, expected, tol, bound):
    """Elementwise ``atol = rtol = tol`` and normalized squared error at most *bound*."""
    from workloads.numerics import assert_normalized_error

    torch.testing.assert_close(got, expected, rtol=tol, atol=tol)
    assert_normalized_error(got, expected, bound=bound)


def _gate_up_swapped(gate_up_index):
    """The reference with gate and up rows exchanged: a gate-order fault."""
    from workloads.numerics import NegativeControl

    def swapped(reference, inputs):
        changed = list(inputs)
        gate, up = changed[gate_up_index].chunk(2, dim=-2)
        changed[gate_up_index] = torch.cat((up, gate), dim=-2)
        return reference(*changed)

    return NegativeControl("gate-up-swapped", swapped)


def _admissible_reference(
    out: torch.Tensor, refs: list[torch.Tensor], tol: float, bound: float
) -> torch.Tensor:
    """The reference row in *refs* that best lets *out* pass ``_moe_close(tol, bound)``.

    Candidates outside the elementwise bound, or whose nonfinite values differ from *out*'s,
    are dropped. Among the rest, the one with the smallest ``E - bound * D`` is taken, where
    ``E`` and ``D`` are the finite elements' squared error and energy; the normalized-error
    bound over the whole output then holds for some choice of rows exactly when it holds
    for this one. With no candidate left, the first is returned and fails the bounds.
    """
    out = out.double()
    finite = torch.isfinite(out)
    best, best_share = refs[0], None
    for ref in refs:
        ref64 = ref.double()
        if not torch.equal(finite, torch.isfinite(ref64)) or not torch.allclose(
            out.masked_fill(finite, 0), ref64.masked_fill(finite, 0), 0, 0, equal_nan=True
        ):
            continue
        a, b = out.masked_fill(~finite, 0), ref64.masked_fill(~finite, 0)
        if ((a - b).abs() > tol + tol * b.abs()).any():
            continue
        share = ((a - b) ** 2).sum() - bound * (a * a + b * b).sum()
        if best_share is None or share < best_share:
            best, best_share = ref, share
    return best


def routed_verification(router, inputs, activation, scale, gate_up_index, shared=False):
    """``moe_verification``'s bounds, against an admissible routing of each near-tied token.

    The output is accepted exactly when some admissible routing of every token yields a
    reference within the bounds.
    """
    from workloads.numerics import Custom

    # Why: the same bounds as moe_verification; only the reference differs.
    tol, bound = 1e-2, 1e-4

    def assemble(got, expected):
        rows = router.ambiguous.nonzero().flatten().tolist()
        if not rows:
            return expected
        assembled = expected.clone()
        for row, refs in _routed_reference_rows(router, rows, inputs, activation, scale).items():
            assembled[row] = _admissible_reference(got[row], refs, tol, bound)
        return assembled

    def validate(got, expected):
        if shared:
            (got_shared, got_routed), (ref_shared, ref_routed) = got, expected
            if ref_shared is None:
                assert got_shared is None
            else:
                _moe_close(got_shared, ref_shared, tol, bound)
            _moe_close(got_routed, assemble(got_routed, ref_routed), tol, bound)
        else:
            _moe_close(got, assemble(got, expected), tol, bound)

    return Custom(
        validate,
        f"elementwise atol/rtol {tol} and normalized squared error <= {bound} under an "
        "admissible routing of each near-tied token",
        controls=(_gate_up_swapped(gate_up_index),),
    )


class FusedMoEWorkload(CallWorkload):
    """Tokens, gating logits and expert weights for one ``FusedMoEFwdOp`` call.

    The logits come from the workload's own generator: they decide the experts the call
    reads, which the roofline prices (docs/design/roofline.md §4.7).
    """

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        hidden, gating, w_gate_up, w_down, bias, *rest = super().gen_inputs()
        g = self.rng(device=gating.device)
        gating = torch.randn(gating.shape, generator=g, device=gating.device)
        if bias is not None:
            bias = torch.randn(bias.shape, generator=g, device=bias.device) * 0.1
        hidden.mul_(0.1)
        w_gate_up.mul_(0.02)
        w_down.mul_(0.02)
        if "b_gate_up" in self.call.signature.inputs:
            _scale_routed(self.call, hidden, w_gate_up, w_down, *rest)
        return (hidden, gating, w_gate_up, w_down, bias, *rest)

    def ref_routed(
        self, hidden, gating, w_gate_up, w_down, bias, b_gate_up=None, b_down=None
    ) -> torch.Tensor:
        p = self.call.params
        # The routing returns unscaled weights: the experts apply the factor, once.
        weights, ids = ref_fused_topk(
            gating,
            bias,
            p["top_k"],
            p["scoring_func"],
            p["renormalize"],
            p["num_expert_group"],
            p["topk_group"],
        )
        return ref_routed_experts(
            hidden,
            w_gate_up,
            w_down,
            weights,
            ids,
            p["activation"],
            p["routed_scaling_factor"],
            b_gate_up=b_gate_up,
            b_down=b_down,
        )

    def ref_program(
        self,
        hidden_states,
        gating_output,
        w_gate_up,
        w_down,
        correction_bias=None,
        b_gate_up=None,
        b_down=None,
    ):
        return self.ref_routed(
            hidden_states, gating_output, w_gate_up, w_down, correction_bias, b_gate_up, b_down
        )

    def _router(self, gating, bias):
        p = self.call.params
        return _Router(
            gating,
            bias,
            p["top_k"],
            p["scoring_func"],
            p["renormalize"],
            p["num_expert_group"],
            p["topk_group"],
        )

    def verification(self, *inputs):
        hidden, gating, w_gate_up, w_down, bias, *rest = inputs
        b_gate_up, b_down = (rest + [None, None])[:2]
        p = self.call.params
        return routed_verification(
            self._router(gating, bias),
            (hidden, w_gate_up, w_down, b_gate_up, b_down),
            p["activation"],
            p["routed_scaling_factor"],
            gate_up_index=2,
        )


class FusedMoESharedExpertWorkload(FusedMoEWorkload):
    """One ``FusedMoESharedExpertFwdOp`` call: FusedMoE's inputs plus the shared weights."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        *routed, shared_w_gate_up, shared_w_down = super().gen_inputs()
        if shared_w_gate_up is not None:
            shared_w_gate_up.mul_(0.02)
            shared_w_down.mul_(0.02)
        return (*routed, shared_w_gate_up, shared_w_down)

    def ref_program(
        self,
        hidden_states,
        gating_output,
        w_gate_up,
        w_down,
        correction_bias=None,
        shared_w_gate_up=None,
        shared_w_down=None,
    ):
        """``(shared_output, routed_output)``; the shared half is this rank's partial sum."""
        routed = self.ref_routed(hidden_states, gating_output, w_gate_up, w_down, correction_bias)
        if shared_w_gate_up is None:
            return None, routed
        p = self.call.params
        ffn = shared_w_down.shape[1]
        shard = ffn // p["tp_size"]
        lo, hi = p["tp_rank"] * shard, (p["tp_rank"] + 1) * shard
        gate_up = torch.cat([shared_w_gate_up[lo:hi], shared_w_gate_up[ffn + lo : ffn + hi]])
        return ref_shared_expert(hidden_states, gate_up, shared_w_down[:, lo:hi]), routed

    def verification(self, *inputs):
        hidden, gating, w_gate_up, w_down, bias, *_ = inputs
        p = self.call.params
        return routed_verification(
            self._router(gating, bias),
            (hidden, w_gate_up, w_down, None, None),
            p["activation"],
            p["routed_scaling_factor"],
            gate_up_index=2,
            shared=True,
        )


class SharedExpertMLPWorkload(CallWorkload):
    """Tokens and the stacked gate/up and down weights of one ``SharedExpertMLPFwdOp`` call."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        hidden, w_gate_up, w_down = super().gen_inputs()
        # Small scales keep fp16 accumulation over H = 7168 finite.
        return hidden.mul_(0.1), w_gate_up.mul_(0.02), w_down.mul_(0.02)

    def ref_program(
        self, hidden_states: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor
    ) -> torch.Tensor:
        """``down(silu(gate) * up)`` in fp32, cast to the hidden dtype."""
        return ref_shared_expert(hidden_states, w_gate_up, w_down)

    def verification(self, *inputs):
        return moe_verification(1)


def ref_shared_expert(hidden, gate_up, down):
    """Shared expert MLP in FP32, narrowed once to the token dtype."""
    act = gated_activation(hidden.float() @ gate_up.float().T, "silu_and_mul")
    return (act @ down.float().T).to(hidden.dtype)


def ref_permute_align(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-Python reference for permute_align."""
    numel = topk_ids.numel()
    flat = topk_ids.flatten().tolist()

    counts = [0] * num_experts
    for eid in flat:
        counts[eid] += 1

    cumsum = [0] * (num_experts + 1)
    for i in range(num_experts):
        padded = math.ceil(counts[i] / block_size) * block_size
        cumsum[i + 1] = cumsum[i] + padded

    total_padded = cumsum[num_experts]
    sorted_token_ids = [numel] * total_padded

    slot = list(cumsum[:-1])
    for flat_idx, eid in enumerate(flat):
        sorted_token_ids[slot[eid]] = flat_idx
        slot[eid] += 1

    num_blocks = total_padded // block_size
    expert_ids_list = []
    for b in range(num_blocks):
        block_start = b * block_size
        lo, hi = 0, num_experts - 1
        eid = num_experts - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if cumsum[mid] <= block_start < cumsum[mid + 1]:
                eid = mid
                break
            elif block_start < cumsum[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        expert_ids_list.append(eid)

    device = topk_ids.device
    return (
        torch.tensor(sorted_token_ids, dtype=torch.int32, device=device),
        torch.tensor(expert_ids_list, dtype=torch.int32, device=device),
        torch.tensor([total_padded], dtype=torch.int32, device=device),
    )


def moe_verification(gate_up_index):
    """Check both individual errors and scale-independent energy, including gate order."""
    from workloads.numerics import Custom

    tol, bound = 1e-2, 1e-4

    def validate(got, expected):
        if isinstance(expected, (tuple, list)):
            for value, target in zip(got, expected, strict=True):
                validate(value, target)
        elif expected is None:
            assert got is None
        else:
            _moe_close(got, expected, tol, bound)

    return Custom(
        validate,
        f"elementwise atol/rtol {tol} and normalized squared error <= {bound}",
        controls=(_gate_up_swapped(gate_up_index),),
    )
