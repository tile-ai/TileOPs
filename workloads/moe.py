import itertools
import math
from typing import NamedTuple

import torch

from workloads.workload_base import CallWorkload

# An expert whose selection score is within this (relative) of the last one kept may be kept in its
# place; one scored further above it is always kept. A group score ties the last kept one within
# this plus an absolute term (_group_tie_slack). Shared because the top-k and fused MoE
# verifications admit the same ties, those FusedTopKFwdOp's and FusedMoEFwdOp's docstrings state.
_TIE_RTOL = 1e-5


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
        ties = _tied_routings(
            gating_output,
            correction_bias,
            p["top_k"],
            p["scoring_func"],
            p["renormalize"],
            p["num_expert_group"],
            p["topk_group"],
        )

        def validate(got, expected):
            # compare_outputs has already matched every output's shape, dtype and device.
            (weights, ids), (_, ref_ids) = got, expected
            returned = ids.long().sort(-1).values
            assert ((returned >= 0) & (returned < gating_output.shape[-1])).all(), (
                "an expert id outside [0, E)"
            )
            assert (returned[:, 1:] != returned[:, :-1]).all(), "a token returns an expert twice"
            tied = torch.tensor(list(ties), dtype=torch.long, device=ids.device)
            untied = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
            untied[tied] = False
            reference = ref_ids.long().sort(-1).values
            assert torch.equal(returned[untied], reference[untied]), (
                "a token with no tie within the documented expert/group band of the last one kept returns other experts than "
                "the float32 reference"
            )
            for row, experts in zip(ties, returned[tied].tolist(), strict=True):
                admitted = [sorted(option.tolist()) for _, option in ties[row]]
                assert experts in admitted, (
                    f"token {row}: experts {experts} exchange an expert or group of the float32 "
                    "reference for one where the two are not both within the documented expert/group band of the last one kept"
                )
            logits = gating_output.float()
            scores = logits.softmax(-1) if p["scoring_func"] == "softmax" else logits.sigmoid()
            selected = scores.gather(1, ids.long())
            if p["renormalize"]:
                selected = selected / selected.sum(-1, keepdim=True)
            torch.testing.assert_close(
                weights,
                selected * p["routed_scaling_factor"],
                rtol=1e-4,
                atol=0,
                msg=lambda detail: (
                    "weights not within 1e-4 (relative) of the unbiased float32 scores of the "
                    f"experts returned, renormalized when asked, times routed_scaling_factor\n{detail}"
                ),
            )

        return Custom(
            validate, "the experts of a routing a tie admits, weights independent of their order"
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
    """``out[rows of g] = a[rows of g] @ b[g]^T + bias[g]`` in fp32; other rows are zero.

    With ``activation``, ``b`` and ``bias`` hold gate and up along ``N`` in the activation's
    layout (:func:`gated_activation`) and the result is the gated activation, half as wide, as
    the fused epilogue writes it. The bias is added before the activation.
    """
    out_dtype = a.dtype if out_dtype is None else out_dtype
    exact = _grouped_gemm(_Bounded(a.float(), None), b, layout_metadata, layout, activation, bias)
    return exact.value.to(out_dtype)


def _group_rows(layout, layout_metadata: torch.Tensor, num_experts: int):
    """``(g, index)`` for each expert: *index* selects its rows of a layout's materialized rows."""
    meta = layout_metadata.tolist()
    if layout.kind == "masked":
        for g, count in enumerate(meta):
            yield g, (g, slice(0, count))
    elif layout.metadata_kind == "per_row":
        ids = layout_metadata.long()
        for g in range(num_experts):
            yield g, ids == g
    else:
        start = 0
        for g, end in enumerate(meta):
            start = -(-start // layout.alignment) * layout.alignment
            yield g, slice(start, end)
            start = end


def _grouped_gemm(
    a: "_Bounded",
    b: torch.Tensor,
    layout_metadata: torch.Tensor,
    layout,
    activation: str | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> "_Bounded":
    """:func:`ref_moe_grouped_gemm` in fp32, with the rounding budget of each element.

    The roundings an implementation may apply: fp32 sums in any order, the activation reading
    the biased gate/up product held in ``b``'s dtype, the activation's own factor held in it,
    and the output's rounding to *out_dtype* (``b``'s dtype by default).
    """
    num_experts, n, _ = b.shape
    width = n // 2 if activation is not None else n
    value = torch.zeros(*a.value.shape[:-1], width, dtype=torch.float32, device=a.value.device)
    budget = None if a.budget is None else torch.zeros_like(value)
    for g, rows in _group_rows(layout, layout_metadata, num_experts):
        x = _Bounded(a.value[rows], None if a.budget is None else a.budget[rows])
        result = _contract(x, b[g], None if bias is None else bias[g])
        if activation is not None:
            result = _activate(_held(result, b.dtype), activation, b.dtype)
        value[rows] = result.value
        if budget is not None:
            budget[rows] = result.budget
    return _held(_Bounded(value, budget), b.dtype if out_dtype is None else out_dtype)


class MoEGroupedGemmWorkload(CallWorkload):
    """Expert-materialized ``a``, per-expert ``b`` and the layout's generated metadata."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        a, b, metadata, bias = super().gen_inputs()
        # A small scale keeps fp16 accumulation over K = 7168 finite. A gate/up result is drawn
        # at :func:`_gate_up_spread`, and a bias takes half the RMS of the product.
        a.mul_(0.02)
        b.mul_(0.02)
        _condition_expert_gemm(a, b, bias, _gate_up_spread(self.call.params["activation"]))
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

        a, b, metadata, *rest = inputs
        bias = rest[0] if rest else None
        p = self.call.params
        out_dtype = None if p["out_dtype"] is None else getattr(torch, p["out_dtype"])
        mask = valid_rows(p["layout"], metadata, a.numel() // a.shape[-1], b.shape[0])

        def validate(got, expected):
            # MoEGroupedGemmFwdOp's docstring, "Accuracy".
            exact = _grouped_gemm(
                _Bounded(a.float(), torch.zeros(a.shape, device=a.device)),
                b,
                metadata,
                p["layout"],
                p["activation"],
                bias,
                out_dtype,
            )
            _assert_within_budget(got, exact, mask, "defined rows")

        return Custom(validate, "defined rows within rounding budgets of the exact grouped GEMM")


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
    """Expert-materialized input, gate/up and down weights, optional biases, generated metadata."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x, w_gate_up, w_down, metadata, b_gate_up, b_down = super().gen_inputs()
        # Small scales keep fp16 accumulation finite. The gate/up result is drawn at
        # :func:`_gate_up_spread`, and each bias takes half the RMS of its product.
        x.mul_(0.02)
        w_gate_up.mul_(0.02)
        w_down.mul_(0.02)
        activation = self.call.params["activation"]
        spread = _gate_up_spread(activation)
        _condition_expert_mlp(x, w_gate_up, w_down, b_gate_up, b_down, activation, spread)
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
        exact = _expert_mlp(
            expert_input,
            w_gate_up,
            w_down,
            layout_metadata,
            p["layout"],
            p["activation"],
            b_gate_up,
            b_down,
            track=False,
            hold_activated=True,
        )
        return exact.value.to(expert_input.dtype)

    def verification(self, *inputs):
        from workloads.numerics import Custom

        x, w_gate_up, w_down, metadata, *biases = inputs
        b_gate_up, b_down = (*biases, None, None)[:2]
        p = self.call.params
        mask = valid_rows(p["layout"], metadata, x.numel() // x.shape[-1], w_down.shape[0])

        def validate(got, expected):
            # MoEExpertMLPFwdOp's docstring, "Accuracy".
            choices = (
                _expert_mlp(
                    x,
                    w_gate_up,
                    w_down,
                    metadata,
                    p["layout"],
                    p["activation"],
                    b_gate_up,
                    b_down,
                    track=True,
                    hold_activated=held,
                )
                for held in (True, False)
            )
            _assert_within_one_budget(got, choices, mask, "defined rows")

        return Custom(validate, "defined rows within rounding budgets of the exact expert MLP")


def _expert_mlp(
    x,
    w_gate_up,
    w_down,
    layout_metadata,
    layout,
    activation,
    b_gate_up,
    b_down,
    *,
    track,
    hold_activated,
) -> "_Bounded":
    """The selected activated-row precision graph and its optional output-error budgets."""
    start = _Bounded(x.float(), torch.zeros(x.shape, device=x.device) if track else None)
    activated = _grouped_gemm(start, w_gate_up, layout_metadata, layout, activation, b_gate_up)
    if hold_activated:
        activated = _Bounded(activated.value.to(x.dtype).float(), activated.budget)
    return _grouped_gemm(activated, w_down, layout_metadata, layout, None, b_down)


class MoEPostPermuteWorkload(CallWorkload):
    """Expert outputs, routing weights and inverse indices of one ``MoEPostPermuteFwdOp`` call."""

    def _scale_and_dtype(self, expert_output: torch.Tensor) -> tuple[float, torch.dtype]:
        p = self.call.params
        scale = 1.0 if p["epilogue"] is None else p["epilogue"].routed_scaling_factor
        dtype = expert_output.dtype if p["out_dtype"] is None else getattr(torch, p["out_dtype"])
        return scale, dtype

    def ref_program(
        self,
        expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        """``out[t] = scale * sum_k w[t, k] * rows[inverse[t * K + k]]`` in fp32, then cast."""
        scale, dtype = self._scale_and_dtype(expert_output)
        exact = _weighted_rows(expert_output, topk_weights, inverse_indices, scale, track=False)
        return exact.value.to(dtype)

    def verification(self, *inputs):
        scale, dtype = self._scale_and_dtype(inputs[0])
        return post_permute_verification(*inputs, routed_scaling_factor=scale, out_dtype=dtype)


def _weighted_rows(
    expert_output: torch.Tensor,
    topk_weights: torch.Tensor,
    inverse_indices: torch.Tensor,
    scale: float,
    out_dtype: torch.dtype | None = None,
    *,
    track: bool,
) -> "_Bounded":
    """The float32 weighted-sum reference and optional error-decision budgets.

    Weighted-term magnitudes contribute input-dtype error budgets. A nonunit scale
    contributes a scaled-accumulator store budget, followed by the output store.
    """
    tokens, top_k = topk_weights.shape
    rows = expert_output.reshape(-1, expert_output.shape[-1])
    terms = rows[inverse_indices.long()].float().view(tokens, top_k, -1)
    terms = terms * topk_weights.float().unsqueeze(-1)
    unit = _UNIT_ROUNDOFF[expert_output.dtype]
    budget = unit * terms.abs().sum(dim=1) * abs(scale) if track else None
    total = _Bounded(terms.sum(dim=1) * scale, budget)
    if scale != 1.0:
        total = _held(total, expert_output.dtype)
    return _held(total, expert_output.dtype if out_dtype is None else out_dtype)


def post_permute_verification(
    expert_output: torch.Tensor,
    topk_weights: torch.Tensor,
    inverse_indices: torch.Tensor,
    routed_scaling_factor: float = 1.0,
    out_dtype: torch.dtype | None = None,
):
    """The accepted outputs of a ``MoEPostPermuteFwdOp`` call: each element within the rounding
    budgets of the exact weighted sum of its token's routed rows (:func:`_weighted_rows`)."""
    from workloads.numerics import Custom

    def validate(got, expected):
        # MoEPostPermuteFwdOp's docstring, "Accuracy".
        exact = _weighted_rows(
            expert_output,
            topk_weights,
            inverse_indices,
            routed_scaling_factor,
            out_dtype,
            track=True,
        )
        _assert_within_budget(got, exact, None, "output")

    return Custom(validate, "each element within rounding budgets of the exact weighted sum")


# gpt-oss fixes swigluoai's sigmoid slope (alpha) and clamp bound (limit); the activation and
# its slopes in _activation_slopes both read them.
_SWIGLUOAI_ALPHA = 1.702
_SWIGLUOAI_LIMIT = 7.0


def _gate_and_up(t: torch.Tensor, activation: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The gate and up columns of *t*'s last axis: interleaved for ``swigluoai``, else stacked."""
    if activation == "swigluoai":
        return t[..., ::2], t[..., 1::2]
    return t.chunk(2, dim=-1)


def gated_activation(gate_up: torch.Tensor, activation: str) -> torch.Tensor:
    """The gated activation over the last axis, half as wide, in the input's dtype:
    ``MoEGroupedGemmFwdOp``'s docstring defines each one and the order of its gate and up
    columns. ``swigluoai`` is computed in fp32."""
    if activation == "swigluoai":
        gate, up = _gate_and_up(gate_up.float(), activation)
        gate = gate.clamp(max=_SWIGLUOAI_LIMIT)
        up = up.clamp(min=-_SWIGLUOAI_LIMIT, max=_SWIGLUOAI_LIMIT)
        glu = gate * torch.sigmoid(gate * _SWIGLUOAI_ALPHA)
        return ((up + 1) * glu).to(gate_up.dtype)
    gate, up = _gate_and_up(gate_up, activation)
    if activation == "silu_and_mul":
        return torch.nn.functional.silu(gate) * up
    if activation == "gelu_and_mul":
        return torch.nn.functional.gelu(gate) * up
    raise ValueError(f"unknown gated activation: {activation}")


def _activation_slopes(
    gate: torch.Tensor,
    up: torch.Tensor,
    activation: str,
    gate_reach: torch.Tensor,
    up_reach: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The gated activation's partial derivatives in its gate input, its up input and its
    sigmoid or Gaussian-CDF factor, in fp32.

    swigluoai is flat past its clamps, but an input within *reach* of a limit can be carried
    across it by the roundings a budget allows, so there the slope on the near side is taken.
    """
    if activation == "swigluoai":
        alpha, limit = _SWIGLUOAI_ALPHA, _SWIGLUOAI_LIMIT
        g, u = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
        s = torch.sigmoid(alpha * g)
        d_gate = torch.where(
            gate < limit + gate_reach, s * (1 + alpha * g * (1 - s)) * (u + 1), 0.0
        )
        d_up = torch.where(up.abs() < limit + up_reach, g * s, 0.0)
        return d_gate, d_up, g * (u + 1)
    if activation == "silu_and_mul":
        s = torch.sigmoid(gate)
        return s * (1 + gate * (1 - s)) * up, gate * s, gate * up
    if activation == "gelu_and_mul":
        cdf = 0.5 * (1 + torch.erf(gate * math.sqrt(0.5)))
        pdf = torch.exp(-0.5 * gate * gate) / math.sqrt(2 * math.pi)
        return (cdf + gate * pdf) * up, gate * cdf, gate * up
    raise ValueError(f"unknown gated activation: {activation}")


# Half the machine epsilon of each supported dtype; the shared MoE budget model uses
# these units for contractions, dtype stores and activation-factor decisions.
_UNIT_ROUNDOFF = {
    dtype: torch.finfo(dtype).eps / 2 for dtype in (torch.bfloat16, torch.float16, torch.float32)
}

# The shared final-error decision bound in the MoE op accuracy contracts; activation
# reach, acceptance and failure diagnostics use the same multiplier.
_BUDGETS = 2


class _Bounded(NamedTuple):
    """A float32 reference value and its contract-defined error-decision budget.

    ``budget`` is ``None`` when the caller requests values without numerical validation.
    """

    value: torch.Tensor
    budget: torch.Tensor | None


def _contract(x: _Bounded, weight: torch.Tensor, bias: torch.Tensor | None) -> _Bounded:
    """``x @ weight^T + bias`` in fp32.

    Input-error decision budgets propagate in quadrature under the shared MoE accuracy
    contract. The float32 contraction contributes its reduction-length budget.
    """
    w = weight.float()
    value = x.value @ w.T
    if bias is not None:
        value = value + bias.float()
    if x.budget is None:
        return _Bounded(value, None)
    inputs = (x.budget.square() @ w.square().T).sqrt()
    terms = x.value.abs() @ w.abs().T
    if bias is not None:
        terms = terms + bias.float().abs()
    accumulation = (w.shape[-1] + 1) * _UNIT_ROUNDOFF[torch.float32] * terms
    return _Bounded(value, inputs + accumulation)


def _held(x: _Bounded, dtype: torch.dtype) -> _Bounded:
    """*x* where an implementation may store it in *dtype*: one rounding more, of its magnitude,
    or of the smallest normal number below that, where the dtype's spacing stops shrinking."""
    if x.budget is None:
        return x
    floor = torch.finfo(dtype).smallest_normal
    return _Bounded(x.value, x.budget + _UNIT_ROUNDOFF[dtype] * (x.value.abs() + floor))


def _activate(x: _Bounded, activation: str, dtype: torch.dtype) -> _Bounded:
    """The gated activation of *x*: its inputs' budgets carried through the activation's slopes,
    its sigmoid or Gaussian CDF evaluated to ``2**-11``, and its gate factor allowed to be held
    in *dtype* before the multiply by up."""
    # The sigmoid's and the Gaussian CDF's absolute error: half a float16 ulp at 1, the bound
    # the in-kernel polynomial erf is built to (tileops/kernels/elementwise/_erf.py). Near 0
    # neither keeps a relative error.
    factor_error = _UNIT_ROUNDOFF[torch.float16]
    value = gated_activation(x.value, activation)
    if x.budget is None:
        return _Bounded(value, None)
    gate, up = _gate_and_up(x.value, activation)
    gate_budget, up_budget = _gate_and_up(x.budget, activation)
    d_gate, d_up, d_factor = _activation_slopes(
        gate, up, activation, _BUDGETS * gate_budget, _BUDGETS * up_budget
    )
    budget = d_gate.abs() * gate_budget + d_up.abs() * up_budget
    budget = budget + factor_error * d_factor.abs()
    return _Bounded(value, budget + _UNIT_ROUNDOFF[dtype] * value.abs())


def _outside_budget(got: torch.Tensor, exact: _Bounded) -> torch.Tensor:
    """Where *got* lies more than :data:`_BUDGETS` budgets from *exact*; a non-finite value does."""
    return ~((got.float() - exact.value).abs() <= _BUDGETS * exact.budget)


def _budget_failure(got: torch.Tensor, exact: _Bounded, outside: torch.Tensor, what: str) -> str:
    """The message naming how many elements lie outside their budgets, and the worst of them."""
    ratio = (got.float() - exact.value).abs() / (_BUDGETS * exact.budget)
    ratio = torch.where(outside, ratio.nan_to_num(nan=math.inf), 0.0)
    worst = tuple(int(i) for i in torch.unravel_index(ratio.argmax(), ratio.shape))
    return (
        f"{what}: {int(outside.sum())} of {outside.numel()} elements lie more than {_BUDGETS} "
        f"rounding budgets from the exact result; the worst, at {worst}, is "
        f"{float(got[worst])} for {float(exact.value[worst])} with a budget of "
        f"{float(exact.budget[worst])}"
    )


def _assert_within_budget(
    got: torch.Tensor, exact: _Bounded, rows: torch.Tensor | None, what: str
) -> None:
    """Every element of *got*, or of its *rows* (a mask over the flattened leading axes), within
    :data:`_BUDGETS` rounding budgets of *exact*."""
    if rows is not None:
        width = got.shape[-1]
        got = got.reshape(-1, width)[rows]
        exact = _Bounded(
            exact.value.reshape(-1, width)[rows], exact.budget.reshape(-1, width)[rows]
        )
    outside = _outside_budget(got, exact)
    if outside.any():
        raise AssertionError(_budget_failure(got, exact, outside, what))


def _assert_within_one_budget(got, choices, rows, what) -> None:
    """Each defined output row must fit one complete reference-precision choice."""
    if rows is not None:
        width = got.shape[-1]
        got = got.reshape(-1, width)[rows]
    remaining, first, first_outside = None, None, None
    for exact in choices:
        if rows is not None:
            exact = _Bounded(
                exact.value.reshape(-1, width)[rows], exact.budget.reshape(-1, width)[rows]
            )
        outside = _outside_budget(got, exact)
        if remaining is None:
            first, first_outside = exact, outside
            remaining = outside.any(dim=-1)
        else:
            remaining &= outside.any(dim=-1)
        if not remaining.any():
            return
    if remaining is None:
        raise ValueError("an accuracy contract must admit at least one precision choice")
    raise AssertionError(
        f"{what}: no complete row-wise activation precision satisfies the accuracy contract\n"
        + _budget_failure(got, first, first_outside & remaining.unsqueeze(-1), what)
    )


def _gate_up_spread(activation: str | None) -> float | None:
    """The RMS a test input's gate/up result is drawn at, for the three workloads that draw one.

    swigluoai samples use RMS 4 against the activation contract's clamp limit 7.
    silu_and_mul and gelu_and_mul samples use RMS 1. A plain GEMM keeps its scale.
    """
    if activation is None:
        return None
    return 4.0 if activation == "swigluoai" else 1.0


def _sample_rms(t: torch.Tensor) -> float:
    return t.float().square().mean().sqrt().item() if t.numel() else 0.0


def _condition_expert_gemm(
    rows: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    spread: float | None,
) -> torch.Tensor:
    """Scale an expert GEMM's weight and standard-normal bias in place; return the result of
    expert 0 on the first rows of *rows*, bias included.

    With *spread*, the weight is scaled so the sampled product has that RMS. The bias takes half the RMS
    of the product it is added to, with the same scale factor shared by all experts.
    """
    sample = rows.reshape(-1, rows.shape[-1])[: rows.shape[-1]].float()
    product = sample @ weight[0].float().T
    magnitude = _sample_rms(product)
    if spread is not None and magnitude > 0:
        weight.mul_(spread / magnitude)
        product = sample @ weight[0].float().T
    if bias is None:
        return product
    bias.mul_(0.5 * _sample_rms(product))
    return product + bias[0].float()


def _condition_expert_mlp(
    rows: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    b_gate_up: torch.Tensor | None,
    b_down: torch.Tensor | None,
    activation: str,
    gate_up_spread: float | None,
) -> None:
    """:func:`_condition_expert_gemm` on both GEMMs of an expert MLP, scaled from expert 0; the
    down GEMM keeps its weight's scale."""
    gate_up = _condition_expert_gemm(rows, w_gate_up, b_gate_up, gate_up_spread)
    activated = gated_activation(gate_up, activation)
    _condition_expert_gemm(activated, w_down, b_down, None)


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
    over each token's routed experts, in fp32 per expert, cast to the hidden dtype.

    Each bias is added to its GEMM's fp32 result; the down bias before the routing weight
    scales it, as gpt-oss and vLLM's fused experts do.
    """
    exact = _routed_experts(
        hidden,
        w_gate_up,
        w_down,
        topk_weights,
        topk_ids,
        activation,
        scale,
        b_gate_up,
        b_down,
        track=False,
        hold_activated=False,
    )
    return exact.value.to(hidden.dtype)


def _routed_experts(
    hidden,
    w_gate_up,
    w_down,
    topk_weights,
    topk_ids,
    activation,
    scale,
    b_gate_up,
    b_down,
    *,
    track: bool,
    hold_activated: bool,
) -> _Bounded:
    """The chosen activated-row precision reference and its optional error budgets.

    Biases join float32 products before stores; route weights and the final scale
    apply once. The shared MoE accuracy model defines the budget propagation.
    """
    dtype = hidden.dtype
    unit = _UNIT_ROUNDOFF[dtype]
    value = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
    budget = torch.zeros_like(value) if track else None
    ids = topk_ids.to(torch.int64)
    for e in range(w_gate_up.shape[0]):
        t_idx, k_idx = (ids == e).nonzero(as_tuple=True)
        if t_idx.numel() == 0:
            continue
        x = hidden[t_idx].float()
        rows = _Bounded(x, torch.zeros_like(x) if track else None)
        gate_up = _contract(rows, w_gate_up[e], None if b_gate_up is None else b_gate_up[e])
        activated = _held(_activate(_held(gate_up, dtype), activation, dtype), dtype)
        if hold_activated:
            activated = _Bounded(activated.value.to(dtype).float(), activated.budget)
        down = _held(_contract(activated, w_down[e], None if b_down is None else b_down[e]), dtype)
        weight = topk_weights[t_idx, k_idx].float().unsqueeze(-1)
        value.index_add_(0, t_idx, down.value * weight)
        if track:
            # The weighted product, or a partial sum it joins, held: one more rounding of it.
            budget.index_add_(0, t_idx, (down.budget + unit * down.value.abs()) * weight.abs())
    total = _Bounded(value * scale, None if budget is None else budget * abs(scale))
    if scale != 1.0:
        total = _held(total, dtype)
    return _held(total, dtype)


def _tied_routings(
    gating, correction_bias, top_k, scoring_func, renormalize, num_expert_group=1, topk_group=1
) -> dict:
    """The other routings a top-k tie admits: token -> ``[(weights, ids), ...]``.

    A selection is tied where experts inside the k-th selection score's band, ``_expert_tie_slack`` wide, outnumber the places left above the band; any of them may fill those places. Weights
    are taken from the unbiased scores of the experts chosen, renormalized when asked.

    With ``num_expert_group > 1`` the kept groups are decided first, the same way: groups inside
    the last kept group score's band (:func:`_group_tie_slack`) that outnumber the group places
    left above it may fill those places, and each such choice of groups is a kept set the expert
    selection, and its own tie, is taken inside.

    Raises:
        ValueError: A token's tie admits more than 64 routings.
    """
    logits = gating.float()
    scores = torch.softmax(logits, dim=-1) if scoring_func == "softmax" else logits.sigmoid()
    select = scores if correction_bias is None else scores + correction_bias
    group_size = select.shape[-1] // num_expert_group
    group_ties = torch.zeros(select.shape[0], dtype=torch.bool, device=select.device)
    if num_expert_group > 1:
        group_scores = ref_group_scores(select, num_expert_group, correction_bias is not None)
        kept = group_scores.sort(dim=-1, descending=True, stable=True).indices[:, :topk_group]
        canonical = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, kept, True)
        last = group_scores.topk(topk_group, dim=-1).values[:, -1:]
        slack = _group_tie_slack(last)
        # One difference decides both sets, so every group is near, above or below.
        gap = group_scores - last
        near_group = gap.abs() <= slack
        above_group = gap > slack
        group_places = topk_group - above_group.sum(-1)
        group_ties = near_group.sum(-1) > group_places
        canonical_select = select.masked_fill(
            ~canonical.repeat_interleave(group_size, dim=-1), float("-inf")
        )
    else:
        canonical_select = select
    kth = canonical_select.topk(top_k, dim=-1).values[:, -1:]
    band = _expert_tie_slack(kth)
    gap = canonical_select - kth
    near = gap.abs() <= band
    places = top_k - (gap > band).sum(-1)
    # A verification limit, not part of the op's contract: an exact tie of n experts for k
    # places admits C(n, k) routings, and each is listed.
    cap = 64
    ties = {}
    for t in ((near.sum(-1) > places) | group_ties).nonzero().flatten().tolist():
        rows = [canonical_select[t]]
        if group_ties[t]:
            fixed = above_group[t].nonzero().flatten().tolist()
            near_groups = near_group[t].nonzero().flatten().tolist()
            _check_tie_count(t, math.comb(len(near_groups), int(group_places[t])), cap)
            rows = []
            for chosen in itertools.combinations(near_groups, int(group_places[t])):
                groups = torch.zeros(num_expert_group, dtype=torch.bool, device=select.device)
                groups[fixed + list(chosen)] = True
                expert_mask = groups.repeat_interleave(group_size)
                rows.append(select[t].masked_fill(~expert_mask, float("-inf")))
        selections = [_tied_selections(row, top_k) for row in rows]
        _check_tie_count(t, sum(math.comb(len(n), top_k - len(f)) for f, n in selections), cap)
        ties[t] = []
        for fixed, near_experts in selections:
            for chosen in itertools.combinations(near_experts, top_k - len(fixed)):
                ids = torch.tensor(fixed + list(chosen), device=select.device)
                weights = scores[t][ids]
                if renormalize:
                    weights = weights / weights.sum()
                ties[t].append((weights, ids))
    return ties


def _expert_tie_slack(last):
    """Expert selection band: relative slack plus one float32 epsilon for score/bias cancellation."""
    return _TIE_RTOL * last.abs() + torch.finfo(torch.float32).eps


def _tied_selections(select, top_k) -> tuple[list, list]:
    """``(above, near)`` of one token's selection row *select*, where ``-inf`` drops an expert:
    the experts above the k-th selection score's band, which every routing keeps, and those
    inside it, which fill the remaining places."""
    kth = select.topk(top_k).values[-1]
    band = _expert_tie_slack(kth)
    gap = select - kth
    near = gap.abs() <= band
    above = gap > band
    return above.nonzero().flatten().tolist(), near.nonzero().flatten().tolist()


def _check_tie_count(token: int, count: int, cap: int) -> None:
    """Refuse to enumerate *token*'s tie when it admits at least *count* routings, over *cap*."""
    if count > cap:
        raise ValueError(
            f"token {token}: its tie admits at least {count} routings, over the {cap} the "
            "verification enumerates"
        )


def routed_experts_verification(
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
    ties: dict | None = None,
    controls: tuple = (),
):
    """The accepted outputs of a routed-experts call (:func:`ref_routed_experts`'s arguments).

    Each element lies within :data:`_BUDGETS` rounding budgets of the exact result under the
    given routing (:func:`_routed_experts`), or, for a token *ties* lists, under one of the
    routings its tie admits.
    """
    from workloads.numerics import Custom

    def exact(weights, ids, rows, hold_activated):
        return _routed_experts(
            hidden[rows],
            w_gate_up,
            w_down,
            weights,
            ids,
            activation,
            scale,
            b_gate_up,
            b_down,
            track=True,
            hold_activated=hold_activated,
        )

    def validate(got, expected):
        # The op docstring's "Accuracy": every element within the budgets of one admissible
        # routing's exact result, chosen per token.
        canonical = exact(topk_weights, topk_ids, slice(None), False)
        outside = _outside_budget(got, canonical)
        for t in outside.any(dim=-1).nonzero().flatten().tolist():
            row = slice(t, t + 1)
            options = [(topk_weights[t], topk_ids[t]), *(ties or {}).get(t, ())]
            if not any(
                not _outside_budget(got[row], exact(w[None], i[None], row, held)).any()
                for w, i in options
                for held in (False, True)
            ):
                raise AssertionError(
                    f"token {t}: no admitted routing and single activation precision fits "
                    "the entire output vector\n"
                    + _budget_failure(
                        got[row],
                        _Bounded(canonical.value[row], canonical.budget[row]),
                        outside[row],
                        "output",
                    )
                )

    return Custom(
        validate, "each element within rounding budgets of the exact routed experts", controls
    )


def _gate_up_swap(names, activation: str) -> tuple:
    """The negative control that exchanges gate and up, in the activation's layout, in the
    ``w_gate_up`` input and in ``b_gate_up`` where the call passes it; *names* are the inputs."""
    from workloads.numerics import NegativeControl

    weight = names.index("w_gate_up")
    bias = names.index("b_gate_up") if "b_gate_up" in names else None

    def swapped(reference, inputs):
        changed = list(inputs)
        changed[weight] = _swap_gate_up(changed[weight], activation, -2)
        if bias is not None and changed[bias] is not None:
            changed[bias] = _swap_gate_up(changed[bias], activation, -1)
        return reference(*changed)

    return (NegativeControl("gate-up-swapped", swapped),)


def _condition_routed_experts(call, hidden, w_gate_up, w_down, b_gate_up=None, b_down=None):
    """Condition a routed call (:func:`_condition_expert_mlp`): the gate/up result is drawn at
    :func:`_gate_up_spread`, and each bias takes half the RMS of its product."""
    activation = call.params.get("activation", "silu_and_mul")
    spread = _gate_up_spread(activation)
    _condition_expert_mlp(hidden, w_gate_up, w_down, b_gate_up, b_down, activation, spread)


class MoEExpertsWorkload(CallWorkload):
    """Tokens, expert weights and their routing for one ``FusedMoEExpertsFwdOp`` call."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        output, hidden, w_gate_up, w_down, topk_weights, topk_ids, *biases = super().gen_inputs()
        # Small scales keep fp16 accumulation over H = 7168 finite; routing weights sum to one.
        hidden.mul_(0.1)
        w_gate_up.mul_(0.02)
        w_down.mul_(0.02)
        _condition_routed_experts(self.call, hidden, w_gate_up, w_down, *biases)
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
        _, hidden, w_gate_up, w_down, topk_weights, topk_ids, *biases = inputs
        b_gate_up, b_down = (*biases, None, None)[:2]
        p = self.call.params
        activation = p.get("activation", "silu_and_mul")
        return routed_experts_verification(
            hidden,
            w_gate_up,
            w_down,
            topk_weights,
            topk_ids,
            activation,
            p["routed_scaling_factor"],
            b_gate_up=b_gate_up,
            b_down=b_down,
            controls=_gate_up_swap(list(self.call.signature.inputs), activation),
        )


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


def _group_tie_slack(last: torch.Tensor) -> torch.Tensor:
    """How far a group score may lie from the last kept one, *last*, and tie it: ``_TIE_RTOL``
    of it, plus ``1e-6`` absolutely, the group bound the op docstrings state."""
    return _TIE_RTOL * last.abs() + 1e-6


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
            _condition_routed_experts(self.call, hidden, w_gate_up, w_down, *rest)
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

    def verification(self, *inputs):
        hidden, gating, w_gate_up, w_down, correction_bias, *biases = inputs
        b_gate_up, b_down = (*biases, None, None)[:2]
        p = self.call.params
        routing = (
            gating,
            correction_bias,
            p["top_k"],
            p["scoring_func"],
            p["renormalize"],
            p["num_expert_group"],
            p["topk_group"],
        )
        # Unscaled, as the routing hands them to the experts, which apply the factor once.
        topk_weights, topk_ids = ref_fused_topk(*routing)
        return routed_experts_verification(
            hidden,
            w_gate_up,
            w_down,
            topk_weights,
            topk_ids,
            p["activation"],
            p["routed_scaling_factor"],
            b_gate_up=b_gate_up,
            b_down=b_down,
            ties=_tied_routings(*routing),
            controls=_gate_up_swap(list(self.call.signature.inputs), p["activation"]),
        )


class FusedMoESharedExpertWorkload(FusedMoEWorkload):
    """One ``FusedMoESharedExpertFwdOp`` call: FusedMoE's inputs plus the shared weights."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        *routed, shared_w_gate_up, shared_w_down = super().gen_inputs()
        hidden, _, w_gate_up, w_down, _ = routed
        # Far from linear, so the gate-up-swapped control lands outside the routed budget.
        _condition_routed_experts(self.call, hidden, w_gate_up, w_down)
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
        from workloads.numerics import Custom

        *routed_inputs, _, _ = inputs
        # The routed half as FusedMoEFwdOp's: its ties and its gate-up-swapped control.
        routed = super().verification(*routed_inputs)

        def validate(got, expected):
            # compare_outputs has already matched every output's shape, dtype and device, and
            # a None shared output to a None reference.
            (shared, routed_output), (shared_reference, routed_reference) = got, expected
            if shared_reference is not None:
                _assert_moe_close(shared, shared_reference)
            routed.validator(routed_output, routed_reference)

        return Custom(
            validate,
            "shared output elementwise atol/rtol 1e-2 and normalized squared error <= 1e-4; "
            f"routed output {routed.reason}",
            routed.controls,
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


def _swap_gate_up(t: torch.Tensor, activation: str, dim: int) -> torch.Tensor:
    """*t* with gate and up exchanged along *dim*, in the activation's layout: the two stacked
    halves, or the two members of each interleaved pair under ``swigluoai``."""
    if activation == "swigluoai":
        dim %= t.dim()
        return t.unflatten(dim, (-1, 2)).flip(dim + 1).flatten(dim, dim + 1)
    gate, up = t.chunk(2, dim=dim)
    return torch.cat((up, gate), dim=dim)


def _assert_moe_close(got: torch.Tensor, expected: torch.Tensor) -> None:
    """*got* within 1e-2 absolute plus 1e-2 relative of *expected* elementwise, and its
    normalized squared error at most 1e-4: the shared expert's bound, as
    ``FusedMoESharedExpertFwdOp``'s docstring states it."""
    from workloads.numerics import assert_normalized_error

    elementwise = 1e-2
    torch.testing.assert_close(
        got,
        expected,
        rtol=elementwise,
        atol=elementwise,
        msg=lambda detail: f"an element past 1e-2 absolute plus 1e-2 relative\n{detail}",
    )
    assert_normalized_error(got, expected, bound=elementwise * elementwise)


def moe_verification(gate_up_index):
    """Check both individual errors and scale-independent energy, including gate order."""
    from workloads.numerics import Custom, NegativeControl

    def swapped(reference, inputs):
        changed = list(inputs)
        gate, up = changed[gate_up_index].chunk(2, dim=-2)
        changed[gate_up_index] = torch.cat((up, gate), dim=-2)
        return reference(*changed)

    return Custom(
        _assert_moe_close,
        "elementwise atol/rtol 1e-2 and normalized squared error <= 1e-4",
        controls=(NegativeControl("gate-up-swapped", swapped),),
    )
