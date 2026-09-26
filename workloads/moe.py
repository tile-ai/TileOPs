import math

import torch

from tileops.manifest import load_adts, load_manifest
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import Call, instantiate
from workloads.workload_base import CallWorkload


def moe_call(op: str, dtype_case: dict | None = None, **row) -> Call:
    """The manifest call of *op* that *row* describes, as a workload row would.

    A test picks its own shapes; the call still takes its metadata from the entry's
    generators and its contract from the entry's signature (docs/design/manifest.md § Rows).
    """
    plan = entry_plan(op, load_manifest()[op], load_adts(), resolve=False)
    return instantiate(plan, {**row, "label": "test"}, dtype_case or {})


class FusedTopKWorkload(CallWorkload):
    """Router logits, and the optional bias, of one ``FusedTopKFwdOp`` call."""

    def ref_program(
        self, gating_output: torch.Tensor, correction_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score, select ``top_k`` (on the biased scores when a bias is passed), renormalize."""
        p = self.call.params
        return ref_fused_topk(
            gating_output, correction_bias, p["top_k"], p["scoring_func"], p["renormalize"]
        )


class MoePermuteAlignWorkload(CallWorkload):
    """The routing ids of one ``MoePermuteAlignFwdOp`` call."""

    def ref_program(
        self, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.call.params
        return ref_permute_align(topk_ids, p["block_size"], p["num_experts"])


class MoePrePermuteWorkload(CallWorkload):
    """Hidden states and local expert ids of one ``MoePrePermuteFwdOp`` call."""

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
) -> torch.Tensor:
    """``out[rows of g] = a[rows of g] @ b[g]^T`` in fp32 per expert; other rows are zero.

    With ``activation``, ``b`` stacks gate and up along ``N`` and the result is
    ``act(gate) * up``, half as wide, as the fused epilogue writes it.
    """
    out_dtype = a.dtype if out_dtype is None else out_dtype
    num_experts, n, _ = b.shape
    out = torch.zeros(*a.shape[:-1], n, dtype=torch.float32, device=a.device)
    meta = layout_metadata.tolist()
    if layout.kind == "masked":
        for g, count in enumerate(meta):
            out[g, :count] = a[g, :count].float() @ b[g].float().T
    elif layout.metadata_kind == "per_row":
        ids = layout_metadata.long()
        for g in range(num_experts):
            selected = ids == g
            out[selected] = a[selected].float() @ b[g].float().T
    else:
        start = 0
        for g, end in enumerate(meta):
            start = -(-start // layout.alignment) * layout.alignment
            out[start:end] = a[start:end].float() @ b[g].float().T
            start = end
    if activation is not None:
        out = gated_activation(out, activation)
    return out.to(out_dtype)


class MoeGroupedGemmWorkload(CallWorkload):
    """Expert-materialized ``a``, per-expert ``b`` and the layout's generated metadata."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        a, b, metadata = super().gen_inputs()
        # A small scale keeps fp16 accumulation over K = 7168 finite and within tolerance.
        return a.mul_(0.02), b.mul_(0.02), metadata

    def ref_program(
        self, a: torch.Tensor, b: torch.Tensor, layout_metadata: torch.Tensor
    ) -> torch.Tensor:
        p = self.call.params
        out_dtype = None if p["out_dtype"] is None else getattr(torch, p["out_dtype"])
        return ref_moe_grouped_gemm(
            a, b, layout_metadata, p["layout"], out_dtype, activation=p["activation"]
        )


class MoeExpertMLPWorkload(CallWorkload):
    """Expert-materialized input, stacked gate/up and down weights, generated metadata."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x, w_gate_up, w_down, metadata = super().gen_inputs()
        return x.mul_(0.02), w_gate_up.mul_(0.02), w_down.mul_(0.02), metadata

    def ref_program(
        self,
        expert_input: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        layout_metadata: torch.Tensor,
    ) -> torch.Tensor:
        p = self.call.params
        activated = ref_moe_grouped_gemm(
            expert_input, w_gate_up, layout_metadata, p["layout"], activation=p["activation"]
        )
        return ref_moe_grouped_gemm(activated, w_down, layout_metadata, p["layout"])


class MoePostPermuteWorkload(CallWorkload):
    """Expert outputs, routing weights and inverse indices of one ``MoePostPermuteFwdOp`` call."""

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


def gated_activation(gate_up: torch.Tensor, activation: str) -> torch.Tensor:
    """``act(gate) * up`` over columns stacked gate then up, in the input's dtype."""
    gate, up = gate_up.chunk(2, dim=-1)
    if activation == "silu_and_mul":
        return torch.nn.functional.silu(gate) * up
    if activation == "gelu_and_mul":
        return torch.nn.functional.gelu(gate) * up
    raise ValueError(f"unknown gated activation: {activation}")


def ref_routed_experts(
    hidden: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu_and_mul",
    scale: float = 1.0,
) -> torch.Tensor:
    """``scale * sum_k w[t, k] * down(act(gate_up(h[t])))`` over each token's routed experts,
    in fp32 per expert, cast to the hidden dtype."""
    output = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
    ids = topk_ids.to(torch.int64)
    for e in range(w_gate_up.shape[0]):
        t_idx, k_idx = (ids == e).nonzero(as_tuple=True)
        if t_idx.numel() == 0:
            continue
        gate_up = hidden[t_idx].float() @ w_gate_up[e].float().T
        down = gated_activation(gate_up, activation) @ w_down[e].float().T
        output.index_add_(0, t_idx, down * topk_weights[t_idx, k_idx].float().unsqueeze(-1))
    return (output * scale).to(hidden.dtype)


class MoeExpertsWorkload(CallWorkload):
    """Tokens, expert weights and their routing for one ``FusedMoEExpertsFwdOp`` call."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        output, hidden, w_gate_up, w_down, topk_weights, topk_ids = super().gen_inputs()
        # Small scales keep fp16 accumulation over H = 7168 finite; routing weights sum to one.
        return (
            output,
            hidden.mul_(0.1),
            w_gate_up.mul_(0.02),
            w_down.mul_(0.02),
            topk_weights.softmax(dim=-1),
            topk_ids,
        )

    def ref_program(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
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
        )


class IndexedExpertMLPWorkload(MoeExpertsWorkload):
    """One ``IndexedExpertMLPFwdOp`` call: the same inputs and reference as the expert MLP."""


def ref_fused_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    top_k: int,
    scoring_func: str,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score, select ``top_k`` (on the biased scores when a bias is passed), renormalize."""
    logits = gating_output.float()
    scores = torch.softmax(logits, dim=-1) if scoring_func == "softmax" else logits.sigmoid()
    select = scores if correction_bias is None else scores + correction_bias
    topk_ids = torch.topk(select, top_k, dim=-1, sorted=False).indices
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids.int()


class FusedMoeWorkload(CallWorkload):
    """Tokens, gating logits and expert weights for one ``FusedMoeFwdOp`` call.

    The logits come from the workload's own generator: they decide the experts the call
    reads, which the roofline prices (docs/design/roofline.md §4.7).
    """

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        hidden, gating, w_gate_up, w_down, bias, *rest = super().gen_inputs()
        g = self.rng(device=gating.device)
        gating = torch.randn(gating.shape, generator=g, device=gating.device)
        if bias is not None:
            bias = torch.randn(bias.shape, generator=g, device=bias.device) * 0.1
        return (hidden.mul_(0.1), gating, w_gate_up.mul_(0.02), w_down.mul_(0.02), bias, *rest)

    def ref_routed(self, hidden, gating, w_gate_up, w_down, bias) -> torch.Tensor:
        p = self.call.params
        weights, ids = ref_fused_topk(gating, bias, p["top_k"], p["scoring_func"], p["renormalize"])
        return ref_routed_experts(
            hidden, w_gate_up, w_down, weights, ids, p["activation"], p["routed_scaling_factor"]
        )

    def ref_program(self, hidden_states, gating_output, w_gate_up, w_down, correction_bias=None):
        return self.ref_routed(hidden_states, gating_output, w_gate_up, w_down, correction_bias)


class FusedMoeSharedExpertWorkload(FusedMoeWorkload):
    """One ``FusedMoeSharedExpertFwdOp`` call: FusedMoe's inputs plus the shared weights."""

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
        act = gated_activation(hidden_states.float() @ gate_up.float().T, "silu_and_mul")
        shared = act @ shared_w_down[:, lo:hi].float().T
        return shared.to(hidden_states.dtype), routed


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
