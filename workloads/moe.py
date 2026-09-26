import math

import torch

from tileops.manifest import load_adts, load_manifest
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import Call, instantiate
from workloads.workload_base import CallWorkload, WorkloadBase


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
        logits = gating_output.float()
        if p["scoring_func"] == "softmax":
            scores = torch.softmax(logits, dim=-1)
        else:
            scores = logits.sigmoid()
        select = scores if correction_bias is None else scores + correction_bias
        topk_ids = torch.topk(select, p["top_k"], dim=-1, sorted=False).indices
        topk_weights = scores.gather(1, topk_ids)
        if p["renormalize"]:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights, topk_ids.int()


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


class FusedMoeWorkload(WorkloadBase):
    """Inputs for a single FusedMoe benchmark configuration."""

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        scoring_func: str,
        renormalize: bool,
        with_correction_bias: bool,
        routed_scaling_factor: float,
        dtype: torch.dtype,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.with_correction_bias = with_correction_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.dtype = dtype

    def gen_inputs(self):
        g = self.rng(device="cuda")
        dev = "cuda"
        hidden = torch.randn(
            self.num_tokens,
            self.hidden_size,
            dtype=self.dtype,
            device=dev,
            generator=g,
        )
        gating = torch.randn(
            self.num_tokens,
            self.num_experts,
            dtype=torch.float32,
            device=dev,
            generator=g,
        )
        correction_bias = (
            torch.randn(self.num_experts, dtype=torch.float32, device=dev, generator=g) * 0.1
            if self.with_correction_bias
            else None
        )
        w_gate_up = (
            torch.randn(
                self.num_experts,
                self.ffn_size * 2,
                self.hidden_size,
                dtype=self.dtype,
                device=dev,
                generator=g,
            )
            * 0.02
        )
        w_down = (
            torch.randn(
                self.num_experts,
                self.hidden_size,
                self.ffn_size,
                dtype=self.dtype,
                device=dev,
                generator=g,
            )
            * 0.02
        )
        return hidden, gating, correction_bias, w_gate_up, w_down


class FusedMoeSharedExpertWorkload(WorkloadBase):
    def __init__(
        self,
        num_tokens,
        num_experts,
        top_k,
        hidden_size,
        ffn_size,
        shared_ffn_size,
        scoring_func,
        renormalize,
        with_correction_bias,
        routed_scaling_factor,
        dtype,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.shared_ffn_size = shared_ffn_size
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.with_correction_bias = with_correction_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.dtype = dtype

    def gen_inputs(self):
        g = self.rng(device="cuda")
        dev = "cuda"
        hidden = torch.randn(
            self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev, generator=g
        )
        gating = torch.randn(
            self.num_tokens, self.num_experts, dtype=self.dtype, device=dev, generator=g
        )
        correction_bias = (
            torch.randn(self.num_experts, dtype=torch.float32, device=dev, generator=g) * 0.1
            if self.with_correction_bias
            else None
        )
        w_gate_up = (
            torch.randn(
                self.num_experts,
                self.ffn_size * 2,
                self.hidden_size,
                dtype=self.dtype,
                device=dev,
                generator=g,
            )
            * 0.02
        )
        w_down = (
            torch.randn(
                self.num_experts,
                self.hidden_size,
                self.ffn_size,
                dtype=self.dtype,
                device=dev,
                generator=g,
            )
            * 0.02
        )
        # Shared expert weights: gate+up concatenated [2*Fs, H], down [H, Fs].
        # ``shared_ffn_size is None`` is the routed-only configuration, where the
        # op takes no shared weights and returns None in their output position.
        if self.shared_ffn_size is None:
            shared_w_gate_up = None
            shared_w_down = None
        else:
            shared_w_gate_up = (
                torch.randn(
                    self.shared_ffn_size * 2,
                    self.hidden_size,
                    dtype=self.dtype,
                    device=dev,
                    generator=g,
                )
                * 0.02
            )
            shared_w_down = (
                torch.randn(
                    self.hidden_size,
                    self.shared_ffn_size,
                    dtype=self.dtype,
                    device=dev,
                    generator=g,
                )
                * 0.02
            )
        return hidden, gating, correction_bias, w_gate_up, w_down, shared_w_gate_up, shared_w_down


class MoeExpertsWorkload(WorkloadBase):
    def __init__(self, num_tokens, num_experts, top_k, hidden_size, ffn_size, dtype):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dtype = dtype

    def gen_inputs(self):
        g = self.rng(device="cuda")
        dev = "cuda"
        hidden = torch.randn(
            self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev, generator=g
        )
        w1 = (
            torch.randn(
                self.num_experts,
                self.ffn_size * 2,
                self.hidden_size,
                dtype=self.dtype,
                device=dev,
                generator=g,
            )
            * 0.02
        )
        w2 = (
            torch.randn(
                self.num_experts,
                self.hidden_size,
                self.ffn_size,
                dtype=self.dtype,
                device=dev,
                generator=g,
            )
            * 0.02
        )
        topk_weights = torch.softmax(
            torch.randn(self.num_tokens, self.top_k, dtype=torch.float32, device=dev, generator=g),
            dim=-1,
        )
        topk_ids = torch.randint(
            0,
            self.num_experts,
            (self.num_tokens, self.top_k),
            dtype=torch.int32,
            device=dev,
            generator=g,
        )
        return hidden, w1, w2, topk_weights, topk_ids


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
