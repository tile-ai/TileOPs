import math
from typing import Any

import torch

from workloads.workload_base import WorkloadBase


class FusedTopKWorkload(WorkloadBase):
    def __init__(
        self,
        num_tokens,
        num_experts,
        top_k,
        scoring_func,
        renormalize,
        dtype,
        with_correction_bias=False,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.dtype = dtype
        self.with_correction_bias = with_correction_bias

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        torch.manual_seed(42)
        gating = torch.randn(self.num_tokens, self.num_experts, dtype=self.dtype, device="cuda")
        if not self.with_correction_bias:
            return (gating,)
        bias = torch.randn(self.num_experts, dtype=torch.float32, device="cuda") * 0.1
        return gating, bias


class MoePermuteWorkload(WorkloadBase):
    def __init__(self, total_tokens, top_k, num_experts, hidden_size, dtype):
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = torch.randn(
            self.total_tokens, self.hidden_size, dtype=self.dtype, device="cuda"
        )
        topk_ids = torch.randint(
            0,
            self.num_experts,
            (self.total_tokens, self.top_k),
            dtype=torch.int32,
            device="cuda",
        )
        return hidden_states, topk_ids

    def ref_program(self, hidden_states, topk_ids):
        return ref_moe_permute_nopad(hidden_states, topk_ids, self.num_experts)


class MoePermuteAlignWorkload(WorkloadBase):
    def __init__(self, total_tokens: int, top_k: int, num_experts: int, block_size: int):
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.block_size = block_size

    def gen_inputs(self) -> tuple[torch.Tensor]:
        topk_ids = torch.randint(
            0,
            self.num_experts,
            (self.total_tokens, self.top_k),
            dtype=torch.int32,
            device="cuda",
        )
        return (topk_ids,)

    def ref_program(
        self, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return ref_permute_align(topk_ids, self.block_size, self.num_experts)


class MoeUnpermuteWorkload(WorkloadBase):
    def __init__(self, total_tokens, top_k, hidden_size, dtype):
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        numel = self.total_tokens * self.top_k
        mm2_pad = torch.randn(numel, self.hidden_size, dtype=self.dtype, device="cuda")
        # fwd_idx: simulate a valid mapping: random shuffle of [0, numel)
        fwd_idx = torch.randperm(numel, dtype=torch.int32, device="cuda")
        topk_weights = torch.rand(self.total_tokens, self.top_k, dtype=torch.float32, device="cuda")
        return mm2_pad, fwd_idx, topk_weights

    def ref_program(self, mm2_pad, fwd_idx, topk_weights):
        return ref_moe_unpermute(mm2_pad, fwd_idx, topk_weights)


def make_expert_sizes_offsets(
    numel: int,
    num_experts: int,
    distribution: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (true_sizes, true_offsets) for a fixed token-to-expert distribution.

    Args:
        numel: Total token-expert pairs (T * top_k).
        num_experts: Number of experts E.
        distribution: "uniform" — evenly split; "skewed" — most tokens on first
            20% of experts (one-token floor for the rest).
        device: CUDA device string.

    Returns:
        true_sizes [E] int32, true_offsets [E] int32.
    """
    if distribution == "uniform":
        base = max(1, numel // num_experts)
        sizes = torch.full((num_experts,), base, dtype=torch.int32, device=device)
        sizes[-1] = numel - base * (num_experts - 1)
    elif distribution == "skewed":
        sizes = torch.ones(num_experts, dtype=torch.int32, device=device)
        extra = numel - num_experts
        top_experts = max(1, num_experts // 5)
        per_top = extra // top_experts
        sizes[:top_experts] += per_top
        sizes[0] += extra - per_top * top_experts
    else:
        raise ValueError(f"unknown distribution: {distribution}")

    offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    assert int(sizes.sum().item()) == numel
    return sizes, offsets


class MoeGroupedGemmNopadWorkload(WorkloadBase):
    """Tight A, per-expert weights B, and the expert size/offset tables."""

    def __init__(
        self,
        numel: int,
        num_experts: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        distribution: str = "uniform",
    ):
        self.numel = numel
        self.num_experts = num_experts
        self.n = n
        self.k = k
        self.dtype = dtype
        self.distribution = distribution

    def gen_inputs(self):
        torch.manual_seed(42)
        dev = "cuda"
        true_sizes, true_offsets = make_expert_sizes_offsets(
            self.numel, self.num_experts, self.distribution, dev
        )
        # Small scale keeps fp16 accumulation well within the parity tolerance.
        a = torch.randn(self.numel, self.k, dtype=self.dtype, device=dev) * 0.02
        b = torch.randn(self.num_experts, self.n, self.k, dtype=self.dtype, device=dev) * 0.02
        return a, b, true_sizes, true_offsets


def make_expert_layout_metadata(
    layout: str,
    rows: int,
    num_experts: int,
    distribution: str,
    device: str,
    *,
    alignment: int = 1,
    max_m: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(layout_metadata, valid_rows)`` for one staged expert layout preset.

    ``rows`` is the materialized row count of the activation (``num_experts * max_m``
    for ``masked``). Per-expert row counts follow ``distribution`` as in
    :func:`make_expert_sizes_offsets`. ``valid_rows`` is a bool mask over the flat
    materialized rows naming the rows a GEMM defines an output for: every row for a
    tight layout, the rows inside an expert's true count for the others.

    * ``tight_physical_psum``: metadata is the running end of each expert's rows.
    * ``aligned_physical_psum``: each expert starts at the previous end rounded up
      to ``alignment``; metadata is the end of its true rows, so a segment's last
      tile is partial.
    * ``aligned_per_row``: the same segments, metadata is the expert id of every
      row with the padding rows carrying the sentinel ``num_experts``; ``rows``
      must hold every rounded segment.
    * ``masked``: metadata is each expert's valid count within its ``max_m`` slab,
      a fifth of the experts full and the rest a quarter full (all full for
      ``uniform``).
    """
    if layout == "masked":
        if max_m is None or rows != num_experts * max_m:
            raise ValueError("masked rows are num_experts * max_m")
        masked_m = torch.full((num_experts,), max_m // 4, dtype=torch.int32, device=device)
        masked_m[: max(1, num_experts // 5)] = max_m
        if distribution == "uniform":
            masked_m.fill_(max_m)
        valid = torch.arange(max_m, device=device)[None, :] < masked_m[:, None]
        return masked_m, valid.reshape(-1)
    if layout == "tight_physical_psum":
        sizes, _ = make_expert_sizes_offsets(rows, num_experts, distribution, device)
        return torch.cumsum(sizes, dim=0).to(torch.int32), torch.ones(
            rows, dtype=torch.bool, device=device
        )
    if layout not in ("aligned_physical_psum", "aligned_per_row"):
        raise ValueError(f"unknown layout preset: {layout}")
    if alignment <= 1 or rows % alignment:
        raise ValueError("aligned layouts need rows to be a multiple of an alignment > 1")
    # Distribute the true rows so that every populated expert ends mid-tile and the
    # rounded segments, plus one whole tile of sentinel padding, fill ``rows``.
    tiles = rows // alignment
    if tiles <= num_experts:
        raise ValueError("aligned layouts need more than one tile per expert")
    tile_counts, _ = make_expert_sizes_offsets(tiles - 1, num_experts, distribution, device)
    valid = torch.zeros(rows, dtype=torch.bool, device=device)
    ends, ids, row = [], torch.full((rows,), num_experts, dtype=torch.int32, device=device), 0
    for g, t in enumerate(tile_counts.tolist()):
        true_rows = t * alignment - alignment // 2
        valid[row : row + true_rows] = True
        ids[row : row + t * alignment] = g
        ends.append(row + true_rows)
        row += t * alignment
    if layout == "aligned_per_row":
        return ids, valid
    return torch.tensor(ends, dtype=torch.int32, device=device), valid


def gated_activation(gate_up: torch.Tensor, activation: str) -> torch.Tensor:
    """``act(gate) * up`` over columns stacked gate then up, in the input's dtype."""
    gate, up = gate_up.chunk(2, dim=-1)
    if activation == "silu_and_mul":
        return torch.nn.functional.silu(gate) * up
    if activation == "gelu_and_mul":
        return torch.nn.functional.gelu(gate) * up
    raise ValueError(f"unknown gated activation: {activation}")


def ref_moe_grouped_gemm_staged(
    a: torch.Tensor,
    b: torch.Tensor,
    layout_metadata: torch.Tensor,
    layout: str,
    out_dtype: torch.dtype | None = None,
    *,
    alignment: int = 1,
    activation: str | None = None,
) -> torch.Tensor:
    """``out[rows of g] = a[rows of g] @ b[g]^T`` in fp32 per expert; other rows are zero.

    With ``activation``, ``b`` stacks gate and up along ``N`` and the result is
    ``act(gate) * up``, half as wide, as the fused epilogue writes it.
    """
    out_dtype = a.dtype if out_dtype is None else out_dtype
    num_experts, n, _ = b.shape
    out = torch.zeros(*a.shape[:-1], n, dtype=torch.float32, device=a.device)
    if layout == "masked":
        for g, valid in enumerate(layout_metadata.tolist()):
            out[g, :valid] = a[g, :valid].float() @ b[g].float().T
    elif layout in ("tight_physical_psum", "aligned_physical_psum"):
        start = 0
        for g, end in enumerate(layout_metadata.tolist()):
            if layout == "aligned_physical_psum":
                start = -(-start // alignment) * alignment
            out[start:end] = a[start:end].float() @ b[g].float().T
            start = end
    elif layout == "aligned_per_row":
        ids = layout_metadata.to(torch.int64)
        for g in range(num_experts):
            rows = ids == g
            out[rows] = a[rows].float() @ b[g].float().T
    else:
        raise ValueError(f"unknown layout preset: {layout}")
    if activation is not None:
        out = gated_activation(out, activation)
    return out.to(out_dtype)


class MoeGroupedGemmStagedWorkload(WorkloadBase):
    """Expert-materialized ``a``, per-expert ``b`` and the layout's metadata tensor.

    ``valid_rows`` (set by ``gen_inputs``) masks the flat rows the GEMM defines an
    output for; a consumer compares those and ignores the layout's padding.
    """

    def __init__(
        self,
        a_shape: tuple[int, ...],
        b_shape: tuple[int, int, int],
        layout: str,
        dtype: torch.dtype,
        *,
        alignment: int = 1,
        max_m: int | None = None,
        activation: str | None = None,
        distribution: str = "skewed",
    ):
        self.a_shape = tuple(a_shape)
        self.b_shape = tuple(b_shape)
        self.layout = layout
        self.dtype = dtype
        self.alignment = alignment
        self.max_m = max_m
        self.activation = activation
        self.distribution = distribution
        self.valid_rows: torch.Tensor | None = None

    @property
    def rows(self) -> int:
        return math.prod(self.a_shape[:-1])

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        dev = "cuda"
        metadata, self.valid_rows = make_expert_layout_metadata(
            self.layout,
            self.rows,
            self.b_shape[0],
            self.distribution,
            dev,
            alignment=self.alignment,
            max_m=self.max_m,
        )
        # Small scale keeps fp16 accumulation well within the parity tolerance.
        a = torch.randn(*self.a_shape, dtype=self.dtype, device=dev) * 0.02
        b = torch.randn(*self.b_shape, dtype=self.dtype, device=dev) * 0.02
        return a, b, metadata

    def ref_program(self, a, b, layout_metadata):
        return ref_moe_grouped_gemm_staged(
            a,
            b,
            layout_metadata,
            self.layout,
            alignment=self.alignment,
            activation=self.activation,
        )


class MoeExpertMLPStagedWorkload(MoeGroupedGemmStagedWorkload):
    """Expert-materialized input, stacked gate/up and down weights, layout metadata."""

    def __init__(
        self,
        expert_input_shape: tuple[int, ...],
        w_gate_up_shape: tuple[int, int, int],
        w_down_shape: tuple[int, int, int],
        layout: str,
        dtype: torch.dtype,
        *,
        alignment: int = 1,
        max_m: int | None = None,
        activation: str = "silu_and_mul",
        distribution: str = "skewed",
    ):
        super().__init__(
            expert_input_shape,
            w_gate_up_shape,
            layout,
            dtype,
            alignment=alignment,
            max_m=max_m,
            activation=activation,
            distribution=distribution,
        )
        self.w_down_shape = tuple(w_down_shape)

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, w_gate_up, metadata = super().gen_inputs()
        w_down = torch.randn(*self.w_down_shape, dtype=self.dtype, device="cuda") * 0.02
        return x, w_gate_up, w_down, metadata

    def ref_program(self, expert_input, w_gate_up, w_down, layout_metadata):
        activated = ref_moe_grouped_gemm_staged(
            expert_input,
            w_gate_up,
            layout_metadata,
            self.layout,
            expert_input.dtype,
            alignment=self.alignment,
            activation=self.activation,
        )
        return ref_moe_grouped_gemm_staged(
            activated, w_down, layout_metadata, self.layout, alignment=self.alignment
        )


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
        torch.manual_seed(42)
        dev = "cuda"
        hidden = torch.randn(
            self.num_tokens,
            self.hidden_size,
            dtype=self.dtype,
            device=dev,
        )
        gating = torch.randn(
            self.num_tokens,
            self.num_experts,
            dtype=torch.float32,
            device=dev,
        )
        correction_bias = (
            torch.randn(self.num_experts, dtype=torch.float32, device=dev) * 0.1
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
            )
            * 0.02
        )
        return hidden, gating, correction_bias, w_gate_up, w_down


class SharedFusedMoeWorkload(WorkloadBase):
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
        torch.manual_seed(42)
        dev = "cuda"
        hidden = torch.randn(self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev)
        gating = torch.randn(self.num_tokens, self.num_experts, dtype=self.dtype, device=dev)
        correction_bias = (
            torch.randn(self.num_experts, dtype=torch.float32, device=dev) * 0.1
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
            )
            * 0.02
        )
        # Shared expert weights: gate+up concatenated [2*Fs, H], down [H, Fs]
        shared_w_gate_up = (
            torch.randn(self.shared_ffn_size * 2, self.hidden_size, dtype=self.dtype, device=dev)
            * 0.02
        )
        shared_w_down = (
            torch.randn(self.hidden_size, self.shared_ffn_size, dtype=self.dtype, device=dev) * 0.02
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
        torch.manual_seed(42)
        dev = "cuda"
        hidden = torch.randn(self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev)
        w1 = (
            torch.randn(
                self.num_experts, self.ffn_size * 2, self.hidden_size, dtype=self.dtype, device=dev
            )
            * 0.02
        )
        w2 = (
            torch.randn(
                self.num_experts, self.hidden_size, self.ffn_size, dtype=self.dtype, device=dev
            )
            * 0.02
        )
        topk_weights = torch.softmax(
            torch.randn(self.num_tokens, self.top_k, dtype=torch.float32, device=dev), dim=-1
        )
        topk_ids = torch.randint(
            0, self.num_experts, (self.num_tokens, self.top_k), dtype=torch.int32, device=dev
        )
        return hidden, w1, w2, topk_weights, topk_ids


class MoeFusedActivationWorkload(WorkloadBase):
    """Workload descriptor for fused vs unfused activation benchmark."""

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        ffn_size: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype,
    ):
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.dtype = dtype
        # Primary shape: (num_tokens, hidden_size) — the token tensor footprint.
        self.shape: tuple[int, int] = (num_tokens, hidden_size)

    def gen_inputs(self) -> tuple[Any, ...]:
        torch.manual_seed(42)
        dev = "cuda"
        hidden = torch.randn(
            self.num_tokens,
            self.hidden_size,
            dtype=self.dtype,
            device=dev,
        )
        w_gate_up = (
            torch.randn(
                self.num_experts,
                self.ffn_size * 2,
                self.hidden_size,
                dtype=self.dtype,
                device=dev,
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
            )
            * 0.02
        )
        topk_weights = torch.softmax(
            torch.randn(self.num_tokens, self.top_k, dtype=torch.float32, device=dev),
            dim=-1,
        )
        topk_ids = torch.randint(
            0,
            self.num_experts,
            (self.num_tokens, self.top_k),
            dtype=torch.int32,
            device=dev,
        )
        return hidden, w_gate_up, w_down, topk_weights, topk_ids


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


def ref_moe_permute_nopad(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for moe_permute (tight, no padding)."""
    T, H = hidden_states.shape
    K = topk_ids.shape[1]
    numel = T * K
    flat_ids = topk_ids.flatten().cpu().tolist()
    dev = hidden_states.device

    counts = [0] * num_experts
    for eid in flat_ids:
        counts[eid] += 1

    offsets = [0] * (num_experts + 1)
    for e in range(num_experts):
        offsets[e + 1] = offsets[e] + counts[e]

    write_ptr = list(offsets[:-1])
    slot_to_row = [0] * numel
    fwd_idx_list = [0] * numel

    for flat_idx, eid in enumerate(flat_ids):
        slot = write_ptr[eid]
        slot_to_row[slot] = flat_idx // K
        fwd_idx_list[flat_idx] = slot
        write_ptr[eid] += 1

    perm_h = torch.empty(numel, H, dtype=hidden_states.dtype, device=dev)
    for slot in range(numel):
        perm_h[slot] = hidden_states[slot_to_row[slot]]

    true_offsets_t = torch.tensor(offsets[:-1], dtype=torch.int32, device=dev)
    true_sizes_t = torch.tensor(counts, dtype=torch.int32, device=dev)
    expert_first_token_offset = torch.tensor(offsets, dtype=torch.int64, device=dev)
    fwd_idx_t = torch.tensor(fwd_idx_list, dtype=torch.int32, device=dev)

    return perm_h, true_offsets_t, true_sizes_t, expert_first_token_offset, fwd_idx_t


def ref_moe_unpermute(
    mm2_pad: torch.Tensor,
    fwd_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference for moe_unpermute."""
    _, H = mm2_pad.shape
    T, K = topk_weights.shape
    dtype = mm2_pad.dtype

    output = torch.zeros(T, H, dtype=torch.float32, device=mm2_pad.device)
    for i in range(T):
        for k in range(K):
            flat_idx = i * K + k
            padded_slot = fwd_idx[flat_idx].item()
            w = topk_weights[i, k].item()
            output[i] += mm2_pad[padded_slot].float() * w

    return output.to(dtype)
