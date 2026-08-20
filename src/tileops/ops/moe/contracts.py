"""Typed contracts shared by the staged Mixture-of-Experts operators."""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import TypeAlias

import torch

__all__ = [
    "AlignmentPolicy",
    "ComputeFamilyKey",
    "ContiguousLayoutSpec",
    "ExpertLayoutMetadata",
    "Fp8OneDOneDComputeSpec",
    "Fp8OneDTwoDComputeSpec",
    "GroupedGemmComputeSpec",
    "InversePermuteContext",
    "MGroupedLayoutSpec",
    "MaskedLayoutSpec",
    "MaskedMetadata",
    "MaterializationPolicy",
    "MaterializedExpertLayout",
    "MetadataKind",
    "NoScaleComputeSpec",
    "PaddingPolicy",
    "PerRowExpertMetadata",
    "PhysicalPsumMetadata",
    "PostPermuteOutput",
    "PrePermuteOutput",
    "ResolvedContiguousLayout",
    "ResolvedMGroupedLayout",
    "ResolvedMaskedLayout",
    "RoutingEpilogueSpec",
    "ScaleLayout",
    "TailReadPolicy",
    "TileBoundaryPolicy",
    "resolve_compute_family",
    "routing_epilogue_reference",
]


class _StringEnum(str, enum.Enum):
    """Python 3.10-compatible string enum."""


class MetadataKind(_StringEnum):
    """Encoding used to associate materialized rows with experts."""

    PHYSICAL_PSUM = "physical_psum"
    PER_ROW_EXPERT_IDS = "per_row_expert_ids"
    MASKED_M = "masked_m"


class MaterializationPolicy(_StringEnum):
    """How contiguous expert segments occupy physical rows."""

    TIGHT = "tight"
    ALIGNED = "aligned"


class AlignmentPolicy(_StringEnum):
    """Rule used to resolve contiguous segment alignment."""

    NONE = "none"
    SM90_128 = "sm90_128"
    SM100_EXPECTED_M = "sm100_expected_m"


class PaddingPolicy(_StringEnum):
    """Whether materialization contains explicitly invalid rows."""

    NONE = "none"
    EXPLICIT_NEGATIVE_ONE = "explicit_negative_one"


class TileBoundaryPolicy(_StringEnum):
    """How a scheduler prevents a tile from crossing expert boundaries."""

    BOUNDARY_AWARE = "boundary_aware"
    ALIGNED = "aligned"


class TailReadPolicy(_StringEnum):
    """Whether masked expert tails may be read."""

    PREDICATED = "predicated"
    ZERO_FILLED = "zero_filled"


class ComputeFamilyKey(_StringEnum):
    """Validated architecture and compute/scale implementation family."""

    SM90_BF16_NO_SCALE = "sm90_bf16_no_scale"
    SM90_FP8_ONE_D_ONE_D = "sm90_fp8_one_d_one_d"
    SM90_FP8_ONE_D_TWO_D = "sm90_fp8_one_d_two_d"


class ScaleLayout(_StringEnum):
    """Reserved scale tensor layout vocabulary for quantized compute specs."""

    ONE_D = "one_d"
    TWO_D = "two_d"


@dataclasses.dataclass(frozen=True)
class ContiguousLayoutSpec:
    """Requested contiguous M-grouped layout semantics."""

    metadata_kind: MetadataKind
    materialization: MaterializationPolicy = MaterializationPolicy.TIGHT
    alignment_policy: AlignmentPolicy = AlignmentPolicy.NONE
    padding_policy: PaddingPolicy = PaddingPolicy.NONE
    tile_boundary_policy: TileBoundaryPolicy = TileBoundaryPolicy.BOUNDARY_AWARE
    expected_m: int | None = None

    def __post_init__(self) -> None:
        if self.metadata_kind is MetadataKind.MASKED_M:
            raise ValueError("contiguous layout cannot use masked metadata")
        if self.expected_m is not None and self.expected_m <= 0:
            raise ValueError("expected_m must be positive when provided")
        if self.materialization is MaterializationPolicy.TIGHT:
            if self.alignment_policy is not AlignmentPolicy.NONE:
                raise ValueError("tight materialization must use alignment policy 'none'")
            if self.padding_policy is not PaddingPolicy.NONE:
                raise ValueError("tight materialization has no padding rows")
            if self.tile_boundary_policy is not TileBoundaryPolicy.BOUNDARY_AWARE:
                raise ValueError("tight materialization requires boundary-aware tiles")
        else:
            if self.alignment_policy is AlignmentPolicy.NONE:
                raise ValueError("aligned materialization requires an alignment policy")
            if self.padding_policy is not PaddingPolicy.EXPLICIT_NEGATIVE_ONE:
                raise ValueError("aligned materialization requires explicit padding")
            if self.tile_boundary_policy is not TileBoundaryPolicy.ALIGNED:
                raise ValueError("aligned materialization requires aligned tiles")

    def resolve(self, *, arch: int) -> "ResolvedContiguousLayout":
        """Resolve architecture-dependent alignment without runtime metadata."""
        alignment = self._resolved_alignment(arch)
        return ResolvedContiguousLayout(spec=self, arch=arch, resolved_alignment=alignment)

    def _resolved_alignment(self, arch: int) -> int:
        if self.materialization is MaterializationPolicy.TIGHT:
            return 1
        if self.alignment_policy is AlignmentPolicy.SM90_128:
            if arch != 90:
                raise ValueError("SM90 alignment policy requires architecture 90")
            return 128
        if arch != 100:
            raise ValueError("SM100 expected-M alignment policy requires architecture 100")
        if self.expected_m is None:
            raise ValueError("SM100 expected-M alignment policy requires expected_m")
        raise ValueError(
            "SM100 alignment vocabulary is reserved, but no SM100 policy resolver "
            "is registered in the current delivery"
        )


@dataclasses.dataclass(frozen=True)
class MaskedLayoutSpec:
    """Requested fixed-capacity masked M-grouped layout semantics."""

    max_m: int
    tail_read_policy: TailReadPolicy = TailReadPolicy.PREDICATED

    def __post_init__(self) -> None:
        if self.max_m < 0:
            raise ValueError("max_m must be non-negative")

    def resolve(self, *, arch: int) -> "ResolvedMaskedLayout":
        """Bind the requested masked contract to an architecture."""
        return ResolvedMaskedLayout(spec=self, arch=arch)


MGroupedLayoutSpec: TypeAlias = ContiguousLayoutSpec | MaskedLayoutSpec


@dataclasses.dataclass(frozen=True)
class ResolvedContiguousLayout:
    """Resolved interpretation contract for contiguous materialization."""

    spec: ContiguousLayoutSpec
    arch: int
    resolved_alignment: int

    def __post_init__(self) -> None:
        if self.arch <= 0:
            raise ValueError("arch must be positive")
        if self.resolved_alignment <= 0:
            raise ValueError("resolved_alignment must be positive")
        expected = self.spec._resolved_alignment(self.arch)
        if self.resolved_alignment != expected:
            raise ValueError(
                f"resolved alignment {self.resolved_alignment} does not match "
                f"policy value {expected}"
            )


@dataclasses.dataclass(frozen=True)
class ResolvedMaskedLayout:
    """Resolved interpretation contract for masked materialization."""

    spec: MaskedLayoutSpec
    arch: int

    def __post_init__(self) -> None:
        if self.arch <= 0:
            raise ValueError("arch must be positive")


ResolvedMGroupedLayout: TypeAlias = ResolvedContiguousLayout | ResolvedMaskedLayout


def _validate_metadata_tensor(tensor: torch.Tensor, *, name: str, length: int) -> None:
    if tensor.dtype is not torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if tensor.ndim != 1 or tensor.shape[0] != length:
        raise ValueError(f"{name} must have shape ({length},), got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


@dataclasses.dataclass(frozen=True)
class PhysicalPsumMetadata:
    """Physical segment ends for each compute expert."""

    physical_ends: torch.Tensor

    def validate_structure(self, *, num_experts: int, device: torch.device) -> None:
        """Validate facts available without reading device-resident values."""
        _validate_metadata_tensor(self.physical_ends, name="physical_ends", length=num_experts)
        if self.physical_ends.device != device:
            raise ValueError("physical_ends must be on the activation device")

    def device_value_guard(
        self,
        layout: ResolvedContiguousLayout,
        *,
        materialized_rows: int,
    ) -> torch.Tensor:
        """Return an asynchronous scalar guard for PSUM ordering and capacity."""
        ends = self.physical_ends
        if ends.numel() == 0:
            return torch.tensor(materialized_rows == 0, dtype=torch.bool, device=ends.device)
        starts = torch.cat((ends.new_zeros(1), ends[:-1]))
        alignment = layout.resolved_alignment
        starts = ((starts + alignment - 1) // alignment) * alignment
        final_rows = ((ends[-1] + alignment - 1) // alignment) * alignment
        return torch.all(ends >= starts) & (final_rows == materialized_rows)


@dataclasses.dataclass(frozen=True)
class PerRowExpertMetadata:
    """Expert ID for every materialized contiguous row."""

    expert_ids: torch.Tensor

    def validate_structure(self, *, materialized_rows: int, device: torch.device) -> None:
        """Validate facts available without reading device-resident values."""
        _validate_metadata_tensor(self.expert_ids, name="expert_ids", length=materialized_rows)
        if self.expert_ids.device != device:
            raise ValueError("expert_ids must be on the activation device")

    def device_value_guard(
        self, *, num_experts: int, layout: ResolvedContiguousLayout
    ) -> torch.Tensor:
        """Guard ID domain, ordered segments, gaps, and aligned starts."""
        ids = self.expert_ids
        allows_padding = layout.spec.padding_policy is PaddingPolicy.EXPLICIT_NEGATIVE_ONE
        lower = -1 if allows_padding else 0
        domain = torch.all((ids >= lower) & (ids < num_experts))
        if ids.numel() == 0:
            return domain

        valid = ids >= 0
        prior_max = torch.cummax(torch.where(valid, ids, ids.new_full((), -1)), dim=0).values
        ordered = torch.all(torch.where(valid, ids >= prior_max, True))
        previous_valid = torch.cat((valid.new_zeros(1), valid[:-1]))
        previous_ids = torch.cat((ids.new_full((1,), -1), ids[:-1]))
        starts = valid & (~previous_valid | (ids != previous_ids))
        row_ids = torch.arange(ids.numel(), device=ids.device)
        aligned_starts = torch.all(
            torch.where(starts, row_ids % layout.resolved_alignment == 0, True)
        )
        resumed = starts & (ids <= torch.cat((ids.new_full((1,), -1), prior_max[:-1])))
        return domain & ordered & aligned_starts & ~torch.any(resumed)


@dataclasses.dataclass(frozen=True)
class MaskedMetadata:
    """Valid row count for each expert in a masked layout."""

    masked_m: torch.Tensor

    def validate_structure(self, *, num_experts: int, device: torch.device) -> None:
        """Validate facts available without reading device-resident values."""
        _validate_metadata_tensor(self.masked_m, name="masked_m", length=num_experts)
        if self.masked_m.device != device:
            raise ValueError("masked_m must be on the activation device")

    def device_value_guard(self, *, max_m: int) -> torch.Tensor:
        """Return an asynchronous scalar guard for masked valid lengths."""
        return torch.all((self.masked_m >= 0) & (self.masked_m <= max_m))


ExpertLayoutMetadata: TypeAlias = PhysicalPsumMetadata | PerRowExpertMetadata | MaskedMetadata


@dataclasses.dataclass(frozen=True)
class MaterializedExpertLayout:
    """A resolved layout bound to exactly one compatible metadata encoding."""

    resolved_layout: ResolvedMGroupedLayout
    metadata: ExpertLayoutMetadata
    num_experts: int
    materialized_rows: int
    materialization_token: object = dataclasses.field(
        default_factory=object, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.num_experts < 0 or self.materialized_rows < 0:
            raise ValueError("num_experts and materialized_rows must be non-negative")
        resolved = self.resolved_layout
        metadata = self.metadata
        if isinstance(resolved, ResolvedMaskedLayout):
            if not isinstance(metadata, MaskedMetadata):
                raise TypeError("masked layout requires MaskedMetadata")
            expected_rows = self.num_experts * resolved.spec.max_m
            if self.materialized_rows != expected_rows:
                raise ValueError(
                    f"masked materialization requires {expected_rows} rows, "
                    f"got {self.materialized_rows}"
                )
            return
        expected_type = {
            MetadataKind.PHYSICAL_PSUM: PhysicalPsumMetadata,
            MetadataKind.PER_ROW_EXPERT_IDS: PerRowExpertMetadata,
        }[resolved.spec.metadata_kind]
        if not isinstance(metadata, expected_type):
            raise TypeError(
                f"{resolved.spec.metadata_kind.value} layout requires {expected_type.__name__}"
            )

    def validate_structure(self, expert_input: torch.Tensor) -> None:
        """Validate activation shape/device and its metadata binding."""
        if not expert_input.is_contiguous():
            raise ValueError("expert_input must be contiguous")
        if expert_input.ndim not in (2, 3):
            raise ValueError("expert_input must be rank 2 (contiguous) or rank 3 (masked)")
        if isinstance(self.resolved_layout, ResolvedContiguousLayout):
            if expert_input.ndim != 2 or expert_input.shape[0] != self.materialized_rows:
                raise ValueError("contiguous expert_input row count does not match layout")
        else:
            expected = (self.num_experts, self.resolved_layout.spec.max_m)
            if expert_input.ndim != 3 or tuple(expert_input.shape[:2]) != expected:
                raise ValueError("masked expert_input leading dimensions do not match layout")
        metadata = self.metadata
        if isinstance(metadata, PhysicalPsumMetadata):
            metadata.validate_structure(num_experts=self.num_experts, device=expert_input.device)
        elif isinstance(metadata, PerRowExpertMetadata):
            metadata.validate_structure(
                materialized_rows=self.materialized_rows, device=expert_input.device
            )
        else:
            metadata.validate_structure(num_experts=self.num_experts, device=expert_input.device)


@dataclasses.dataclass(frozen=True)
class InversePermuteContext:
    """Opaque invocation-bound state consumed only by post-permute."""

    inverse_indices: torch.Tensor
    resolved_layout: ResolvedMGroupedLayout
    materialization_token: object
    num_tokens: int
    top_k: int
    materialized_rows: int

    def __post_init__(self) -> None:
        if self.num_tokens < 0 or self.top_k <= 0 or self.materialized_rows < 0:
            raise ValueError("invalid inverse-permute dimensions")
        _validate_metadata_tensor(
            self.inverse_indices,
            name="inverse_indices",
            length=self.num_tokens * self.top_k,
        )

    def device_value_guard(self) -> torch.Tensor:
        """Return an asynchronous guard for local rows and non-local sentinels."""
        return torch.all(
            (self.inverse_indices >= -1) & (self.inverse_indices < self.materialized_rows)
        )


@dataclasses.dataclass(frozen=True)
class PrePermuteOutput:
    """Materialized expert input and state required by later stages."""

    expert_input: torch.Tensor
    expert_layout: MaterializedExpertLayout
    inverse_permute_context: InversePermuteContext
    ready_event: torch.cuda.Event | None = None

    def __post_init__(self) -> None:
        self.expert_layout.validate_structure(self.expert_input)
        context = self.inverse_permute_context
        if context.resolved_layout != self.expert_layout.resolved_layout:
            raise ValueError("inverse context and expert layout resolve different contracts")
        if context.materialized_rows != self.expert_layout.materialized_rows:
            raise ValueError("inverse context and expert layout have different row counts")
        if context.materialization_token is not self.expert_layout.materialization_token:
            raise ValueError("inverse context belongs to a different materialization")
        if context.inverse_indices.device != self.expert_input.device:
            raise ValueError("inverse context must be on the expert input device")


@dataclasses.dataclass(frozen=True)
class PostPermuteOutput:
    """Token-order output and optional readiness event."""

    output: torch.Tensor
    ready_event: torch.cuda.Event | None = None


@dataclasses.dataclass(frozen=True)
class RoutingEpilogueSpec:
    """Exactly-once local routing epilogue semantics."""

    apply_routing_weight: bool = True
    reduce_topk: bool = True
    routed_scaling_factor: float = 1.0
    accumulation_dtype: torch.dtype = torch.float32
    output_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if not self.apply_routing_weight or not self.reduce_topk:
            raise ValueError("post-permute must apply routing weights and reduce top-k")
        if self.accumulation_dtype is not torch.float32:
            raise ValueError("routing reduction accumulation must be torch.float32")
        if not math.isfinite(self.routed_scaling_factor) or self.routed_scaling_factor <= 0:
            raise ValueError("routed_scaling_factor must be finite and positive")


class GroupedGemmComputeSpec:
    """Marker base for caller-declared grouped-GEMM compute semantics."""


@dataclasses.dataclass(frozen=True)
class NoScaleComputeSpec(GroupedGemmComputeSpec):
    """No-scale grouped GEMM with explicit accumulation and output types."""

    accumulation_dtype: torch.dtype = torch.float32
    output_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if self.accumulation_dtype is not torch.float32:
            raise ValueError("NoScale accumulation currently requires torch.float32")
        if self.output_dtype is not torch.bfloat16:
            raise ValueError("NoScale output currently requires torch.bfloat16")


@dataclasses.dataclass(frozen=True)
class Fp8OneDOneDComputeSpec(GroupedGemmComputeSpec):
    """Reserved FP8 contract with one-dimensional A and B scales."""

    accumulation_dtype: torch.dtype = torch.float32
    output_dtype: torch.dtype = torch.bfloat16
    a_scale_layout: ScaleLayout = ScaleLayout.ONE_D
    b_scale_layout: ScaleLayout = ScaleLayout.ONE_D

    def __post_init__(self) -> None:
        if self.a_scale_layout is not ScaleLayout.ONE_D:
            raise ValueError("Fp8OneDOneDComputeSpec requires one-dimensional A scales")
        if self.b_scale_layout is not ScaleLayout.ONE_D:
            raise ValueError("Fp8OneDOneDComputeSpec requires one-dimensional B scales")


@dataclasses.dataclass(frozen=True)
class Fp8OneDTwoDComputeSpec(GroupedGemmComputeSpec):
    """Reserved FP8 contract with one-dimensional A and two-dimensional B scales."""

    accumulation_dtype: torch.dtype = torch.float32
    output_dtype: torch.dtype = torch.bfloat16
    a_scale_layout: ScaleLayout = ScaleLayout.ONE_D
    b_scale_layout: ScaleLayout = ScaleLayout.TWO_D

    def __post_init__(self) -> None:
        if self.a_scale_layout is not ScaleLayout.ONE_D:
            raise ValueError("Fp8OneDTwoDComputeSpec requires one-dimensional A scales")
        if self.b_scale_layout is not ScaleLayout.TWO_D:
            raise ValueError("Fp8OneDTwoDComputeSpec requires two-dimensional B scales")


def resolve_compute_family(
    compute: GroupedGemmComputeSpec,
    *,
    arch: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    scales: object | None,
) -> ComputeFamilyKey:
    """Validate operands and scales, returning one candidate-selection key."""
    if isinstance(compute, NoScaleComputeSpec):
        if scales is not None:
            raise ValueError("NoScaleComputeSpec forbids scales")
        if arch != 90 or a_dtype is not torch.bfloat16 or b_dtype is not torch.bfloat16:
            raise ValueError("NoScale currently supports only SM90 BF16 operands")
        return ComputeFamilyKey.SM90_BF16_NO_SCALE
    if isinstance(compute, (Fp8OneDOneDComputeSpec, Fp8OneDTwoDComputeSpec)):
        raise ValueError("FP8 compute specs are reserved and have no registered capability")
    raise TypeError(f"unsupported grouped-GEMM compute spec {type(compute).__name__}")


def routing_epilogue_reference(
    expert_output: torch.Tensor,
    inverse_permute_context: InversePermuteContext,
    topk_weights: torch.Tensor,
    epilogue: RoutingEpilogueSpec,
) -> torch.Tensor:
    """Apply inverse permutation and the routing epilogue in declared order."""
    context = inverse_permute_context
    if expert_output.ndim not in (2, 3):
        raise ValueError("expert_output must be contiguous rank 2 or masked rank 3")
    physical_rows = expert_output.numel() // expert_output.shape[-1]
    if physical_rows != context.materialized_rows:
        raise ValueError("expert_output must match the inverse context's materialized rows")
    if tuple(topk_weights.shape) != (context.num_tokens, context.top_k):
        raise ValueError("topk_weights shape does not match the inverse context")
    if topk_weights.dtype is not torch.float32:
        raise TypeError("topk_weights must have dtype torch.float32")
    if expert_output.device != context.inverse_indices.device:
        raise ValueError("expert_output and inverse indices must share a device")
    if expert_output.device != topk_weights.device:
        raise ValueError("expert_output and topk_weights must share a device")

    flat_indices = context.inverse_indices.to(torch.int64)
    local = flat_indices >= 0
    safe_indices = flat_indices.clamp_min(0)
    flat_output = expert_output.reshape(context.materialized_rows, expert_output.shape[-1])
    gathered = flat_output.index_select(0, safe_indices)
    gathered = gathered.reshape(context.num_tokens, context.top_k, -1)
    local = local.reshape(context.num_tokens, context.top_k, 1)
    weighted = gathered.to(epilogue.accumulation_dtype) * topk_weights.unsqueeze(-1)
    weighted = torch.where(local, weighted, torch.zeros_like(weighted))
    reduced = weighted.sum(dim=1, dtype=epilogue.accumulation_dtype)
    scaled = reduced * epilogue.routed_scaling_factor
    return scaled.to(epilogue.output_dtype)
