"""Typed contracts shared by the staged Mixture-of-Experts operators."""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import TypeAlias

import torch

__all__ = [
    "ContiguousLayoutSpec",
    "ExpertLayoutMetadata",
    "InversePermuteContext",
    "MGroupedLayoutSpec",
    "MaskedLayoutSpec",
    "MaskedMetadata",
    "MaterializedExpertLayout",
    "NoScaleComputeSpec",
    "PerRowExpertMetadata",
    "PhysicalPsumMetadata",
    "PrePermuteOutput",
    "RoutingEpilogueSpec",
    "routing_epilogue_reference",
]


class _LayoutKind(str, enum.Enum):
    """Concrete layout specializations understood by kernel candidates."""

    TIGHT_PHYSICAL_PSUM = "tight_physical_psum"
    TIGHT_PER_ROW = "tight_per_row"
    MASKED_PREDICATED = "masked_predicated"


@dataclasses.dataclass(frozen=True, init=False)
class ContiguousLayoutSpec:
    """Requested tight contiguous M-grouped layout semantics.

    Use a named constructor so callers select one supported semantic instead of
    assembling alignment, padding, and tile-boundary policies independently.
    Tight materialization always has alignment one, no padding rows, and
    requires a boundary-aware consumer.
    """

    _kind: _LayoutKind = dataclasses.field(repr=False)

    def __init__(self, kind: _LayoutKind) -> None:
        if kind not in (
            _LayoutKind.TIGHT_PHYSICAL_PSUM,
            _LayoutKind.TIGHT_PER_ROW,
        ):
            raise ValueError(f"{kind!r} is not a contiguous layout kind")
        object.__setattr__(self, "_kind", kind)

    @classmethod
    def tight_physical_psum(cls) -> "ContiguousLayoutSpec":
        """Use tight rows described by per-expert physical segment ends."""
        return cls(_LayoutKind.TIGHT_PHYSICAL_PSUM)

    @classmethod
    def tight_per_row(cls) -> "ContiguousLayoutSpec":
        """Use tight rows described by one expert ID per materialized row."""
        return cls(_LayoutKind.TIGHT_PER_ROW)

    @property
    def selection_key(self) -> str:
        """Return the concrete specialization key recorded in CallSpecs."""
        return self._kind.value

    @property
    def max_m(self) -> None:
        """Contiguous layouts have no fixed per-expert capacity."""
        return None

    def __repr__(self) -> str:
        constructor = {
            _LayoutKind.TIGHT_PHYSICAL_PSUM: "tight_physical_psum",
            _LayoutKind.TIGHT_PER_ROW: "tight_per_row",
        }[self._kind]
        return f"ContiguousLayoutSpec.{constructor}()"


@dataclasses.dataclass(frozen=True)
class MaskedLayoutSpec:
    """Requested fixed-capacity masked M-grouped layout semantics."""

    max_m: int

    def __post_init__(self) -> None:
        if self.max_m < 0:
            raise ValueError("max_m must be non-negative")

    @property
    def selection_key(self) -> str:
        """Return the concrete specialization key recorded in CallSpecs."""
        return _LayoutKind.MASKED_PREDICATED.value


MGroupedLayoutSpec: TypeAlias = ContiguousLayoutSpec | MaskedLayoutSpec


def _validate_metadata_tensor(tensor: torch.Tensor, *, name: str, length: int) -> None:
    if tensor.dtype is not torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if tensor.ndim != 1 or tensor.shape[0] != length:
        raise ValueError(f"{name} must have shape ({length},), got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


@dataclasses.dataclass(frozen=True)
class PhysicalPsumMetadata:
    """Physical segment ends for tightly materialized compute experts."""

    physical_ends: torch.Tensor

    def validate_structure(self, *, num_experts: int, device: torch.device) -> None:
        """Validate facts available without reading device-resident values."""
        _validate_metadata_tensor(self.physical_ends, name="physical_ends", length=num_experts)
        if self.physical_ends.device != device:
            raise ValueError("physical_ends must be on the activation device")

    def device_value_guard(self, *, materialized_rows: int) -> torch.Tensor:
        """Return an asynchronous guard for tight PSUM ordering and capacity."""
        ends = self.physical_ends
        if ends.numel() == 0:
            return torch.tensor(materialized_rows == 0, dtype=torch.bool, device=ends.device)
        starts = torch.cat((ends.new_zeros(1), ends[:-1]))
        return torch.all(ends >= starts) & (ends[-1] == materialized_rows)


@dataclasses.dataclass(frozen=True)
class PerRowExpertMetadata:
    """Expert ID for every tightly materialized contiguous row."""

    expert_ids: torch.Tensor

    def validate_structure(self, *, materialized_rows: int, device: torch.device) -> None:
        """Validate facts available without reading device-resident values."""
        _validate_metadata_tensor(self.expert_ids, name="expert_ids", length=materialized_rows)
        if self.expert_ids.device != device:
            raise ValueError("expert_ids must be on the activation device")

    def device_value_guard(self, *, num_experts: int) -> torch.Tensor:
        """Guard the expert-ID domain and ordered tight segments."""
        ids = self.expert_ids
        domain = torch.all((ids >= 0) & (ids < num_experts))
        if ids.numel() == 0:
            return domain
        ordered = torch.all(ids[1:] >= ids[:-1])
        return domain & ordered


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
        """Return an asynchronous guard for masked valid lengths."""
        return torch.all((self.masked_m >= 0) & (self.masked_m <= max_m))


ExpertLayoutMetadata: TypeAlias = PhysicalPsumMetadata | PerRowExpertMetadata | MaskedMetadata


@dataclasses.dataclass(frozen=True)
class MaterializedExpertLayout:
    """One materialization's layout semantic, metadata, and physical capacity."""

    layout: MGroupedLayoutSpec
    metadata: ExpertLayoutMetadata
    num_experts: int
    materialized_rows: int
    _materialization_token: object = dataclasses.field(
        default_factory=object,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.num_experts < 0 or self.materialized_rows < 0:
            raise ValueError("num_experts and materialized_rows must be non-negative")
        if isinstance(self.layout, MaskedLayoutSpec):
            if not isinstance(self.metadata, MaskedMetadata):
                raise TypeError("masked layout requires MaskedMetadata")
            _validate_metadata_tensor(
                self.metadata.masked_m,
                name="masked_m",
                length=self.num_experts,
            )
            expected_rows = self.num_experts * self.layout.max_m
            if self.materialized_rows != expected_rows:
                raise ValueError(
                    f"masked materialization requires {expected_rows} rows, "
                    f"got {self.materialized_rows}"
                )
            return
        expected_type = {
            _LayoutKind.TIGHT_PHYSICAL_PSUM: PhysicalPsumMetadata,
            _LayoutKind.TIGHT_PER_ROW: PerRowExpertMetadata,
        }[self.layout._kind]
        if not isinstance(self.metadata, expected_type):
            raise TypeError(f"{self.layout.selection_key} layout requires {expected_type.__name__}")
        if isinstance(self.metadata, PhysicalPsumMetadata):
            _validate_metadata_tensor(
                self.metadata.physical_ends,
                name="physical_ends",
                length=self.num_experts,
            )
        else:
            _validate_metadata_tensor(
                self.metadata.expert_ids,
                name="expert_ids",
                length=self.materialized_rows,
            )

    @classmethod
    def from_physical_psum(
        cls,
        physical_ends: torch.Tensor,
        *,
        materialized_rows: int,
        num_experts: int | None = None,
    ) -> "MaterializedExpertLayout":
        """Build a tight Physical-PSUM binding for external materialization."""
        experts = physical_ends.numel() if num_experts is None else num_experts
        return cls(
            layout=ContiguousLayoutSpec.tight_physical_psum(),
            metadata=PhysicalPsumMetadata(physical_ends),
            num_experts=experts,
            materialized_rows=materialized_rows,
        )

    @classmethod
    def from_per_row_ids(
        cls,
        expert_ids: torch.Tensor,
        *,
        num_experts: int,
    ) -> "MaterializedExpertLayout":
        """Build a tight per-row-ID binding for external materialization."""
        return cls(
            layout=ContiguousLayoutSpec.tight_per_row(),
            metadata=PerRowExpertMetadata(expert_ids),
            num_experts=num_experts,
            materialized_rows=expert_ids.numel(),
        )

    @classmethod
    def from_masked_m(
        cls,
        masked_m: torch.Tensor,
        *,
        max_m: int,
        num_experts: int | None = None,
    ) -> "MaterializedExpertLayout":
        """Build a predicated masked binding for external materialization."""
        experts = masked_m.numel() if num_experts is None else num_experts
        return cls(
            layout=MaskedLayoutSpec(max_m=max_m),
            metadata=MaskedMetadata(masked_m),
            num_experts=experts,
            materialized_rows=experts * max_m,
        )

    @property
    def selection_key(self) -> str:
        """Return the concrete specialization key recorded in CallSpecs."""
        return self.layout.selection_key

    @property
    def max_m(self) -> int | None:
        """Return masked capacity per expert, or ``None`` for contiguous layouts."""
        return self.layout.max_m

    def validate_structure(self, expert_input: torch.Tensor) -> None:
        """Validate activation shape/device and its metadata binding."""
        if not expert_input.is_contiguous():
            raise ValueError("expert_input must be contiguous")
        if isinstance(self.layout, ContiguousLayoutSpec):
            if expert_input.ndim != 2 or expert_input.shape[0] != self.materialized_rows:
                raise ValueError("contiguous expert_input row count does not match layout")
        else:
            expected = (self.num_experts, self.layout.max_m)
            if expert_input.ndim != 3 or tuple(expert_input.shape[:2]) != expected:
                raise ValueError("masked expert_input leading dimensions do not match layout")
        if isinstance(self.metadata, PhysicalPsumMetadata):
            self.metadata.validate_structure(
                num_experts=self.num_experts, device=expert_input.device
            )
        elif isinstance(self.metadata, PerRowExpertMetadata):
            self.metadata.validate_structure(
                materialized_rows=self.materialized_rows, device=expert_input.device
            )
        else:
            self.metadata.validate_structure(
                num_experts=self.num_experts, device=expert_input.device
            )


@dataclasses.dataclass(frozen=True)
class InversePermuteContext:
    """Opaque invocation-bound state consumed only by post-permute."""

    inverse_indices: torch.Tensor
    layout: MGroupedLayoutSpec
    num_experts: int
    num_tokens: int
    top_k: int
    materialized_rows: int
    _materialization_token: object = dataclasses.field(
        default_factory=object,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            self.num_experts < 0
            or self.num_tokens < 0
            or self.top_k <= 0
            or self.materialized_rows < 0
        ):
            raise ValueError("invalid inverse-permute dimensions")
        if isinstance(self.layout, MaskedLayoutSpec):
            expected_rows = self.num_experts * self.layout.max_m
            if self.materialized_rows != expected_rows:
                raise ValueError(
                    f"masked inverse context requires {expected_rows} rows, "
                    f"got {self.materialized_rows}"
                )
        _validate_metadata_tensor(
            self.inverse_indices,
            name="inverse_indices",
            length=self.num_tokens * self.top_k,
        )

    @classmethod
    def for_layout(
        cls,
        inverse_indices: torch.Tensor,
        expert_layout: MaterializedExpertLayout,
        *,
        num_tokens: int,
        top_k: int,
    ) -> "InversePermuteContext":
        """Bind inverse indices to one materialized expert layout."""
        context = cls(
            inverse_indices=inverse_indices,
            layout=expert_layout.layout,
            num_experts=expert_layout.num_experts,
            num_tokens=num_tokens,
            top_k=top_k,
            materialized_rows=expert_layout.materialized_rows,
        )
        object.__setattr__(context, "_materialization_token", expert_layout._materialization_token)
        return context

    @property
    def selection_key(self) -> str:
        """Return the concrete specialization key recorded in CallSpecs."""
        return self.layout.selection_key

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

    def __post_init__(self) -> None:
        self.expert_layout.validate_structure(self.expert_input)
        context = self.inverse_permute_context
        if context.layout != self.expert_layout.layout:
            raise ValueError("inverse context and expert layout use different contracts")
        if context.materialized_rows != self.expert_layout.materialized_rows:
            raise ValueError("inverse context and expert layout have different row counts")
        if context.num_experts != self.expert_layout.num_experts:
            raise ValueError("inverse context and expert layout have different expert counts")
        if context._materialization_token is not self.expert_layout._materialization_token:
            raise ValueError("inverse context belongs to a different materialization")
        if context.inverse_indices.device != self.expert_input.device:
            raise ValueError("inverse context must be on the expert input device")


@dataclasses.dataclass(frozen=True)
class RoutingEpilogueSpec:
    """Exactly-once local routing epilogue with fixed reduction/cast semantics."""

    routed_scaling_factor: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.routed_scaling_factor) or self.routed_scaling_factor <= 0:
            raise ValueError("routed_scaling_factor must be finite and positive")

    @property
    def accumulation_dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def output_dtype(self) -> torch.dtype:
        return torch.bfloat16


@dataclasses.dataclass(frozen=True)
class NoScaleComputeSpec:
    """SM90 BF16 grouped GEMM with FP32 accumulation and BF16 output."""

    @property
    def accumulation_dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def output_dtype(self) -> torch.dtype:
        return torch.bfloat16


def routing_epilogue_reference(
    expert_output: torch.Tensor,
    inverse_permute_context: InversePermuteContext,
    topk_weights: torch.Tensor,
    epilogue: RoutingEpilogueSpec,
) -> torch.Tensor:
    """Apply inverse permutation and the routing epilogue in declared order."""
    context = inverse_permute_context
    if not expert_output.is_contiguous():
        raise ValueError("expert_output must be contiguous")
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
    flat_output = expert_output.reshape(context.materialized_rows, expert_output.shape[-1])
    if context.materialized_rows == 0:
        gathered = expert_output.new_zeros(
            (context.num_tokens * context.top_k, expert_output.shape[-1])
        )
    else:
        gathered = flat_output.index_select(0, flat_indices.clamp_min(0))
    gathered = gathered.reshape(context.num_tokens, context.top_k, -1)
    local = local.reshape(context.num_tokens, context.top_k, 1)
    weighted = gathered.to(epilogue.accumulation_dtype) * topk_weights.unsqueeze(-1)
    weighted = torch.where(local, weighted, torch.zeros_like(weighted))
    reduced = weighted.sum(dim=1, dtype=epilogue.accumulation_dtype)
    scaled = reduced * epilogue.routed_scaling_factor
    return scaled.to(epilogue.output_dtype)
