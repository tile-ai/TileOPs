"""Typed contracts shared by the staged Mixture-of-Experts operators."""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import TypeAlias

import torch

__all__ = [
    "ContiguousMetadata",
    "ContiguousPacking",
    "ContiguousLayoutSpec",
    "MGroupedLayoutSpec",
    "MaskedLayoutSpec",
    "MaskedMetadata",
    "PerRowExpertMetadata",
    "PhysicalPsumMetadata",
    "RoutingEpilogueSpec",
    "layout_from_preset",
    "layout_value_guard",
]


class ContiguousPacking(str, enum.Enum):
    """How expert-contiguous rows are physically packed."""

    TIGHT = "tight"
    ALIGNED = "aligned"


class ContiguousMetadata(str, enum.Enum):
    """Metadata ABI that describes expert-contiguous rows."""

    PHYSICAL_PSUM = "physical_psum"
    PER_ROW = "per_row"


class _LayoutKind(str, enum.Enum):
    """Non-contiguous layout specializations understood by kernel candidates."""

    MASKED_PREDICATED = "masked_predicated"


@dataclasses.dataclass(frozen=True)
class ContiguousLayoutSpec:
    """Compile-time packing and metadata policy for contiguous expert rows."""

    packing: ContiguousPacking
    metadata_kind: ContiguousMetadata
    alignment: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.packing, ContiguousPacking):
            raise TypeError("packing must be ContiguousPacking")
        if not isinstance(self.metadata_kind, ContiguousMetadata):
            raise TypeError("metadata_kind must be ContiguousMetadata")
        if self.packing is ContiguousPacking.TIGHT and self.alignment != 1:
            raise ValueError("tight contiguous packing requires alignment == 1")
        if self.packing is ContiguousPacking.ALIGNED and self.alignment <= 1:
            raise ValueError("aligned contiguous packing requires alignment > 1")

    @classmethod
    def tight_physical_psum(cls) -> "ContiguousLayoutSpec":
        """Use tight rows described by per-expert physical segment ends."""
        return cls(ContiguousPacking.TIGHT, ContiguousMetadata.PHYSICAL_PSUM)

    @classmethod
    def tight_per_row(cls) -> "ContiguousLayoutSpec":
        """Use tight rows described by one expert ID per materialized row."""
        return cls(ContiguousPacking.TIGHT, ContiguousMetadata.PER_ROW)

    @classmethod
    def aligned_per_row(cls, alignment: int) -> "ContiguousLayoutSpec":
        """Use aligned expert segments described by one expert ID per row."""
        return cls(ContiguousPacking.ALIGNED, ContiguousMetadata.PER_ROW, alignment)

    @classmethod
    def aligned_physical_psum(cls, alignment: int) -> "ContiguousLayoutSpec":
        """Use aligned expert segments described by per-expert physical segment ends."""
        return cls(ContiguousPacking.ALIGNED, ContiguousMetadata.PHYSICAL_PSUM, alignment)

    @property
    def selection_key(self) -> str:
        """Return the concrete specialization key recorded in CallSpecs."""
        return f"{self.packing.value}_{self.metadata_kind.value}"

    @property
    def kind(self) -> str:
        """The layout family a kernel candidate reads first: ``"contiguous"``."""
        return "contiguous"

    @property
    def max_m(self) -> None:
        """Contiguous layouts have no fixed per-expert capacity."""
        return None

    def metadata_length(self, *, rows: int, num_experts: int) -> int:
        """Rows of ``layout_metadata`` this layout describes ``rows`` rows with."""
        if self.metadata_kind is ContiguousMetadata.PER_ROW:
            return rows
        return num_experts

    def __repr__(self) -> str:
        if self.packing is ContiguousPacking.ALIGNED:
            return f"ContiguousLayoutSpec.{self.selection_key}({self.alignment})"
        return f"ContiguousLayoutSpec.{self.selection_key}()"


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

    @property
    def kind(self) -> str:
        """The layout family a kernel candidate reads first: ``"masked"``."""
        return "masked"

    def metadata_length(self, *, rows: int, num_experts: int) -> int:
        """Masked metadata carries one valid-row count per expert."""
        return num_experts


MGroupedLayoutSpec: TypeAlias = ContiguousLayoutSpec | MaskedLayoutSpec


def layout_from_preset(
    name: str, *, alignment: int | None = None, max_m: int | None = None
) -> MGroupedLayoutSpec:
    """Resolve a manifest workload's ``layout`` preset name into a layout spec.

    ``tight_physical_psum`` and ``tight_per_row`` take nothing else;
    ``aligned_per_row`` and ``aligned_physical_psum`` take ``alignment``;
    ``masked`` takes ``max_m``. A missing or surplus argument is an error, so a
    workload row cannot silently mean a different layout than it names.
    """
    if name in ("tight_physical_psum", "tight_per_row"):
        if alignment is not None or max_m is not None:
            raise ValueError(f"{name} takes neither alignment nor max_m")
        return getattr(ContiguousLayoutSpec, name)()
    if name in ("aligned_per_row", "aligned_physical_psum"):
        if alignment is None or max_m is not None:
            raise ValueError(f"{name} takes alignment and no max_m")
        return getattr(ContiguousLayoutSpec, name)(alignment)
    if name == "masked":
        if max_m is None or alignment is not None:
            raise ValueError("masked takes max_m and no alignment")
        return MaskedLayoutSpec(max_m=max_m)
    raise ValueError(f"unknown layout preset {name!r}")


@dataclasses.dataclass(frozen=True)
class PhysicalPsumMetadata:
    """Physical segment ends for tightly materialized compute experts."""

    physical_ends: torch.Tensor

    def device_value_guard(self, *, materialized_rows: int) -> torch.Tensor:
        """Return an asynchronous guard for tight PSUM ordering and capacity."""
        ends = self.physical_ends
        if ends.numel() == 0:
            return torch.tensor(materialized_rows == 0, dtype=torch.bool, device=ends.device)
        starts = torch.cat((ends.new_zeros(1), ends[:-1]))
        return torch.all(ends >= starts) & (ends[-1] == materialized_rows)


@dataclasses.dataclass(frozen=True)
class PerRowExpertMetadata:
    """Expert ID for every materialized contiguous row."""

    expert_ids: torch.Tensor

    def device_value_guard(
        self, *, num_experts: int, allow_capacity_sentinel: bool = False
    ) -> torch.Tensor:
        """Guard ordered expert IDs and an optional trailing capacity sentinel."""
        ids = self.expert_ids
        upper = num_experts + int(allow_capacity_sentinel)
        domain = torch.all((ids >= 0) & (ids < upper))
        if ids.numel() == 0:
            return domain
        ordered = torch.all(ids[1:] >= ids[:-1])
        return domain & ordered


@dataclasses.dataclass(frozen=True)
class MaskedMetadata:
    """Valid row count for each expert in a masked layout."""

    masked_m: torch.Tensor

    def device_value_guard(self, *, max_m: int) -> torch.Tensor:
        """Return an asynchronous guard for masked valid lengths."""
        return torch.all((self.masked_m >= 0) & (self.masked_m <= max_m))


@dataclasses.dataclass(frozen=True)
class RoutingEpilogueSpec:
    """Exactly-once local routing epilogue with fixed reduction/cast semantics."""

    routed_scaling_factor: float = 1.0
    output_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.routed_scaling_factor) or self.routed_scaling_factor <= 0:
            raise ValueError("routed_scaling_factor must be finite and positive")
        if self.output_dtype not in (None, torch.bfloat16, torch.float16):
            raise ValueError("output_dtype must be None, torch.bfloat16, or torch.float16")

    @property
    def accumulation_dtype(self) -> torch.dtype:
        return torch.float32

    def resolve_output_dtype(self, input_dtype: torch.dtype) -> torch.dtype:
        """Use an explicit final dtype, or preserve the expert-output dtype."""
        return input_dtype if self.output_dtype is None else self.output_dtype


def layout_value_guard(
    layout: MGroupedLayoutSpec,
    layout_metadata: torch.Tensor,
    *,
    rows: int,
    num_experts: int,
) -> torch.Tensor:
    """Return an asynchronous bool tensor: does ``layout_metadata`` satisfy ``layout``?

    These are the invariants only the device-resident values can answer — a host
    check would synchronise, so ``forward`` never runs this. Tests and benchmarks
    consume it through ``torch._assert_async``. ``rows`` is the materialized row
    count of the activation (``E * max_m`` for masked layouts).

    * tight physical psum: ends non-decreasing, first end non-negative, last end
      equal to ``rows``.
    * aligned physical psum: ends non-decreasing; each end at or after its
      segment start, the previous end rounded up to ``alignment``; last end at
      most ``rows``.
    * tight per-row: ids non-decreasing in ``[0, num_experts)``.
    * aligned per-row: ids non-decreasing in ``[0, num_experts]`` (``num_experts``
      is the padding sentinel); every id change lands on a multiple of
      ``alignment``.
    * masked: every valid count in ``[0, max_m]``.
    """
    meta = layout_metadata
    if isinstance(layout, MaskedLayoutSpec):
        return MaskedMetadata(meta).device_value_guard(max_m=layout.max_m)
    if layout.metadata_kind is ContiguousMetadata.PER_ROW:
        ordered = PerRowExpertMetadata(meta).device_value_guard(
            num_experts=num_experts,
            allow_capacity_sentinel=layout.packing is ContiguousPacking.ALIGNED,
        )
        if layout.packing is ContiguousPacking.TIGHT or meta.numel() < 2:
            return ordered
        change = meta[1:] != meta[:-1]
        position = torch.arange(1, meta.numel(), device=meta.device)
        on_boundary = (position % layout.alignment) == 0
        return ordered & torch.all(~change | on_boundary)
    if layout.packing is ContiguousPacking.TIGHT:
        return PhysicalPsumMetadata(meta).device_value_guard(materialized_rows=rows)
    ends = meta
    if ends.numel() == 0:
        return torch.tensor(rows == 0, dtype=torch.bool, device=ends.device)
    alignment = layout.alignment
    prev = torch.cat((ends.new_zeros(1), ends[:-1]))
    starts = (prev + alignment - 1) // alignment * alignment
    return torch.all(ends >= starts) & (ends[0] >= 0) & (ends[-1] <= rows)
