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
    "RoutingEpilogueSpec",
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


MGroupedLayoutSpec: TypeAlias = ContiguousLayoutSpec | MaskedLayoutSpec


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
