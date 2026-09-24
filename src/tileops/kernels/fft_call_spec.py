"""Call record and service regions for the FFT kernels."""

from __future__ import annotations

import dataclasses

import torch

from tileops.kernels.call_spec import CallSpec

__all__ = ["FFTC2CCall", "four_step_region", "single_cta_region"]


@dataclasses.dataclass(frozen=True)
class FFTC2CCall(CallSpec):
    """What decides which C2C implementation serves a call.

    The transform length and the complex dtype, which together name one plan, on
    top of the device facts ``CallSpec`` carries. The batch extent is not here:
    every kernel takes it as a symbolic dimension, so it changes neither the
    choice nor the compilation.
    """

    n: int = 0
    dtype: torch.dtype = torch.complex64
    device_index: "int | None" = None


def _record(call: FFTC2CCall, plans: "dict[tuple, object]") -> object:
    """The plan record for *call*, or ``None`` when no plan covers it."""
    return plans.get((call.n, str(call.dtype).split(".")[-1]))


def single_cta_region(call: FFTC2CCall, plans: "dict[tuple, object]") -> bool:
    """The region one CTA serves in one launch: a plan of a single factor.

    Widening it means adding a record to the plan table, not changing this shape.
    A record that does not list the device's architecture is outside the region:
    what that list states is whether the record's tiles fit the shared memory a
    block can be given there.
    """
    plan = _record(call, plans)
    return plan is not None and not plan.decomposed and call.arch in plan.archs


def four_step_region(call: FFTC2CCall, plans: "dict[tuple, object]") -> bool:
    """The region the four-step decomposition serves: a plan of two or more factors.

    Keyed by (length, dtype), not by length alone: 16384 is decomposed at
    complex128, whose four-pass plan does not fit in shared memory, and kept in
    one CTA at complex64. The two regions are complementary by construction --
    one record per call and the factor count decides which side it falls on --
    so no call can be claimed by both.
    """
    plan = _record(call, plans)
    return plan is not None and plan.decomposed and call.arch in plan.archs
