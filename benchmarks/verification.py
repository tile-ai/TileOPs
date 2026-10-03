"""What establishes that a benchmark tag computes what the op computes.

A ratio against a baseline computing something else is precise and meaningless, which is
worse than no ratio. Every tag therefore carries evidence, and the four kinds below are the
whole vocabulary: a tag is checked elementwise against the reference, checked by a validator
the case supplies, timed without a ratio because it implements a different function, or
measured with no reference available and said so.

`Exact` is the default, so a tag that can take an elementwise check needs no declaration.
"""

import dataclasses
from typing import Any, Callable, Optional

__all__ = [
    "Custom",
    "Evidence",
    "Exact",
    "Noncomparable",
    "Partial",
    "ReferenceInfeasible",
    "Unestablished",
    "set_verifying",
    "verifying",
]

# Set from --tileops-verify by benchmarks/conftest.py. A run either measures or verifies:
# a reference resident while the timer runs perturbs the measurement it exists to take.
_VERIFYING = False


def verifying() -> bool:
    """Whether this process verifies rather than times."""
    return _VERIFYING


def set_verifying(value: bool) -> None:
    """Put the run into verification mode. The benchmark conftest owns this."""
    global _VERIFYING
    _VERIFYING = value


@dataclasses.dataclass(frozen=True)
class Exact:
    """The tag's outputs equal the reference's.

    ``rtol`` and ``atol`` override the dtype's table entry, for an op whose accumulation
    order makes that table too tight: a complex FFT has no entry at all, and the table is
    the test's decision rather than the shared layer's.

    ``reference`` names the one to check against, for a workload shared by more than one
    op: `layer-boundaries.md` puts the reference on the narrowest class naming a single
    operator, so a call class serving both a forward and a backward carries neither. A
    backward's reference must differentiate its own forward, since one checked against
    state the op's own forward produced validates a matching pair of errors.
    """

    rtol: Optional[float] = None
    atol: Optional[float] = None
    reference: Optional[Callable] = None
    kind: str = "exact"

    def tolerance(self, default: dict) -> dict:
        """The tolerance to assert with, the declaration winning over the dtype's."""
        named = {k: v for k, v in (("rtol", self.rtol), ("atol", self.atol)) if v is not None}
        return named or default


@dataclasses.dataclass(frozen=True)
class Partial:
    """The reference establishes the first *outputs* of the tag's result and no more.

    An op whose signature declares intermediates the reference has no counterpart for —
    DeltaNet's chunk buffers, a saved state — would otherwise have them pass unchecked
    under `Exact`, which claims the whole result was established. *reason* names what is
    left, so a reader of the row knows which outputs nothing stands behind.
    """

    outputs: int
    reason: str
    rtol: Optional[float] = None
    atol: Optional[float] = None
    reference: Optional[Callable] = None
    kind: str = "partial"

    def __post_init__(self) -> None:
        # Zero outputs keeps the ratio while nothing was compared.
        if self.outputs < 1:
            raise ValueError(
                f"Partial(outputs={self.outputs}) establishes nothing; a tag no reference "
                "reaches is Noncomparable or ReferenceInfeasible"
            )

    def tolerance(self, default: dict) -> dict:
        """The tolerance to assert with, the declaration winning over the dtype's."""
        named = {k: v for k, v in (("rtol", self.rtol), ("atol", self.atol)) if v is not None}
        return named or default


@dataclasses.dataclass(frozen=True)
class Unestablished:
    """Nothing available can check this tag, and no one has said why.

    What a tag defaults to when its workload carries no reference. Distinct from
    `ReferenceInfeasible`, which is a judgement someone made; this is the absence of one.
    The row is timed, carries no ratio, and the nightly report counts it.
    """

    kind: str = "unestablished"


@dataclasses.dataclass(frozen=True)
class Custom:
    """The tag is checked by *validator*, for an op an elementwise check cannot judge.

    A sampling op is the case: its outputs are a draw, and `tests/ops/test_sampling.py`
    compares support, structure and distribution instead. The validator takes the tag's
    output and the reference's and raises on disagreement.
    """

    validator: Callable[[Any, Any], None]
    reason: str
    kind: str = "custom"


@dataclasses.dataclass(frozen=True)
class Noncomparable:
    """The tag implements a different function, so it is timed and publishes no ratio.

    fla's NSA block selection against `NSATopkVarlenFwdOp` is the case: different pinning
    and tie rules, so no tolerance applies and a ratio would compare two functions.
    """

    reason: str
    kind: str = "noncomparable"


@dataclasses.dataclass(frozen=True)
class ReferenceInfeasible:
    """Nothing available can establish this tag at this case.

    Distinct from `Noncomparable`: the tag computes the right function and nothing here can
    currently say so. *missing* names what would close it, so the row states the gap rather
    than a tracker number that goes stale while the gap does not.
    """

    missing: str
    reason: str
    kind: str = "reference_infeasible"


Evidence = Exact | Partial | Custom | Noncomparable | ReferenceInfeasible | Unestablished


def ratio_allowed(evidence: Evidence) -> bool:
    """Whether a row carrying *evidence* may publish a ratio against the op."""
    return evidence.kind in ("exact", "custom", "partial")


def describe(evidence: Evidence) -> Optional[str]:
    """The text a published row carries, or None where the tag was checked."""
    if evidence.kind == "exact":
        return None
    if evidence.kind == "custom":
        return f"custom: {evidence.reason}"
    if evidence.kind == "partial":
        return f"first {evidence.outputs} output(s) checked: {evidence.reason}"
    if evidence.kind == "noncomparable":
        return f"noncomparable: {evidence.reason}"
    if evidence.kind == "reference_infeasible":
        return f"unestablished, needs {evidence.missing}: {evidence.reason}"
    return "unestablished: the workload carries no reference"
