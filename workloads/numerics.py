"""Numerical comparison mechanics shared by tests and benchmarks.

What belongs here is how a quantity is measured, not what bound it must meet.
Tolerances, which outputs a case compares and which deviations it allows stay
with the consumer; two copies of the measurement itself drift apart in silence,
and a benchmark then reports a ratio against a quantity no test computes the
same way.
"""

import dataclasses
from typing import Any, Callable, Optional

import torch

__all__ = [
    "Custom",
    "Evidence",
    "Exact",
    "NegativeControl",
    "Noncomparable",
    "Partial",
    "ReferenceInfeasible",
    "Unestablished",
    "assert_normalized_error",
    "assert_quantized",
    "describe",
    "logit_mask_validator",
    "zeroed_input",
]


def assert_normalized_error(got: Any, expected: Any, bound: float = 1e-3) -> None:
    """Bound squared error / combined energy, with identical nonfinite values.

    Args:
        got: Tensor, or a tuple or list of them, to check.
        expected: The reference, matching *got* in structure, shape and dtype.
        bound: Upper bound on the normalized squared error. Not an rtol or
            atol: those compare element by element, this one ratio over the
            whole output.

    Raises:
        AssertionError: When a shape, dtype or nonfinite entry disagrees, or
            the normalized squared error exceeds *bound*.
    """
    if isinstance(got, (tuple, list)) or isinstance(expected, (tuple, list)):
        outputs = got if isinstance(got, (tuple, list)) else (got,)
        targets = expected if isinstance(expected, (tuple, list)) else (expected,)
        for output, target in zip(outputs, targets, strict=True):
            assert_normalized_error(output, target, bound)
        return
    assert got.shape == expected.shape and got.dtype == expected.dtype
    error = torch.zeros((), device=got.device, dtype=torch.float64)
    energy = torch.zeros_like(error)
    # Reduced a megabyte of elements at a time: a long attention output reduced
    # whole holds an FP64 temporary the size of the input, and the sum is the
    # same either way.
    chunk = 1 << 20
    for a, b in zip(got.reshape(-1).split(chunk), expected.reshape(-1).split(chunk), strict=True):
        a, b = a.double(), b.double()
        finite = torch.isfinite(a)
        assert torch.equal(finite, torch.isfinite(b)), "nonfinite mask mismatch"
        torch.testing.assert_close(
            a.masked_fill(finite, 0), b.masked_fill(finite, 0), rtol=0, atol=0, equal_nan=True
        )
        a, b = a.masked_fill(~finite, 0), b.masked_fill(~finite, 0)
        error += ((a - b) ** 2).sum()
        energy += (a * a + b * b).sum()
    assert error <= bound * energy, f"normalized squared error: {error / energy}"


def logit_mask_validator(logits: torch.Tensor, near: torch.Tensor) -> Callable:
    """Compare a masked logit row, allowing the mask to differ only at the boundary.

    Args:
        logits: The unmasked logits the op was given.
        near: True where a row's value sits within the caller's margin of the
            threshold, which is where two correct implementations may disagree
            on whether an element is kept.

    Returns:
        A ``validate(got, expected)`` for a comparison that keeps the caller's
        margin out of the shared code.
    """

    def validate(got: torch.Tensor, expected: torch.Tensor) -> None:
        assert got.shape == expected.shape and got.dtype == expected.dtype
        assert got.dtype == logits.dtype, "masking changed the logit dtype"
        taken, kept = got != -float("inf"), expected != -float("inf")
        assert not ((taken ^ kept) & ~near).any(), "mask differs away from boundary"
        assert torch.equal(got[taken], logits[taken]), "retained logits changed"

    return validate


def assert_quantized(got: Any, expected: Any, scale_rtol: float = 1e-6) -> None:
    """Check scales and allow one adjacent FP8/INT8 code at rounding boundaries.

    The one-code allowance is the semantics of quantization, not a decision:
    two correct implementations rounding a value that sits on a boundary land
    on neighbouring codes. What the caller decides is how close the scales must
    be, which is *scale_rtol*.
    """
    for output, target in zip(got, expected, strict=True):
        assert output.shape == target.shape and output.dtype == target.dtype
        if output.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            step = (output.view(torch.uint8).int() - target.view(torch.uint8).int()).abs()
            assert step.max().item() <= 1, "FP8 quantization differs by more than one code"
            assert torch.isfinite(output.float()).all() and torch.isfinite(target.float()).all()
        elif output.dtype == torch.int8:
            assert (output.int() - target.int()).abs().max().item() <= 1, "INT8 code differs by > 1"
        else:
            torch.testing.assert_close(output, target, rtol=scale_rtol, atol=0)


@dataclasses.dataclass(frozen=True)
class NegativeControl:
    """A named, deliberately faulty computation evaluated against a row's oracle."""

    name: str
    run: Callable[[Callable, tuple], Any]


def zeroed_input(index: int, name: str) -> NegativeControl:
    """Drop one tensor input without changing the call's shape or dtype."""

    def run(reference: Callable, inputs: tuple) -> Any:
        changed = list(inputs)
        changed[index] = torch.zeros_like(changed[index])
        return reference(*changed)

    return NegativeControl(name, run)


@dataclasses.dataclass(frozen=True)
class Exact:
    """Compare every returned output with an independent reference.

    Explicit tolerances override dtype defaults; ``reference`` overrides the workload oracle."""

    rtol: Optional[float] = None
    atol: Optional[float] = None
    reference: Optional[Callable] = None
    controls: tuple[NegativeControl, ...] = ()
    kind: str = "exact"

    def tolerance(self, default: dict) -> dict:
        """Use explicit tolerances, or the dtype defaults."""
        named = {k: v for k, v in (("rtol", self.rtol), ("atol", self.atol)) if v is not None}
        return named or default


@dataclasses.dataclass(frozen=True)
class Partial:
    """Check the first ``outputs`` results; ``reason`` identifies unchecked outputs.

    For a reference that never produces the rest. One that produces them for
    some inputs and not others returns None in their place, which is decided
    per call rather than declared once.
    """

    outputs: int
    reason: str
    rtol: Optional[float] = None
    atol: Optional[float] = None
    reference: Optional[Callable] = None
    controls: tuple[NegativeControl, ...] = ()
    kind: str = "partial"

    def __post_init__(self) -> None:
        # Zero outputs keeps the ratio while nothing was compared.
        if self.outputs < 1:
            raise ValueError(
                f"Partial(outputs={self.outputs}) establishes nothing; a tag no reference "
                "reaches is Noncomparable or ReferenceInfeasible"
            )

    def tolerance(self, default: dict) -> dict:
        """Use explicit tolerances, or the dtype defaults."""
        named = {k: v for k, v in (("rtol", self.rtol), ("atol", self.atol)) if v is not None}
        return named or default


@dataclasses.dataclass(frozen=True)
class Unestablished:
    """No reference contract is available; publish timing without a ratio."""

    kind: str = "unestablished"


@dataclasses.dataclass(frozen=True)
class Custom:
    """Validate outputs against the oracle with a case-specific assertion function."""

    validator: Callable[[Any, Any], None]
    reason: str
    controls: tuple[NegativeControl, ...] = ()
    kind: str = "custom"


@dataclasses.dataclass(frozen=True)
class Noncomparable:
    """Different operator semantics; publish timing without a ratio."""

    reason: str
    kind: str = "noncomparable"


@dataclasses.dataclass(frozen=True)
class ReferenceInfeasible:
    """Reference unavailable for this case; record the missing capability and omit ratios."""

    missing: str
    reason: str
    kind: str = "reference_infeasible"


Evidence = Exact | Partial | Custom | Noncomparable | ReferenceInfeasible | Unestablished


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
    return "unestablished: no benchmark reference contract declared or available"
