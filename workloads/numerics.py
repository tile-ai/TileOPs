"""The shared correctness protocol for workloads, tests and benchmarks.

Workloads declare Evidence. Only compare_outputs interprets it; verify executes
callables with isolated inputs and uses that same comparison for negative controls.
Consumers record CheckResult and own pytest/timing policy, never numerical policy.
"""

import dataclasses
from typing import Any, Callable, Optional

import torch

__all__ = [
    "CheckResult",
    "Custom",
    "Evidence",
    "Exact",
    "NegativeControl",
    "Noncomparable",
    "Partial",
    "ReferenceInfeasible",
    "Request",
    "Unestablished",
    "assert_close",
    "assert_normalized_error",
    "assert_quantized",
    "assert_rounded",
    "check_inputs_preserved",
    "compare_outputs",
    "describe",
    "logit_mask_validator",
    "reference_tolerance",
    "verify",
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
        passed = taken & ~logits.isnan()
        bits = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
            logits.element_size()
        ]
        assert torch.equal(got[passed].view(bits), logits[passed].view(bits)), (
            "retained logits changed"
        )
        assert got[taken & logits.isnan()].isnan().all(), "retained NaN changed"

    return validate


def assert_quantized(got: Any, expected: Any, scale_rtol: float = 1e-6) -> None:
    """Check a tensor or output sequence, allowing one adjacent FP8/INT8 code.

    The one-code allowance is the semantics of quantization, not a decision:
    two correct implementations rounding a value that sits on a boundary land
    on neighbouring codes. What the caller decides is how close the scales must
    be, which is *scale_rtol*.
    """
    outputs = got if isinstance(got, (tuple, list)) else (got,)
    targets = expected if isinstance(expected, (tuple, list)) else (expected,)
    for output, target in zip(outputs, targets, strict=True):
        assert output.shape == target.shape and output.dtype == target.dtype
        if output.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            step = (output.view(torch.uint8).int() - target.view(torch.uint8).int()).abs()
            assert not step.numel() or step.max().item() <= 1, (
                "FP8 quantization differs by more than one code"
            )
            assert torch.isfinite(output.float()).all() and torch.isfinite(target.float()).all()
        elif output.dtype == torch.int8:
            step = (output.int() - target.int()).abs()
            assert not step.numel() or step.max().item() <= 1, "INT8 code differs by > 1"
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

    Explicit tolerances override dtype defaults. ``normalized`` adds a bound on each
    output's normalized squared error, for an atol scaled past the output itself."""

    rtol: Optional[float] = None
    atol: Optional[float] = None
    controls: tuple[NegativeControl, ...] = ()
    normalized: Optional[float] = None
    kind: str = "exact"

    def tolerance(self, default: dict) -> dict:
        """Use explicit tolerances, or the dtype defaults."""
        named = {k: v for k, v in (("rtol", self.rtol), ("atol", self.atol)) if v is not None}
        return {"rtol": 0.0, "atol": 0.0} | named if named else default


@dataclasses.dataclass(frozen=True)
class Partial:
    """Check the first ``outputs`` results; ``reason`` identifies unchecked outputs.

    Use this when the reference does not establish the remaining outputs.
    Workloads may choose the prefix per call; None never means unchecked.
    """

    outputs: int
    reason: str
    rtol: Optional[float] = None
    atol: Optional[float] = None
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
        return {"rtol": 0.0, "atol": 0.0} | named if named else default


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
    probe: Optional[Callable[[Callable, tuple], None]] = None
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


@dataclasses.dataclass(frozen=True)
class CheckResult:
    """Completed comparisons, independent of pytest, Op attribution or timing."""

    checked_outputs: int
    total_outputs: int | None
    max_abs_err: float | None
    unchecked_reason: str | None = None


def reference_tolerance(dtype: torch.dtype) -> dict[str, float]:
    """One default table, applied to each output's dtype independently."""
    tol = {
        torch.float16: 1e-3,
        torch.bfloat16: 1.6e-2,
        torch.float32: 1e-5,
        torch.float64: 1e-7,
        torch.complex64: 1e-5,
        torch.complex128: 1e-7,
    }.get(dtype, 0.0)
    return {"rtol": tol, "atol": tol}


def _outputs(value: Any) -> tuple:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _tensor_pairs(got: Any, expected: Any):
    """Validate structure before any numerical strategy can accept an output."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(got, torch.Tensor), "expected a tensor output"
        assert got.shape == expected.shape, f"shape mismatch: {got.shape} != {expected.shape}"
        assert got.dtype == expected.dtype, f"dtype mismatch: {got.dtype} != {expected.dtype}"
        assert got.device == expected.device, f"device mismatch: {got.device} != {expected.device}"
        yield got, expected
    elif isinstance(expected, dict):
        assert isinstance(got, dict) and got.keys() == expected.keys(), "mapping outputs differ"
        for key in expected:
            yield from _tensor_pairs(got[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert isinstance(got, (tuple, list)) and len(got) == len(expected), (
            "output structure mismatch"
        )
        for a, b in zip(got, expected, strict=True):
            yield from _tensor_pairs(a, b)
    elif expected is None:
        assert got is None, "None is an output value, not an unchecked tensor"
    else:
        assert got == expected, f"non-tensor output mismatch: {got!r} != {expected!r}"


def compare_outputs(produced: Any, expected: Any, evidence: Evidence) -> CheckResult:
    """The only interpreter of output coverage, structure and numerical policy.

    Exact checks every output. Partial explicitly checks a prefix. Custom receives
    the complete result after structural checks; it cannot bypass those checks.
    None is an actual output value, not an implicit permission to omit validation.
    """
    if evidence.kind not in ("exact", "partial", "custom"):
        return CheckResult(0, None, None, describe(evidence))
    outputs, targets = _outputs(produced), _outputs(expected)
    width = len(outputs)
    claimed = evidence.outputs if isinstance(evidence, Partial) else width
    if not width or not targets:
        raise ValueError("the reference or implementation returned no outputs")
    if isinstance(evidence, Partial):
        if claimed > min(width, len(targets)):
            raise ValueError(
                f"claims {claimed} outputs, implementation returns {width} and reference {len(targets)}"
            )
    elif width != len(targets):
        raise ValueError(
            f"reference establishes {len(targets)} of {width} outputs; declare Partial explicitly"
        )
    pairs = [
        list(_tensor_pairs(a, b)) for a, b in zip(outputs[:claimed], targets[:claimed], strict=True)
    ]
    if isinstance(evidence, Custom):
        evidence.validator(produced, expected)
    else:
        for output_pairs in pairs:
            for got, target in output_pairs:
                torch.testing.assert_close(
                    got,
                    target,
                    equal_nan=True,
                    **evidence.tolerance(reference_tolerance(got.dtype)),
                )
                if getattr(evidence, "normalized", None) is not None:
                    assert_normalized_error(got, target, evidence.normalized)
    # Metrics are diagnostics computed only after all comparisons succeed.
    maximum = 0.0
    checked = 0
    for output_pairs in pairs:
        if not output_pairs:
            continue
        checked += 1
        for got, target in output_pairs:
            if not got.numel():
                continue
            dtype = torch.complex128 if got.is_complex() else torch.float64
            for a, b in zip(
                got.reshape(-1).split(1 << 20), target.reshape(-1).split(1 << 20), strict=True
            ):
                a, b = a.to(dtype), b.to(dtype)
                finite = torch.isfinite(a) & torch.isfinite(b)
                if finite.any():
                    maximum = max(maximum, (a[finite] - b[finite]).abs().max().item())
    reason = describe(evidence) if isinstance(evidence, Partial) else None
    if not checked:
        reason = "no numerical output was compared"
    return CheckResult(checked, width, maximum if checked else None, reason)


def _tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def _copy_outputs(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_copy_outputs(v) for v in value)
    if isinstance(value, list):
        return [_copy_outputs(v) for v in value]
    if isinstance(value, dict):
        return {k: _copy_outputs(v) for k, v in value.items()}
    return value


@dataclasses.dataclass(frozen=True)
class Request:
    """One call to check against the reference: ``reset()``, then ``run(*args)``.

    ``preserve_inputs`` marks a call made on the shared inputs themselves, which it must
    leave as it found them.
    """

    run: Callable
    args: tuple
    reset: Optional[Callable[[], None]] = None
    preserve_inputs: bool = False


def _snapshot(tensors: list) -> list:
    return [t.detach().clone() for t in tensors]


def _restore(tensors: list, snapshot: list) -> None:
    with torch.no_grad():
        for tensor, source in zip(tensors, snapshot, strict=True):
            target = tensor
            # Expanded read-only inputs share storage along zero-stride axes.
            # Restore each stored element once, preserving the original view.
            for axis in range(tensor.ndim - 1, -1, -1):
                if tensor.stride(axis) == 0 and tensor.shape[axis] > 1:
                    target, source = target.select(axis, 0), source.select(axis, 0)
            target.copy_(source)


def _unchanged(tensors: list, snapshot: list) -> bool:
    try:
        for tensor, source in zip(tensors, snapshot, strict=True):
            torch.testing.assert_close(tensor, source, rtol=0, atol=0, equal_nan=True)
    except AssertionError:
        return False
    return True


def _unique(value: Any) -> list:
    return list({id(t): t for t in _tensors(value)}.values())


def check_inputs_preserved(request: Request, inputs: tuple) -> None:
    """Check an unverified call's shared-input contract without judging its output."""
    shared = _unique(inputs)
    pristine = _snapshot(shared)
    own = [t for t in _unique(request.args) if all(t is not s for s in shared)]
    own_pristine = _snapshot(own)
    try:
        if request.reset is not None:
            request.reset()
        with torch.no_grad():
            request.run(*request.args)
        if not _unchanged(shared, pristine):
            raise AssertionError(
                "overwrote the shared inputs; give it private arguments and a reset"
            )
    finally:
        _restore(shared, pristine)
        _restore(own, own_pristine)


def _check_requests_preserve(requests: dict[str, Request], inputs: tuple) -> None:
    """Run each request that must leave the shared inputs unchanged, naming the one that did not."""
    for name, request in requests.items():
        if request.preserve_inputs:
            try:
                check_inputs_preserved(request, inputs)
            except AssertionError as exc:
                raise AssertionError(f"{name}: {exc}") from exc


def verify(
    reference: Callable | None,
    inputs: tuple,
    *,
    evidence: Evidence,
    requests: dict[str, Request],
    preserve_reference_inputs: bool = False,
) -> dict[str, CheckResult]:
    """Check every request against one run of the reference, restoring inputs even on failure.

    The reference runs once and every request is compared with its copied result, which
    stays allocated while the requests run so no request's output can be carved from
    memory that already holds the expected values. Negative controls run once; a Custom
    probe takes the call under test, so it runs once per request. A failure names its
    request. Reference OOM establishes nothing; request failures always propagate.
    """
    if evidence.kind not in ("exact", "partial", "custom"):
        if preserve_reference_inputs and reference is not None:
            try:
                check_inputs_preserved(Request(reference, inputs), inputs)
            except AssertionError as exc:
                raise AssertionError(f"reference: {exc}") from exc
        _check_requests_preserve(requests, inputs)
        return {name: CheckResult(0, None, None, describe(evidence)) for name in requests}
    if reference is None:
        raise ValueError(
            "a checked workload must supply ref_program; declare Unestablished explicitly"
        )
    shared = _unique(inputs)
    pristine = _snapshot(shared)
    try:
        try:
            produced_by_reference = reference(*inputs)
            expected = _copy_outputs(produced_by_reference)
        except torch.OutOfMemoryError:
            # Nothing is compared, but every request is still timed on the shared inputs.
            _restore(shared, pristine)
            _check_requests_preserve(requests, inputs)
            return {
                name: CheckResult(0, None, None, "reference ran out of memory") for name in requests
            }
        if preserve_reference_inputs and not _unchanged(shared, pristine):
            raise AssertionError(
                "reference overwrote the shared inputs; time it on private arguments"
            )
        _restore(shared, pristine)
        results = {}
        for name, request in requests.items():
            own = [t for t in _unique(request.args) if all(t is not s for s in shared)]
            own_pristine = _snapshot(own)
            try:
                if request.reset is not None:
                    request.reset()
                with torch.no_grad():
                    produced = _copy_outputs(request.run(*request.args))
                if request.preserve_inputs and not _unchanged(shared, pristine):
                    raise AssertionError(
                        "overwrote the shared inputs; give it private arguments and a reset"
                    )
                results[name] = compare_outputs(produced, expected, evidence)
                del produced
                if isinstance(evidence, Custom) and evidence.probe is not None:
                    evidence.probe(request.run, request.args)
            except (AssertionError, ValueError) as exc:
                raise type(exc)(f"{name}: {exc}") from exc
            finally:
                _restore(shared, pristine)
                _restore(own, own_pristine)
                own_pristine.clear()
        del produced_by_reference
        for control in evidence.controls:
            faulty = _copy_outputs(control.run(reference, inputs))
            _restore(shared, pristine)
            try:
                compare_outputs(faulty, expected, evidence)
            except AssertionError:
                pass
            else:
                raise ValueError(f"negative control {control.name!r} was accepted")
        return results
    finally:
        _restore(shared, pristine)
        # A caught negative-control failure keeps this frame in a reference cycle
        # until the next gc pass; drop the snapshots now so they never pile up.
        pristine.clear()
        shared.clear()


def assert_rounded(
    got: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float = 0
) -> None:
    """Allow one adjacent stored value, then enforce the arithmetic error bound.

    Independent reductions can straddle a half/bfloat rounding boundary even when
    their FP32 error is smaller than a storage step. This allowance applies only
    to the narrowed output; callers verify FP32 state without it.
    """
    adjacent = torch.nextafter(expected, got)
    torch.testing.assert_close(got.float(), adjacent.float(), atol=atol, rtol=rtol, equal_nan=True)


def assert_close(got: Any, expected: Any, *, atol: float, rtol: float) -> None:
    """A numerical strategy for an explicitly bounded complete output structure."""
    for a, b in _tensor_pairs(got, expected):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=True)
