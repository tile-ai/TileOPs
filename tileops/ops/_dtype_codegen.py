"""Synthesize ``_validate_dtypes`` bodies from manifest signatures.

The L1 ``Op`` base declares ``_validate_dtypes`` as a staged-rollout stub
that raises ``NotImplementedError``. Per ``docs/design/ops-design.md``
§Step 5, every concrete op with ``status: implemented`` must override
the stub with a body derived from its manifest ``signature.inputs``
dtype unions and ``same_as`` references.

This module provides a single codegen entry point — ``synthesize_validate_dtypes``
— that emits an equivalent function from the manifest signature, and an
``Op.__init_subclass__`` hook (installed in ``tileops.ops.op_base``) that
auto-applies the generated method to subclasses that do not supply their
own override.

Manifest constructs handled:

- Plain dtype tokens (``"float16"``).
- Pipe-separated unions (``"float16 | bfloat16 | float32"``).
- ``same_as(ref)`` — the input's dtype must equal ``ref``'s dtype.
- ``same_as`` inside a union (e.g. ``"float32 | same_as(input)"``) — accept
  the listed concrete tokens or the ref's dtype.

The synthesized function raises ``ValueError`` on any deviation. Its
keyword-argument names mirror ``signature.inputs`` so the manifest
validator's parity probes (which bind via ``inspect.signature``) work
unchanged.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import torch

_SAME_AS_RE = re.compile(r"^same_as\(\s*(\w+)\s*\)$")


def _parse_tokens(dtype_str: str) -> list[str]:
    """Split a dtype expression into ``|``-separated tokens."""
    return [t.strip() for t in dtype_str.split("|") if t.strip()]


def _classify_tokens(
    tokens: list[str],
) -> tuple[list[torch.dtype], list[str]]:
    """Partition tokens into concrete torch dtypes and ``same_as`` refs.

    Returns:
        (concrete_dtypes, same_as_refs)
    """
    concrete: list[torch.dtype] = []
    refs: list[str] = []
    for tok in tokens:
        m = _SAME_AS_RE.match(tok)
        if m:
            refs.append(m.group(1))
            continue
        dt = getattr(torch, tok, None)
        if not isinstance(dt, torch.dtype):
            raise ValueError(
                f"unknown dtype token {tok!r} in manifest signature"
            )
        concrete.append(dt)
    return concrete, refs


def _parse_dtype_combos(
    op_name: str,
    combos: Any,
    input_names: list[str],
) -> list[dict[str, torch.dtype]] | None:
    """Validate and normalize ``signature.dtype_combos`` rows.

    Each row is a mapping ``{input_name: dtype_token}`` listing the
    concrete dtype for combo-axis inputs. Rows MUST NOT reference inputs
    declared via ``same_as(ref)`` (per manifest R6: such inputs do not
    contribute an independent axis to the combo space).

    Returns:
        A list of normalized rows, or ``None`` when *combos* is absent.

    Raises:
        ValueError: when an entry is malformed or names an unknown input.
    """
    if combos is None:
        return None
    if not isinstance(combos, list) or not combos:
        raise ValueError(
            f"{op_name}: signature.dtype_combos must be a non-empty list "
            f"when present"
        )
    normalized: list[dict[str, torch.dtype]] = []
    for idx, row in enumerate(combos):
        if not isinstance(row, dict) or not row:
            raise ValueError(
                f"{op_name}: signature.dtype_combos[{idx}] must be a "
                f"non-empty mapping"
            )
        norm: dict[str, torch.dtype] = {}
        for name, tok in row.items():
            if name not in input_names:
                raise ValueError(
                    f"{op_name}: signature.dtype_combos[{idx}] references "
                    f"unknown input {name!r}"
                )
            if not isinstance(tok, str):
                raise ValueError(
                    f"{op_name}: signature.dtype_combos[{idx}][{name!r}] "
                    f"must be a string dtype token"
                )
            if _SAME_AS_RE.match(tok.strip()):
                raise ValueError(
                    f"{op_name}: signature.dtype_combos[{idx}][{name!r}] "
                    f"may not use same_as(...); list concrete dtypes only"
                )
            dt = getattr(torch, tok.strip(), None)
            if not isinstance(dt, torch.dtype):
                raise ValueError(
                    f"{op_name}: signature.dtype_combos[{idx}][{name!r}] "
                    f"unknown dtype token {tok!r}"
                )
            norm[name] = dt
        normalized.append(norm)
    return normalized


def synthesize_validate_dtypes(
    op_name: str, sig: dict[str, Any],
) -> Callable[..., None]:
    """Build a ``_validate_dtypes`` function from a manifest signature.

    Args:
        op_name: Manifest op name; used in error messages.
        sig: The ``signature`` block from the manifest entry. Must contain
            an ``inputs`` mapping; each entry's ``dtype`` is the union
            expression to enforce. May also contain ``dtype_combos`` —
            when present, it is the exhaustive list of accepted
            cross-tensor dtype rows (per ``docs/design/manifest.md`` R6).

    Returns:
        A function with signature ``(self, **inputs) -> None`` that
        raises ``ValueError`` when any input's dtype lies outside the
        declared union, when a ``same_as(ref)`` constraint is violated,
        or when ``dtype_combos`` is present and the observed combo row
        is not listed.
    """
    inputs = sig.get("inputs") or {}
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError(
            f"{op_name}: signature.inputs is missing or empty; cannot "
            f"synthesize _validate_dtypes"
        )

    # Pre-parse every input's dtype expression so the generated body
    # does minimal work on the hot path.
    per_input: dict[str, tuple[list[torch.dtype], list[str], str]] = {}
    for name, attrs in inputs.items():
        if not isinstance(attrs, dict):
            raise ValueError(
                f"{op_name}: signature.inputs[{name!r}] must be a mapping"
            )
        dtype_str = attrs.get("dtype", "")
        tokens = _parse_tokens(dtype_str)
        if not tokens:
            raise ValueError(
                f"{op_name}: signature.inputs[{name!r}].dtype is empty"
            )
        concrete, refs = _classify_tokens(tokens)
        per_input[name] = (concrete, refs, dtype_str)

    input_names = list(inputs.keys())
    combos = _parse_dtype_combos(
        op_name, sig.get("dtype_combos"), input_names,
    )
    # When dtype_combos is present, freeze each row to a hashable key
    # over the combo-axis inputs. ``same_as`` inputs do not contribute
    # an axis (R6); they are still validated by the per-input loop.
    combo_keys: set[tuple] | None = None
    combo_axis_names: list[str] | None = None
    if combos is not None:
        # Combo axes are the union of names that appear in any row.
        seen: list[str] = []
        for row in combos:
            for n in row:
                if n not in seen:
                    seen.append(n)
        combo_axis_names = seen
        combo_keys = {
            tuple(row.get(n) for n in combo_axis_names) for row in combos
        }

    def _validate_dtypes(self, **kwargs: torch.Tensor) -> None:  # noqa: D401
        # Verify all manifest-declared inputs are present.
        for name in input_names:
            if name not in kwargs:
                raise TypeError(
                    f"{op_name}._validate_dtypes() missing required "
                    f"keyword argument {name!r}"
                )
        for name in input_names:
            concrete, refs, dtype_str = per_input[name]
            t = kwargs[name]
            actual = t.dtype
            if actual in concrete:
                continue
            # Resolve ``same_as`` refs against the actual tensors passed.
            matched_ref = False
            for ref in refs:
                ref_tensor = kwargs.get(ref)
                if ref_tensor is None:
                    raise ValueError(
                        f"{op_name}: input {name!r} declares "
                        f"same_as({ref}) but {ref!r} was not supplied"
                    )
                if actual == ref_tensor.dtype:
                    matched_ref = True
                    break
            if matched_ref:
                continue
            raise ValueError(
                f"{op_name}: input {name!r} has dtype {actual}, "
                f"expected {dtype_str!r}"
            )
        # dtype_combos, when present, is exhaustive: only the listed
        # cross-tensor combinations are valid.
        if combo_keys is not None and combo_axis_names is not None:
            observed = tuple(
                kwargs[n].dtype for n in combo_axis_names
            )
            if observed not in combo_keys:
                pairs = ", ".join(
                    f"{n}={d}"
                    for n, d in zip(
                        combo_axis_names, observed, strict=True,
                    )
                )
                raise ValueError(
                    f"{op_name}: dtype combination ({pairs}) is not "
                    f"listed in signature.dtype_combos"
                )

    _validate_dtypes.__name__ = "_validate_dtypes"
    _validate_dtypes.__qualname__ = f"{op_name}._validate_dtypes"
    _validate_dtypes.__doc__ = (
        f"Synthesized from manifest signature for {op_name}."
    )
    # Rebuild the function so its inspect.signature reflects the
    # manifest's named inputs as keyword-only parameters. The manifest
    # validator probes via kwargs only.
    _validate_dtypes = _rebuild_with_named_kwargs(
        _validate_dtypes, input_names,
    )
    return _validate_dtypes


def _rebuild_with_named_kwargs(
    fn: Callable[..., None], input_names: list[str],
) -> Callable[..., None]:
    """Wrap *fn* so its ``inspect.signature`` exposes ``input_names`` as
    keyword-or-positional parameters.

    The original implementation accepts ``**kwargs`` for simplicity; the
    validator binds via ``inspect.signature(...).bind(self, **tensors)``,
    which already succeeds with a ``**kwargs`` body. Exposing the named
    parameters also lets callers introspect the manifest contract.
    """
    import inspect

    params = [
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]
    for n in input_names:
        params.append(
            inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
    new_sig = inspect.Signature(parameters=params, return_annotation=None)

    def wrapper(self, *args, **kwargs):  # noqa: ANN001, ANN002
        bound = new_sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        call_kwargs = {n: bound.arguments[n] for n in input_names}
        return fn(self, **call_kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = fn.__qualname__
    wrapper.__doc__ = fn.__doc__
    wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper


def _lookup_manifest_entry(op_name: str) -> dict[str, Any] | None:
    """Return the manifest entry for *op_name* or None if absent.

    Lazy-imports the manifest loader to avoid pulling YAML I/O in the
    ``Op`` base import path when no subclass has been declared yet.
    Failures (missing key, load errors) downgrade to ``None`` so an
    incomplete manifest never blocks op-class construction.
    """
    try:
        from tileops.manifest import load_manifest
    except Exception:  # noqa: BLE001
        return None
    try:
        ops = load_manifest()
    except Exception:  # noqa: BLE001
        return None
    entry = ops.get(op_name)
    if not isinstance(entry, dict):
        return None
    return entry


def maybe_install_validator(cls: type) -> None:
    """Install a synthesized ``_validate_dtypes`` on *cls* when warranted.

    Resolution order for the manifest source:

    1. Class-attached ``__manifest_signature__`` + ``__manifest_status__``
       (used by unit tests and by callers that want to bypass the YAML
       loader).
    2. Manifest entry whose key matches ``cls.__name__``.

    Conditions for installation:

    - Resolved status is ``"implemented"`` (spec-only entries
      intentionally leave the L1 stub in place).
    - The class did not already define ``_validate_dtypes`` in its own
      ``__dict__`` (manual overrides are honored verbatim).
    - The manifest signature has a non-empty ``inputs`` mapping the
      codegen recognizes.
    """
    if "_validate_dtypes" in cls.__dict__:
        return

    sig = getattr(cls, "__manifest_signature__", None)
    status = getattr(cls, "__manifest_status__", None)
    if sig is None or status is None:
        entry = _lookup_manifest_entry(cls.__name__)
        if entry is None:
            return
        sig = entry.get("signature")
        status = entry.get("status")
    if status != "implemented":
        return
    if not isinstance(sig, dict):
        return
    try:
        fn = synthesize_validate_dtypes(cls.__name__, sig)
    except ValueError:
        # Manifest signature too irregular to synthesize from; leave the
        # base stub in place rather than mask the gap.
        return
    cls._validate_dtypes = fn  # type: ignore[assignment]
