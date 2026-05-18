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

    Each row is a mapping ``{input_name: dtype_token}`` where each value
    is either a concrete torch dtype token (``"float16"``) or a
    ``same_as(ref)`` expression naming another input in the same row.
    ``same_as`` tokens are resolved against their sibling within the row
    before tuple comparison (manifest.md R4/R6; see
    ``scripts/validate_manifest.py``'s ``check_l3_dtype_combos_data``).

    Returns:
        A list of normalized rows, or ``None`` when *combos* is absent.

    Raises:
        ValueError: when an entry is malformed, names an unknown input,
            or contains a ``same_as`` reference that cannot be resolved
            within the row (dangling sibling, cycle, or union expression).
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
        # First pass: type-check entries and capture raw tokens.
        raw: dict[str, str] = {}
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
            tok_stripped = tok.strip()
            if "|" in tok_stripped:
                raise ValueError(
                    f"{op_name}: signature.dtype_combos[{idx}][{name!r}] "
                    f"= {tok!r} — combo values must be a single concrete "
                    f"dtype, not a union"
                )
            raw[name] = tok_stripped
        # Second pass: resolve same_as references to siblings in the row.
        norm: dict[str, torch.dtype] = {}
        # Iterative resolution: a same_as may chain through several
        # siblings before reaching a concrete dtype. Bail on cycles.
        for name in raw:
            seen: list[str] = []
            cur = name
            while True:
                if cur in seen:
                    chain = " -> ".join(seen + [cur])
                    raise ValueError(
                        f"{op_name}: signature.dtype_combos[{idx}] has "
                        f"a same_as cycle ({chain})"
                    )
                seen.append(cur)
                tok = raw[cur]
                m = _SAME_AS_RE.match(tok)
                if not m:
                    dt = getattr(torch, tok, None)
                    if not isinstance(dt, torch.dtype):
                        raise ValueError(
                            f"{op_name}: signature.dtype_combos[{idx}]"
                            f"[{cur!r}] unknown dtype token {tok!r}"
                        )
                    norm[name] = dt
                    break
                ref = m.group(1)
                if ref not in raw:
                    raise ValueError(
                        f"{op_name}: signature.dtype_combos[{idx}]"
                        f"[{cur!r}] = {tok!r} references sibling "
                        f"{ref!r} which is not present in the same "
                        f"combo row"
                    )
                cur = ref
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
    # Validate every same_as(ref) names a sibling in the same signature.
    # Doing this at synthesis time turns typos into class-construction
    # errors instead of deferring them to a runtime fallback path.
    for name, (_concrete, refs, _dtype_str) in per_input.items():
        for ref in refs:
            if ref not in input_names:
                raise ValueError(
                    f"{op_name}: signature.inputs[{name!r}].dtype "
                    f"references same_as({ref}) but {ref!r} is not "
                    f"declared in signature.inputs"
                )
    combos = _parse_dtype_combos(
        op_name, sig.get("dtype_combos"), input_names,
    )
    # When dtype_combos is present, every row enumerates every declared
    # input (manifest validator R6, scripts/validate_manifest.py R6
    # combo-row completeness check). The observed combo key is built
    # over the full input_names tuple, with same_as-bound inputs included
    # at their resolved concrete dtype.
    combo_keys: set[tuple] | None = None
    if combos is not None:
        input_names_set = set(input_names)
        for idx, row in enumerate(combos):
            row_keys = set(row.keys())
            if row_keys != input_names_set:
                missing = input_names_set - row_keys
                extra = row_keys - input_names_set
                detail_parts: list[str] = []
                if missing:
                    detail_parts.append(f"missing {sorted(missing)!r}")
                if extra:
                    detail_parts.append(f"extra {sorted(extra)!r}")
                detail = "; ".join(detail_parts)
                raise ValueError(
                    f"{op_name}: signature.dtype_combos[{idx}] keys "
                    f"{sorted(row_keys)!r} do not cover every declared "
                    f"signature.inputs name {sorted(input_names_set)!r} "
                    f"({detail}); every combo row must enumerate every "
                    f"declared input"
                )
        combo_keys = {
            tuple(row[n] for n in input_names) for row in combos
        }

    # Generate the validator with explicit named parameters via ``exec``
    # so its native ``inspect.signature`` reports the manifest inputs and
    # no per-call ``inspect.Signature.bind`` is paid on the hot path.
    # ``_validate_dtypes`` is invoked on every ``forward()``; using a
    # ``**kwargs`` body with a wrapper that calls ``Signature.bind`` per
    # call adds measurable overhead.
    closure: dict[str, Any] = {
        "per_input": per_input,
        "input_names": input_names,
        "combo_keys": combo_keys,
        "ValueError": ValueError,
        "op_name": op_name,
    }
    params_src = ", ".join(input_names)
    src_lines = [
        f"def _validate_dtypes(self, {params_src}):",
        f'    """Synthesized from manifest signature for {op_name}."""',
        "    _locals = locals()",
        "    for _name in input_names:",
        "        _concrete, _refs, _dtype_str = per_input[_name]",
        "        _actual = _locals[_name].dtype",
        "        if _actual in _concrete:",
        "            continue",
        "        _matched = False",
        "        for _ref in _refs:",
        "            _ref_tensor = _locals.get(_ref)",
        "            if _ref_tensor is None:",
        "                raise ValueError(",
        "                    f\"{op_name}: input {_name!r} declares \"",
        "                    f\"same_as({_ref}) but {_ref!r} was not supplied\"",
        "                )",
        "            if _actual == _ref_tensor.dtype:",
        "                _matched = True",
        "                break",
        "        if _matched:",
        "            continue",
        "        raise ValueError(",
        "            f\"{op_name}: input {_name!r} has dtype {_actual}, \"",
        "            f\"expected {_dtype_str!r}\"",
        "        )",
        "    if combo_keys is not None:",
        "        _observed = tuple(_locals[_n].dtype for _n in input_names)",
        "        if _observed not in combo_keys:",
        "            _pairs = \", \".join(",
        "                f\"{_n}={_d}\" for _n, _d in zip(input_names, _observed)",
        "            )",
        "            raise ValueError(",
        "                f\"{op_name}: dtype combination ({_pairs}) is not \"",
        "                f\"listed in signature.dtype_combos\"",
        "            )",
    ]
    exec("\n".join(src_lines), closure)
    _validate_dtypes = closure["_validate_dtypes"]
    _validate_dtypes.__name__ = "_validate_dtypes"
    _validate_dtypes.__qualname__ = f"{op_name}._validate_dtypes"
    return _validate_dtypes


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
