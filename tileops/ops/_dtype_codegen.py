"""Synthesize ``_validate_dtypes`` bodies from manifest signatures.

The L1 ``Op`` base declares ``_validate_dtypes`` as a stub; per
docs/design/ops-design.md §Step 5 every ``status: implemented`` op must
override it from its manifest ``signature.inputs``.
``synthesize_validate_dtypes`` emits that body, and an
``Op.__init_subclass__`` hook (in ``tileops.ops.op_base``) installs it on
subclasses that supply no override.

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

from typing import Any, Callable

import torch

from tileops.manifest import try_load_entry
from tileops.manifest.dtype_rules import parse_tokens, same_as_ref


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
        ref = same_as_ref(tok)
        if ref is not None:
            refs.append(ref)
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
                ref = same_as_ref(tok)
                if ref is None:
                    dt = getattr(torch, tok, None)
                    if not isinstance(dt, torch.dtype):
                        raise ValueError(
                            f"{op_name}: signature.dtype_combos[{idx}]"
                            f"[{cur!r}] unknown dtype token {tok!r}"
                        )
                    norm[name] = dt
                    break
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
        tokens = parse_tokens(dtype_str)
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
    # R6 guarantees every combo row enumerates every declared input, so the
    # observed key spans all of input_names, same_as-bound ones resolved.
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

    # ``exec`` with explicit named params so ``inspect.signature`` reports the
    # manifest inputs natively. A ``**kwargs`` body would need a per-call
    # ``Signature.bind``, which is measurable on this ``forward()`` hot path.
    closure: dict[str, Any] = {
        "per_input": per_input,
        "input_names": input_names,
        "combo_keys": combo_keys,
        "ValueError": ValueError,
        "op_name": op_name,
    }
    params_src = ", ".join(input_names)
    # Unrolled per input, with each parameter referenced by name. A `locals()`
    # lookup or a loop over `input_names` would read the same values, but
    # `torch.compile` cannot trace `locals()`, and this body runs inside
    # `forward()` — so an op that validates dtypes would lose `fullgraph`.
    src_lines = [
        f"def _validate_dtypes(self, {params_src}):",
        f'    """Synthesized from manifest signature for {op_name}."""',
    ]
    for name in input_names:
        concrete, refs, dtype_str = per_input[name]
        closure[f"_concrete_{name}"] = frozenset(concrete)
        closure[f"_dtype_str_{name}"] = dtype_str
        src_lines.append(f"    _actual = {name}.dtype")
        src_lines.append(f"    if _actual not in _concrete_{name}:")
        # Every `same_as(ref)` was checked above to name a sibling input, so
        # each ref is in scope as a parameter here.
        if refs:
            cond = " or ".join(f"_actual == {r}.dtype" for r in refs)
            src_lines.append(f"        if not ({cond}):")
            indent = "            "
        else:
            indent = "        "
        src_lines += [
            f"{indent}raise ValueError(",
            f'{indent}    f"{{op_name}}: input {name!r} has dtype {{_actual}}, "',
            f"{indent}    f\"expected {{_dtype_str_{name}!r}}\"",
            f"{indent})",
        ]
    if combo_keys is not None:
        observed = ", ".join(f"{n}.dtype" for n in input_names)
        trailing = "," if len(input_names) == 1 else ""
        src_lines += [
            f"    _observed = ({observed}{trailing})",
            "    if _observed not in combo_keys:",
            "        _pairs = \", \".join(",
            "            f\"{_n}={_d}\" for _n, _d in zip(input_names, _observed)",
            "        )",
            "        raise ValueError(",
            "            f\"{op_name}: dtype combination ({_pairs}) is not \"",
            "            f\"listed in signature.dtype_combos\"",
            "        )",
        ]
    exec("\n".join(src_lines), closure)
    _validate_dtypes = closure["_validate_dtypes"]
    _validate_dtypes.__name__ = "_validate_dtypes"
    _validate_dtypes.__qualname__ = f"{op_name}._validate_dtypes"
    return _validate_dtypes




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
      ``__dict__`` (manual overrides are honored verbatim). Note this
      differs from ``_roofline_codegen.maybe_install_eval_roofline``,
      which honors an override anywhere above L1 in the MRO: a manual
      ``_validate_dtypes`` on an intermediate family base is shadowed by
      the synthesized one, so bind it in the concrete class body.
    - The manifest signature has a non-empty ``inputs`` mapping the
      codegen recognizes.
    """
    if "_validate_dtypes" in cls.__dict__:
        return

    sig = getattr(cls, "__manifest_signature__", None)
    status = getattr(cls, "__manifest_status__", None)
    if sig is None or status is None:
        entry = try_load_entry(cls.__name__)
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
