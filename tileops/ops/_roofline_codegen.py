"""Synthesize ``eval_roofline`` bodies from manifest ``roofline`` entries.

The L1 ``Op`` base declares ``eval_roofline`` as a staged-rollout stub
that raises ``NotImplementedError``. Per ``docs/design/roofline.md`` §4.4,
every concrete op with ``status: implemented`` must override the stub
with a body derived from its manifest ``roofline`` block.

This module provides:

- ``synthesize_eval_roofline`` — emit an ``eval_roofline`` function from
  a manifest ``roofline`` block (inline or func mode).
- ``maybe_install_eval_roofline`` — ``Op.__init_subclass__`` hook that
  auto-applies the generated method when the subclass advertises the
  manifest metadata and does not supply its own override.

The two roofline modes follow ``docs/design/roofline.md`` §2.2 and §4.4.2:

- **Func** — ``roofline.func`` points at ``module.path.callable``. The
  emitted body is ``return <func>(self)``. Codegen resolves the dotted
  path at synthesis time so a typo fails class construction rather than
  the first benchmark call.
- **Inline** — ``roofline.vars`` (optional) + ``flops`` + ``bytes`` are
  Python expression strings. Codegen emits two sequential blocks: a vars
  block that binds tensor inputs, signature params, ``elem_bytes``, and
  any explicit ``vars`` entries, and a return block that evaluates the
  ``flops`` / ``bytes`` expressions over those locals.

The L1 stub is preserved for ``status: spec-only`` entries — codegen
re-evaluates them once the status flips.
"""

from __future__ import annotations

import importlib
from math import prod
from typing import Any, Callable


def _resolve_func_path(path: str) -> Callable[..., Any]:
    """Resolve ``module.path.callable`` to a Python callable.

    Raises ``ValueError`` if the module or attribute is absent — codegen
    is the authoritative gate for ``func`` correctness
    (``docs/design/roofline.md`` §4.4).
    """
    if not isinstance(path, str) or "." not in path:
        raise ValueError(
            f"roofline.func must be a dotted module.attr path, got {path!r}"
        )
    mod_path, _, attr = path.rpartition(".")
    try:
        mod = importlib.import_module(mod_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"cannot resolve roofline.func {path!r}: import {mod_path!r} "
            f"failed ({exc})"
        ) from exc
    fn = getattr(mod, attr, None)
    if not callable(fn):
        raise ValueError(
            f"cannot resolve roofline.func {path!r}: {attr!r} is not a "
            f"callable on {mod_path!r}"
        )
    return fn


# Names bound into the vars-layer evaluation namespace per
# ``docs/design/roofline.md`` §4.4.4. Shared with the arithmetic layer
# (a strict subset is sufficient there but the same dict is reused).
def _vars_namespace() -> dict[str, Any]:
    import math
    return {
        "product": prod,
        "isinstance": isinstance,
        "len": len,
        "set": set,
        "tuple": tuple,
        "list": list,
        "range": range,
        "int": int,
        "float": float,
        "bool": bool,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "log2": math.log2,
        "ceil": math.ceil,
        "floor": math.floor,
    }


def _synthesize_func_mode(
    op_name: str, func_path: str,
) -> Callable[..., tuple[int, int]]:
    """Build an ``eval_roofline`` that delegates to a human-authored func.

    Codegen resolves the dotted path eagerly so the closure captures the
    callable directly — subsequent ``op.eval_roofline()`` calls skip the
    import machinery on the hot path. Per ``docs/design/roofline.md``
    §4.4.2, the emitted body is ``return <func>(self)``; if the author
    writes a non-``func(op)`` signature, the resulting TypeError surfaces
    to the caller as designed.
    """
    fn = _resolve_func_path(func_path)

    def eval_roofline(self):
        return fn(self)

    eval_roofline.__name__ = "eval_roofline"
    eval_roofline.__qualname__ = f"{op_name}.eval_roofline"
    eval_roofline.__doc__ = (
        f"Synthesized from manifest roofline.func={func_path!r}."
    )
    return eval_roofline


def _synthesize_inline_mode(
    op_name: str,
    roofline: dict[str, Any],
    signature: dict[str, Any] | None,
) -> Callable[..., tuple[int, int]]:
    """Build an ``eval_roofline`` from inline ``flops`` / ``bytes`` exprs.

    The emitted body materializes the manifest's two-layer contract
    (``docs/design/roofline.md`` §4.4.3):

    - Bind every ``signature.inputs`` name as a local (read from
      ``self.<name>``). Arbitrary-rank ops are expected to have bound
      these in ``forward()`` before ``eval_roofline()`` runs.
    - Bind every ``signature.params`` name as a local (``self.<name>``).
    - Bind ``elem_bytes`` from ``self.dtype.itemsize``.
    - Evaluate each ``vars`` entry in declaration order; each may
      reference earlier locals.
    - Return ``(<flops>, <bytes>)`` over the resolved namespace.

    Expressions are evaluated with ``eval`` against a namespace built
    from the §4.4.4 helper table. Codegen does not parse the expression
    strings — the manifest validator owns structural checks.
    """
    flops_expr = roofline.get("flops")
    bytes_expr = roofline.get("bytes")
    if not isinstance(flops_expr, str) or not isinstance(bytes_expr, str):
        raise ValueError(
            f"{op_name}: inline-mode roofline must declare both "
            f"flops and bytes as strings"
        )
    vars_block = roofline.get("vars") or {}
    if not isinstance(vars_block, dict):
        raise ValueError(
            f"{op_name}: roofline.vars must be a mapping when present"
        )
    sig = signature or {}
    input_names = list((sig.get("inputs") or {}).keys())
    param_names = list((sig.get("params") or {}).keys())

    # Pre-compile each expression to bytecode so the hot path does no
    # parsing. The codegen layer remains the authoritative form gate
    # because compilation errors fail class construction.
    helpers = _vars_namespace()
    try:
        vars_compiled = {
            name: compile(expr, f"<{op_name}.roofline.vars.{name}>", "eval")
            for name, expr in vars_block.items()
        }
    except SyntaxError as exc:
        raise ValueError(
            f"{op_name}: roofline.vars contains an unparsable expression "
            f"({exc})"
        ) from exc
    try:
        flops_compiled = compile(
            flops_expr, f"<{op_name}.roofline.flops>", "eval",
        )
        bytes_compiled = compile(
            bytes_expr, f"<{op_name}.roofline.bytes>", "eval",
        )
    except SyntaxError as exc:
        raise ValueError(
            f"{op_name}: roofline.flops/bytes unparsable ({exc})"
        ) from exc

    def eval_roofline(self):
        ns: dict[str, Any] = dict(helpers)
        # Bind tensor inputs and signature params from the op instance.
        # ``getattr`` is used directly so a missing binding surfaces as
        # an AttributeError naming the manifest input — preferable to a
        # NameError swallowed inside ``eval``.
        for n in input_names:
            ns[n] = getattr(self, n, None)
        for n in param_names:
            if hasattr(self, n):
                ns[n] = getattr(self, n)
        ns["elem_bytes"] = self.dtype.itemsize
        for name, code in vars_compiled.items():
            ns[name] = eval(code, ns)  # noqa: S307
        flops = eval(flops_compiled, ns)  # noqa: S307
        nbytes = eval(bytes_compiled, ns)  # noqa: S307
        return int(flops), int(nbytes)

    eval_roofline.__name__ = "eval_roofline"
    eval_roofline.__qualname__ = f"{op_name}.eval_roofline"
    eval_roofline.__doc__ = (
        f"Synthesized from manifest inline roofline for {op_name}."
    )
    return eval_roofline


def synthesize_eval_roofline(
    op_name: str,
    *,
    roofline: dict[str, Any] | None,
    signature: dict[str, Any] | None,
) -> Callable[..., tuple[int, int]]:
    """Build an ``eval_roofline`` function from a manifest roofline block.

    Args:
        op_name: Manifest op name; used in error messages and __qualname__.
        roofline: The ``roofline`` block from the manifest entry.
        signature: The ``signature`` block; consumed for inline-mode
            input/param bindings. May be ``None`` for func mode.

    Returns:
        A method-shaped callable ``eval_roofline(self) -> tuple[int, int]``.

    Raises:
        ValueError: When ``roofline`` is absent or malformed (missing
            both modes, mixing both modes, unresolvable func path,
            inline missing ``flops``/``bytes``).
    """
    if not isinstance(roofline, dict) or not roofline:
        raise ValueError(
            f"{op_name}: manifest roofline is missing or empty; cannot "
            f"synthesize eval_roofline"
        )
    has_func = "func" in roofline
    has_inline = "flops" in roofline or "bytes" in roofline or "vars" in roofline
    if has_func and has_inline:
        raise ValueError(
            f"{op_name}: roofline cannot mix func and inline modes"
        )
    if has_func:
        return _synthesize_func_mode(op_name, roofline["func"])
    return _synthesize_inline_mode(op_name, roofline, signature)


def _lookup_manifest_entry(op_name: str) -> dict[str, Any] | None:
    """Return the manifest entry for *op_name* or ``None`` if absent."""
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


def maybe_install_eval_roofline(cls: type) -> None:
    """Install a synthesized ``eval_roofline`` on *cls* when warranted.

    Resolution order mirrors ``_dtype_codegen.maybe_install_validator``:

    1. Class-attached ``__manifest_roofline__`` + ``__manifest_status__``
       + ``__manifest_signature__`` (used by unit tests and bypass paths).
    2. Manifest entry whose key matches ``cls.__name__``.

    Conditions for installation:

    - Resolved status is ``"implemented"``.
    - The class did not already define ``eval_roofline`` in its own
      ``__dict__`` (manual overrides are honored verbatim).
    - The manifest roofline block parses successfully under
      ``synthesize_eval_roofline``.

    Synthesis failures are swallowed so an irregular manifest entry
    leaves the base stub in place rather than blocking class
    construction; the validator catches the resulting C7 error.
    """
    if "eval_roofline" in cls.__dict__:
        return

    roofline = getattr(cls, "__manifest_roofline__", None)
    sig = getattr(cls, "__manifest_signature__", None)
    status = getattr(cls, "__manifest_status__", None)
    if roofline is None or status is None:
        entry = _lookup_manifest_entry(cls.__name__)
        if entry is None:
            return
        roofline = entry.get("roofline")
        sig = entry.get("signature")
        status = entry.get("status")
    if status != "implemented":
        return
    if roofline is None:
        return
    try:
        fn = synthesize_eval_roofline(
            cls.__name__, roofline=roofline, signature=sig,
        )
    except ValueError:
        return
    cls.eval_roofline = fn  # type: ignore[assignment]
