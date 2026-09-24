"""Evaluating a manifest entry's ``shape_rules``.

One evaluator serves both readers of the rules: the manifest validator, which probes
them against mock inputs, and the op layer, which holds a call a target serves to them.
A rule is a Python expression over the signature's inputs, params, and the dim names
its ``shape`` strings bind, evaluated with ``RULE_BUILTINS`` as its only builtins.
"""

from __future__ import annotations

import ast
import re

from .shape_rules import dim_range_validity, dim_uniqueness, reduced_axes, reduced_shape

__all__ = [
    "RULE_BUILTINS",
    "bind_declared_shapes",
    "broadcast_shapes",
    "eval_shape_rule",
    "is_broadcastable_to",
]


def broadcast_shapes(*shapes: object) -> tuple:
    """Pure-Python equivalent of ``torch.broadcast_shapes``.

    Shapes are right-aligned; each dimension must be equal, or one of
    them must be 1 (or missing). Returns ``()`` when called with no
    arguments.

    Raises:
        ValueError: If the shapes are not broadcast-compatible.
    """
    if not shapes:
        return ()
    normalized = [tuple(int(d) for d in s) for s in shapes]
    ndim = max((len(s) for s in normalized), default=0)
    out: list[int] = []
    for axis in range(ndim):
        # Right-align: walk from the trailing dim back.
        dim = 1
        for s in normalized:
            i = len(s) - ndim + axis
            if i < 0:
                # This shape has no entry at this axis (treat as 1).
                continue
            d = s[i]
            if d == 1 or d == dim:
                continue
            if dim == 1:
                dim = d
                continue
            raise ValueError(
                f"shapes {shapes!r} are not broadcast-compatible at axis {axis}",
            )
        out.append(dim)
    return tuple(out)


def is_broadcastable_to(src: object, dst: object) -> bool:
    """Return True if ``src`` is broadcastable *to* ``dst`` (unidirectional).

    Unlike the symmetric ``broadcast_shapes``, this predicate fixes the
    destination shape: each ``src`` dim (right-aligned) must equal the
    matching ``dst`` dim or be 1, and ``src`` may not have more
    dimensions than ``dst``.
    """
    src_t = tuple(int(d) for d in src)
    dst_t = tuple(int(d) for d in dst)
    if len(src_t) > len(dst_t):
        return False
    offset = len(dst_t) - len(src_t)
    for i, s_dim in enumerate(src_t):
        d_dim = dst_t[offset + i]
        if s_dim == d_dim or s_dim == 1:
            continue
        return False
    return True


# Safe builtins for shape_rules eval — the R11 / R11a helper set. Widening
# it widens the rule language, so keep it aligned with the manifest spec.
# An explicit pair list (not a dict merge) makes a name collision raise at
# import time instead of silently shadowing a primitive.
_RULE_BUILTIN_PAIRS = [
    ("len", len),
    ("isinstance", isinstance),
    ("int", int),
    # ``float`` lets manifest rules spell sentinels like
    # ``ord == float('inf')``. Add new callables only when an existing
    # manifest rule needs them and the semantics are obviously bounded.
    ("float", float),
    ("tuple", tuple),
    ("list", list),
    ("type", type),
    ("all", all),
    ("any", any),
    ("range", range),
    ("set", set),
    ("abs", abs),
    ("min", min),
    ("max", max),
    ("broadcast_shapes", broadcast_shapes),
    ("is_broadcastable_to", is_broadcastable_to),
    ("dim_range_validity", dim_range_validity),
    ("dim_uniqueness", dim_uniqueness),
    ("reduced_axes", reduced_axes),
    ("reduced_shape", reduced_shape),
]
RULE_BUILTINS: dict = {}
for _entry_name, _entry_fn in _RULE_BUILTIN_PAIRS:
    if _entry_name in RULE_BUILTINS:
        raise RuntimeError(
            f"shape_rule builtin name collision: {_entry_name!r} is "
            f"registered twice. Two callables cannot share the same "
            f"name in the rule eval scope; rename one or unify them."
        )
    RULE_BUILTINS[_entry_name] = _entry_fn


def eval_shape_rule(
    rule: str,
    ctx: dict,
) -> tuple[bool, str | None]:
    """Evaluate a single shape_rule in *ctx*.

    Returns (ok, failure_reason). ``ok=False`` with reason=None means the
    rule evaluated to a falsy non-exception value; a non-None reason
    indicates the rule could not be evaluated (treated as skipped, not a
    parity error).

    The eval globals expose the ``RULE_BUILTINS`` helper set so
    R11 / R11a-style rules can be evaluated against the mock context
    instead of being silently skipped. Context names (inputs / outputs /
    params) are injected into both eval globals and locals: comprehension
    scopes only see globals, so rules like
    ``all(d % x.ndim in ... for d in dim)`` still resolve ``x`` / ``dim``.
    """
    # Defense-in-depth: even though manifest content is trusted (PR review
    # gates it), parse the rule first and reject any dunder attribute
    # access. This closes the classic ``().__class__.__mro__[1].
    # __subclasses__()`` sandbox-escape against the restricted builtins.
    try:
        tree = ast.parse(rule, mode="eval")
    except SyntaxError as exc:
        return False, f"eval error: SyntaxError: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or node.attr.endswith("__")
        ):
            return False, (f"eval error: dunder attribute access not permitted ({node.attr!r})")

    eval_globals = {"__builtins__": RULE_BUILTINS}
    eval_globals.update(ctx)
    # A ctx key literally named ``__builtins__`` would overwrite the
    # sandboxed mapping installed above and re-expose the unrestricted
    # builtins; reinstate the sandbox after the update.
    eval_globals["__builtins__"] = RULE_BUILTINS
    try:
        result = eval(
            rule,
            eval_globals,
            ctx,
        )
    except Exception as exc:
        return False, f"eval error: {exc.__class__.__name__}: {exc}"
    try:
        return bool(result), None
    except Exception as exc:
        return False, f"non-boolean result: {exc}"


_SHAPE_EQ_RE = re.compile(r"^\s*([A-Za-z_]\w*)\.shape\s*==\s*\(([^)]*)\)\s*$")


def bind_declared_shapes(signature: dict, tensors: dict) -> dict:
    """The dim names a signature declares, read off the call's *tensors*.

    Two forms name a tensor's axes: its ``shape`` string, such as ``"[N, C_in, H, W]"``,
    and a rule ``x.shape == (M, N)`` over bare names (R13a). The same name on two
    tensors is one extent (R11). ``...`` in a ``shape`` string stands for any number of
    axes, so ``"[..., n]"`` binds the last one. An axis written as an expression binds
    nothing, and a tensor the call did not pass is skipped.

    Args:
        signature: The entry's ``signature``.
        tensors: The call's tensors, by input name.

    Returns:
        Each bound name and its extent.

    Raises:
        ValueError: A tensor's rank is not its declaration's, or one name has two extents.
    """
    declared = []
    for name, attrs in (signature.get("inputs") or {}).items():
        text = ((attrs or {}).get("shape") or "").strip()
        if text.startswith("[") and text.endswith("]"):
            declared.append((name, [a.strip() for a in text[1:-1].split(",") if a.strip()], text))
    for rule in signature.get("shape_rules") or ():
        match = _SHAPE_EQ_RE.match(rule) if isinstance(rule, str) else None
        if match is not None:
            axes = [a.strip() for a in match.group(2).split(",") if a.strip()]
            if all(a.isidentifier() for a in axes):
                declared.append((match.group(1), axes, rule))
    bound: dict = {}
    for name, axes, text in declared:
        tensor = tensors.get(name)
        if tensor is None:
            continue
        shape = tuple(tensor.shape)
        if "..." in axes:
            cut = axes.index("...")
            lead, trail = axes[:cut], axes[cut + 1 :]
            if tensor.ndim < len(lead) + len(trail):
                raise ValueError(f"{name} has rank {tensor.ndim}; the signature declares {text}")
            axes = lead + trail
            shape = shape[: len(lead)] + shape[tensor.ndim - len(trail) :]
        elif tensor.ndim != len(axes):
            raise ValueError(f"{name} has rank {tensor.ndim}; the signature declares {text}")
        for axis, extent in zip(axes, shape, strict=True):
            if not axis.isidentifier():
                continue
            if bound.setdefault(axis, int(extent)) != int(extent):
                raise ValueError(f"{axis} is {bound[axis]} elsewhere but {int(extent)} in {name}")
    return bound
