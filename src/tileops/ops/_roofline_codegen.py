"""Install a manifest-derived ``eval_roofline`` on each concrete op.

The judging happens in :mod:`tileops.manifest.roofline_analysis` and the writing
in :mod:`tileops.ops._roofline_emit`; this module is the seam between them and
the class being created, plus the two helpers the generated body calls back
into, which need an op instance and so cannot live on the analysis side.
"""

from __future__ import annotations

from typing import Any, Callable

from tileops.manifest import try_load_entry
from tileops.manifest.roofline_analysis import analyze_roofline
from tileops.ops._roofline_emit import emit_eval_roofline


class _ShapeProxy:
    """Synthetic tensor stand-in exposing only ``shape`` and ``ndim``.

    Inline-mode roofline expressions reference ``<tensor>.shape`` and
    ``<tensor>.ndim``. Op classes are not required to retain the original
    tensor argument on ``self`` — many keep only derived state such as
    ``self.shape`` (the input shape tuple) or ``self.N_total`` (a flat
    element count). When the op does not store the tensor itself,
    ``_resolve_tensor_binding`` constructs a ``_ShapeProxy`` from the
    derived state so vars-layer expressions resolve uniformly.
    """

    __slots__ = ("shape", "ndim")

    def __init__(self, shape: tuple) -> None:
        """Build the op. Shapes and dtype are taken from the first call."""
        self.shape = tuple(shape)
        self.ndim = len(self.shape)


def _resolve_tensor_binding(
    op: Any,
    name: str,
    op_name: str,
    *,
    optional: bool = False,
) -> Any:
    """Bind ``name`` for inline-mode synthesis from op-instance state.

    Two accepted conventions, in order:

    1. ``self.<name>`` exposes ``.shape`` (a real tensor or any object
       exposing ``.shape`` / ``.ndim``).
    2. ``self.<name>_shape`` is a shape tuple/list; wrapped in a
       `_ShapeProxy` for uniform ``.shape``/``.ndim`` access.

    An op that exposes the binding but has not filled it raises `RuntimeError`
    naming the op: forward() has not run. An op that exposes neither attribute
    raises `ValueError`, which is the author's wiring, not the caller's
    sequencing; either beats a vacuous ``'NoneType' object has no attribute
    'shape'`` from inside the generated body.

    An ``optional: true`` input binds to ``None`` when the op exposes it as
    ``None`` under either convention — the call did not pass it, and R18.1
    limits what the expression may then do with the name to a presence test.
    Exposing neither attribute still raises: silently reading "absent" off an
    op that forgot to expose the input would under-count the roofline.

    Op-family-specific aliases (``self.shape`` / ``self.num_channels``
    / ``self.N_total``) are *not* consulted; ops opting into inline
    roofline declare bindings explicitly per ``docs/design/roofline.md``
    §4.4.3.
    """
    _unset = object()
    direct = getattr(op, name, _unset)
    # Tier 1 requires both ``.shape`` and ``.ndim`` so a partially
    # conformant object (e.g. exposes ``.shape`` only) does not slip
    # past and die later when the generated body reads ``.ndim``.
    if (
        direct is not _unset
        and direct is not None
        and hasattr(direct, "shape")
        and hasattr(direct, "ndim")
    ):
        return direct
    shape_attr = getattr(op, f"{name}_shape", _unset)
    if isinstance(shape_attr, (tuple, list)):
        return _ShapeProxy(tuple(shape_attr))
    if optional and (direct is None or shape_attr is None):
        return None
    if direct is None or shape_attr is None:
        # The op exposes the binding and has not filled it: the caller has not
        # run forward() yet. An absent attribute is the author's wiring and
        # falls through to the ValueError below.
        raise RuntimeError(
            f"{op_name}.eval_roofline() requires a prior forward() call to bind {name!r}"
        )
    raise ValueError(
        f"{op_name}: cannot resolve roofline input {name!r}; expected "
        f"either self.{name} (with .shape/.ndim) or self.{name}_shape "
        f"(shape tuple) on the op instance"
    )


def _output_dtype_on_call(op: Any, name: str, fallback: Any) -> Any:
    """Resolve an output's dtype, importing the resolver when the body runs.

    Synthesis binds this name into the generated globals. Checking a formula's
    names and forms needs no torch, and the resolver pulls it in, so the import
    belongs to the call and not to the check.
    """
    from tileops.ops._output_dtype import output_dtype

    return output_dtype(op, name, fallback)


def synthesize_eval_roofline(
    op_name: str,
    *,
    roofline: dict[str, Any] | None,
    signature: dict[str, Any] | None,
) -> Callable[..., tuple[int, int]]:
    """Analyse a roofline block and emit its method, raising on the first defect.

    One call and an exception, over :func:`analyze_roofline` plus
    :func:`emit_eval_roofline`. A caller wanting every defect calls the analyzer;
    this raises the first, in the order the analysis found it.

    Raises:
        ValueError: The block carries a defect, or a fact the formula needs
            could not be read.
    """
    result = analyze_roofline(op_name, roofline=roofline, signature=signature)
    if result.plan is None:
        blocking = result.blocking
        if blocking:
            raise ValueError(blocking[0].message)
        missing = ", ".join(sorted({u.missing for u in result.unjudged}))
        raise ValueError(
            f"{op_name}: roofline cannot be evaluated; {missing or 'a required fact'} "
            f"could not be read"
        )
    return emit_eval_roofline(result.plan)


def maybe_install_eval_roofline(cls: type) -> None:
    """Install the manifest-derived ``eval_roofline`` for an implemented op.

    Called for every subclass; the entry decides whether anything is installed.
    Class-attached manifest metadata takes precedence over the entry named by
    ``cls.__name__``.

    Two outcomes leave the class alone: a subclass with no manifest entry is not
    an op -- intermediate bases such as ``UnaryOp`` sit here -- and an entry that
    is not ``implemented`` has nothing to evaluate yet.

    An entry the analysis cannot build a plan for binds the abstract
    ``Op.eval_roofline``, which makes the class uninstantiable and is what
    ``check_c6`` names. Binding it is what keeps MRO lookup from answering with a
    parent's evaluator, which would price this op by another's formula. Raising
    instead would take down the import of whichever module defines the op. For an
    entry read from the manifest, ``check_roofline_synthesis`` reports every
    defect under the op's name; the validator reads the manifest, so a formula a
    class attached to itself is not among them.

    An implemented entry with no usable ``roofline`` is neither. The field is
    required of every entry regardless of status, so an absent, empty or
    non-mapping block means the manifest was never validated, and that raises
    rather than passing for a configuration.
    """
    from tileops.ops.op_base import Op

    roofline = getattr(cls, "__manifest_roofline__", None)
    sig = getattr(cls, "__manifest_signature__", None)
    status = getattr(cls, "__manifest_status__", None)
    if roofline is None or status is None:
        entry = try_load_entry(cls.__name__)
        if entry is None:
            return
        roofline = entry.get("roofline")
        sig = entry.get("signature")
        status = entry.get("status")
    if status != "implemented":
        return
    # Absence, not content: an empty or non-mapping block is as missing as no
    # key at all, and both are states the manifest is not allowed to be in.
    if not isinstance(roofline, dict) or not roofline:
        raise ValueError(
            f"{cls.__name__}: entry is implemented but declares no roofline block. "
            "roofline is required of every entry (validate_manifest.py, _REQUIRED_TOP), "
            "so reaching here means the manifest has not been validated"
        )
    # No catch. The analysis is total over entry data, so an exception is a
    # defect in it rather than in the entry, and it says so where it happened.
    # Swallowing one would leave a valid op abstract with no diagnostic
    # anywhere, which is what a refused entry already looks like.
    result = analyze_roofline(cls.__name__, roofline=roofline, signature=sig)
    if result.plan is None:
        cls.eval_roofline = Op.eval_roofline  # type: ignore[assignment]
        return
    cls.eval_roofline = emit_eval_roofline(result.plan)  # type: ignore[assignment]
