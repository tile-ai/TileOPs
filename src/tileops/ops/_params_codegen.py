"""Which params an op hands a backend, taken from its manifest entry.

A backend's ``build_kernel`` is called with the op's ``signature.params`` by keyword
(docs/design/manifest.md), so the op layer has to know those names. They come from the
manifest rather than from a hand-written list per op: the manifest is the contract, and a
second copy would drift from it.

Only the names are generated. Reading them off the instance is one loop in
``Op._manifest_params``, so there is no function body to synthesize — unlike
``_dtype_codegen``, whose rules per input need one.
"""

from __future__ import annotations

from tileops.manifest import try_load_entry

#: Attached to a class when its manifest entry declares ``signature.params``. The empty
#: tuple is a real answer: plenty of ops take no params.
ATTRIBUTE = "__manifest_param_names__"


def maybe_install_param_names(cls: type) -> None:
    """Attach the op's manifest param names to *cls*.

    Resolution mirrors :func:`tileops.ops._dtype_codegen.maybe_install_validator`: a
    class-attached ``__manifest_signature__`` first, then the manifest entry keyed by class
    name. A name attached in the class body wins.

    Every class gets its own answer, never an inherited one. Params are exactly this op's
    ``signature.params``, so a class whose entry declares none — or has no entry at all —
    must hand a backend nothing rather than whatever its base declared.
    """
    if ATTRIBUTE in cls.__dict__:
        return

    sig = getattr(cls, "__manifest_signature__", None)
    if sig is None:
        entry = try_load_entry(cls.__name__)
        sig = entry.get("signature") if entry is not None else None
    params = sig.get("params") if isinstance(sig, dict) else None
    setattr(cls, ATTRIBUTE, tuple(params) if isinstance(params, dict) else ())
