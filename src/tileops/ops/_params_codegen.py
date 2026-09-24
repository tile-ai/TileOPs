"""Which params an op hands a backend, taken from its manifest entry.

``build_kernel`` is called with the op's ``signature.params`` by keyword, so the op layer
needs those names. Only the names are attached here; ``Op._manifest_params`` reads the values
off the instance.
"""

from __future__ import annotations

import inspect

from tileops.manifest import try_load_entry

# Attached to a class when its manifest entry declares ``signature.params``. The empty
# tuple is a real answer: plenty of ops take no params.
ATTRIBUTE = "__manifest_param_names__"


def maybe_install_param_names(cls: type) -> None:
    """Attach the op's manifest param names to *cls*.

    Resolution mirrors `tileops.ops._dtype_codegen.maybe_install_validator`: a
    class-attached ``__manifest_signature__`` first, then the manifest entry keyed by class
    name. A name in the class body wins.

    Every class gets its own answer, never an inherited one: params are exactly this op's
    ``signature.params``, and a class with no entry hands a backend nothing. A param
    ``forward`` takes, such as a caller-supplied ``out`` buffer, belongs to one call and
    reaches the kernel with it, so it is not among them.
    """
    if ATTRIBUTE in cls.__dict__:
        return

    sig = getattr(cls, "__manifest_signature__", None)
    if sig is None:
        entry = try_load_entry(cls.__name__)
        sig = entry.get("signature") if entry is not None else None
    params = sig.get("params") if isinstance(sig, dict) else None
    per_call = set(inspect.signature(cls.forward).parameters)
    names = tuple(p for p in params if p not in per_call) if isinstance(params, dict) else ()
    setattr(cls, ATTRIBUTE, names)
