"""Binding a forward call from the manifest signature.

An op's forward arity, and which of its arguments are tensors, is per-op knowledge the base
class cannot have: across this repo forwards take zero to twelve positionals, plenty of them
not tensors. But the manifest already states it — ``signature.inputs`` names the tensors in
call order — so the binding is generated, joining ``_validate_dtypes`` and the roofline
bodies as manifest-derived code installed on the class.

What comes out is the pair a backend is asked with: the tensors, and the params. Tensors stay
tensors here; describing them is the caller's job, after any lowering it does, so that the
description always matches what the kernel is actually handed.
"""

from __future__ import annotations

from typing import Any, Callable

from tileops.manifest import try_load_entry

#: Params whose value the op holds under the manifest's own name. A manifest param is a
#: construction-time decision — ``normalized_shape``, ``eps`` — so the instance is where it
#: lives by the time forward runs.
_MISSING = object()


def synthesize_bind_call(op_name: str, sig: dict[str, Any]) -> Callable[..., tuple]:
    """Build a ``_bind_call`` for the op described by *sig*.

    Args:
        op_name: Manifest op name, used in error messages.
        sig: The manifest ``signature`` block. ``inputs`` gives the tensor parameter names
            in call order; ``params`` gives the names to read off the instance.

    Returns:
        ``_bind_call(self, *args, **kwargs) -> (tensors, params)``, accepting the tensors
        positionally or by their manifest names.

    Raises:
        ValueError: ``signature.inputs`` is missing or not a mapping. An op with no tensor
            input has nothing to bind, and no device to dispatch from either.
    """
    inputs = sig.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError(f"{op_name}: signature.inputs is missing or empty")
    params = sig.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"{op_name}: signature.params must be a mapping")

    names = tuple(inputs)
    param_names = tuple(params)

    def _bind_call(self, *args: Any, **kwargs: Any) -> tuple:
        if len(args) > len(names):
            raise TypeError(
                f"{op_name} takes {len(names)} tensor inputs {names}, got {len(args)}"
            )
        bound = dict(zip(names, args, strict=False))
        for name, value in kwargs.items():
            if name not in names:
                raise TypeError(f"{op_name} has no tensor input {name!r}; expected {names}")
            if name in bound:
                raise TypeError(f"{op_name} got two values for {name!r}")
            bound[name] = value
        missing = [name for name in names if name not in bound]
        if missing:
            raise TypeError(f"{op_name} is missing tensor inputs {tuple(missing)}")

        values = {}
        for name in param_names:
            value = getattr(self, name, _MISSING)
            if value is _MISSING:
                raise AttributeError(
                    f"{op_name} declares manifest param {name!r} but the instance has no "
                    f"attribute of that name to bind it from"
                )
            values[name] = value
        return tuple(bound[name] for name in names), values

    _bind_call.__doc__ = (
        f"Bind a forward call to ({names}, params) for {op_name}. Generated from the "
        f"manifest signature."
    )
    return _bind_call


def maybe_install_bind_call(cls: type) -> None:
    """Install a generated ``_bind_call`` on *cls* when the manifest supports one.

    Resolution mirrors :func:`tileops.ops._dtype_codegen.maybe_install_validator`: a
    class-attached ``__manifest_signature__`` first, then the manifest entry keyed by class
    name. A hand-written ``_bind_call`` in the class body wins, and a signature too
    irregular to bind leaves the gap visible rather than masked.
    """
    if "_bind_call" in cls.__dict__:
        return

    sig = getattr(cls, "__manifest_signature__", None)
    status = getattr(cls, "__manifest_status__", None)
    if sig is None or status is None:
        entry = try_load_entry(cls.__name__)
        if entry is None:
            return
        sig = entry.get("signature")
        status = entry.get("status")
    if status != "implemented" or not isinstance(sig, dict):
        return
    try:
        cls._bind_call = synthesize_bind_call(cls.__name__, sig)
    except ValueError:
        return
