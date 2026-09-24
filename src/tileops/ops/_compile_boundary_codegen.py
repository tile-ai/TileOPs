"""An op's compile-boundary operators, generated from its manifest entry.

A traced ``forward`` is one call to an opaque operator, and that operator's schema is
already written down: its tensor arguments are ``signature.inputs`` in order, what it
returns is ``signature.outputs``, and which arguments it writes is the inputs marked
``mutated: true``. Generating the registration from the entry keeps the two from drifting
apart, and makes adding an op cost no registration code.

An op joins by declaring one :class:`OperatorSpec` per operator it registers. A spec says
which of the three kinds its operator is and, for the two writing kinds, which argument it
writes; the entry says the rest. Each output's dtype comes from
:func:`tileops.ops._output_dtype.output_dtype`.
"""

from __future__ import annotations

import dataclasses
import re
import string
from typing import Literal

import torch

from tileops.manifest import try_load_entry

from ._output_dtype import output_dtype
from .compile_boundary import get_instance

__all__ = ["OperatorSpec", "maybe_install_compile_boundary"]

ATTRIBUTE = "compile_boundary"


@dataclasses.dataclass(frozen=True)
class OperatorSpec:
    """One operator an op registers.

    Attributes:
        kind: ``"default"`` returns the declared outputs; ``"inplace"`` copies the result
            into an argument; ``"out"`` hands a caller-supplied buffer to
            the operator body under that keyword. Both writing kinds return nothing.
        argument: Which argument the two writing kinds write. ``None`` for ``"default"``.
    """

    kind: Literal["default", "inplace", "out"] = "default"
    argument: "str | None" = None

    @classmethod
    def inplace(cls, argument: str) -> "OperatorSpec":
        """The companion that writes its result back into *argument*.

        The kernel writes a fresh buffer, so the operator copies the result into the
        argument it was handed.
        """
        return cls(kind="inplace", argument=argument)

    @classmethod
    def writes_out(cls, argument: str) -> "OperatorSpec":
        """The operator that writes into a caller-supplied buffer.

        *argument* is either a declared input, which the operator body already receives
        in its own position, or a tensor-typed manifest param, which it receives under
        that keyword. The entry says which, so the spec does not.
        """
        return cls(kind="out", argument=argument)

    @property
    def writes(self) -> bool:
        """Whether this operator's result reaches the caller through an argument."""
        return self.kind != "default"


def maybe_install_compile_boundary(cls: type) -> None:
    """Register *cls*'s operators and publish their names on the class.

    A class with no ``compile_boundary`` is left alone, and so is one the manifest does
    not name: a family's base class states the spec its leaves share, and only a leaf the
    manifest names has an operator to register. That a manifest op registers no operator
    by hand is checked where source is read, by the manifest validator.

    Raises:
        TypeError: The class declares a boundary but implements no
            ``_infer_output_shapes``, so its fake has no output shape to give.
    """
    specs = getattr(cls, ATTRIBUTE, ())
    if not specs:
        return
    entry = try_load_entry(cls.__name__)
    if entry is None:
        return
    if not any("_infer_output_shapes" in base.__dict__ for base in cls.__mro__[:-1]):
        raise TypeError(
            f"{cls.__name__} declares a compile boundary but implements no "
            "_infer_output_shapes; its fake has no output shape to give"
        )
    cls.compile_op_names = tuple(_register(cls, entry, spec, specs) for spec in specs)


def _snake(name: str) -> str:
    """``"FusedAddRMSNormFwdOp"`` -> ``"fused_add_rms_norm_fwd"``."""
    spaced = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", spaced).lower().removesuffix("_op")


def operator_name(family: str, class_name: str) -> str:
    """``("normalization", "RMSNormFwdOp")`` -> ``"normalization_rms_norm_fwd"``.

    The family is not repeated: a class whose own name already opens with it, such as
    ``MoePrePermuteFwdOp``, names it once.
    """
    stem = _snake(class_name)
    if stem == family or stem.startswith(f"{family}_"):
        return stem
    return f"{family}_{stem}"


def written_arguments(entry: dict, spec: OperatorSpec, specs: "tuple[OperatorSpec, ...]") -> tuple:
    """Which arguments one operator writes.

    An operator writes the inputs the manifest marks ``mutated: true``, and a writing one
    also writes the argument it names. The exception is a default operator whose op also
    registers a writing one: that one does the writing and this one does none. That is why
    this reads the whole spec set rather than one spec.
    """
    if not spec.writes and any(other.writes for other in specs):
        return ()
    inputs = entry["signature"]["inputs"]
    mutated = tuple(
        name
        for name, attrs in inputs.items()
        if isinstance(attrs, dict) and attrs.get("mutated") is True
    )
    if not spec.writes or spec.argument in mutated:
        return mutated
    return mutated + (spec.argument,)


def operator_schema(entry: dict, spec: OperatorSpec, mutates: tuple) -> str:
    """The operator's schema, in the torch library's own grammar.

    Declared rather than assembled as Python source: ``torch.library.custom_op`` takes a
    schema directly, so one ``*args`` callable serves every op.
    """
    # One alias symbol per written argument: sharing one would say two arguments may be
    # the same tensor.
    aliases = iter(string.ascii_lowercase)
    args = []
    for name, attrs in entry["signature"]["inputs"].items():
        optional = "?" if isinstance(attrs, dict) and attrs.get("optional") else ""
        written = f"({next(aliases)}!)" if name in mutates else ""
        args.append(f"Tensor{written}{optional} {name}")
    if spec.kind == "out" and spec.argument not in entry["signature"]["inputs"]:
        args.append(f"Tensor({next(aliases)}!) {spec.argument}")
    args.append("str instance_key")
    if spec.writes:
        returns = "()"
    else:
        count = len(entry["signature"]["outputs"])
        returns = "Tensor" if count == 1 else "(" + ", ".join(["Tensor"] * count) + ")"
    return f"({', '.join(args)}) -> {returns}"


def _register(cls: type, entry: dict, spec: OperatorSpec, specs: tuple) -> str:
    """Register one operator for *cls* and return its qualified name."""
    declared = tuple(entry["signature"]["inputs"])
    outputs = tuple(entry["signature"]["outputs"])
    mutates = written_arguments(entry, spec, specs)
    # What the kernel writes: an inplace operator copies the result back itself.
    kernel_writes = frozenset(m for m in mutates if spec.kind != "inplace" or m != spec.argument)
    # The suffix tells two operators of one op apart, so a lone one takes no suffix.
    companion = spec.writes and any(not other.writes for other in specs)
    suffix = "_inplace" if companion else ""
    name = f"tileops::{operator_name(entry['family'], cls.__name__)}{suffix}"

    def operator(*args):
        *tensors, key = args
        op = get_instance(key)
        if spec.kind == "out":
            if spec.argument in declared:
                op._serve(*tensors, _written=kernel_writes)
                return None
            *passed, buffer = tensors
            op._serve(*passed, _written=kernel_writes, **{spec.argument: buffer})
            return None
        if spec.kind == "inplace":
            written = tensors[declared.index(spec.argument)]
            written.copy_(op._serve(*tensors, _written=kernel_writes).reshape(written.shape))
            return None
        result = op._serve(*tensors, _written=kernel_writes)
        # A body handing back a list satisfies the schema the same way.
        return result if len(outputs) == 1 else tuple(result)

    def fake(*args):
        *tensors, key = args
        op = get_instance(key)
        # An absent optional input arrives as ``None``, so presence stays a fact
        # ``_infer_output_shapes`` reads off the slot.
        shapes = op._infer_output_shapes(
            *(None if t is None else tuple(t.shape) for t in tensors[: len(declared)])
        )
        by_name = dict(zip(declared, tensors, strict=False))
        built = tuple(_output_tensor(op, out, by_name, shapes[out]) for out in outputs)
        return built[0] if len(outputs) == 1 else built

    operator.__name__ = name.replace("::", "_")
    fake.__name__ = f"{operator.__name__}_fake"
    registered = torch.library.custom_op(
        name, mutates_args=mutates, schema=operator_schema(entry, spec, mutates)
    )(operator)
    if not spec.writes:
        registered.register_fake(fake)
    setattr(cls, "_wrapped_inplace" if companion else "_wrapped", staticmethod(registered))
    return name


def _output_tensor(op, output: str, tensors: dict, shape) -> torch.Tensor:
    """An empty tensor of the shape the op inferred and the dtype whoever decides it gives.

    ``new_empty``, not ``empty_like``: the real path writes fresh contiguous storage, and
    a non-contiguous input's strides in the fake fail the graph's assertion.
    """
    entry = try_load_entry(type(op).__name__)
    source = _fallback_source(entry["signature"]["outputs"][output]["dtype"], tensors)
    return source.new_empty(shape, dtype=output_dtype(op, output, source.dtype))


def _fallback_source(expr: str, tensors: dict) -> torch.Tensor:
    """The input the fallback rule names, or any present one when it names none.

    The tensor that comes back supplies the device either way.
    """
    from tileops.manifest.dtype_rules import promote_int_to_float_ref, same_as_ref

    referenced = same_as_ref(expr) or promote_int_to_float_ref(expr)
    named = tensors.get(referenced) if referenced else None
    if named is not None:
        return named
    return next(tensor for tensor in tensors.values() if tensor is not None)
