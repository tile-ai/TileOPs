"""The dtype an op writes an output in, from whichever of its two origins applies.

An output's dtype is the caller's where the entry marks the output ``caller_stated``, and
the entry's own resolved declaration everywhere else. Both the fake and the eager path ask
here, so the two cannot answer differently, and neither asks a kernel class — that would
make the compiled graph depend on which target served the op.
"""

import torch

from tileops.manifest import load_manifest
from tileops.manifest.dtype_rules import promote_int_to_float_ref, same_as_ref

__all__ = ["output_dtype", "resolve_output_dtype"]

# The one name a caller-stated output dtype travels under, and the flag on the outputs it
# states (docs/design/manifest.md R23).
OUT_DTYPE_PARAM = "out_dtype"
CALLER_STATED_FLAG = "caller_stated"

# What ``promote_int_to_float`` promotes an integral input to.
_PROMOTED_FLOAT_DTYPE = torch.float32


def _declared_expr(op_class_name: str, output: "str | None") -> str:
    """The manifest ``signature.outputs`` dtype expression for one output.

    Args:
        op_class_name: Op class name, which is the manifest entry key.
        output: Which output to read, or ``None`` for an op that declares one.

    Returns:
        The declared expression, e.g. ``"same_as(input)"`` or ``"bool"``.

    Raises:
        KeyError: The manifest has no entry for *op_class_name*, or no such output.
        ValueError: *output* is ``None`` and the entry declares more than one.
    """
    entry = load_manifest().get(op_class_name)
    if entry is None:
        raise KeyError(
            f"{op_class_name} has no manifest entry; the output dtype is "
            "declared under signature.outputs"
        )
    outputs = entry["signature"]["outputs"]
    if output is not None:
        if output not in outputs:
            raise KeyError(f"{op_class_name} declares no output named {output!r}")
        return outputs[output]["dtype"]
    if len(outputs) != 1:
        raise ValueError(
            f"{op_class_name} declares {len(outputs)} outputs, so the caller has to say "
            "which one's dtype it wants"
        )
    return next(iter(outputs.values()))["dtype"]


def resolve_output_dtype(
    op_class_name: str,
    input_dtype: torch.dtype,
    output: "str | None" = None,
) -> torch.dtype:
    """Resolve an op's output dtype from its manifest declaration.

    Args:
        op_class_name: Op class name, which is the manifest entry key.
        input_dtype: Dtype of the input the declaration refers to.
        output: Which output to resolve. ``None`` for an op that declares one.

    Returns:
        The output dtype. ``same_as(...)`` follows the input;
        ``promote_int_to_float(...)`` promotes integral inputs to float32; a
        bare dtype name resolves to that dtype.

    Raises:
        ValueError: The declared expression names an unknown dtype, or names a set of
            them — an output resolves to one dtype (manifest.md R23), so a union here
            would be an answer this function invented.
    """
    expr = _declared_expr(op_class_name, output)
    if "|" in expr:
        raise ValueError(
            f"{op_class_name}: manifest output dtype {expr!r} names a set; an output "
            "declares the one dtype it falls back to (R23)"
        )
    if same_as_ref(expr) is not None:
        return input_dtype
    if promote_int_to_float_ref(expr) is not None:
        if input_dtype.is_floating_point:
            return input_dtype
        return _PROMOTED_FLOAT_DTYPE
    resolved = getattr(torch, expr, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"{op_class_name}: manifest output dtype {expr!r} is not a torch dtype")
    return resolved


def output_dtype(op: object, output: str, input_dtype: "torch.dtype | None") -> torch.dtype:
    """The dtype *op* writes *output* in.

    Args:
        op: The op instance, read for its ``out_dtype`` where the entry marks the output.
        output: Which output to answer for.
        input_dtype: Dtype of the input the output's declaration refers to, used when the
            caller states nothing. ``None`` where the op takes no tensor input, whose
            declaration names a dtype outright.

    Returns:
        The caller's dtype where the entry marks *output* ``caller_stated: true`` and the
        caller passed one, otherwise what the declaration resolves to.
    """
    op_class_name = type(op).__name__
    outputs = load_manifest()[op_class_name]["signature"]["outputs"]
    if outputs[output].get(CALLER_STATED_FLAG):
        stated = getattr(op, OUT_DTYPE_PARAM, None)
        if stated is not None:
            return stated
    return resolve_output_dtype(op_class_name, input_dtype, output)
