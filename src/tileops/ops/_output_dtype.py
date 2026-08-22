"""The dtype an op's fake reports, read off the manifest.

A registered fake is all the compiler learns about an op's node, and the only statement
about output dtype that holds for every target is the manifest's. So the fake reads it from
here rather than from a kernel class, which would make the compiled graph depend on which
target served the op.
"""

import torch

from tileops.manifest import load_manifest
from tileops.manifest.dtype_rules import promote_int_to_float_ref, same_as_ref

__all__ = ["resolve_output_dtype"]

#: What ``promote_int_to_float`` promotes an integral input to.
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
        The output dtype. ``same_as(...)`` and dtype unions follow the input;
        ``promote_int_to_float(...)`` promotes integral inputs to float32; a
        bare dtype name resolves to that dtype.

    Raises:
        ValueError: The declared expression names an unknown dtype.
    """
    expr = _declared_expr(op_class_name, output)
    if same_as_ref(expr) is not None or "|" in expr:
        return input_dtype
    if promote_int_to_float_ref(expr) is not None:
        if input_dtype.is_floating_point:
            return input_dtype
        return _PROMOTED_FLOAT_DTYPE
    resolved = getattr(torch, expr, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"{op_class_name}: manifest output dtype {expr!r} is not a torch dtype")
    return resolved
