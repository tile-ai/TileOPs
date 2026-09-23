"""Emit an ``eval_roofline`` from a plan. Decides nothing.

Which locals to bind, whether the formula reads ``elem_bytes``, which output
prices the write: all of it is decided in
:mod:`tileops.manifest.roofline_analysis` and travels in the plan. Nothing here
reads a manifest block, re-parses an expression, or produces a message an entry
author would act on. A defect reaching this module is a defect in the analysis,
which is what the assertions state.
"""

from __future__ import annotations

from typing import Any, Callable

from tileops.manifest.roofline_analysis import VARS_HELPERS, RooflinePlan

SYNTHESIZED_ATTR = "__tileops_synthesized_roofline__"


def _emit_func_mode(plan: RooflinePlan) -> Callable[..., tuple[int, int]]:
    """Delegate to the callable the plan resolved."""
    fn = plan.func
    assert fn is not None, "func-mode plan without a resolved callable"

    def eval_roofline(self):
        return fn(self)

    eval_roofline.__name__ = "eval_roofline"
    eval_roofline.__qualname__ = f"{plan.op_name}.eval_roofline"
    eval_roofline.__doc__ = f"Synthesized from manifest roofline.func={plan.func_path!r}."
    return eval_roofline


def _emit_inline_mode(plan: RooflinePlan) -> Callable[..., tuple[int, int]]:
    """Write the plan out as a plain function and compile it.

    The expressions are copied verbatim: the generated body parses nothing and
    evaluates no string at call time.
    """
    from tileops.ops._roofline_codegen import _output_dtype_on_call, _resolve_tensor_binding

    assert plan.flops_expr is not None and plan.bytes_expr is not None

    src_lines: list[str] = [
        "def eval_roofline(self):",
        f'    """Synthesized from manifest inline roofline for {plan.op_name}."""',
    ]
    for binding in plan.bindings:
        if binding.kind == "input":
            src_lines.append(
                f"    {binding.name} = _resolve_tensor_binding("
                f"self, {binding.name!r}, {plan.op_name!r}, optional={binding.optional})"
            )
        else:
            src_lines.append(f"    {binding.name} = self.{binding.name}")
    if plan.bind_elem_bytes:
        src_lines.append("    elem_bytes = self.dtype.itemsize")
    if plan.out_elem_bytes_output is not None:
        # Through ``output_dtype``, so a ``caller_stated`` output follows the
        # dtype the call asked for.
        src_lines.append(
            f"    out_elem_bytes = _output_dtype("
            f"self, {plan.out_elem_bytes_output!r}, self.dtype).itemsize"
        )
    for name, expr in plan.vars_program:
        src_lines.append(f"    {name} = {expr}")
    src_lines.append(f"    _flops = {plan.flops_expr}")
    src_lines.append(f"    _bytes = {plan.bytes_expr}")
    src_lines.append("    return int(_flops), int(_bytes)")

    globs: dict[str, Any] = dict(VARS_HELPERS)
    globs["_resolve_tensor_binding"] = _resolve_tensor_binding
    if plan.out_elem_bytes_output is not None:
        globs["_output_dtype"] = _output_dtype_on_call
    globs["__builtins__"] = {
        "int": int,
        "float": float,
        "bool": bool,
        "ValueError": ValueError,
    }

    src = "\n".join(src_lines)
    code = compile(src, f"<{plan.op_name}.eval_roofline>", "exec")
    local_ns: dict[str, Any] = {}
    exec(code, globs, local_ns)
    fn = local_ns["eval_roofline"]
    fn.__name__ = "eval_roofline"
    fn.__qualname__ = f"{plan.op_name}.eval_roofline"
    return fn


def emit_eval_roofline(plan: RooflinePlan) -> Callable[..., tuple[int, int]]:
    """Build the method this plan describes."""
    fn = _emit_func_mode(plan) if plan.mode == "func" else _emit_inline_mode(plan)
    setattr(fn, SYNTHESIZED_ATTR, True)
    return fn
