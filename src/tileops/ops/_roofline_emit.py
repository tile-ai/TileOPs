"""Emit an ``eval_roofline`` from a plan. Reads no manifest block and judges none.

Which locals to bind, whether the formula reads ``elem_bytes``, which output
prices the write: all of it is decided in
:mod:`tileops.manifest.roofline_analysis` and travels in the plan. Nothing here
reads a manifest block, re-parses an expression, or produces a message an entry
author would act on. A defect reaching this module is a defect in the analysis,
which is what the assertions state.

One judgment does live here, and it is about a value rather than an entry: what
a ``roofline.func`` callable returns is settled only by calling it, so the
wrapper holds the result to the pair-of-ints contract.
"""

from __future__ import annotations

from typing import Any, Callable

from tileops.manifest.roofline_analysis import (
    VARS_HELPERS,
    RooflinePlan,
    render_inline_source,
)

SYNTHESIZED_ATTR = "__tileops_synthesized_roofline__"


def _emit_func_mode(plan: RooflinePlan) -> Callable[..., tuple[int, int]]:
    """Delegate to the callable the plan resolved, and hold it to the contract.

    What the callable returns is settled only by calling it, so the analysis
    cannot rule on it and the wrapper does: a formula that answers with one
    number, or with something that is not a pair of ints, says so under the
    op's name rather than travelling as a roofline reading.
    """
    fn = plan.func
    assert fn is not None, "func-mode plan without a resolved callable"

    def eval_roofline(self):
        result = fn(self)
        if not (
            isinstance(result, tuple)
            and len(result) == 2
            and all(isinstance(part, int) and not isinstance(part, bool) for part in result)
        ):
            raise TypeError(
                f"{plan.op_name}: roofline.func {plan.func_path!r} returned "
                f"{result!r}; a roofline is a (flops, bytes) pair of ints"
            )
        return result

    eval_roofline.__name__ = "eval_roofline"
    eval_roofline.__qualname__ = f"{plan.op_name}.eval_roofline"
    eval_roofline.__doc__ = f"Synthesized from manifest roofline.func={plan.func_path!r}."
    return eval_roofline


def _emit_inline_mode(plan: RooflinePlan) -> Callable[..., tuple[int, int]]:
    """Compile and run the source the plan renders to. Builds no text of its own."""
    from tileops.ops._roofline_codegen import _output_dtype_on_call, _resolve_tensor_binding

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

    code = compile(render_inline_source(plan), f"<{plan.op_name}.eval_roofline>", "exec")
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
