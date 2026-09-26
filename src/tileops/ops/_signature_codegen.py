"""Methods generated from a parametric signature (docs/design/manifest.md § Call Semantics).

`install` gives an op class, from its manifest entry, `_check_construction`, the call checks,
`_validate_dtypes`, `_infer_output_shapes`, `eval_roofline` and, when the class declares a
compile boundary, one operator per effect branch. Each check is emitted as Python source when
the class is created, one per discriminant point, so a call parses no expression string and a
traced call only looks its check up. What construction can decide is checked there, once; the
call check continues from what construction solved. Under SymInt the emitted code lowers
`and`, `or`, `not`, conditionals and sequence equality to their symbolic forms, and a
refinement becomes `torch._check`. A failure names the declaration it came from.
"""

from __future__ import annotations

import abc
import ast
import copy
import inspect
import itertools
import string
from dataclasses import dataclass

import torch
from torch._guards import detect_fake_mode
from torch.fx.experimental.symbolic_shapes import sym_and, sym_or

from tileops.backend import OpNotAvailableError
from tileops.manifest import load_adts, try_load_entry
from tileops.manifest.dtype_rules import DTYPE_BITS
from tileops.manifest.expr import SignatureError, fold, infer_kinds, names, parse, value_at
from tileops.manifest.plan import EntryPlan, PlanBranch, entry_plan
from tileops.manifest.primitives import namespace
from tileops.manifest.signature import (
    Signature,
    complete_point,
    discriminant_axes,
    expand,
    is_legacy,
    kind_env,
    output_emitted,
    param_kind,
    rejecting_rule,
    tensor_passed,
    unification,
)
from tileops.manifest.values import convert
from tileops.manifest.workload import CallView

from ._compile_boundary_codegen import operator_name
from .compile_boundary import get_instance

__all__ = ["CheckError", "SignatureCall", "install", "maybe_install_signature"]


# ---------------------------------------------------------------- run-time helpers


class CheckError(ValueError):
    """A call or construction outside its signature; the message names the declaration."""


def _and(value, *rest):
    for thunk in rest:
        if isinstance(value, bool):
            if not value:
                return False
            value = thunk()
        else:
            value = sym_and(value, thunk())
    return value


def _or(value, *rest):
    for thunk in rest:
        if isinstance(value, bool):
            if value:
                return True
            value = thunk()
        else:
            value = sym_or(value, thunk())
    return value


def _not(value):
    return (not value) if isinstance(value, bool) else torch.sym_not(value)


def _ite(test, body, orelse):
    if isinstance(test, bool):
        return body() if test else orelse()
    return torch.sym_ite(test, body(), orelse())


def _eq(a, b):
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False
        return _and(True, *(lambda x=x, y=y: _eq(x, y) for x, y in zip(a, b, strict=True)))
    return a == b


def _require(value, message: str) -> None:
    if isinstance(value, torch.SymBool):
        torch._check(value, lambda: message)
    elif not value:
        raise CheckError(message)


def _solve(size, axis, message: str):
    """The integer `v` with `axis(v) == size`, for an affine `axis` with a positive slope."""
    lo = axis(0)
    step = axis(1) - lo
    _require((size - lo) % step == 0, message)
    value = (size - lo) // step
    _require(value >= 0, message)
    return value


def _dname(dtype) -> str | None:
    """A dtype's registry name: `torch.float16` and `"float16"` are both `"float16"`."""
    return None if dtype is None else str(dtype).removeprefix("torch.")


def _field(value, name: str):
    """An ADT field as expressions read it: an enum field by its `.value`."""
    field = getattr(value, name)
    return getattr(field, "value", field)


def _call_device(name: str, op, tensors: tuple, cpu: tuple, ctor: tuple):
    """The call device (docs/design/manifest.md § Call Semantics)."""
    devices = {t.device for t in tensors}
    if len(devices) > 1:
        raise CheckError(
            f"{name} needs every tensor on one device; got {sorted(map(str, devices))}"
        )
    for t in cpu:
        if t.device.type != "cpu":
            raise CheckError(f"{name}: a tensor declaring `device: cpu` is on {t.device}")
    if devices:
        return devices.pop()
    declared = op._declared_device() if hasattr(op, "_declared_device") else None
    if declared is not None:
        return declared
    held = {t.device for t in ctor}
    if len(held) > 1:
        raise CheckError(f"{name}: construction-time tensors are on {sorted(map(str, held))}")
    if held:
        return held.pop()
    # FIXME(staged-rollout): the target's device classes are not consulted
    #
    # Broken invariant: without tensors or a `device` parameter, the call device is the
    #   choice of the explicit or process-default target among its device classes.
    # Why: `Target` declares no device classes yet.
    # Cleanup: read them here once `tileops.backend` exposes them.
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None


def _placed(op, name: str, device, dtype: str):
    """A construction-time tensor in its signature dtype, on the call device unless *device* is
    None (a `device: cpu` tensor), kept on the op."""
    tensor = getattr(op, name)
    target = getattr(torch, dtype)
    if detect_fake_mode() is not None or torch.compiler.is_compiling():
        return tensor
    device = tensor.device if device is None else device
    if tensor.device != device or tensor.dtype != target:
        tensor = tensor.to(device=device, dtype=target)
        setattr(op, name, tensor)
    return tensor


def _failed(name: str, where: str, exc: Exception) -> ValueError:
    """The one contextual failure a generated method raises for an evaluation that failed."""
    return ValueError(f"{name}: {where}: {exc}")


_GLOBALS = {
    **namespace(),
    # The builtins generated code calls; expressions themselves reach only primitives.
    "__builtins__": {
        "AttributeError": AttributeError,
        "ArithmeticError": ArithmeticError,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "NameError": NameError,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "frozenset": frozenset,
        "getattr": getattr,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "tuple": tuple,
    },
    "_and": _and,
    "_or": _or,
    "_not": _not,
    "_ite": _ite,
    "_eq": _eq,
    "_require": _require,
    "_solve": _solve,
    "_dname": _dname,
    "_field": _field,
    "_call_device": _call_device,
    "_placed": _placed,
    "_failed": _failed,
    "CheckError": CheckError,
    "torch": torch,
}


@dataclass(frozen=True)
class SignatureCall(CallView):
    """One checked call: `ix`, every present tensor's shape and dtype, its effects, the
    metadata tensors whose values decide its traffic, and the checked calls its sub-ops
    completed during it (docs/design/roofline.md)."""

    ix: dict
    # Present tensors, outputs included, as `(shape, dtype name)`.
    tensors: dict
    # `(tensor, reads, writes)` per tensor the effect rules charge.
    traffic: tuple[tuple[str, int, int], ...]
    device: object = None
    # The inputs this call writes, and whether a caller passed `out`.
    written: frozenset = frozenset()
    out: bool = False
    # The discriminant point the call took, as its plan keys it.
    key: tuple = ()
    # The metadata tensors the call passed: inputs declaring `values`.
    metadata: dict = None
    # The checked calls the op's sub-ops completed during this call, by stage.
    stages: dict = None

    def values(self, name: str) -> list:
        """The contents of metadata tensor *name*.

        Raises:
            OpNotAvailableError: The call ran on meta tensors, which hold no values.
        """
        tensor = self.metadata[name]
        if tensor.device.type == "meta":
            raise OpNotAvailableError(f"{name} is a meta tensor: a meta call holds no values")
        return tensor.tolist()

    def derived_bytes(self) -> int:
        return sum(self.bytes(t) * (r + w) for t, r, w in self.traffic)


# ---------------------------------------------------------------- lowering


def _call(fn: str, args: list) -> ast.Call:
    return ast.Call(func=ast.Name(fn, ast.Load()), args=args, keywords=[])


def _thunk(body: ast.expr) -> ast.Lambda:
    no_args = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
    return ast.Lambda(args=no_args, body=body)


class _Lower(ast.NodeTransformer):
    """Rewrite a checked expression into code that runs on ints and on SymInts."""

    def __init__(self, sig: Signature, present=None):
        """Lower for *sig*; *present* names the helper `present(t)` calls, when they remain."""
        self.sig, self.present = sig, present
        self.maybes = {p for p in sig.params if sig.kind(p).tag == "Maybe"}

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        fn = "_and" if isinstance(node.op, ast.And) else "_or"
        return _call(fn, [node.values[0], *map(_thunk, node.values[1:])])

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        return _call("_not", [node.operand]) if isinstance(node.op, ast.Not) else node

    def visit_IfExp(self, node):
        self.generic_visit(node)
        return _call("_ite", [node.test, _thunk(node.body), _thunk(node.orelse)])

    def visit_Compare(self, node):
        self.generic_visit(node)
        parts, left = [], node.left
        for op, right in zip(node.ops, node.comparators, strict=True):
            if isinstance(op, (ast.Eq, ast.NotEq)):
                part = _call("_eq", [left, right])
                parts.append(_call("_not", [part]) if isinstance(op, ast.NotEq) else part)
            else:
                parts.append(ast.Compare(left, [op], [right]))
            left = right
        return parts[0] if len(parts) == 1 else _call("_and", [parts[0], *map(_thunk, parts[1:])])

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if (
            node.attr == "value"
            and isinstance(node.value, ast.Name)
            and node.value.id in self.maybes
        ):
            return node.value
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in {*self.sig.params, *self.sig.let}
            and node.attr != "kind"
        ):
            # An ADT field, on a parameter or a `let` holding one, reads an enum by its `.value`.
            return _call("_field", [node.value, ast.Constant(node.attr)])
        return node

    def visit_Name(self, node):
        declared = {*self.sig.forall, *self.sig.params, *self.sig.let}
        if node.id in DTYPE_BITS and node.id not in declared:
            return ast.Constant(node.id)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        callee = node.func.id if isinstance(node.func, ast.Name) else None
        if callee == "present" and self.present is not None:
            (arg,) = node.args
            if arg.id in self.maybes:
                return ast.Compare(arg, [ast.IsNot()], [ast.Constant(None)])
            return _call(self.present, [ast.Constant(arg.id)])
        if callee == "bytes" and self.present is not None:
            return _call("_c.bytes", [ast.Constant(node.args[0].id)])
        return node


def _code(sig: Signature, node: ast.expr, present=None) -> str:
    return ast.unparse(ast.fix_missing_locations(_Lower(sig, present).visit(copy.deepcopy(node))))


def _compiled(name: str, source: str, where: str, extra: dict | None = None):
    scope = {**_GLOBALS, **(extra or {})}
    exec(compile(source, f"<{where}>", "exec"), scope)  # noqa: S102
    return scope[name]


# ---------------------------------------------------------------- effects


def _written(sig: Signature, point: dict, present: set[str]) -> frozenset:
    """The inputs a call at `point` writes: `mutated` where its condition holds, `write_only`."""
    return frozenset(
        t.name
        for t in sig.call_tensors.values()
        if t.name in present
        and (
            t.write_only
            or t.mutated is True
            or (isinstance(t.mutated, str) and value_at(parse(t.mutated), point) is True)
        )
    )


def _traffic(sig: Signature, point: dict, present: set[str]) -> tuple:
    """Reads and writes the effect rules charge each present tensor (docs/design/roofline.md)."""
    written = _written(sig, point, present)
    out = [
        (t.name, 0 if t.write_only else 1, int(t.name in written))
        for t in sig.call_tensors.values()
        if t.name in present
    ]
    out += [
        (t.name, 0, 1) for t in sig.outputs.values() if t.name in present and t.alias not in written
    ]
    return tuple(out)


# ---------------------------------------------------------------- construction


class _Construction:
    """What construction decides at one construction point: the obligations it checks and the
    names it solves, which the call check at every point above it continues from."""

    def __init__(self, plan: EntryPlan, point: dict, env):
        """Decide, at construction point *point* of *plan*, what construction checks and solves."""
        sig = plan.sig
        self.point, self.rejected = point, rejecting_rule(sig, point)
        self.known = set(sig.params)
        # Construction-time tensors construction unifies.
        self.tensors = {}
        for t in sig.ctor_tensors.values():
            if point.get(f"present({t.name})") is not True:
                continue
            try:
                self.tensors[t.name] = fold(expand(sig, t.shape, point), point)
            except SignatureError:
                continue
        self.dtypes = {}
        for t in self.tensors:
            node = fold(parse(sig.ctor_tensors[t].dtype), point)
            if isinstance(node, ast.Name) and node.id in sig.forall and node.id not in self.known:
                self.dtypes[t] = node.id
                self.known.add(node.id)
        self.lets = {}
        for n, e in sig.let.items():
            try:
                self.lets[n] = fold(parse(e), point, f"let {n}")
            except SignatureError:
                continue
        self.steps = unification(self.tensors, self.known, self.lets)
        self.known |= {s.name for s in self.steps if s.name}
        # The construction-time tensor elements construction bound or checked.
        self.done = {(s.tensor, s.index) for s in self.steps if s.tensor is not None}
        # Tensors whose rank construction can check: every spliced length is known here.
        self.ranked = {
            t
            for t, node in self.tensors.items()
            if all(names(e.value) <= self.known for e in node.elts if isinstance(e, ast.Starred))
        }
        # Refinements whose activation and value construction has.
        self.rules = {}
        for i, r in enumerate(sig.rules):
            try:
                node = fold(parse(r), point, f"shape_rules[{i}]")
            except SignatureError:
                continue
            if names(node) <= self.known and not (
                isinstance(node, ast.Constant) and node.value is True
            ):
                self.rules[i] = node
        # Axes of inputs and outputs present here that are not a `Dim` and that construction has.
        self.axes = {}
        for t in (*sig.inputs.values(), *sig.outputs.values()):
            if point.get(f"present({t.name})") is not True:
                continue
            try:
                node = fold(expand(sig, t.shape, point), point)
            except SignatureError:
                continue
            for i, e in enumerate(node.elts):
                value = e.value if isinstance(e, ast.Starred) else e
                if names(value) <= self.known and _signed(value, e, env):
                    self.axes[(t.name, i)] = e

    def source(self, sig: Signature) -> str:
        e = _Emitter(sig)
        for p, decl in sig.params.items():
            dtype = param_kind(decl.get("type"), sig.adts).payload().tag == "DType"
            e.emit(f"{p} = {f'_dname(self.{p})' if dtype else f'self.{p}'}")
        for t in sig.ctor_tensors:
            present = self.point.get(f"present({t})")
            if present is True:
                e.require(f"self.{t} is not None", f"construction-time tensor {t!r} is required")
                e.require(
                    f"isinstance(self.{t}, torch.Tensor)",
                    f"construction-time tensor {t!r} is not a tensor",
                )
            elif present is False:
                e.require(
                    f"self.{t} is None", f"{t!r} is given where its presence condition is false"
                )
        if self.rejected is not None:
            e.emit(f"raise CheckError({f'{sig.name}: refinement fails: {self.rejected}'!r})")
            return e.function("construct", "self")
        for t in self.tensors:
            e.emit(f"_s_{t} = tuple(self.{t}.shape)")
        for t, index in self.dtypes.items():
            e.bind_dtype(index, f"_dname(self.{t}.dtype)", t)
        e.unify(self.tensors, self.steps, self.lets, self.ranked, set())
        for i, rule in self.rules.items():
            e.where(f"shape_rules[{i}]")
            e.require(_code(sig, rule), f"shape_rules[{i}]: {sig.rules[i]}")
        for (t, i), axis in self.axes.items():
            e.nonnegative(t, i, axis)
        e.emit(f"return {{{', '.join(f'{n!r}: {n}' for n in sorted(self.known))}}}")
        return e.function("construct", "self")


def _signed(value: ast.expr, element: ast.expr, env) -> bool:
    """Whether an axis carries a non-negativity obligation: its kind is not a `Dim`."""
    kind, _ = infer_kinds(value, env, "")
    if isinstance(element, ast.Starred):
        kind = kind.sequence().item if kind is not None and kind.sequence() is not None else None
    return not (kind is not None and kind.tag == "Int" and kind.nonneg)


# ---------------------------------------------------------------- the check at one point


class _Emitter:
    """Python source for one generated function, statement by statement.

    Every statement that evaluates a declaration is preceded by its name, and the function's
    one handler turns any failure an evaluation raises into a failure naming it.
    """

    def __init__(self, sig: Signature):
        """Start an empty function body for *sig*."""
        self.sig, self.lines = sig, []

    def emit(self, line: str) -> None:
        self.lines.append(f"        {line}")

    def where(self, text: str) -> None:
        self.emit(f"_w = {text!r}")

    def require(self, cond: str, message: str) -> None:
        self.emit(f"_require({cond}, {f'{self.sig.name}: {message}'!r})")

    def function(self, name: str, args: str) -> str:
        return "\n".join(
            [
                f"def {name}({args}):",
                "    _w = 'the signature'",
                "    try:",
                *(self.lines or ["        pass"]),
                "    except CheckError:",
                "        raise",
                "    except (ValueError, TypeError, ArithmeticError, IndexError, KeyError, AttributeError, NameError) as _e:",
                f"        raise _failed({self.sig.name!r}, _w, _e) from None",
            ]
        )

    def bind_dtype(self, index: str, value: str, tensor: str) -> None:
        members = sorted(self.sig.kind(index).values or ())
        self.emit(f"{index} = {value}")
        self.require(f"{index} in {tuple(members)!r}", f"{tensor} dtype is outside {members}")

    def bind_input_dtypes(
        self, b, inputs: list[str], known: set[str], absent: bool = False
    ) -> list:
        """Bind or check each `DType` index an input's dtype `_d_<t>` names; the inputs whose
        dtype is another expression are returned. With *absent* a dtype may be None, and an
        index no passed dtype names is left unbound."""
        sig, deferred, indices = self.sig, [], set()
        for t in inputs:
            node = b.dtypes[t]
            if not (
                isinstance(node, ast.Name)
                and node.id in sig.forall
                and sig.kind(node.id).tag == "DType"
            ):
                deferred.append((t, node))
            elif absent:
                message = f"{sig.name}: {t} dtype differs from {node.id}"
                self.emit(f"if _d_{t} is not None:")
                self.emit(
                    f"    _require(_dt.setdefault({node.id!r}, _d_{t}) == _d_{t}, {message!r})"
                )
                indices.add(node.id)
            elif node.id in known:
                self.require(f"_d_{t} == {node.id}", f"{t} dtype differs from {node.id}")
            else:
                self.bind_dtype(node.id, f"_d_{t}", t)
                known.add(node.id)
        for index in sorted(indices - known):
            members = sorted(sig.kind(index).values or ())
            message = f"{sig.name}: {index} is outside {members}"
            self.emit(f"if {index!r} in _dt:")
            self.emit(f"    {index} = _dt[{index!r}]")
            self.emit(f"    _require({index} in {tuple(members)!r}, {message!r})")
            known.add(index)
        return deferred

    def nonnegative(self, t: str, i: int, axis: ast.expr) -> None:
        self.where(f"tensor {t!r} shape")
        message = f"axis {i} of {t}, {ast.unparse(axis)}, is negative"
        if isinstance(axis, ast.Starred):
            self.emit(f"for _v in {_code(self.sig, axis.value)}:")
            self.emit(f"    _require(_v >= 0, {f'{self.sig.name}: {message}'!r})")
        else:
            self.require(f"{_code(self.sig, axis)} >= 0", message)

    def unify(self, nodes: dict, steps: list, lets: dict, ranked: set, skip: set) -> None:
        """Rank checks of `ranked`, the plan's bindings, then every remaining axis equality;
        elements in `skip` were bound or checked at construction."""
        sig = self.sig
        for t in ranked:
            node = nodes[t]
            fixed = sum(not isinstance(e, ast.Starred) for e in node.elts)
            op = ">=" if fixed < len(node.elts) else "=="
            self.require(f"len(_s_{t}) {op} {fixed}", f"{t} needs rank {op} {fixed}")

        def length(parts) -> str:
            return (
                " + ".join(
                    f"len({_code(sig, p.value)})" if isinstance(p, ast.Starred) else "1"
                    for p in parts
                )
                or "0"
            )

        checks = []
        for step in steps:
            if step.action == "let":
                self.where(f"let {step.name}")
                self.emit(f"{step.name} = {_code(sig, lets[step.name])}")
                continue
            if (step.tensor, step.index) in skip:
                continue
            t, e = step.tensor, nodes[step.tensor].elts[step.index]
            self.where(f"tensor {t!r} shape")
            start = length(step.before) if step.before is not None else None
            end = f"len(_s_{t}) - ({length(step.after)})" if step.after is not None else None
            if isinstance(e, ast.Starred):
                part = f"_s_{t}[{start}:{end}]"
                if step.action == "bind":
                    self.emit(f"{step.name} = {part}")
                else:
                    checks.append((f"_eq({part}, tuple({_code(sig, e.value)}))", t, e))
                continue
            index = start if start is not None else f"{end} - 1"
            if step.action == "check":
                checks.append((f"_eq(_s_{t}[{index}], {_code(sig, e)})", t, e))
            elif isinstance(e, ast.Name):
                self.emit(f"{e.id} = _s_{t}[{index}]")
            else:
                message = f"{sig.name}: {t} axis {step.index} is not {ast.unparse(e)}"
                self.emit(
                    f"{step.name} = _solve(_s_{t}[{index}], lambda {step.name}: {_code(sig, e)}, {message!r})"
                )
        for t in ranked:
            if any(isinstance(e, ast.Starred) for e in nodes[t].elts):
                self.require(f"len(_s_{t}) == {length(nodes[t].elts)}", f"{t} has the wrong rank")
        for cond, t, e in checks:
            self.where(f"tensor {t!r} shape")
            self.require(cond, f"{t} shape does not match {ast.unparse(e)}")


class _CallCheck:
    """The call check at one discriminant point, as Python source.

    With `shapes_only` it takes shape tuples and yields output shapes: no dtype or device.
    """

    def __init__(
        self, plan: EntryPlan, point: dict, key: tuple, built: _Construction, env, shapes_only: bool
    ):
        """The check of *plan* at *point*, keyed *key*, continuing from construction *built*."""
        self.plan, self.point, self.key, self.built, self.env = plan, point, key, built, env
        self.shapes_only = shapes_only

    def source(self) -> str:  # noqa: C901 - one pass per stage of the check
        plan, point, built, shapes_only = self.plan, self.point, self.built, self.shapes_only
        sig = plan.sig
        e = _Emitter(sig)
        name = "shapes" if shapes_only else "check"
        b = plan.branch(point)
        present = set(b.shapes)
        buffer = point.get("present(out)", False)
        e.emit("_k = self._construction_ix")
        for n in sorted(built.known):
            e.emit(f"{n} = _k[{n!r}]")
        for t in sig.inputs:
            e.emit(f"{t} = tensors.get({t!r})")
            if t in present:
                e.require(f"{t} is not None", f"{t!r} is required")
                if not shapes_only:
                    e.require(f"isinstance({t}, torch.Tensor)", f"{t!r} is not a tensor")
            else:
                e.require(f"{t} is None", f"{t!r} is given where its presence condition is false")
        rejected = rejecting_rule(sig, point)
        if rejected is not None:
            e.emit(f"raise CheckError({f'{sig.name}: refinement fails: {rejected}'!r})")
            return e.function(name, "self, tensors, dtypes" if shapes_only else "self, tensors")
        if buffer:
            e.emit("out = tensors['out']")
            if not shapes_only:
                e.require("isinstance(out, torch.Tensor)", "'out' is not a tensor")
        inputs = [t for t in sig.inputs if t in present]
        ctor = [t for t in sig.ctor_tensors if t in present]
        known = set(built.known)
        dtype_indices = {n for n in sig.forall if sig.kind(n).tag == "DType"}
        if shapes_only:
            for t in inputs:
                e.emit(f"_s_{t} = tuple({t})")
                e.emit(f"_d_{t} = _dname(dtypes.get({t!r}))")
            for t in ctor:
                e.emit(f"_s_{t} = tuple(self.{t}.shape)")
            e.emit("_dt = {}")
            e.bind_input_dtypes(b, inputs, known, absent=True)
        else:
            self._device_and_dtypes(e, b, inputs, ctor, known)
        nodes = {t: b.shapes[t] for t in (*inputs, *ctor)}
        ranked = {t for t in nodes if t not in built.ranked}
        e.unify(nodes, unification(nodes, known, b.lets), b.lets, ranked, built.done)
        for i, rule in enumerate(b.rules):
            if (isinstance(rule, ast.Constant) and rule.value is True) or i in built.rules:
                continue
            if shapes_only and names(rule) & dtype_indices:
                continue
            e.where(f"shape_rules[{i}]")
            e.require(_code(sig, rule), f"shape_rules[{i}]: {sig.rules[i]}")
        outputs = [o for o in sig.outputs if o in present]
        for o in outputs:
            e.where(f"tensor {o!r} shape")
            axes = [
                f"*tuple({_code(sig, a.value)})" if isinstance(a, ast.Starred) else _code(sig, a)
                for a in b.shapes[o].elts
            ]
            e.emit(f"_s_{o} = ({', '.join(axes)}{',' if len(axes) == 1 else ''})")
            for i, a in enumerate(b.shapes[o].elts):
                value = a.value if isinstance(a, ast.Starred) else a
                if (o, i) not in built.axes and _signed(value, a, self.env):
                    e.nonnegative(o, i, a)
            if shapes_only:
                continue
            e.where(f"tensor {o!r} dtype")
            e.emit(f"_d_{o} = {_code(sig, b.dtypes[o])}")
            if buffer and sig.outputs[o].buffer:
                e.require(f"_eq(tuple(out.shape), _s_{o})", f"out does not have the shape of {o}")
                e.require(f"_dname(out.dtype) == _d_{o}", f"out does not have the dtype of {o}")
                e.require("out.device == _device", "out is not on the call device")
                if sig.outputs[o].contiguous:
                    e.require("out.is_contiguous()", "out must be contiguous")
        if shapes_only:
            e.emit(f"return {{{', '.join(f'{o!r}: _s_{o}' for o in outputs)}}}")
            return e.function(name, "self, tensors, dtypes")
        values = {n for n, k in sig.forall.items() if k == "Seq[Int]"}
        solved = {s.name for s in unification(nodes, known, b.lets) if s.name}
        ix = sorted((known | solved) - values)
        e.emit(f"_ix = {{{', '.join(f'{n!r}: {n}' for n in ix)}}}")
        shapes = ", ".join(f"{t!r}: (_s_{t}, _d_{t})" for t in (*inputs, *ctor, *outputs))
        metadata = ", ".join(
            f"{t!r}: {t}" for t in (*inputs, *ctor) if sig.call_tensors[t].values is not None
        )
        e.emit(
            f"return _SignatureCall(_ix, {{{shapes}}}, {_traffic(sig, point, present)!r}, _device, "
            f"frozenset({sorted(_written(sig, point, present))!r}), {buffer!r}, {self.key!r}, {{{metadata}}})"
        )
        return e.function(name, "self, tensors")

    def _device_and_dtypes(
        self, e: _Emitter, b: PlanBranch, inputs: list[str], ctor: list[str], known: set[str]
    ) -> None:
        """The call device, then dtypes: inputs bind them, construction-time tensors are cast."""
        sig = self.plan.sig
        cpu = [t for t in inputs if sig.inputs[t].cpu]
        held_cpu = [f"self.{t}" for t in ctor if sig.ctor_tensors[t].cpu]
        args = ["".join(f"{t}, " for t in inputs if t not in cpu)]
        args.append("".join(f"{t}, " for t in [*cpu, *held_cpu]))
        args.append("".join(f"self.{t}, " for t in ctor if not sig.ctor_tensors[t].cpu))
        e.emit(f"_device = _call_device({sig.name!r}, self, ({args[0]}), ({args[1]}), ({args[2]}))")
        for t in inputs:
            e.emit(f"_s_{t} = tuple({t}.shape)")
            e.emit(f"_d_{t} = _dname({t}.dtype)")
            if sig.inputs[t].contiguous:
                e.require(f"{t}.is_contiguous()", f"{t} must be contiguous")
        for t, node in e.bind_input_dtypes(b, inputs, known):
            e.where(f"tensor {t!r} dtype")
            e.require(f"_d_{t} == {_code(sig, node)}", f"{t} dtype is not {ast.unparse(node)}")
        for t in ctor:
            node = b.dtypes[t]
            if isinstance(node, ast.Name) and node.id in sig.forall and node.id not in known:
                e.bind_dtype(node.id, f"_dname(self.{t}.dtype)", t)
                known.add(node.id)
            e.where(f"tensor {t!r} dtype")
            device = "None" if sig.ctor_tensors[t].cpu else "_device"
            e.emit(f"{t} = _placed(self, {t!r}, {device}, {_code(sig, node)})")
            e.emit(f"_s_{t} = tuple({t}.shape)")
            e.emit(f"_d_{t} = {_code(sig, node)}")
            if sig.ctor_tensors[t].contiguous:
                e.require(f"{t}.is_contiguous()", f"{t} must be contiguous")
        if sig.dtype_combos:
            columns = sorted(sig.dtype_combos[0])
            rows = sorted({tuple(r[c] for c in columns) for r in sig.dtype_combos})
            e.require(
                f"({', '.join(columns)},) in {rows!r}", f"{columns} is not a dtype_combos row"
            )


def _roofline_source(sig: Signature, b: PlanBranch) -> str:
    """`roofline(_c)` at one point: its folded inline formula over the call's `ix`."""
    tensors = {*sig.call_tensors, *sig.outputs, "out"}
    exprs = [n for n in b.roofline.values() if n is not None]
    read = sorted(set().union(set(), *(names(n) for n in exprs)) - tensors)
    flops = _code(sig, b.roofline["flops"], present="_c.present")
    moved = (
        _code(sig, b.roofline["bytes"], present="_c.present")
        if b.roofline["bytes"] is not None
        else "_c.derived_bytes()"
    )
    return "\n".join(
        [
            "def roofline(_c):",
            *(f"    {n} = _c.ix[{n!r}]" for n in read),
            f"    return ({flops}, {moved})",
        ]
    )


def _read_names(sig: Signature) -> set[str]:
    """Every name the signature's expressions read, and every tensor whose presence varies.

    A type family's discriminants reach it as arguments of the shapes that apply it.
    """
    texts = [*sig.rules, *sig.let.values()]
    for t in (*sig.call_tensors.values(), *sig.outputs.values()):
        conditions = (t.optional, t.nullable, t.mutated)
        texts += [t.shape, t.dtype, *(c for c in conditions if isinstance(c, str))]
    read = {t.name for t in sig.call_tensors.values() if t.optional is True} | {"out"}
    return read.union(*(names(parse(text)) for text in texts))


def _rejecting(sig: Signature, message: str):
    def check(self, *args):
        raise CheckError(f"{sig.name}: {message}")

    return check


class _Plan:
    """One entry's discriminant axes and the checks emitted for each of their points.

    Every point is emitted when the class is installed, so a traced call only looks one up.
    """

    def __init__(self, plan: EntryPlan):
        """Emit, for every point of *plan*'s signature, a check, a shape-only check, the effect
        branch and the roofline; and for every construction point, the construction check.

        A point is keyed by its axes; the presence of tensors whose condition reads them is
        settled here, once.
        """
        sig = self.sig = plan.sig
        self.entry = plan
        env, _ = kind_env(sig, {n: parse(e) for n, e in sig.let.items()})
        # An axis nothing in the signature reads cannot change its checks.
        read = _read_names(sig)
        self.axes = {a: v for a, v in discriminant_axes(sig).items() if a in read}
        self.built_axes = {
            a: v for a, v in self.axes.items() if a in sig.params or a in sig.ctor_tensors
        }
        self.keys, self.built_keys = self._keys(self.axes), self._keys(self.built_axes)
        self.constructions, self.checks, self.shapes, self.effects, self.roofs = {}, {}, {}, {}, {}
        built = {}
        for base in self._points(self.built_axes):
            key = self.built_key(base)
            try:
                point = complete_point(sig, base, strict=False)
                built[key] = _Construction(plan, point, env)
                self.constructions[key] = _compiled(
                    "construct", built[key].source(sig), f"{sig.name} construction"
                )
            except SignatureError as exc:
                self.constructions[key] = _rejecting(sig, str(exc))
        for base in self._points(self.axes):
            key = self.key(base)
            construction = built.get(self.built_key(base))
            try:
                point = complete_point(sig, base)
                if construction is None:
                    raise SignatureError("its construction point is outside the signature")
                present = {t for t in sig.call_tensors if tensor_passed(sig.call_tensors[t], point)}
                emitted = frozenset(o for o in sig.outputs if output_emitted(sig.outputs[o], point))
                self.effects[key] = (
                    _written(sig, point, present),
                    point.get("present(out)", False),
                    emitted,
                )
                for table, shapes_only in ((self.checks, False), (self.shapes, True)):
                    source = _CallCheck(plan, point, key, construction, env, shapes_only).source()
                    table[key] = _compiled(
                        "shapes" if shapes_only else "check",
                        source,
                        f"{sig.name} check",
                        {"_SignatureCall": SignatureCall},
                    )
                if (
                    plan.roofline is not None
                    and "flops" in plan.roofline
                    and rejecting_rule(sig, point) is None
                ):
                    self.roofs[key] = _compiled(
                        "roofline",
                        _roofline_source(sig, plan.branch(point)),
                        f"{sig.name} roofline",
                    )
            except SignatureError as exc:
                self.checks[key] = self.shapes[key] = _rejecting(sig, str(exc))

    @staticmethod
    def _keys(axes: dict) -> list:
        return sorted(
            {key for key, _ in axes.values() if key is not None}
            | {k for key, values in axes.values() if key is None for v in values for k in v}
        )

    @staticmethod
    def _points(axes: dict):
        for combo in itertools.product(*(values for _, values in axes.values())):
            base = {}
            for (key, _), value in zip(axes.values(), combo, strict=True):
                base.update(value if key is None else {key: value})
            yield base

    def key(self, point: dict) -> tuple:
        return tuple(point.get(k) for k in self.keys)

    def built_key(self, point: dict) -> tuple:
        return tuple(point.get(k) for k in self.built_keys)

    def point(self, op, tensors: dict, axes: dict | None = None) -> dict:
        point = {}
        for name, (key, values) in (self.axes if axes is None else axes).items():
            if name in self.sig.params:
                value = getattr(op, name)
                if key is None:
                    # An ADT, or a Maybe[ADT] whose payload fields are keyed `name.value.`.
                    maybe = any(f"present({name})" in v for v in values)
                    prefix = f"{name}.value." if maybe else f"{name}."
                    if maybe:
                        point[f"present({name})"] = value is not None
                    if value is None:
                        continue
                    point[f"{prefix}kind"] = value.kind
                    fields = {k[len(prefix) :] for v in values for k in v if k.startswith(prefix)}
                    for field in fields - {"kind"}:
                        if hasattr(value, field):
                            point[f"{prefix}{field}"] = _field(value, field)
                elif key == name:
                    point[name] = getattr(value, "value", value)
                else:
                    point[key] = value is not None
            elif name in self.sig.ctor_tensors:
                point[key] = getattr(op, name) is not None
            else:
                point[key] = tensors.get(name) is not None
        return point

    def construct(self, op) -> None:
        """Check what construction decides and keep what it solved on *op*."""
        key = self.built_key(self.point(op, {}, self.built_axes))
        fn = self.constructions.get(key)
        if fn is None:
            raise CheckError(f"{self.sig.name}: discriminants {key} are outside their types")
        op._construction_ix = fn(op)

    def check(self, op, tensors: dict) -> SignatureCall:
        return self._lookup(self.checks, op, tensors)(op, tensors)

    def output_shapes(self, op, shapes: dict, dtypes: dict) -> dict:
        """Output shapes from input shapes; `DType` indices bind from *dtypes* where given."""
        return self._lookup(self.shapes, op, shapes)(op, shapes, dtypes)

    def effect(self, op, tensors: dict) -> tuple:
        """The inputs this call writes, whether it passes `out`, and the outputs it emits."""
        return self._lookup(self.effects, op, tensors)

    def roofline(self, call: SignatureCall) -> tuple[int, int]:
        """`(flops, bytes)` of a checked call (docs/design/roofline.md)."""
        if self.entry.roofline is not None and "func" in self.entry.roofline:
            return _counts(self.sig.name, self.entry.roofline["func"](call))
        return _counts(self.sig.name, self.roofs[call.key](call))

    def _lookup(self, table: dict, op, tensors: dict):
        point = self.point(op, tensors)
        fn = table.get(self.key(point))
        if fn is None:
            raise CheckError(f"{self.sig.name}: discriminants {point} are outside their types")
        return fn


def check_result(
    sig: Signature, call: SignatureCall, result, tensors: dict, held: tuple = ()
) -> None:
    """What the implementation returned against the checked call and its effects.

    Each output slot is checked on its own: an absent one is None, a buffered one is the `out`
    passed, one aliasing an input the call writes is that input; each returned tensor then has
    its predicted shape, dtype, device and layout.
    """
    outputs = list(sig.outputs)
    if not outputs:
        _require(result is None, f"{sig.name}: forward returns a value but declares no output")
        return
    items = tuple(result) if len(outputs) > 1 and isinstance(result, (tuple, list)) else (result,)
    _require(len(items) == len(outputs), f"{sig.name}: forward returns {len(items)} outputs")
    # Storage a fresh output may not share: every tensor passed or held, and each output before it.
    taken = [v for v in tensors.values() if isinstance(v, torch.Tensor)]
    taken += [v for v in held if isinstance(v, torch.Tensor)]
    for name, item in zip(outputs, items, strict=True):
        decl = sig.outputs[name]
        if name not in call.tensors:
            _require(item is None, f"{sig.name}: {name} is returned where it is absent")
            continue
        if decl.buffer and call.out:
            _require(item is tensors["out"], f"{sig.name}: {name} must be the `out` it was given")
        if decl.alias in call.written:
            _require(
                item is tensors[decl.alias], f"{sig.name}: {name} must be the input {decl.alias}"
            )
        shape, dtype = call.tensors[name]
        _require(isinstance(item, torch.Tensor), f"{sig.name}: {name} is not a tensor")
        _require(_eq(tuple(item.shape), shape), f"{sig.name}: {name} has shape {tuple(item.shape)}")
        _require(_dname(item.dtype) == dtype, f"{sig.name}: {name} has dtype {item.dtype}")
        _require(
            call.device is None or item.device == call.device,
            f"{sig.name}: {name} is not on the call device",
        )
        if decl.contiguous:
            _require(item.is_contiguous(), f"{sig.name}: {name} must be contiguous")
        fresh = not (decl.buffer and call.out) and decl.alias not in call.written
        _require(
            not fresh or not any(_shares_storage(item, other) for other in taken),
            f"{sig.name}: {name} shares storage with a tensor it does not declare as its alias",
        )
        taken.append(item)


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    if detect_fake_mode() is not None or torch.compiler.is_compiling():
        return False
    if a.numel() == 0 or b.numel() == 0 or a.is_meta or b.is_meta:
        return False
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def _construction_check(plan: _Plan):
    """`_check_construction`: parameter values against their `type`, construction-time tensor
    presence, then what the construction point decides."""
    sig = plan.sig
    dtypes = {
        p
        for p, d in sig.params.items()
        if param_kind(d.get("type"), sig.adts).payload().tag == "DType"
    }

    def check(self) -> None:
        for p, decl in sig.params.items():
            value = getattr(self, p)
            try:
                convert(_dname(value) if p in dtypes else value, decl.get("type"), sig.adts)
            except ValueError as exc:
                raise CheckError(f"{sig.name}: {p} = {exc}") from None
        plan.construct(self)

    return check


def _counts(name: str, result) -> tuple[int, int]:
    """`(flops, bytes)` as two integers (docs/design/roofline.md)."""
    if not (
        isinstance(result, tuple)
        and len(result) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in result)
    ):
        raise TypeError(f"{name}: roofline yields {result!r}, not (flops: int, bytes: int)")
    return result


def _last_call(op) -> SignatureCall:
    call = getattr(op, "_signature_call", None)
    if call is None:
        raise RuntimeError(f"{type(op).__name__}: eval_roofline needs a completed call")
    return call


# ---------------------------------------------------------------- compile boundary


class _Boundary:
    """The compile-boundary operators of one class: one per effect branch.

    A branch is the set of inputs a call writes, whether it passes `out`, and the outputs it
    emits; its operator's schema, written arguments and fake follow from the signature there.
    """

    def __init__(self, cls: type, plan: _Plan, family: str):
        """Register one operator per effect branch of *plan*'s signature.

        `forward`'s code-defined execution parameters follow the signature's arguments in every
        schema, typed by their annotations.
        """
        self.plan, self.operators = plan, {}
        prefix = {"self", "out", *plan.sig.inputs}
        parameters = [
            p for p in inspect.signature(cls.forward).parameters.values() if p.name not in prefix
        ]
        self.execution = [(p.name, _schema_type(cls, p)) for p in parameters]
        stem = f"tileops::{operator_name(family, cls.__name__)}"
        registered = []
        branches = set(plan.effects.values())
        for effect in sorted(branches, key=lambda k: (sorted(k[0]), k[1], sorted(k[2]))):
            written, out, emitted = effect
            suffix = (
                "".join(f"_writes_{w}" for w in sorted(written))
                + ("_out" if out else "")
                + "".join(f"_without_{o}" for o in plan.sig.outputs if o not in emitted)
            )
            self.operators[effect] = self._register(f"{stem}{suffix}", *effect)
            registered.append(f"{stem}{suffix}")
        cls.compile_op_names = tuple(registered)

    def _schema(self, written: frozenset, out: bool, emitted: frozenset) -> str:
        sig, aliases = self.plan.sig, iter(string.ascii_lowercase)
        args = []
        for name, t in sig.inputs.items():
            mark = f"({next(aliases)}!)" if name in written else ""
            args.append(f"Tensor{mark}{'?' if t.optional is not False else ''} {name}")
        if out:
            args.append(f"Tensor({next(aliases)}!) out")
        args += [f"{kind} {name}" for name, kind in self.execution]
        args.append("str instance_key")
        returned = self._returned(written, out, emitted)
        if not returned:
            return f"({', '.join(args)}) -> ()"
        kinds = ["Tensor"] * len(returned)
        return f"({', '.join(args)}) -> {kinds[0] if len(kinds) == 1 else '(' + ', '.join(kinds) + ')'}"

    def _returned(self, written: frozenset, out: bool, emitted: frozenset) -> list[str]:
        """The outputs an operator returns: emitted ones that are not an input or the `out`."""
        return [
            o
            for o, t in self.plan.sig.outputs.items()
            if o in emitted and t.alias not in written and not (out and t.buffer)
        ]

    def _register(self, name: str, written: frozenset, out: bool, emitted: frozenset):
        sig, count = self.plan.sig, len(self.plan.sig.inputs)
        returned = self._returned(written, out, emitted)
        schema = self._schema(written, out, emitted)

        tail = count + int(out)

        def operator(*args):
            *values, key = args
            op = get_instance(key)
            writes = {"out": values[count]} if out else {}
            execution = dict(zip((n for n, _ in self.execution), values[tail:], strict=True))
            result = op._serve(*values[:count], _written=written, _execution=execution, **writes)
            if not returned:
                return None
            if len(sig.outputs) == 1:
                return result
            items = dict(zip(sig.outputs, result, strict=True))
            picked = tuple(items[o] for o in returned)
            return picked[0] if len(picked) == 1 else picked

        def fake(*args):
            *tensors, key = args
            op = get_instance(key)
            passed = dict(zip(sig.inputs, tensors[:count], strict=True))
            call = self.plan.check(op, passed | ({"out": tensors[count]} if out else {}))
            built = tuple(
                torch.empty(
                    call.tensors[o][0], dtype=getattr(torch, call.tensors[o][1]), device=call.device
                )
                for o in returned
            )
            if detect_fake_mode() is None and not torch.compiler.is_compiling():
                # An eager call on meta tensors completes here, not in `operator`; it runs no
                # sub-op, so it opens and closes its call at once.
                op._open_call()
                op._keep_call(call)
            return None if not built else built[0] if len(built) == 1 else built

        operator.__name__ = name.replace("::", "_")
        registered = torch.library.custom_op(
            name, mutates_args=tuple(sorted(written)) + (("out",) if out else ()), schema=schema
        )(operator)
        # Registered even for an operator that returns nothing, so its eager call on meta
        # tensors runs the checks and completes like any other.
        registered.register_fake(fake)
        return registered

    def call(self, op, inputs: tuple, writes: dict, execution: dict):
        """Call the operator of this call's effect branch, and return what `forward` returns."""
        sig = self.plan.sig
        _require(
            all(isinstance(v, torch.Tensor) for v in writes.values()),
            f"{sig.name}: 'out' is not a tensor",
        )
        tensors = {**dict(zip(sig.inputs, inputs, strict=True)), **writes}
        written, out, emitted = self.plan.effect(op, tensors)
        result = self.operators[(written, out, emitted)](
            *inputs,
            *([writes["out"]] if out else []),
            *(execution[n] for n, _ in self.execution),
            op._instance_key,
        )
        returned = self._returned(written, out, emitted)
        items = (
            dict(zip(returned, result if len(returned) > 1 else (result,), strict=True))
            if returned
            else {}
        )
        values = [
            None
            if o not in emitted
            else tensors[t.alias]
            if t.alias in written
            else writes["out"]
            if out and t.buffer
            else items[o]
            for o, t in sig.outputs.items()
        ]
        return None if not values else values[0] if len(values) == 1 else tuple(values)

    def binder(self, cls: type):
        """`_call_boundary` for *cls*: `forward`'s parameters bound by plain Python arguments."""
        parameters = list(inspect.signature(cls.forward).parameters.values())[1:]
        defaults = {f"_d{i}": p.default for i, p in enumerate(parameters)}
        formal = ", ".join(
            p.name if p.default is inspect.Parameter.empty else f"{p.name}=_d{i}"
            for i, p in enumerate(parameters)
        )
        inputs = "".join(f"{n}, " for n in self.plan.sig.inputs)
        writes = (
            "{'out': out} if out is not None else {}"
            if "out" in (p.name for p in parameters)
            else "{}"
        )
        execution = ", ".join(f"{n!r}: {n}" for n, _ in self.execution)
        source = (
            f"def _call_boundary(self, {formal}):\n"
            f"    return _boundary.call(self, ({inputs}), {writes}, {{{execution}}})"
        )
        scope = {**defaults, "_boundary": self}
        exec(compile(source, f"<{cls.__name__} boundary>", "exec"), scope)  # noqa: S102
        return scope["_call_boundary"]


_SCHEMA_TYPES = {int: "SymInt", float: "float", bool: "bool", str: "str"}


def _schema_type(cls: type, parameter: inspect.Parameter) -> str:
    """The operator-schema type of one execution parameter, from its annotation."""
    annotation = parameter.annotation
    if isinstance(annotation, str):
        annotation = {"int": int, "float": float, "bool": bool, "str": str}.get(annotation)
    if annotation not in _SCHEMA_TYPES:
        raise TypeError(
            f"{cls.__name__}.forward: execution parameter {parameter.name!r} needs an int, "
            "float, bool or str annotation to cross the compile boundary"
        )
    return _SCHEMA_TYPES[annotation]


# ---------------------------------------------------------------- installation


def _input_binder(sig: Signature, name: str, body, dtypes: bool = False):
    """A method taking the signature's inputs as `forward` does, that calls `body(self, tensors)`;
    with *dtypes* it also takes keyword-only `dtypes`, an input name to dtype mapping, and calls
    `body(self, tensors, dtypes)`."""
    formal = [
        "self",
        *(f"{t}=None" if d.optional is not False else t for t, d in sig.inputs.items()),
    ]
    formal += ["*", "dtypes=None"] if dtypes else []
    tensors = ", ".join(f"{t!r}: {t}" for t in sig.inputs)
    tail = ", dtypes or {}" if dtypes else ""
    source = f"def {name}({', '.join(formal)}):\n    return _body(self, {{{tensors}}}{tail})"
    scope = {"_body": body}
    exec(compile(source, f"<{sig.name} {name}>", "exec"), scope)  # noqa: S102
    return scope[name]


def install(cls: type, entry: dict, adts: dict | None = None) -> bool:
    """Give `cls` the methods its entry's signature generates; False when the signature is malformed."""
    try:
        entry_plan_ = entry_plan(cls.__name__, entry, load_adts() if adts is None else adts)
    except SignatureError:
        return False
    plan = _Plan(entry_plan_)
    sig = plan.sig
    cls._signature = plan
    cls._check_construction = _construction_check(plan)
    cls._validate_dtypes = _input_binder(
        sig, "_validate_dtypes", lambda self, ts: plan.check(self, ts)
    )
    cls._infer_output_shapes = _input_binder(
        sig,
        "_infer_output_shapes",
        lambda self, ss, ds: plan.output_shapes(self, ss, ds),
        dtypes=True,
    )
    if entry_plan_.roofline is not None:
        cls.eval_roofline = lambda self: plan.roofline(_last_call(self))
    # Declaring a compile boundary is the class's claim that it supports `fullgraph=True`.
    if getattr(cls, "compile_boundary", ()) is True:
        if not sig.inputs:
            raise TypeError(f"{cls.__name__}: compile_boundary needs a call-time tensor input")
        boundary = _Boundary(cls, plan, entry["family"])
        cls._call_boundary = boundary.binder(cls)
    abc.update_abstractmethods(cls)
    return True


def maybe_install_signature(cls: type) -> bool:
    """Install for an implemented parametric entry; the manifest is read leniently."""
    entry = try_load_entry(cls.__name__)
    if not isinstance(entry, dict) or entry.get("status") != "implemented" or is_legacy(entry):
        return False
    return install(cls, entry)
