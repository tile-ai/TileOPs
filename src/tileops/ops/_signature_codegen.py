"""Methods generated from a parametric signature (docs/design/manifest.md § Call Semantics).

`install` gives an op class, from its manifest entry, the call checks, `_validate_dtypes`,
`_infer_output_shapes`, `_check_construction`, `eval_roofline` and, when the class declares a
compile boundary, one operator per effect branch. The check of each discriminant point is
emitted as Python source when the class is created, so a call parses no expression string and
a traced call only looks its check up. Under SymInt the emitted code lowers `and`, `or`,
`not`, conditionals and sequence equality to their symbolic forms, and a refinement becomes
`torch._check`.
"""

from __future__ import annotations

import abc
import ast
import copy
import importlib
import inspect
import itertools
import math
import string
from dataclasses import dataclass

import torch
from torch._guards import detect_fake_mode
from torch.fx.experimental.symbolic_shapes import sym_and, sym_or

from tileops.manifest import LEGACY_FAMILIES, load_adts, try_load_entry
from tileops.manifest.dtype_rules import DTYPE_BITS
from tileops.manifest.primitives import namespace
from tileops.manifest.signature import (
    Signature,
    SignatureError,
    _discriminant_axes,
    _emitted,
    _field_kind,
    _parse,
    _passed,
    _value_at,
    branch,
    complete_point,
    names,
    param_kind,
    parse_signature,
    rejecting_rule,
    roofline_plan,
    unification,
)
from tileops.manifest.workload import RowError, _param_value

from ._compile_boundary_codegen import operator_name
from .compile_boundary import get_instance

__all__ = ["SignatureCall", "install", "maybe_install_signature"]


# ---------------------------------------------------------------- run-time helpers


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
        raise ValueError(message)


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


def _call_device(name: str, op, tensors: tuple, cpu: tuple, ctor: tuple):
    """The call device (docs/design/manifest.md § Call Semantics)."""
    devices = {t.device for t in tensors}
    if len(devices) > 1:
        raise ValueError(
            f"{name} needs every tensor on one device; got {sorted(map(str, devices))}"
        )
    for t in cpu:
        if t.device.type != "cpu":
            raise ValueError(f"{name}: a tensor declaring `device: cpu` is on {t.device}")
    if devices:
        return devices.pop()
    declared = op._declared_device() if hasattr(op, "_declared_device") else None
    if declared is not None:
        return declared
    held = {t.device for t in ctor}
    if len(held) > 1:
        raise ValueError(f"{name}: construction-time tensors are on {sorted(map(str, held))}")
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


_GLOBALS = {
    **namespace(),
    # The builtins generated code calls; expressions themselves reach only primitives.
    "__builtins__": {
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
    "_call_device": _call_device,
    "torch": torch,
    "_placed": _placed,
}


@dataclass(frozen=True)
class SignatureCall:
    """What one checked call bound: `ix`, every present tensor's shape and dtype, its effects."""

    ix: dict
    # Present tensors, outputs included, as `(shape, dtype name)`.
    tensors: dict
    # `(tensor, reads, writes)` per tensor the effect rules charge.
    traffic: tuple[tuple[str, int, int], ...]
    device: object = None
    # The inputs this call writes, and whether a caller passed `out`.
    written: frozenset = frozenset()
    out: bool = False

    def bytes(self, name: str) -> int:
        shape, dtype = self.tensors[name]
        return (math.prod(shape) * DTYPE_BITS[dtype] + 7) // 8

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
            and node.value.id in self.sig.params
            and node.attr != "kind"
        ):
            # An ADT enum field reads as its `.value`.
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


def _field(value, name: str):
    field = getattr(value, name)
    return getattr(field, "value", field)


_GLOBALS["_field"] = _field


def _code(sig: Signature, node: ast.expr, present=None) -> str:
    return ast.unparse(ast.fix_missing_locations(_Lower(sig, present).visit(copy.deepcopy(node))))


# ---------------------------------------------------------------- the check at one point


def _written(sig: Signature, point: dict, present: set[str]) -> frozenset:
    """The inputs a call at `point` writes: `mutated` where its condition holds, `write_only`."""
    return frozenset(
        t.name
        for t in sig.call_tensors.values()
        if t.name in present
        and (
            t.write_only
            or t.mutated is True
            or (isinstance(t.mutated, str) and _value_at(_parse(t.mutated), point) is True)
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


class _Emitter:
    """The check at one discriminant point, as Python source.

    With `shapes_only` it takes shape tuples and yields output shapes: no dtype or device.
    """

    def __init__(
        self, sig: Signature, point: dict, shapes_only: bool = False, rejected: str | None = None
    ):
        """Emit the check *sig* states at *point*; a *rejected* point fails after presence."""
        self.sig, self.point, self.shapes_only, self.lines = sig, point, shapes_only, []
        self.rejected = rejected

    def emit(self, line: str) -> None:
        self.lines.append(f"    {line}")

    def require(self, cond: str, message: str) -> None:
        self.emit(f"_require({cond}, {f'{self.sig.name}: {message}'!r})")

    def source(self) -> str:  # noqa: C901 - one pass per stage of the check
        sig, point, shapes_only = self.sig, self.point, self.shapes_only
        present = {
            t.name
            for t in (*sig.call_tensors.values(), *sig.outputs.values())
            if (_emitted if t.name in sig.outputs else _passed)(sig, t, point)
        }
        buffer = point.get("present(out)", False)
        known = set(sig.params)
        for p, decl in sig.params.items():
            dtype = param_kind(decl.get("type"), sig.adts).payload().tag == "DType"
            read = f"_dname(self.{p})" if dtype else f"self.{p}"
            self.emit(f"{p} = {read}")
        for t in sig.inputs:
            self.emit(f"{t} = tensors.get({t!r})")
            if t in present:
                self.require(f"{t} is not None", f"{t!r} is required")
                if not shapes_only:
                    self.require(f"isinstance({t}, torch.Tensor)", f"{t!r} is not a tensor")
            else:
                self.require(
                    f"{t} is None", f"{t!r} is given where its presence condition is false"
                )
        for t in sig.ctor_tensors:
            if t in present:
                self.require(f"self.{t} is not None", f"construction-time tensor {t!r} is required")
        if self.rejected is not None:
            self.emit(f"raise ValueError({f'{sig.name}: refinement fails: {self.rejected}'!r})")
            return f"def {'shapes' if shapes_only else 'check'}(self, tensors):\n" + "\n".join(
                self.lines
            )
        b = branch(sig, point)
        if buffer:
            self.emit("out = tensors['out']")
            if not shapes_only:
                self.require("isinstance(out, torch.Tensor)", "'out' is not a tensor")
        inputs = [t for t in sig.inputs if t in present]
        ctor = [t for t in sig.ctor_tensors if t in present]
        if shapes_only:
            for t in inputs:
                self.emit(f"_s_{t} = tuple({t})")
            for t in ctor:
                self.emit(f"_s_{t} = tuple(self.{t}.shape)")
        else:
            self._device_and_dtypes(inputs, ctor, known, buffer)
        self._shapes(b, [*inputs, *ctor], known)
        for i, rule in enumerate(b.rules):
            if isinstance(rule, ast.Constant) and rule.value is True:
                continue
            if shapes_only and names(rule) & {n for n in sig.forall if sig.kind(n).tag == "DType"}:
                continue
            self.require(_code(sig, rule), f"shape_rules[{i}]: {sig.rules[i]}")
        outputs = [o for o in sig.outputs if o in present]
        for o in outputs:
            axes = [
                f"*tuple({_code(sig, a.value)})" if isinstance(a, ast.Starred) else _code(sig, a)
                for a in b.shapes[o].elts
            ]
            self.emit(f"_s_{o} = ({', '.join(axes)}{',' if len(axes) == 1 else ''})")
            self.emit(f"for _v in _s_{o}:")
            self.emit(f"    _require(_v >= 0, {f'{sig.name}: an axis of {o} is negative'!r})")
            if shapes_only:
                continue
            self.emit(f"_d_{o} = {_code(sig, _parse(sig.outputs[o].dtype))}")
            if buffer and sig.outputs[o].buffer:
                self.require(
                    f"_eq(tuple(out.shape), _s_{o})", f"out does not have the shape of {o}"
                )
                self.require(f"_dname(out.dtype) == _d_{o}", f"out does not have the dtype of {o}")
                self.require("out.device == _device", "out is not on the call device")
                if sig.outputs[o].contiguous:
                    self.require("out.is_contiguous()", "out must be contiguous")
        if shapes_only:
            self.emit(f"return {{{', '.join(f'{o!r}: _s_{o}' for o in outputs)}}}")
            return "def shapes(self, tensors):\n" + "\n".join(self.lines)
        solved = sorted(known - set(sig.params))
        self.emit(f"_ix = {{{', '.join(f'{n!r}: {n}' for n in (*sig.params, *solved))}}}")
        shapes = ", ".join(f"{t!r}: (_s_{t}, _d_{t})" for t in (*inputs, *ctor, *outputs))
        written = _written(sig, point, present)
        traffic = _traffic(sig, point, present)
        self.emit(
            f"return _SignatureCall(_ix, {{{shapes}}}, {traffic!r}, _device, "
            f"frozenset({sorted(written)!r}), {buffer!r})"
        )
        return "def check(self, tensors):\n" + "\n".join(self.lines)

    def _device_and_dtypes(
        self, inputs: list[str], ctor: list[str], known: set[str], buffer
    ) -> None:
        """The call device, then dtypes: inputs bind them, construction-time tensors are cast."""
        sig = self.sig
        cpu = [t for t in inputs if sig.inputs[t].cpu]
        held_cpu = [f"self.{t}" for t in ctor if sig.ctor_tensors[t].cpu]
        args = ["".join(f"{t}, " for t in inputs if t not in cpu)]
        args.append("".join(f"{t}, " for t in [*cpu, *held_cpu]))
        args.append("".join(f"self.{t}, " for t in ctor if not sig.ctor_tensors[t].cpu))
        self.emit(
            f"_device = _call_device({sig.name!r}, self, ({args[0]}), ({args[1]}), ({args[2]}))"
        )
        for t in inputs:
            self.emit(f"_s_{t} = tuple({t}.shape)")
            self.emit(f"_d_{t} = _dname({t}.dtype)")
            if sig.inputs[t].contiguous:
                self.require(f"{t}.is_contiguous()", f"{t} must be contiguous")
        deferred = []
        for t in inputs:
            node = _parse(sig.inputs[t].dtype)
            if (
                isinstance(node, ast.Name)
                and node.id in sig.forall
                and sig.kind(node.id).tag == "DType"
            ):
                if node.id in known:
                    self.require(f"_d_{t} == {node.id}", f"{t} dtype differs from {node.id}")
                else:
                    self._bind_dtype(node.id, f"_d_{t}", t)
                    known.add(node.id)
            else:
                deferred.append((t, node))
        for t, node in deferred:
            self.require(f"_d_{t} == {_code(sig, node)}", f"{t} dtype is not {ast.unparse(node)}")
        for t in ctor:
            node = _parse(sig.ctor_tensors[t].dtype)
            unbound = isinstance(node, ast.Name) and node.id in sig.forall and node.id not in known
            if unbound:
                self._bind_dtype(node.id, f"_dname(self.{t}.dtype)", t)
                known.add(node.id)
            device = "None" if sig.ctor_tensors[t].cpu else "_device"
            self.emit(f"{t} = _placed(self, {t!r}, {device}, {_code(sig, node)})")
            self.emit(f"_s_{t} = tuple({t}.shape)")
            self.emit(f"_d_{t} = {_code(sig, node)}")
            if sig.ctor_tensors[t].contiguous:
                self.require(f"{t}.is_contiguous()", f"{t} must be contiguous")
        if sig.dtype_combos:
            columns = sorted(sig.dtype_combos[0])
            rows = sorted({tuple(r[c] for c in columns) for r in sig.dtype_combos})
            self.require(
                f"({', '.join(columns)},) in {rows!r}", f"{columns} is not a dtype_combos row"
            )

    def _bind_dtype(self, index: str, value: str, tensor: str) -> None:
        members = sorted(self.sig.kind(index).values or ())
        self.emit(f"{index} = {value}")
        self.require(f"{index} in {tuple(members)!r}", f"{tensor} dtype is outside {members}")

    def _shapes(self, b, tensors: list[str], known: set[str]) -> None:
        """Rank checks, the inference plan's bindings, then every remaining axis equality."""
        sig = self.sig
        nodes = {t: b.shapes[t] for t in tensors}
        for t, node in nodes.items():
            fixed = sum(not isinstance(e, ast.Starred) for e in node.elts)
            op = ">=" if fixed < len(node.elts) else "=="
            self.require(f"len(_s_{t}) {op} {fixed}", f"{t} needs rank {op} {fixed}")

        def length(parts) -> str:
            lengths = [
                f"len({_code(sig, p.value)})" if isinstance(p, ast.Starred) else "1" for p in parts
            ]
            return " + ".join(lengths) or "0"

        checks = []
        for step in unification(nodes, known, b.lets):
            if step.name:
                known.add(step.name)
            if step.action == "let":
                self.emit(f"{step.name} = {_code(sig, b.lets[step.name])}")
                continue
            t, e = step.tensor, nodes[step.tensor].elts[step.index]
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
                    f"{step.name} = _solve(_s_{t}[{index}], lambda {step.name}: {_code(sig, e)}, "
                    f"{message!r})"
                )
        for t, node in nodes.items():
            if any(isinstance(e, ast.Starred) for e in node.elts):
                self.require(f"len(_s_{t}) == {length(node.elts)}", f"{t} has the wrong rank")
        for cond, t, e in checks:
            self.require(cond, f"{t} shape does not match {ast.unparse(e)}")


def _read_names(sig: Signature) -> set[str]:
    """Every name the signature's expressions read, and every tensor whose presence varies.

    A type family's discriminants reach it as arguments of the shapes that apply it.
    """
    texts = [*sig.rules, *sig.let.values()]
    for t in (*sig.call_tensors.values(), *sig.outputs.values()):
        conditions = (t.optional, t.nullable, t.mutated)
        texts += [t.shape, t.dtype, *(c for c in conditions if isinstance(c, str))]
    read = {t.name for t in sig.call_tensors.values() if t.optional is True} | {"out"}
    return read.union(*(names(_parse(text)) for text in texts))


def _rejecting(sig: Signature, message: str):
    def check(self, tensors):
        raise ValueError(f"{sig.name}: {message}")

    return check


class _Plan:
    """One entry's discriminant axes and the checks emitted for each of their points.

    Every point is emitted when the class is installed, so a traced call only looks one up.
    """

    def __init__(self, sig: Signature):
        """Emit a check, a shape-only check and the effect branch of every point of *sig*.

        A point is keyed by its axes; the presence of tensors whose condition reads them is
        settled here, once.
        """
        self.sig = sig
        # An axis nothing in the signature reads cannot change its checks.
        read = _read_names(sig)
        self.axes = {a: v for a, v in _discriminant_axes(sig).items() if a in read}
        self.keys = sorted(
            {key for key, _ in self.axes.values() if key is not None}
            | {k for key, values in self.axes.values() if key is None for v in values for k in v}
        )
        self.points, self.checks, self.shapes, self.effects = [], {}, {}, {}
        for combo in itertools.product(*(values for _, values in self.axes.values())):
            base = {}
            for (key, _), value in zip(self.axes.values(), combo, strict=True):
                base.update(value if key is None else {key: value})
            key = self.key(base)
            try:
                point = complete_point(sig, base)
            except SignatureError as exc:
                self.checks[key] = self.shapes[key] = _rejecting(sig, str(exc))
                continue
            present = {t for t in sig.call_tensors if _passed(sig, sig.call_tensors[t], point)}
            emitted = frozenset(o for o, d in sig.outputs.items() if _emitted(sig, d, point))
            self.points.append(point)
            self.effects[key] = (
                _written(sig, point, present),
                point.get("present(out)", False),
                emitted,
            )
            self.checks[key] = self._emit(point, shapes_only=False)
            self.shapes[key] = self._emit(point, shapes_only=True)

    def key(self, point: dict) -> tuple:
        return tuple(point.get(k) for k in self.keys)

    def point(self, op, tensors: dict) -> dict:
        point = {}
        for name, (key, values) in self.axes.items():
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

    def check(self, op, tensors: dict) -> SignatureCall:
        return self._lookup(self.checks, op, tensors)(op, tensors)

    def output_shapes(self, op, shapes: dict) -> dict:
        return self._lookup(self.shapes, op, shapes)(op, shapes)

    def effect(self, op, tensors: dict) -> tuple:
        """The inputs this call writes, whether it passes `out`, and the outputs it emits."""
        return self._lookup(self.effects, op, tensors)

    def _lookup(self, table: dict, op, tensors: dict):
        point = self.point(op, tensors)
        fn = table.get(self.key(point))
        if fn is None:
            raise ValueError(f"{self.sig.name}: discriminants {point} are outside their types")
        return fn

    def _emit(self, point: dict, shapes_only: bool):
        rule = rejecting_rule(self.sig, point)
        try:
            source = _Emitter(self.sig, point, shapes_only, rule).source()
        except SignatureError as exc:
            return _rejecting(self.sig, str(exc))
        scope = {**_GLOBALS, "_SignatureCall": SignatureCall}
        exec(compile(source, f"<{self.sig.name} check>", "exec"), scope)  # noqa: S102
        return scope["shapes" if shapes_only else "check"]


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
    if a.numel() == 0 or b.numel() == 0:
        return False
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


# ---------------------------------------------------------------- construction


def _resolve(path: str):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _adt_classes(adt: dict) -> dict:
    """The Python class of each constructor, and of each enum field declared with one."""
    classes = {}
    for ctor, spec in adt["sum"].items():
        fields = {
            f: _resolve(d["python"])
            for f, d in (spec.get("fields") or {}).items()
            if isinstance(d, dict)
        }
        classes[ctor] = (_resolve(spec["python"]), fields)
    return classes


def _adt_ok(adt: dict, classes: dict, value) -> bool:
    """A Python ADT object against its declaration: its class, constructor and field kinds."""
    ctor = getattr(value, "kind", None)
    if ctor not in adt["sum"] or not isinstance(value, classes[ctor][0]):
        return False
    for f, decl in (adt["sum"][ctor].get("fields") or {}).items():
        if not hasattr(value, f):
            return False
        raw, kind = getattr(value, f), _field_kind(decl)
        if f in classes[ctor][1] and not isinstance(raw, classes[ctor][1][f]):
            return False
        v = getattr(raw, "value", raw)
        if kind.tag == "Str":
            ok = v in kind.values
        elif kind.tag == "Bool":
            ok = isinstance(v, bool)
        else:
            ok = isinstance(v, int) and not isinstance(v, bool) and (v >= 0 or not kind.nonneg)
        if not ok:
            return False
    return True


def _construction_check(sig: Signature):
    """Parameter values against their `type`, construction-time tensor presence, ADT invariants."""
    shaped = set().union(
        set(),
        *(names(_parse(t.shape)) for t in (*sig.call_tensors.values(), *sig.outputs.values())),
    )
    kinds = {p: param_kind(d.get("type"), sig.adts).payload() for p, d in sig.params.items()}
    adts = {p: k.name for p, k in kinds.items() if k.tag == "ADT"}
    classes = {name: _adt_classes(sig.adts[name]) for name in set(adts.values())}
    invariants = {
        p: {
            ctor: compile(spec["invariant"], f"<{p} invariant>", "eval")
            for ctor, spec in sig.adts[name]["sum"].items()
            if "invariant" in spec
        }
        for p, name in adts.items()
    }

    def check(self) -> None:
        for p, decl in sig.params.items():
            value, kind = getattr(self, p), kinds[p]
            if p in adts:
                name = adts[p]
                if value is not None and not _adt_ok(sig.adts[name], classes[name], value):
                    raise ValueError(f"{sig.name}: {p} is not a {name} value")
            else:
                plain = _dname(value) if kind.tag == "DType" else value
                plain = list(plain) if isinstance(plain, tuple) else plain
                try:
                    _param_value(sig, p, decl.get("type"), plain)
                except RowError as exc:
                    raise ValueError(f"{sig.name}: {exc}") from None
            values = value if isinstance(value, (list, tuple)) else [value]
            integral = kind.tag == "Int" or (
                kind.tag == "Seq" and getattr(kind.item, "tag", None) == "Int"
            )
            if p in shaped and integral and any(v is not None and v < 0 for v in values):
                raise ValueError(f"{sig.name}: {p} = {value} is negative")
            code = invariants.get(p, {}).get(getattr(value, "kind", None))
            fields = {k: _field(value, k) for k in vars(value)} if code else {}
            if code is not None and not eval(code, dict(_GLOBALS), fields):  # noqa: S307
                raise ValueError(f"{sig.name}: {p} breaks its invariant")
        for t, decl in sig.ctor_tensors.items():
            value = getattr(self, t)
            if decl.optional is False and value is None:
                raise ValueError(f"{sig.name}: construction-time tensor {t!r} is required")
            if value is not None and not isinstance(value, torch.Tensor):
                raise ValueError(f"{sig.name}: construction-time tensor {t!r} is not a tensor")

    return check


# ---------------------------------------------------------------- roofline


def _roofline_method(sig: Signature, plan: dict):
    """`eval_roofline` over the `ix` of the op's last call, from `roofline_plan`."""
    if "func" in plan:
        fn = plan["func"]

        def eval_roofline(self):
            return _counts(sig.name, fn(_last_call(self).ix, self))

        return eval_roofline
    tensors = {*sig.call_tensors, *sig.outputs}
    exprs = [n for n in (plan["flops"], plan["bytes"]) if n is not None]
    read = sorted(set().union(set(), *(names(n) for n in exprs)) - tensors - {"out"})
    body = [f"    {n} = _c.ix[{n!r}]" for n in read]
    flops = _code(sig, plan["flops"], present="_has")
    moved = (
        _code(sig, plan["bytes"], present="_has")
        if plan["bytes"] is not None
        else "_c.derived_bytes()"
    )
    source = "\n".join(
        [
            "def eval_roofline(self):",
            "    _c = _last_call(self)",
            "    _has = lambda t: _c.out if t == 'out' else t in _c.tensors",
            *body,
            f"    return _counts({sig.name!r}, ({flops}, {moved}))",
        ]
    )
    scope = {**_GLOBALS, "_last_call": _last_call, "_counts": _counts}
    exec(compile(source, f"<{sig.name} roofline>", "exec"), scope)  # noqa: S102
    return scope["eval_roofline"]


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
            return built[0] if len(built) == 1 else built

        operator.__name__ = name.replace("::", "_")
        registered = torch.library.custom_op(
            name, mutates_args=tuple(sorted(written)) + (("out",) if out else ()), schema=schema
        )(operator)
        if returned:
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


def _input_binder(sig: Signature, name: str, body):
    """A method taking the signature's inputs as `forward` does, that calls `body(self, tensors)`."""
    formal = ", ".join(f"{t}=None" if d.optional is not False else t for t, d in sig.inputs.items())
    tensors = ", ".join(f"{t!r}: {t}" for t in sig.inputs)
    source = f"def {name}(self, {formal}):\n    return _body(self, {{{tensors}}})"
    scope = {"_body": body}
    exec(compile(source, f"<{sig.name} {name}>", "exec"), scope)  # noqa: S102
    return scope[name]


def install(cls: type, entry: dict, adts: dict | None = None) -> bool:
    """Give `cls` the methods its entry's signature generates; False when the signature is malformed."""
    try:
        sig = parse_signature(cls.__name__, entry, load_adts() if adts is None else adts)
    except SignatureError:
        return False
    plan = _Plan(sig)
    cls._signature = plan
    cls._check_construction = _construction_check(sig)

    cls._validate_dtypes = _input_binder(
        sig, "_validate_dtypes", lambda self, ts: plan.check(self, ts)
    )
    cls._infer_output_shapes = _input_binder(
        sig, "_infer_output_shapes", lambda self, ss: plan.output_shapes(self, ss)
    )
    errors, roofline = roofline_plan(sig, entry.get("roofline"))
    if not errors:
        cls.eval_roofline = _roofline_method(sig, roofline)
    if getattr(cls, "compile_boundary", ()) and sig.inputs:
        boundary = _Boundary(cls, plan, entry["family"])
        cls._call_boundary = boundary.binder(cls)
    abc.update_abstractmethods(cls)
    return True


def maybe_install_signature(cls: type) -> bool:
    """Install for an implemented entry of a converted family; the manifest is read leniently."""
    entry = try_load_entry(cls.__name__)
    if not isinstance(entry, dict) or entry.get("status") != "implemented":
        return False
    if entry.get("family") in LEGACY_FAMILIES:
        return False
    return install(cls, entry)
