#!/usr/bin/env python3
"""Lint module-level constants in ``src/tileops/``: their spelling and their readers.

A name bound once at module scope to a value written out in the source is ``UPPER_SNAKE``,
with or without a leading underscore. A lowercase name reads as state a reader may rebind,
and a short one hides what it holds: ``_pc`` and ``_cf`` in ``gqa/dense.py`` were byte-identical
copies of ``_PASS_CONFIGS`` and ``_COMPILE_FLAGS`` defined 25 lines above them.

Written out means a literal, a dotted name, or a tuple, list, set or dict of those. A value a
call produces is not, so a logger or a device object is left alone. A name any function
declares ``global``, binds twice, or annotates ``TypeAlias`` is not a constant either.

A constant is also judged by its readers (``.claude/rules/code-style.md``). A value one
function reads is a local of that function, beside its reason; a module constant is for a
value several readers must agree on, and the comment above it, or above the run of
constants it opens, says why they share it. A reader is a top-level function or class, or
the module's own top-level code, that loads the name: in the defining module, or in another
module that imports the name or reaches it as an attribute of the defining module. Every
tracked ``.py`` file in the repository is searched. A name some ``__all__`` lists is public
API, read by callers outside the repository, and is not judged by its readers.

Usage: ``module_constant_lint.py [--base REF] [FILE ...]``. With no files, scans
``src/tileops/``. The reader rule judges a constant a diff adds, so it runs only with
``--base``, over the constants whose binding line is added since the merge base of REF and
HEAD (``lint_diff.py``); the hook passes ``--base auto`` to find the TileOPs upstream,
overridable with ``TILEOPS_LINT_BASE``. Exits 1 on a finding and 2 when an explicit base
has no merge base with HEAD.
"""

import argparse
import ast
import io
import subprocess
import sys
import tokenize
from pathlib import Path

from lint_diff import BASE_ENV, NoMergeBaseError, added_lines, relative, resolve_base

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED = "src/tileops"


def _is_written_out(node: ast.expr) -> bool:
    """Whether *node* spells its value out, rather than computing one."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Attribute):
        # A dotted name such as ``tilelang.PassConfigKey.TL_ENABLE_FAST_MATH``, but not an
        # attribute of whatever a call returned.
        while isinstance(node, ast.Attribute):
            node = node.value
        return isinstance(node, ast.Name)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_written_out(e.value if isinstance(e, ast.Starred) else e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            k is not None and _is_written_out(k) and _is_written_out(v)
            for k, v in zip(node.keys, node.values, strict=True)
        )
    if isinstance(node, ast.UnaryOp):
        return _is_written_out(node.operand)
    if isinstance(node, ast.JoinedStr):
        return all(isinstance(part, ast.Constant) for part in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return all(
            isinstance(side, ast.Constant) and isinstance(side.value, str)
            for side in (node.left, node.right)
        )
    return False


def _is_type_alias(node: ast.stmt) -> bool:
    if not isinstance(node, ast.AnnAssign) or node.annotation is None:
        return False
    annotation = node.annotation
    if isinstance(annotation, ast.Attribute):
        return annotation.attr == "TypeAlias"
    return isinstance(annotation, ast.Name) and annotation.id == "TypeAlias"


def _module_statements(body: list[ast.stmt]):
    """Every statement that binds at module scope, through the blocks that open no scope."""
    nesting = (ast.If, ast.Try, ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While)
    for node in body:
        yield node
        if isinstance(node, nesting):
            nested = list(node.body) + list(getattr(node, "orelse", []))
            nested += list(getattr(node, "finalbody", []))
            for handler in getattr(node, "handlers", []):
                nested += list(handler.body)
            yield from _module_statements(nested)


def _bound_names(target: ast.expr) -> list[ast.Name]:
    """Every name a target binds, through tuple, list and starred targets."""
    if isinstance(target, ast.Name):
        return [target]
    if isinstance(target, ast.Starred):
        return _bound_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in _bound_names(element)]
    return []


def _candidates(tree: ast.Module) -> list[tuple[str, int]]:
    """Names bound exactly once at module scope to a value written out in the source."""
    rebound = {
        name for node in ast.walk(tree) if isinstance(node, ast.Global) for name in node.names
    }
    bindings: dict[str, int] = {}
    for node in _module_statements(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            rebound.add(node.name)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            rebound.update(a.asname or a.name.split(".")[0] for a in node.names)
            continue
        if isinstance(node, (ast.AugAssign, ast.For, ast.AsyncFor)):
            rebound.update(name.id for name in _bound_names(node.target))
            continue
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if _is_type_alias(node):
            rebound.update(name.id for t in targets for name in _bound_names(t))
            continue
        for target in targets:
            # A destructuring target takes the matching element of a written-out sequence;
            # against anything else, every name it binds is state rather than a constant.
            if isinstance(target, (ast.Tuple, ast.List)) and isinstance(
                node.value, (ast.Tuple, ast.List)
            ):
                pairs = list(zip(target.elts, node.value.elts, strict=False))
            else:
                pairs = [(target, node.value)]
            for name_node, value in pairs:
                names = _bound_names(name_node)
                if len(names) != 1 or names[0] is not name_node or not _is_written_out(value):
                    rebound.update(name.id for name in names)
                    continue
                if name_node.id in bindings:
                    rebound.add(name_node.id)
                else:
                    bindings[name_node.id] = name_node.lineno
    return sorted((n, line) for n, line in bindings.items() if n not in rebound)


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def findings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    return [
        f"{rel}:{line}: module constant {name} is not UPPER_SNAKE"
        for name, line in _candidates(tree)
        if not _is_dunder(name) and name.lstrip("_") != name.lstrip("_").upper()
    ]


def _is_constant_name(name: str) -> bool:
    bare = name.lstrip("_")
    return bool(bare) and not _is_dunder(name) and bare == bare.upper() and bare[0].isalpha()


class _Bindings(ast.NodeVisitor):
    """Names local to one lexical scope, without descending into nested scopes."""

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.globals: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_arg(self, node: ast.arg) -> None:
        self.names.add(node.arg)

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(alias.asname or alias.name for alias in node.names)

    def visit_Global(self, node: ast.Global) -> None:
        self.globals.update(node.names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass

    def visit_ListComp(self, node: ast.ListComp) -> None:
        pass

    def visit_SetComp(self, node: ast.SetComp) -> None:
        pass

    def visit_DictComp(self, node: ast.DictComp) -> None:
        pass

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        pass


def _scope_bindings(body: list[ast.stmt], args: ast.arguments | None = None) -> set[str]:
    visitor = _Bindings()
    if args is not None:
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            visitor.visit(arg)
        if args.vararg:
            visitor.visit(args.vararg)
        if args.kwarg:
            visitor.visit(args.kwarg)
    for statement in body:
        visitor.visit(statement)
    return visitor.names - visitor.globals


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for element in target.elts for name in _target_names(element)}
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return set()


class _Reader(ast.NodeVisitor):
    """Whether one top-level owner reads a module binding rather than a shadow."""

    def __init__(self, local: set[str], modules: set[str], attribute: str) -> None:
        self.local = local
        self.modules = modules
        self.attribute = attribute
        self.scopes: list[set[str]] = []
        self.found = False

    def _shadowed(self, name: str) -> bool:
        return any(name in scope for scope in self.scopes)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in self.local and not self._shadowed(node.id):
            self.found = True

    def visit_Attribute(self, node: ast.Attribute) -> None:
        value = ast.unparse(node.value)
        root = value.split(".", 1)[0]
        if (
            isinstance(node.ctx, ast.Load)
            and node.attr == self.attribute
            and value in self.modules
            and not self._shadowed(root)
        ):
            self.found = True
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for expression in (*node.decorator_list, *node.args.defaults, *node.args.kw_defaults):
            if expression is not None:
                self.visit(expression)
        if node.returns:
            self.visit(node.returns)
        self.scopes.append(_scope_bindings(node.body, node.args))
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for expression in (*node.args.defaults, *node.args.kw_defaults):
            if expression is not None:
                self.visit(expression)
        self.scopes.append(_scope_bindings([], node.args))
        self.visit(node.body)
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        bindings = _scope_bindings(node.body)
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(statement)
            else:
                self.scopes.append(bindings)
                self.visit(statement)
                self.scopes.pop()

    def _visit_comprehension(self, node: ast.ListComp | ast.SetComp | ast.GeneratorExp) -> None:
        self.visit(node.generators[0].iter)
        self.scopes.append(set())
        for index, generator in enumerate(node.generators):
            if index:
                self.visit(generator.iter)
            self.scopes[-1] |= _target_names(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        self.visit(node.elt)
        self.scopes.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.visit(node.generators[0].iter)
        self.scopes.append(set())
        for index, generator in enumerate(node.generators):
            if index:
                self.visit(generator.iter)
            self.scopes[-1] |= _target_names(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        self.visit(node.key)
        self.visit(node.value)
        self.scopes.pop()


def _module_name(path: Path, root: Path) -> str | None:
    try:
        rel = path.resolve().relative_to((root / "src").resolve())
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _exports(tree: ast.Module, name: str) -> bool:
    """Whether an ``__all__`` binding in *tree* lists *name*."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        if node.value is not None and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in targets
        ):
            listed = ast.walk(node.value)
            if any(isinstance(c, ast.Constant) and c.value == name for c in listed):
                return True
    return False


def _readers_in(tree: ast.Module, name: str, module: str | None, own: bool) -> set[str]:
    """The readers of *name* in one module; *own* is whether that module defines it."""
    local = {name} if own else set()
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == name:
                    local.add(alias.asname or name)
                elif module and f"{node.module}.{alias.name}" == module:
                    modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import) and module:
            for alias in node.names:
                if alias.name == module:
                    modules.add(alias.asname or alias.name)

    readers = set()
    for statement in tree.body:
        owner = (
            statement.name
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            else "<module>"
        )
        visitor = _Reader(local, modules, name)
        visitor.visit(statement)
        if visitor.found:
            readers.add(owner)
    return readers


def _explained(source: str, tree: ast.Module, line: int) -> bool:
    """Whether a comment sits on *line*, above it, or above the run of constants it is in."""
    commented = set()
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                commented.add(token.start[0])
    except (tokenize.TokenError, SyntaxError):
        return False
    lines = source.splitlines()
    spans = {
        node.end_lineno: node
        for node in _module_statements(tree.body)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    }
    statement = next((n for n in spans.values() if n.lineno <= line <= n.end_lineno), None)
    first, last = (statement.lineno, statement.end_lineno) if statement else (line, line)
    if commented & set(range(first, last + 1)):
        return True
    above = first - 1
    while above >= 1:
        if above in commented and lines[above - 1].lstrip().startswith("#"):
            return True
        previous = spans.get(above)
        targets = [] if previous is None else _targets(previous)
        if not targets or not all(_is_constant_name(n.id) for n in targets):
            return False
        above = previous.lineno - 1
    return False


def _targets(node: ast.Assign | ast.AnnAssign) -> list[ast.Name]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [name for target in targets for name in _bound_names(target)]


def _grep(name: str, root: Path) -> list[Path]:
    done = subprocess.run(
        ["git", "grep", "-l", "-w", "-F", name, "--", "*.py"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return [root / p for p in done.stdout.splitlines()]


def reader_findings(path: Path, added: set[int] | None = None, root: Path = REPO_ROOT) -> list[str]:
    """Reader-rule findings for the constants of *path* bound on *added* lines (None: all)."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    module = _module_name(path, root)
    rel = path.relative_to(root) if path.is_relative_to(root) else path
    out = []
    for name, line in _candidates(tree):
        if not _is_constant_name(name) or (added is not None and line not in added):
            continue
        if _exports(tree, name):
            continue
        readers = _readers_in(tree, name, module, own=True)
        exported = False
        for other in _grep(name, root):
            if other.resolve() == path.resolve():
                continue
            try:
                other_tree = ast.parse(other.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue
            if _exports(other_tree, name):
                exported = True
                break
            readers |= {
                f"{other.relative_to(root)}:{r}"
                for r in _readers_in(other_tree, name, module, own=False)
            }
        if exported:
            continue
        if not readers:
            out.append(f"{rel}:{line}: module constant {name} has no reader — delete it")
        elif len(readers) == 1:
            (reader,) = readers
            out.append(
                f"{rel}:{line}: module constant {name} has one reader, {reader} — "
                f"make it a local of {reader}"
            )
        elif not _explained(source, tree, line):
            out.append(
                f"{rel}:{line}: module constant {name} is shared by {len(readers)} readers "
                "and no comment above it says why they share it"
            )
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="judge the constants added since the merge base")
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    paths = [
        p
        for p in (args.files or sorted((REPO_ROOT / SCANNED).rglob("*.py")))
        if p.suffix == ".py" and "__pycache__" not in p.parts
    ]
    errors = [e for p in paths for e in findings(p)]
    if args.base:
        try:
            base = resolve_base(args.base, REPO_ROOT)
            if base is None:
                print(
                    "module_constant_lint: skipping reader check: no fetched "
                    f"tile-ai/TileOPs main remote; set {BASE_ENV} to a base ref",
                    file=sys.stderr,
                )
                return 1 if errors else 0
            added = added_lines(paths, base, REPO_ROOT)
        except NoMergeBaseError as exc:
            print(f"module_constant_lint: {exc}", file=sys.stderr)
            return 2
        for path in paths:
            lines = added.get(relative(path, REPO_ROOT))
            if lines != set():
                errors += reader_findings(path, lines)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
