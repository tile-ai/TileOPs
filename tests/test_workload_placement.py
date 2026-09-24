"""Structural gates for the workloads layer.

Input construction and the op's reference computation live in
``workloads/<family>.py`` so both the test stage and the benchmark stage read one
definition. Tolerances, checks and roofline numbers do not: those are decisions,
and a decision placed there reaches the other stage.

See docs/design/trust-model.md §Test and §Workloads Layer.
"""

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

# Tolerance and roofline names — decisions, not definitions.
NOT_IN_WORKLOADS = ("check", "calculate_flops", "calculate_memory")


def _methods_named(root: Path, wanted) -> dict[str, list[str]]:
    """Map each file under ``root`` to the wanted methods any class defines."""
    offenders = {}
    for path in sorted(root.rglob("*.py")):
        if path.name == "__init__.py" or path.resolve() == SELF:
            continue
        hits = [
            f"{node.name}.{m.name} (line {m.lineno})"
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
            if isinstance(node, ast.ClassDef)
            for m in node.body
            if isinstance(m, ast.FunctionDef) and wanted(m.name)
        ]
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = hits
    return offenders


@pytest.mark.smoke
def test_tests_do_not_author_gen_inputs() -> None:
    assert _methods_named(REPO_ROOT / "tests", lambda n: n == "gen_inputs") == {}


@pytest.mark.smoke
def test_workloads_carry_no_decisions() -> None:
    assert _methods_named(REPO_ROOT / "workloads", NOT_IN_WORKLOADS.__contains__) == {}


def _seeds_global_rng(node: ast.AST) -> bool:
    """Whether *node* is a call that seeds the global RNG.

    ``torch.manual_seed(...)`` and the imported ``manual_seed(...)`` do.
    ``generator.manual_seed(...)`` does not: it seeds a generator the workload
    owns, which is what the rule asks for.
    """
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id == "manual_seed"
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "manual_seed"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    )


def _global_seed_calls(root: Path) -> dict[str, list[str]]:
    """Map each file under *root* to the calls that seed the global RNG."""
    offenders = {}
    for path in sorted(root.rglob("*.py")):
        hits = [
            f"line {node.lineno}"
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
            if _seeds_global_rng(node)
        ]
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = hits
    return offenders


@pytest.mark.smoke
def test_workloads_do_not_seed_the_global_rng() -> None:
    """A workload that reseeds the global RNG moves every later draw in the session.

    The conftests seed it once per test; an input that must not move with the
    stream takes ``WorkloadBase.rng()`` instead.
    """
    assert _global_seed_calls(REPO_ROOT / "workloads") == {}


def _names_cuda(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and re.fullmatch(r"cuda(:\d+)?", node.value) is not None
    )


def _cuda_placements(source: str) -> list[int]:
    """Lines of *source* that place a tensor on CUDA themselves.

    A ``"cuda"`` or ``"cuda:<n>"`` string anywhere counts — ``device="cuda"``,
    ``.to("cuda")``, ``torch.device("cuda")``, a module constant — as do a
    ``.cuda()`` call and any ``torch.cuda`` use. The one exception is the default
    of a parameter named ``device``, which is what lets every existing caller keep
    its CUDA inputs.
    """
    tree = ast.parse(source)
    defaults = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = fn.args
            positional = args.posonlyargs + args.args
            for arg, default in zip(
                positional[len(positional) - len(args.defaults) :], args.defaults, strict=True
            ):
                if arg.arg == "device":
                    defaults.add(id(default))
            for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                if arg.arg == "device" and default is not None:
                    defaults.add(id(default))
    lines = set()
    for node in ast.walk(tree):
        cuda_call = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "cuda"
        )
        torch_cuda = (
            isinstance(node, ast.Attribute)
            and node.attr == "cuda"
            and isinstance(node.value, ast.Name)
            and node.value.id == "torch"
        )
        if (_names_cuda(node) and id(node) not in defaults) or cuda_call or torch_cuda:
            lines.add(node.lineno)
    return sorted(lines)


@pytest.mark.smoke
def test_workloads_place_tensors_where_the_caller_asks() -> None:
    """``gen_inputs(device=...)`` reaches every tensor a workload builds.

    A backend checks its kernels on its own device with these inputs, so a
    workload that pins one tensor to CUDA breaks that run for the whole op.
    """
    offenders = {
        str(path.relative_to(REPO_ROOT)): hits
        for path in sorted((REPO_ROOT / "workloads").rglob("*.py"))
        if (hits := _cuda_placements(path.read_text()))
    }
    assert offenders == {}


@pytest.mark.smoke
def test_the_cuda_placement_check_sees_every_form() -> None:
    source = "\n".join(
        [
            "import torch",
            'DEV = "cuda"',
            "def f(x, *, device='cuda'):",
            '    a = torch.empty(1, device="cuda")',
            '    b = x.to("cuda")',
            '    c = x.to(device="cuda:0")',
            '    d = torch.device("cuda")',
            '    g = torch.Generator(device="cuda")',
            "    e = x.cuda()",
            "    n = torch.cuda.device_count()",
            '    msg = "needs a cuda device"',
            '    arch = "cuda:sm90"',
            "    flag = config.cuda",
            "    return torch.empty(1, device=device)",
        ]
    )
    assert _cuda_placements(source) == [2, 4, 5, 6, 7, 8, 9, 10]


@pytest.mark.smoke
def test_a_drawn_domain_follows_the_requested_device() -> None:
    """A workload that picks its draw from a table still hands it the device."""
    import torch

    from workloads.elementwise import (
        BROADCAST_DOMAINS,
        PAIR_DOMAINS,
        BinaryBenchCase,
        BroadcastBenchCase,
        RandnFlatWorkload,
        draw_positive_away_from_zero,
    )

    cases = [BinaryBenchCase((8,), torch.float32, torch.float32, domain) for domain in PAIR_DOMAINS]
    cases += [
        BroadcastBenchCase((4, 8), (1, 8), torch.float32, torch.float32, domain)
        for domain in BROADCAST_DOMAINS
    ]
    cases.append(
        RandnFlatWorkload(
            8,
            torch.float32,
            gen_fn=lambda n, d, *, device: draw_positive_away_from_zero((n,), d, device=device)[0],
        )
    )
    for case in cases:
        assert {t.device.type for t in case.gen_inputs(device="cpu")} == {"cpu"}, case
