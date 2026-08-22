"""Third-party baselines the bench files time next to their torch reference.

A resolver raises when its library is absent rather than let a tag that names a
library report torch: ``.github/runner/Dockerfile`` installs every one of them
non-fatally, so a degraded image has to fail the row it degraded. A kernel that
cannot express the case is the bench file's call: that row carries no tag for the
library and says why.

Importing this module also arms :class:`_FlagGemsImportOrder`.
"""

import contextlib
import importlib
import importlib.abc
import sys
from typing import Any, Callable, Optional

import torch

__all__ = [
    "FLAGGEMS_TAG",
    "FLASHINFER_TAG",
    "TORCH_COMPILE_TAG",
    "VLLM_TAG",
    "assert_matches_reference",
    "compiled_reference",
    "flaggems_dims",
    "flaggems_group_norm",
    "flaggems_op",
    "flashinfer_op",
    "reference_tolerance",
    "vllm_op",
]

# docs/design/testing.md's per-dtype tolerances. A baseline is checked at the same
# strength a test checks an op: the question is the same one.
_TOLERANCES = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1.6e-2, 1.6e-2),
    torch.float32: (1e-5, 1e-5),
    torch.float64: (1e-7, 1e-7),
}

TORCH_COMPILE_TAG = "torch-compile"
FLAGGEMS_TAG = "flaggems"
FLASHINFER_TAG = "flashinfer"
VLLM_TAG = "vllm"


def compiled_reference(fn: Callable, *, dynamic: bool = False) -> Callable:
    """Return *fn* compiled by inductor, resetting dynamo first.

    dynamo caches eight graphs per code object and every case of a bench file
    shares one reference callable, so without the reset the later cases would
    time eager under a tag that says compiled.
    """
    torch._dynamo.reset()
    return torch.compile(fn, dynamic=dynamic)


def _resolve(module: str, attr: str, library: str) -> Any:
    """Return ``module.attr``, naming the library in both failure messages."""
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(
            f"{library} is a selected baseline for this case; install it "
            f"(see .github/runner/Dockerfile) or drop the tag from the bench file"
        ) from exc
    target = mod
    for part in attr.split("."):
        target = getattr(target, part, None)
        if target is None:
            raise RuntimeError(
                f"{library} {getattr(mod, '__version__', '?')} does not expose "
                f"{module}.{attr}; the adapter needs updating for this version"
            )
    return target


def _claim_registry_before_flaggems() -> None:
    """Import vllm's custom ops, if installed, before flag_gems registers any.

    In the other order the process aborts: flag_gems claims schemas that vllm's
    ``_moe_C`` then re-defines, and the failed ``aoti_torch_library_def`` throws
    through a C++ static initializer, past any ``except``. Costs the vllm import
    (5.9s, measured in the runner image) in every process that reaches flag_gems.
    """
    with contextlib.suppress(ImportError):
        importlib.import_module("vllm._custom_ops")


class _FlagGemsImportOrder(importlib.abc.MetaPathFinder):
    """Hold that order for every importer, not just :func:`flaggems_op`.

    Returns ``None`` always: it claims no module, it only runs first.
    """

    _importing = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "flag_gems" and not fullname.startswith("flag_gems."):
            return None
        if _FlagGemsImportOrder._importing:
            return None
        _FlagGemsImportOrder._importing = True
        try:
            _claim_registry_before_flaggems()
        finally:
            _FlagGemsImportOrder._importing = False
        return None


def _install_flaggems_import_order() -> None:
    """Arm the guard once. ``benchmarks/conftest.py`` imports this module for it."""
    if any(isinstance(finder, _FlagGemsImportOrder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _FlagGemsImportOrder())


_install_flaggems_import_order()


def _builds_pointwise_kernel(fn: Callable) -> bool:
    """Is *fn* built by flag_gems' ``pointwise_dynamic``?

    Structural rather than a list of names: the module defining such an op holds a
    ``pointwise_dynamic`` kernel object, and no other flag_gems module does.
    """
    module = sys.modules.get(getattr(fn, "__module__", ""))
    if module is None:
        return False
    return any(
        type(value).__module__.startswith("flag_gems.utils.pointwise_dynamic")
        for value in vars(module).values()
    )


def flaggems_op(name: str) -> Callable:
    """Return the ``flag_gems.ops`` entry point *name*.

    Its parameters follow the aten schema, not the ``torch.nn.functional``
    signature: ``softmax(self, dim, half_to_float=False)``,
    ``group_norm(input, weight, bias, N, C, HxW, group, eps)``.

    Raises:
        RuntimeError: When *name* is built by ``pointwise_dynamic``, whose second
            launch aborts the process.
    """
    fn = _resolve("flag_gems.ops", name, "flag_gems")
    if _builds_pointwise_kernel(fn):
        raise RuntimeError(
            f"flag_gems.ops.{name} goes through LibEntry, whose argument cache "
            "misaligns under triton 3.7: it returns once, then aborts the process on "
            "the second launch, which a timing loop reaches immediately. Reaching such "
            "a kernel means launching it through triton's own Autotuner, the way "
            "benchmarks/ops/bench_pool.py does for two named pooling kernels"
        )
    return fn


def flaggems_dims(dim) -> list:
    """Wrap a manifest row's ``dim`` into the list flag_gems' reductions take."""
    return list(dim) if isinstance(dim, (list, tuple)) else [dim]


def flaggems_group_norm(n: int, c: int, hxw: int, groups: int, eps: float) -> Callable:
    """Return flag_gems' group_norm bound to this geometry, output only.

    It takes the geometry rather than reading it off the input, and ``None`` for
    an unaffine row.
    """
    fn = flaggems_op("group_norm")

    def baseline_fn(x, weight=None, bias=None):
        return fn(x, weight, bias, n, c, hxw, groups, eps)[0]

    return baseline_fn


def flashinfer_op(name: str) -> Callable:
    """Return the ``flashinfer`` entry point *name*, dots allowed for submodules."""
    return _resolve("flashinfer", name, "flashinfer")


def vllm_op(name: str) -> Callable:
    """Return the ``vllm._custom_ops`` entry point *name*.

    Most of them write into a caller-allocated out tensor and return ``None``,
    so an adapter has to allocate before the timed region, not inside it.

    Raises:
        RuntimeError: When flag_gems got to the registry first, which
            :func:`_claim_registry_before_flaggems` explains. Importing vllm here
            would abort the process, so this reports it instead.
    """
    if "flag_gems" in sys.modules and "vllm._custom_ops" not in sys.modules:
        raise RuntimeError(
            "flag_gems was imported before vllm, and importing vllm now would abort "
            "the process inside a C++ static initializer. Import benchmarks.baselines "
            "before anything that imports flag_gems (benchmarks/conftest.py does), or "
            "resolve flag_gems through flaggems_op"
        )
    return _resolve("vllm._custom_ops", name, "vllm")


def reference_tolerance(dtype: torch.dtype) -> dict[str, float]:
    """Return ``rtol``/``atol`` for *dtype*, ready to splat into an assertion.

    A dtype outside the table takes no tolerance override, leaving
    ``assert_close`` on its own defaults.
    """
    rtol_atol = _TOLERANCES.get(dtype)
    if rtol_atol is None:
        return {}
    return {"rtol": rtol_atol[0], "atol": rtol_atol[1]}


def assert_matches_reference(
    fn: Callable,
    reference: Callable,
    *inputs: Any,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
) -> None:
    """Check a baseline against the reference, output by output.

    A baseline may return more than the reference does — saved statistics an aten
    signature carries — and those extra outputs go unchecked.

    Raises:
        AssertionError: When an output disagrees, or the baseline returns fewer
            outputs than the reference.
    """
    got, expected = fn(*inputs), reference(*inputs)
    tolerances = {}
    if rtol is not None:
        tolerances["rtol"] = rtol
    if atol is not None:
        tolerances["atol"] = atol
    if tolerances:
        tolerances.setdefault("rtol", 0.0)
        tolerances.setdefault("atol", 0.0)

    if not isinstance(expected, (tuple, list)):
        got = got[0] if isinstance(got, (tuple, list)) else got
        torch.testing.assert_close(got, expected, **tolerances)
        return
    if not isinstance(got, (tuple, list)) or len(got) < len(expected):
        raise AssertionError(
            f"baseline returned {1 if not isinstance(got, (tuple, list)) else len(got)} "
            f"output(s), the reference {len(expected)}"
        )
    for index, (got_i, expected_i) in enumerate(zip(got[: len(expected)], expected, strict=True)):
        torch.testing.assert_close(
            got_i,
            expected_i,
            msg=lambda message, index=index: f"output {index}: {message}",
            **tolerances,
        )
