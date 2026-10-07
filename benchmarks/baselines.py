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
from typing import Any, Callable

import torch

from benchmarks.api import Implementation

__all__ = [
    "DEEPGEMM_TAG",
    "DEEPSPEED_TAG",
    "FLAGGEMS_TAG",
    "FLASHINFER_TAG",
    "FLA_TAG",
    "QUACK_TAG",
    "TORCH_COMPILE_TAG",
    "VLLM_TAG",
    "backward_of",
    "compiled_reference",
    "deepgemm_op",
    "deepspeed_op",
    "fla_op",
    "flaggems_dims",
    "flaggems_group_norm",
    "flaggems_op",
    "flashinfer_op",
    "private_float32_logits",
    "private_inputs",
    "quack_op",
    "vllm_op",
]

QUACK_TAG = "quack"
DEEPGEMM_TAG = "deepgemm"
DEEPSPEED_TAG = "deepspeed"
TORCH_COMPILE_TAG = "torch-compile"
FLAGGEMS_TAG = "flaggems"
FLASHINFER_TAG = "flashinfer"
FLA_TAG = "fla"
VLLM_TAG = "vllm"


def compiled_reference(
    fn: Callable, *, dynamic: bool = False, preserve_precision: bool = False
) -> Callable:
    """Compile one full graph without an extra eager execution of stateful or random code.

    Reset Dynamo for shared reference code; ``preserve_precision`` retains rounding casts."""
    torch._dynamo.reset()
    compiled: list[Callable] = []

    def compiled_fn(*args: Any, **kwargs: Any) -> Any:
        if not compiled:
            options = {"emulate_precision_casts": True} if preserve_precision else None
            compiled.append(torch.compile(fn, dynamic=dynamic, fullgraph=True, options=options))
        try:
            return compiled[0](*args, **kwargs)
        except (torch._dynamo.exc.Unsupported, torch._dynamo.exc.UserError) as exc:
            raise AssertionError(
                f"{TORCH_COMPILE_TAG}: {getattr(fn, '__name__', fn)} must compile as one full graph"
            ) from exc

    return compiled_fn


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


def quack_op(name: str, module: str = "quack") -> Callable:
    """Resolve a QuACK CuTeDSL kernel from the runner image."""
    return _resolve(module, name, "quack")


def deepgemm_op(name: str) -> Callable:
    """Return the ``deep_gemm`` entry point *name*.

    Its GEMMs write into a caller-allocated output and return ``None``, so an adapter
    allocates before it calls.
    """
    return _resolve("deep_gemm", name, "deepgemm")


def deepspeed_op(name: str) -> Any:
    """Resolve an entry of DeepSpeed's prebuilt quantizer extension."""
    return _resolve("deepspeed.ops.quantizer.quantizer_op", name, DEEPSPEED_TAG)


def flashinfer_op(name: str, module: str = "") -> Callable:
    """Return the ``flashinfer`` entry point *name*, dots allowed for submodules."""
    return _resolve(f"flashinfer.{module}" if module else "flashinfer", name, "flashinfer")


def fla_op(name: str) -> Callable:
    """Return the ``fla`` entry point *name*, dots allowed for submodules."""
    return _resolve("fla", name, "fla")


def vllm_op(name: str, module: str = "_custom_ops") -> Callable:
    """Return the entry point *name* of ``vllm.<module>``, ``vllm._custom_ops`` by default.

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
    return _resolve(f"vllm.{module}", name, "vllm")


def backward_of(output: torch.Tensor) -> Any:
    """Return a callable running *output*'s backward on the thread that calls it.

    How a baseline reaches its gradients: ``Tensor.backward`` hands the graph to
    autograd's engine thread, whose kernels carry no iteration id for the timer to
    attribute them to, and charges the baseline for engine overhead a tileops backward
    op never pays. Takes one gradient per output of the op that produced *output*, so
    one returning ``(out, lse)`` is driven with ``(grad, None)``.
    """
    node = output.grad_fn
    if node is None:
        raise ValueError(
            f"{type(output).__name__} has no grad_fn; build the graph under "
            "enable_grad on inputs that require grad before timing its backward."
        )
    # A Python autograd.Function's node exposes apply() and is not callable; a node
    # built in C++ is callable and has no apply(). Neither offers the other's form.
    return getattr(node, "apply", None) or node


def private_inputs(run: Callable, inputs: tuple, *positions: int) -> Implementation:
    """*run* on *inputs*, with the ones at *positions* replaced by copies it may overwrite.

    The copies are restored from *inputs* before every round, outside the timed call.
    """
    args = list(inputs)
    owned = [i for i in positions if inputs[i] is not None]
    for i in owned:
        args[i] = inputs[i].clone()

    def reset() -> None:
        for i in owned:
            args[i].copy_(inputs[i])

    return Implementation(run=run, args=tuple(args), reset=reset)


def private_float32_logits(
    run: Callable, logits: torch.Tensor, *args: Any, **kwargs: Any
) -> Implementation:
    """*run* on a float32 copy of *logits* it masks in place, restored before every round.

    Takes ``(logits_fp32, *args)``; *kwargs* go to the :class:`Implementation`.
    """
    private = torch.empty_like(logits, dtype=torch.float32)

    def reset() -> None:
        private.copy_(logits)

    return Implementation(run=run, args=(private, *args), reset=reset, **kwargs)
