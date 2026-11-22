"""Target identity, target-aware kernel builder caching, and shared CUDA build options."""

import contextlib
import functools
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

import torch

from tileops._utils.device import current_device, device_context, on_current_index

__all__ = [
    "CUDA_COMPILE_FLAGS",
    "BuildTarget",
    "building_on",
    "chain_cuda_postproc",
    "current_target",
    "target_cache",
    "target_of",
]

# nvcc flags for generated CUDA that uses the half and bfloat16 operators and conversions,
# constexpr functions on the device and extended lambdas, with asserts compiled out.
CUDA_COMPILE_FLAGS: tuple[str, ...] = (
    "-O3",
    "-Wno-deprecated-declarations",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-DNDEBUG",
)


class BuildTarget(NamedTuple):
    """The device facts a build compiles for, with their device type."""

    kind: str | None
    arch: int
    sm_count: int
    smem_budget: int
    calibration: str | None
    threads_per_sm: int
    regs_per_sm: int
    smem_per_sm: int
    l2_bytes: int


def target_of(device: torch.device | None) -> BuildTarget:
    """The build target of *device*; ``None`` names the current device."""
    from tileops.backend._dispatch import device_facts_and_tuning

    device = current_device() if device is None else on_current_index(device)
    facts, tuning = device_facts_and_tuning(device)
    kind = None if device is None else device.type
    return BuildTarget(kind, facts.arch, facts.sm_count, facts.smem_budget, *tuning)


def current_target() -> BuildTarget:
    """The build target of the current device: what a build running now compiles for."""
    return target_of(None)


@contextlib.contextmanager
def building_on(device: torch.device | None) -> Iterator[None]:
    """Make *device* current and set its compilation target.

    Match TileLang's architecture spelling so explicit and detected targets share cache entries.
    """
    target = target_of(device)
    with device_context(device):
        if target.kind != "cuda" or target.arch <= 0:
            yield
            return
        from tilelang.contrib.nvcc import get_target_arch
        from tvm.target import Target

        unroll_pipelined_waits()
        sm = get_target_arch((target.arch // 10, target.arch % 10))
        with Target({"kind": "cuda", "arch": f"sm_{sm}"}):
            yield


def target_cache(maxsize: int | None = 128) -> Callable[[Callable], Callable]:
    """LRU cache keyed by arguments and the current build target.

    Compile JIT objects directly to bypass their argument-only cache. Every argument of the
    builder, a ``target`` keyword included, passes to it unchanged.
    """

    def decorate(fn: Callable) -> Callable:
        from tilelang.jit import JITImpl

        jit = isinstance(fn, JITImpl)

        @functools.lru_cache(maxsize=maxsize)
        def cached(target: BuildTarget, /, *args: Any, **kwargs: Any) -> Any:
            with building_on(None):
                unroll_pipelined_waits()
                return fn.compile(*args, **kwargs) if jit else fn(*args, **kwargs)

        @functools.wraps(fn, updated=())
        def keyed(*args: Any, **kwargs: Any) -> Any:
            return cached(current_target(), *args, **kwargs)

        keyed.cache_clear = cached.cache_clear
        keyed.cache_info = cached.cache_info
        return keyed

    return decorate


def chain_cuda_postproc(rewrite: Callable[[str], str]) -> None:
    """Run *rewrite* on generated CUDA after whatever holds TileLang's one post-processing slot.

    TileLang keeps a single CUDA post-processing callback, so each rewrite wraps the one
    registered before it instead of replacing it.
    """
    import tilelang
    import tvm_ffi

    previous = tvm_ffi.get_global_func("tilelang_callback_cuda_postproc", allow_missing=True)

    def postproc(code: str, target: Any) -> str:
        if previous is not None:
            code = previous(code, target)
        return rewrite(code)

    tilelang.register_cuda_postproc(postproc)


def unroll_pipelined_waits() -> None:
    """Unroll each constant-extent unrolled loop holding a cp.async wait on a loop variable.

    TileLang's CUDA codegen rejects the variable ``ptx_wait_group`` count a pipelined
    epilogue leaves; per-iteration copies let Simplify fold each count to a constant.
    Cleanup: drop this hook once tile-ai/tilelang#4424 is fixed.
    """
    import importlib
    import sys

    from tilelang import tvm

    tirx = tvm.tirx
    importlib.import_module("tilelang.engine.lower")
    lower = sys.modules["tilelang.engine.lower"]
    prepare = lower._prepare_device_codegen_mod
    if getattr(prepare, "unrolls_pipelined_waits", False):
        return
    wait = tvm.ir.Op.get("tirx.ptx_wait_group")

    def waits_on_a_variable(stmt: Any) -> bool:
        found = []

        def visit(node: Any) -> None:
            if (
                isinstance(node, tirx.Call)
                and node.op.same_as(wait)
                and not isinstance(node.args[0], tirx.IntImm)
            ):
                found.append(node)

        tirx.stmt_functor.post_order_visit(stmt, visit)
        return bool(found)

    def unroll(node: Any) -> Any:
        if (
            isinstance(node, tirx.For)
            and node.kind == tirx.ForKind.UNROLLED
            and isinstance(node.extent, tirx.IntImm)
            and waits_on_a_variable(node.body)
        ):
            var = node.loop_var
            return tirx.SeqStmt(
                [
                    tirx.stmt_functor.substitute(
                        node.body, {var: node.min + tirx.const(i, var.dtype)}
                    )
                    for i in range(node.extent.value)
                ]
            )
        return None

    def prepare_unrolled(device_mod: Any) -> Any:
        functions = {}
        for gvar, func in device_mod.functions.items():
            if isinstance(func, tirx.PrimFunc) and waits_on_a_variable(func.body):
                func = func.with_body(tirx.stmt_functor.ir_transform(func.body, None, unroll))
            functions[gvar] = func
        # Each unrolled copy redefines the loops it holds; ConvertSSA renames them apart.
        unrolled = tirx.transform.ConvertSSA()(tvm.IRModule(functions, attrs=device_mod.attrs))
        return prepare(unrolled)

    prepare_unrolled.unrolls_pipelined_waits = True
    lower._prepare_device_codegen_mod = prepare_unrolled
