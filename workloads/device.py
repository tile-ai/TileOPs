"""The device a test or benchmark run places its tensors on.

The run names it (``--tileops-device`` in ``tests/conftest.py``); every workload and every test
that is not ``cuda_only`` reads it here. Read it when a tensor is built: a module constant or a
default argument would keep the value from before the run set it.
"""

from __future__ import annotations

import functools

import torch

_device: torch.device | str = "cuda"


def run_device() -> torch.device | str:
    """The device this run places tensors on; ``"cuda"`` unless the run named another."""
    return _device


def run_device_is_cuda() -> bool:
    """Whether this run places tensors on a CUDA device."""
    return torch.device(_device).type == "cuda"


def set_run_device(device: torch.device | str) -> None:
    """Place this run's tensors on *device* from now on."""
    global _device
    _device = device


def run_device_available() -> bool:
    """Whether this process can place tensors on the run device."""
    kind = torch.device(_device).type
    if kind in ("cpu", "meta"):
        return True
    return getattr(torch, kind).is_available()


@functools.cache
def torch_compile_unavailable(device: torch.device) -> str | None:
    """Why Inductor cannot compile for *device*, or ``None`` when it can.

    A device type with no Inductor backend registered is not compiled for. Otherwise compiles
    ``x + 1`` once, and only a missing Triton counts; any other failure raises. CUDA is not
    probed: the runner image must compile.
    """
    if device.type == "cuda":
        return None
    from torch._inductor.codegen.common import get_scheduling_for_device, init_backend_registration

    init_backend_registration()
    if get_scheduling_for_device(device.type) is None:
        return f"no Inductor backend is registered for {device.type}"
    x = torch.ones(1, device=device)
    try:
        torch.compile(lambda t: t + 1, fullgraph=True)(x)
    except Exception as exc:
        verdict = _device_verdict(exc)
        if verdict is None:
            raise
        return f"{type(verdict).__name__}: {verdict}"
    return None


def _device_verdict(exc: BaseException) -> BaseException | None:
    """The error in *exc*'s chain saying Triton, which Inductor compiles with, is missing."""
    from torch._inductor.exc import TritonMissing

    seen: set[int] = set()
    pending: list[BaseException | None] = [exc]
    while pending:
        err = pending.pop()
        if err is None or id(err) in seen:
            continue
        seen.add(id(err))
        if isinstance(err, TritonMissing):
            return err
        if isinstance(err, ModuleNotFoundError) and (err.name or "").split(".")[0] == "triton":
            return err
        # Inductor's and Dynamo's compiler errors keep what they wrap as ``inner_exception``.
        context = None if err.__suppress_context__ else err.__context__
        pending += [err.__cause__, context, getattr(err, "inner_exception", None)]
    return None
