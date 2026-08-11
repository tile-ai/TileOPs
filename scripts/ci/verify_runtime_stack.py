#!/usr/bin/env python3
"""Fail the runner-image build unless the baked stack is coherent. Runs GPU-free.

Each check covers a failure a plain `import tilelang` misses:

  1. tilelang imports — an apache-tvm-ffi too new double-registers and calls abort().
  2. the installed apache-tvm-ffi satisfies tilelang's declared range — too old crashes only
     when a kernel first compiles under GPU, which a no-GPU import never reaches.
  3. torch is still the cu132 build — a baseline pulling torch from PyPI breaks every
     prebuilt c10-ABI extension.
  4. cupti-python is present and left cuda-bindings on torch's pin — it must go in with
     --no-deps; pip only warns when a resolved install moves it.
"""
import importlib.metadata as md
import sys

import tilelang
import torch
from packaging.requirements import Requirement


def _torch_pin(name: str) -> Requirement | None:
    """Return torch's own requirement on *name*, ignoring environment markers."""
    for raw in md.requires("torch") or []:
        req = Requirement(raw)
        if req.name == name:
            return req
    return None

# Bump together with the base image's CUDA major.minor.
EXPECTED_TORCH_CUDA = "13.2"

installed = md.version("apache-tvm-ffi")
ffi_req = next(
    (Requirement(r) for r in (md.requires("tilelang") or [])
     if Requirement(r).name == "apache-tvm-ffi"),
    None,
)
if ffi_req is None:
    sys.exit("FAIL: tilelang declares no apache-tvm-ffi requirement; cannot verify the ABI pin")
if not ffi_req.specifier.contains(installed, prereleases=True):
    sys.exit(
        f"FAIL: apache-tvm-ffi {installed} violates tilelang's requirement {ffi_req.specifier}. "
        "Pin a version inside that range in constraints.txt."
    )

if torch.version.cuda != EXPECTED_TORCH_CUDA:
    sys.exit(
        f"FAIL: torch CUDA is {torch.version.cuda}, expected {EXPECTED_TORCH_CUDA} (cu132). "
        "A bench baseline pulled torch from PyPI; reinstall it from the cu132 index in that layer."
    )

try:
    cupti_version = md.version("cupti-python")
except md.PackageNotFoundError:
    sys.exit(
        "FAIL: cupti-python is missing; the benchmark layer times kernels through it."
    )

# Not imported: no GPU here, and the check is about the resolver, not the driver.
bindings_pin = _torch_pin("cuda-bindings")
bindings_installed = md.version("cuda-bindings")
if bindings_pin is not None and not bindings_pin.specifier.contains(
    bindings_installed, prereleases=True
):
    sys.exit(
        f"FAIL: cuda-bindings {bindings_installed} violates torch's requirement "
        f"{bindings_pin.specifier}. Install cupti-python with --no-deps."
    )

print(
    f"runtime-stack OK: tilelang {tilelang.__version__} | "
    f"torch {torch.__version__} (cuda {torch.version.cuda}) | "
    f"apache-tvm-ffi {installed} satisfies {ffi_req.specifier} | "
    f"cupti-python {cupti_version} with cuda-bindings {bindings_installed}"
)
