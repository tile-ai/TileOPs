#!/usr/bin/env python3
"""Build-time guard for the runner image: fail the build unless the baked stack is coherent.

Runs GPU-free, so it works during `docker build` (no GPU attached). Catches the failure
modes that a plain `import tilelang` smoke check misses:

  1. tilelang imports at all — catches gross ABI breakage that aborts on import (e.g. an
     apache-tvm-ffi too new for the baked wheel, which double-registers and calls abort()).
  2. the installed apache-tvm-ffi satisfies tilelang's own declared requirement — catches
     the case that imports lazily here but crashes the first time a kernel compiles under
     GPU (apache-tvm-ffi too old: `undefined symbol: tvm::ffi::ReprPrint`). A no-GPU import
     does not load the compiler library, so only the version range exposes this at build.
  3. torch is still the cu132 build — a bench baseline that pulls torch from PyPI silently
     swaps it to cu128, which breaks prebuilt c10-ABI extensions (e.g. vllm's `_C`:
     `undefined symbol: c10::cuda::c10_cuda_check_implementation`).
  4. cupti-python is present and did not drag cuda-bindings off torch's pin — it carries its
     own CUDA-runtime requirements, so it is installed with --no-deps; a resolved install that
     moved cuda-bindings only warns, and the benchmark timing path breaks later.
  5. the importable tilelang is the one the tilelang stage installed — vllm depends on an exact
     tilelang release, so a PyPI wheel is present earlier in the build and the source build is
     what must survive. Both are importable, so nothing else in this file tells them apart.
"""
import importlib.metadata as md
import sys
from pathlib import Path

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

# Matches the cu132 base image; bump together with the base/torch CUDA major.minor.
EXPECTED_TORCH_CUDA = "13.2"
# Written by the tilelang stage from the wheel it built (or the release it was asked for).
EXPECTED_TILELANG_VERSION_FILE = Path("/tmp/tilelang-expected-version")

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
        "A bench baseline pulled torch from PyPI; reinstall torch from the cu132 index in "
        "that layer so the c10 ABI stays consistent."
    )

try:
    cupti_version = md.version("cupti-python")
except md.PackageNotFoundError:
    sys.exit(
        "FAIL: cupti-python is missing; the benchmark layer times kernels through it. "
        "Install it with --no-deps (see the post-fa3 stage)."
    )

# No import here: the build has no GPU, and the point of this check is the resolver,
# not the driver. A broken binding surfaces at benchmark time, which fails closed.
bindings_pin = _torch_pin("cuda-bindings")
bindings_installed = md.version("cuda-bindings")
if bindings_pin is not None and not bindings_pin.specifier.contains(
    bindings_installed, prereleases=True
):
    sys.exit(
        f"FAIL: cuda-bindings {bindings_installed} violates torch's requirement "
        f"{bindings_pin.specifier}. cupti-python carries its own cuda-bindings requirement, so "
        "it must be installed with --no-deps; a resolved install moves it and pip only warns."
    )

if not EXPECTED_TILELANG_VERSION_FILE.is_file():
    sys.exit(
        f"FAIL: {EXPECTED_TILELANG_VERSION_FILE} is missing; the tilelang stage records the "
        "version it installed there, so this guard cannot tell a source build from the PyPI "
        "wheel vllm depends on."
    )
expected_tilelang = EXPECTED_TILELANG_VERSION_FILE.read_text().strip()
installed_tilelang = md.version("tilelang")
if installed_tilelang != expected_tilelang:
    sys.exit(
        f"FAIL: tilelang {installed_tilelang} is importable, but this build installed "
        f"{expected_tilelang}. A later step reinstalled tilelang from PyPI — most likely a "
        "bench baseline whose exact pin pip chose to satisfy."
    )

print(
    f"runtime-stack OK: tilelang {tilelang.__version__} | "
    f"torch {torch.__version__} (cuda {torch.version.cuda}) | "
    f"apache-tvm-ffi {installed} satisfies {ffi_req.specifier} | "
    f"cupti-python {cupti_version} with cuda-bindings {bindings_installed} | "
    f"tilelang {installed_tilelang} is the build's own"
)
