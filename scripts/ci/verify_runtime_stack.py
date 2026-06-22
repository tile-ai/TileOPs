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
  3. torch is still the cu129 build — a bench baseline that pulls torch from PyPI silently
     swaps it to cu128, which breaks prebuilt c10-ABI extensions (e.g. vllm's `_C`:
     `undefined symbol: c10::cuda::c10_cuda_check_implementation`).
"""
import importlib.metadata as md
import sys

import tilelang
import torch
from packaging.requirements import Requirement

# Matches the cu129 base image; bump together with the base/torch CUDA major.minor.
EXPECTED_TORCH_CUDA = "12.9"

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
        f"FAIL: torch CUDA is {torch.version.cuda}, expected {EXPECTED_TORCH_CUDA} (cu129). "
        "A bench baseline pulled torch from PyPI; reinstall torch from the cu129 index in "
        "that layer so the c10 ABI stays consistent."
    )

print(
    f"runtime-stack OK: tilelang {tilelang.__version__} | "
    f"torch {torch.__version__} (cuda {torch.version.cuda}) | "
    f"apache-tvm-ffi {installed} satisfies {ffi_req.specifier}"
)
