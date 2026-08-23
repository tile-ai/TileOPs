#!/usr/bin/env python3
"""Fail the runner-image build unless the baked stack is coherent. Runs GPU-free.

Two checks are not self-evident. The bare `import tilelang` is one: an apache-tvm-ffi too new
double-registers and calls abort(). The apache-tvm-ffi range check is the other: too old crashes
only when a kernel first compiles under GPU, which no import here reaches. Each failure below
names its own fix.
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
    (
        Requirement(r)
        for r in (md.requires("tilelang") or [])
        if Requirement(r).name == "apache-tvm-ffi"
    ),
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
    sys.exit("FAIL: cupti-python is missing; the benchmark layer times kernels through it.")

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

# A missing baseline costs the column, not the run, so nothing else notices. FA3 is
# imported because a returned 0 has meant an installed wrapper with no extension behind
# it; the rest are checked for presence, since flag_gems and flashinfer want a device.
try:
    import flash_attn_interface

    assert flash_attn_interface.flash_attn_func is not None
except Exception as exc:  # noqa: BLE001 - a half-built FA3 raises several ways
    sys.exit(
        f"FAIL: flash_attn_interface does not import ({exc}). Eleven attention "
        "benchmarks name `fa3` and would silently drop the column."
    )

for dist, why in (
    ("flash-attn", "the FA2 columns"),
    ("flag_gems", "the flaggems columns"),
    ("flashinfer-python", "the flashinfer columns"),
    ("flash-linear-attention", "the linear-attention columns"),
):
    try:
        md.version(dist)
    except md.PackageNotFoundError:
        sys.exit(f"FAIL: {dist} is not installed; {why} would be missing.")

print(
    f"runtime-stack OK: tilelang {tilelang.__version__} | "
    f"torch {torch.__version__} (cuda {torch.version.cuda}) | "
    f"apache-tvm-ffi {installed} satisfies {ffi_req.specifier} | "
    f"cupti-python {cupti_version} with cuda-bindings {bindings_installed}"
)
