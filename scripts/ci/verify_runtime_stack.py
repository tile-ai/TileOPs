#!/usr/bin/env python3
"""Fail the runner-image build unless the baked stack is coherent. Runs GPU-free.

Two checks are not self-evident. The bare `import tilelang` is one: an apache-tvm-ffi too new
double-registers and calls abort(). The apache-tvm-ffi range check is the other: too old crashes
only when a kernel first compiles under GPU, which no import here reaches. Each failure below
names its own fix.
"""

import importlib.metadata as md
import importlib.util
import site
import sys
from pathlib import Path

import tilelang
import torch
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# The pins the image replaces for every install.
OVERRIDES = Path("/tmp/overrides.txt")
# Requirements of a package installed --no-deps that the image leaves unmet, by (package,
# requirement). Any other unmet requirement fails the build.
NO_DEPS_GAPS = {
    # flag_gems 5.0.2 pins exact versions of three packages the lock resolves otherwise.
    ("flag-gems", "numpy"),
    ("flag-gems", "sqlalchemy"),
    ("flag-gems", "packaging"),
}


def _torch_pin(name: str) -> Requirement | None:
    """Return torch's own requirement on *name*, ignoring environment markers."""
    for raw in md.requires("torch") or []:
        req = Requirement(raw)
        if req.name == name:
            return req
    return None


def _overridden() -> set[str]:
    """Names of the requirements overrides.txt replaces for every install."""
    if not OVERRIDES.exists():
        sys.exit(f"FAIL: {OVERRIDES} is missing; the Dockerfile copies it beside the constraints.")
    lines = (line.split("#", 1)[0].strip() for line in OVERRIDES.read_text().splitlines())
    return {canonicalize_name(Requirement(line).name) for line in lines if line}


def _unsatisfied() -> list[str]:
    """Every declared requirement of an installed package that the environment does not meet.

    A requirement naming extras, such as ``nvidia-cutlass-dsl[cu13]``, also checks what those
    extras of the named package require.
    """
    # The interpreter's site directories only: importing setuptools puts its vendored copies of
    # packaging and wheel on sys.path, and those are setuptools' own, not installs. The first copy
    # found wins, as for an import; a second one is reported.
    dists: dict[str, md.Distribution] = {}
    found = []
    for dist in md.distributions(path=site.getsitepackages()):
        name = canonicalize_name(dist.metadata["Name"])
        first = dists.setdefault(name, dist)
        if first is not dist and first.locate_file("") != dist.locate_file(""):
            found.append(
                f"{name} is installed twice: {first.version} at {first.locate_file('')} "
                f"and {dist.version} at {dist.locate_file('')}"
            )
    overridden = _overridden()
    pending = [(name, "") for name in dists]
    seen = set(pending)
    while pending:
        name, extra = pending.pop()
        for raw in dists[name].requires or []:
            req = Requirement(raw)
            if req.marker is not None and not req.marker.evaluate({"extra": extra}):
                continue
            if extra and (req.marker is None or "extra" not in str(req.marker)):
                continue  # Already checked with the package's base requirements.
            dep = canonicalize_name(req.name)
            have = dists[dep].version if dep in dists else None
            if dep in overridden or (name, dep) in NO_DEPS_GAPS:
                continue
            if have is None or not req.specifier.contains(have, prereleases=True):
                where = f"{name}[{extra}]" if extra else name
                found.append(f"{where} requires {req}, found {have or 'nothing'}")
                continue
            for wanted in req.extras:
                key = (dep, canonicalize_name(wanted))
                if key not in seen:
                    seen.add(key)
                    pending.append(key)
    return sorted(found)


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

# install_tileops.sh builds tileops with --no-build-isolation, which resolves no backend itself.
for dist in ("setuptools", "setuptools-scm", "wheel"):
    try:
        md.version(dist)
    except md.PackageNotFoundError:
        sys.exit(f"FAIL: {dist} is missing; tileops cannot build with --no-build-isolation.")

# A missing baseline costs the column, not the run, so nothing else notices. FA3 is
# imported because a returned 0 has meant an installed wrapper with no extension behind
# it; the rest are checked for presence, since flag_gems and flashinfer want a device.
try:
    import flash_attn_interface

    assert flash_attn_interface.flash_attn_func is not None
except Exception as exc:
    sys.exit(
        f"FAIL: flash_attn_interface does not import ({exc}). Eleven attention "
        "benchmarks name `fa3` and would silently drop the column."
    )

for dist, why in (
    ("flash-attn", "the FA2 columns"),
    ("flag_gems", "the flaggems columns"),
    ("flashinfer-python", "the flashinfer columns"),
    ("flash-linear-attention", "the linear-attention columns"),
    ("deepspeed", "the fused INT4 quantization baseline"),
    ("nvmath-python", "the cuBLASLt-search GEMM baseline"),
    ("vllm", "the vllm columns, and the stack every other version follows"),
    ("tile-kernels", "the TileKernels columns"),
):
    try:
        md.version(dist)
    except md.PackageNotFoundError:
        sys.exit(f"FAIL: {dist} is not installed; {why} would be missing.")

try:
    from deepspeed.ops.quantizer import quantizer_op

    assert callable(quantizer_op.quantize)
except Exception as exc:
    sys.exit(
        f"FAIL: DeepSpeed's prebuilt quantizer does not import ({exc}). "
        "Build it with DS_BUILD_QUANTIZER=1 in the runner image."
    )

# vllm's wheel carries the image's only DeepGEMM; a second, top-level one would win vllm's import.
if importlib.util.find_spec("deep_gemm") is not None:
    sys.exit(
        "FAIL: a top-level deep_gemm is installed; the image's DeepGEMM is vllm's vendored copy."
    )
if importlib.util.find_spec("vllm.third_party.deep_gemm") is None:
    sys.exit("FAIL: vllm.third_party.deep_gemm is missing; the deepgemm columns would be missing.")

unsatisfied = _unsatisfied()
if unsatisfied:
    sys.exit(
        "FAIL: requirements left unsatisfied outside overrides.txt:\n  " + "\n  ".join(unsatisfied)
    )

print(
    f"runtime-stack OK: tilelang {tilelang.__version__} | "
    f"torch {torch.__version__} (cuda {torch.version.cuda}) | "
    f"apache-tvm-ffi {installed} satisfies {ffi_req.specifier} | "
    f"cupti-python {cupti_version} with cuda-bindings {bindings_installed}"
)
