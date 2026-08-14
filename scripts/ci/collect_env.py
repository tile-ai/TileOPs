#!/usr/bin/env python3
"""Record the stack a benchmark run executed on, as JSON on stdout.

A fact that cannot be read is omitted, never guessed.

    python scripts/ci/collect_env.py > env.json
"""

import importlib.metadata as md
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Our stack, then every baseline the suite times against.
_PACKAGES = {
    "tilelang": "tilelang",
    "triton": "triton",
    "flashinfer": "flashinfer-python",
    "flash-attn": "flash_attn",
    "vllm": "vllm",
    "mamba-ssm": "mamba_ssm",
    "deep_gemm": "deep_gemm",
    "fla": "flash-linear-attention",
}


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _driver() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.stdout.splitlines()[0].strip() if out.returncode == 0 else None


def collect() -> dict:
    env: dict[str, object] = {}
    image = os.environ.get("TILEOPS_RUNNER_IMAGE", "").strip()
    if image:
        env["image"] = image

    try:
        import torch
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
        if torch.version.cuda:
            env["cuda"] = torch.version.cuda
        # __version__ keeps the local segment naming the CUDA build.
        env["torch"] = torch.__version__
    except ImportError:
        pass

    driver = _driver()
    if driver:
        env["driver"] = driver

    for label, dist in _PACKAGES.items():
        version = _version(dist)
        if version:
            env[label] = version

    # Run as a script, sys.path[0] is scripts/ci, not the repo root.
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    try:
        from benchmarks.timing import DRY_RUN_MS, REPEAT_MS
        env["warmup_ms"] = DRY_RUN_MS
        env["repeat_ms"] = REPEAT_MS
    except ImportError:
        pass
    return env


def main() -> int:
    env = collect()
    json.dump(env, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    missing = [k for k in ("image", "gpu", "driver", "cuda", "torch", "tilelang")
               if k not in env]
    if missing:
        print("collect_env: not recorded: " + ", ".join(missing), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
