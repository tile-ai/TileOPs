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


# What the card was set to: a run at a lower cap or clock is a different
# measurement, and nothing else in the snapshot says so.
#
# The clock the card ran at, and the one it could have: a card held below its
# own maximum is the case this has to show, and it does not slow every
# implementation by the same amount. `clocks.applications.*` used to carry
# this and now answers "Requested functionality has been deprecated" on the
# runner's driver -- an answer the reader below drops, so the clock went
# unrecorded rather than wrong.
_GPU_FIELDS = (
    ("driver", "driver_version"),
    ("power_limit_w", "power.limit"),
    ("sm_clock_mhz", "clocks.current.graphics"),
    ("sm_clock_max_mhz", "clocks.max.graphics"),
    ("memory_clock_mhz", "clocks.current.memory"),
    ("mig", "mig.mode.current"),
)


def _gpu_state() -> dict:
    """What nvidia-smi reports for card 0. A field it cannot answer is dropped."""
    query = ",".join(field for _, field in _GPU_FIELDS)
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if out.returncode != 0 or not out.stdout.strip():
        return {}
    values = [v.strip() for v in out.stdout.splitlines()[0].split(",")]
    if len(values) != len(_GPU_FIELDS):
        print(
            f"collect_env: nvidia-smi answered {len(values)} of "
            f"{len(_GPU_FIELDS)} fields; card state not recorded",
            file=sys.stderr,
        )
        return {}
    state = {}
    for (name, _), value in zip(_GPU_FIELDS, values, strict=True):
        # `[N/A]`, `[Requested functionality has been deprecated]` — anything
        # the driver answers in brackets is not a fact.
        if not value or value.startswith("[") or value == "N/A":
            continue
        state[name] = float(value) if name.endswith(("_w", "_mhz")) else value
    return state


def collect() -> dict:
    env: dict[str, object] = {}
    image = os.environ.get("TILEOPS_RUNNER_IMAGE", "").strip()
    if image:
        env["image"] = image
    # A tag can be pushed again; the digest cannot. A container cannot read
    # its own, so the host passes it in.
    digest = os.environ.get("TILEOPS_RUNNER_IMAGE_DIGEST", "").strip()
    if digest:
        env["image_digest"] = digest

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

    # tilelang beside torch, not only in the inventory below: the two compile
    # every kernel these numbers describe.
    try:
        import tilelang

        env["tilelang"] = tilelang.__version__
    except ImportError:
        pass

    env.update(_gpu_state())

    # Every installed distribution, not a chosen few: which library matters to
    # a number is the reader's question, and a list written here would answer it
    # for a suite that has since added a baseline.
    env["packages"] = {
        dist.metadata["Name"]: dist.version
        for dist in md.distributions()
        if dist.metadata.get("Name")
    }

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
    missing = [
        k for k in ("image", "image_digest", "gpu", "driver", "cuda", "torch") if k not in env
    ]
    if missing:
        print("collect_env: not recorded: " + ", ".join(missing), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
