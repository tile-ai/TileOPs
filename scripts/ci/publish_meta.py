#!/usr/bin/env python3
"""Write the meta.json that captions a published benchmark snapshot.

The docs site reads it for the commit, the run, and the environment the
benchmark job recorded. A missing environment stays missing.
"""

import argparse
import contextlib
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-json", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    env = {}
    if os.path.exists(args.env_json):
        # A truncated file must not block publishing the numbers.
        with contextlib.suppress(json.JSONDecodeError), open(args.env_json, encoding="utf-8") as f:
            env = json.load(f)
    if not env:
        print(
            "::warning::env.json missing or empty; the Benchmarks page will "
            "report the run environment as unpublished",
            file=sys.stderr,
        )

    # The installed set sits beside the environment rather than inside it: the
    # environment is a table a reader scans, and this is an inventory to search.
    packages = env.pop("packages", None)
    meta = {
        "commit": args.commit,
        "date": args.date,
        "gpu": env.get("gpu", "NVIDIA H200"),
        "run_id": args.run_id,
    }
    if env:
        meta["environment"] = env
    if packages:
        meta["packages"] = packages
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
