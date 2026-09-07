#!/usr/bin/env python3
"""Audit manifest ``bytes`` formulas against NCU DRAM counters.

Spec: docs/design/roofline.md §4.5. For each audited op, one ``forward()``
runs under Nsight Compute with cache control on, and ``dram__bytes_read.sum``
over the call's kernels is compared against the formula's read half:

- measured_read < read_bytes × (1 − EPS)  → FAIL  (read-side overestimate)
- measured_read > read_bytes × OVER       → WARN  (multi-pass / replay inflation)
- missing metric, empty range, or a
  declared read half of zero              → ERROR (never a verdict)
- read half undeclared                    → NO-VERDICT (not a pass)

Write traffic is measured and reported, never judged: lines still dirty in L2
when the kernel ends are written back outside the profiled range, so
``dram__bytes_write.sum`` undercounts by up to the L2 capacity.

Coverage: ops reachable through the manifest single-tensor-input contract run
generically; multi-input ops run through ``INPUT_BUILDERS``; everything else
is reported SKIPPED with the reason — silent gaps would read as audited.

Usage:
    python scripts/validate_roofline_bytes.py [--op OpName] [--family NAME] [--out DIR]
    python scripts/validate_roofline_bytes.py --check-counters
    python scripts/validate_roofline_bytes.py --child OpName --row JSON --dtype bf16
"""

import argparse
import csv
import importlib
import io
import json
import subprocess
import sys
from math import prod
from pathlib import Path

EPS = 0.05  # counter noise allowance below the formula
OVER = 1.5  # informational ceiling above the formula
NVTX_RANGE = "tileops_roofline"
METRICS = "dram__bytes_read.sum,dram__bytes_write.sum"
# Workloads at least this large keep fixed sector/TLB overheads inside EPS.
SMALL_WORKLOAD_BYTES = 32 * 2**20


def _op_class(op_name: str, entry: dict):
    mod_path = entry["source"]["op"].removesuffix(".py").replace("/", ".")
    return getattr(importlib.import_module(mod_path), op_name)


def _single_input_case(op_name: str, entry: dict, row: dict, dtype):
    """(op, inputs) via the manifest single-tensor-input contract, or None."""
    import torch

    from tileops.manifest import single_input_workload_contract

    contract = single_input_workload_contract(entry.get("signature") or {})
    if contract is None:
        return None
    shape_key, _ = contract
    if shape_key not in row:
        return None
    reserved = {"label", "dtypes", "bench_skip_reason", shape_key}
    params = {k: v for k, v in row.items() if k not in reserved and not k.startswith("__")}
    op = _op_class(op_name, entry)(**params)
    # Positive, away from zero: valid for every unary domain (log, rsqrt, ...);
    # the counters read traffic, not values.
    x = torch.rand(tuple(row[shape_key]), dtype=dtype, device="cuda") + 0.5
    return op, (x,)


def _gemm_case(op_name: str, entry: dict, row: dict, dtype):
    import torch

    op = _op_class(op_name, entry)()
    a = torch.randn(row["m"], row["k"], dtype=dtype, device="cuda")
    b = torch.randn(row["k"], row["n"], dtype=dtype, device="cuda")
    return op, (a, b)


def _bmm_case(op_name: str, entry: dict, row: dict, dtype):
    import torch

    op = _op_class(op_name, entry)()
    a = torch.randn(row["batch"], row["m"], row["k"], dtype=dtype, device="cuda")
    b = torch.randn(row["batch"], row["k"], row["n"], dtype=dtype, device="cuda")
    return op, (a, b)


# Multi-input ops the audit can build. Extend per family; an op absent here
# and outside the single-input contract is SKIPPED, visibly.
INPUT_BUILDERS = {
    "GemmFwdOp": _gemm_case,
    "BmmFwdOp": _bmm_case,
}


def _build_case(op_name: str, entry: dict, row: dict, dtype):
    builder = INPUT_BUILDERS.get(op_name)
    if builder is not None:
        return builder(op_name, entry, row, dtype)
    return _single_input_case(op_name, entry, row, dtype)


def _branch_signature(row: dict) -> tuple:
    """Rows sharing this key exercise the same formula branches."""
    return (
        tuple(sorted(row.get("dtypes", []))),
        row.get("backend"),
        tuple(sorted(k for k, v in row.items() if v is None)),
        tuple(sorted(k for k in row if isinstance(row[k], bool))),
    )


def _pick_workloads(entry: dict, cap: int = 6) -> list[tuple[dict, str]]:
    """One row per branch signature, largest first, at most *cap*."""
    picked: dict[tuple, tuple[dict, str]] = {}
    for row in entry.get("workloads") or []:
        if row.get("bench_skip_reason"):
            continue
        for dtype_str in row.get("dtypes", []):
            key = (*_branch_signature(row), dtype_str)
            size = 1
            for v in row.values():
                if isinstance(v, list) and v and all(isinstance(x, int) for x in v):
                    size *= prod(v)
                elif isinstance(v, int) and not isinstance(v, bool) and v > 1:
                    size *= v  # scalar dims (m/n/k/batch) rank GEMM-style rows
            held = picked.get(key)
            if held is None or size > held[0].get("__size", -1):
                picked[key] = ({**row, "__size": size}, dtype_str)
    ranked = sorted(picked.values(), key=lambda p: -p[0]["__size"])
    return [({k: v for k, v in r.items() if k != "__size"}, d) for r, d in ranked[:cap]]


def _declared_read_bytes(op, inputs) -> int | None:
    """The formula's read half, or None when the op declares none.

    An op §4.7 routes to this audit declares the half itself. The fallback is
    the logical extent of the call's distinct input tensors, which matches the
    formula's read half only for ops that read each input once at its own
    shape — the ops the oracle already covers exactly.
    """
    import torch

    declared = getattr(op, "eval_roofline_read_bytes", None)
    if callable(declared):
        return int(declared())
    seen: dict[int, int] = {}
    for value in inputs:
        if isinstance(value, torch.Tensor):
            seen[value.data_ptr()] = value.numel() * value.element_size()
    return sum(seen.values()) or None


def run_child(op_name: str, row_json: str, dtype_str: str) -> None:
    """Run one op's forward inside an NVTX range; print the formula values."""
    import torch

    from tileops.manifest import load_manifest

    entry = load_manifest()[op_name]
    dtype = getattr(torch, dtype_str)
    case = _build_case(op_name, entry, json.loads(row_json), dtype)
    if case is None:
        print(json.dumps({"error": "no input builder"}))
        sys.exit(3)
    op, inputs = case
    with torch.no_grad():
        op(*inputs)  # bind input-inferred roofline vars; build kernels
        torch.cuda.synchronize()
        flops, nbytes = op.eval_roofline()
        read_bytes = _declared_read_bytes(op, inputs)
        torch.cuda.nvtx.range_push(NVTX_RANGE)
        op(*inputs)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "formula_flops": int(flops),
                "formula_bytes": int(nbytes),
                "read_bytes": None if read_bytes is None else int(read_bytes),
            }
        )
    )


def _parse_ncu_csv(path: Path) -> tuple[tuple[float, float] | None, int]:
    """((read bytes, write bytes) over profiled kernels, kernel count).

    The two directions stay apart: only the read side carries a verdict
    (§4.5). None on any gap — an absent metric is never read as zero.
    """
    text = path.read_text(errors="replace")
    lines = [ln for ln in text.splitlines() if ln.startswith('"')]
    if not lines:
        return None, 0
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    per_kernel: dict[tuple, dict[str, float]] = {}
    for r in rows:
        name = r.get("Metric Name", "")
        if name not in ("dram__bytes_read.sum", "dram__bytes_write.sum"):
            continue
        kid = (r.get("ID"), r.get("Kernel Name"))
        raw = (r.get("Metric Value") or "").replace(",", "")
        try:
            per_kernel.setdefault(kid, {})[name] = float(raw)
        except ValueError:
            return None, len(per_kernel)  # n/a: never read as zero
    if not per_kernel:
        return None, 0
    for metrics in per_kernel.values():
        if len(metrics) != 2:
            return None, len(per_kernel)
    read = sum(m["dram__bytes_read.sum"] for m in per_kernel.values())
    write = sum(m["dram__bytes_write.sum"] for m in per_kernel.values())
    return (read, write), len(per_kernel)


def read_side_verdict(measured_read: float, read_bytes: int | None) -> str:
    """§4.5's verdict table. Write traffic never reaches it."""
    if read_bytes is None:
        return "NO-VERDICT"
    if read_bytes <= 0:
        return "ERROR"  # a declaration of zero reads is a broken one
    if measured_read < read_bytes * (1 - EPS):
        return "FAIL"
    if measured_read > read_bytes * OVER:
        return "WARN"
    return "PASS"


def audit_one(op_name: str, entry: dict, out_dir: Path) -> list[dict]:
    results = []
    cases = _pick_workloads(entry)
    if not cases:
        return [{"op": op_name, "verdict": "SKIPPED", "reason": "no workloads"}]
    for row, dtype_str in cases:
        label = row.get("label", "workload")
        csv_path = out_dir / f"{op_name}.{label}.{dtype_str}.csv"
        cmd = [
            "ncu",
            "--nvtx",
            f"--nvtx-include={NVTX_RANGE}/",
            "--metrics", METRICS,
            "--cache-control", "all",
            "--target-processes", "all",
            "--csv",
            "--log-file", str(csv_path),
            sys.executable, __file__,
            "--child", op_name,
            "--row", json.dumps(row),
            "--dtype", dtype_str,
        ]  # fmt: skip
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        base = {"op": op_name, "workload": label, "dtype": dtype_str}
        if proc.returncode == 3:
            results.append({**base, "verdict": "SKIPPED", "reason": "no input builder"})
            continue
        if proc.returncode != 0:
            reason = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["?"]
            results.append({**base, "verdict": "ERROR", "reason": reason[0][:200]})
            continue
        try:
            emitted = json.loads(proc.stdout.strip().splitlines()[-1])
            formula = emitted["formula_bytes"]
            read_bytes = emitted["read_bytes"]
        except (ValueError, KeyError, IndexError):
            results.append({**base, "verdict": "ERROR", "reason": "child emitted no formula"})
            continue
        measured, n_kernels = _parse_ncu_csv(csv_path)
        if measured is None:
            results.append(
                {**base, "verdict": "ERROR", "reason": f"metric missing (kernels={n_kernels})"}
            )
            continue
        measured_read, measured_write = measured
        verdict = read_side_verdict(measured_read, read_bytes)
        row_out = {
            **base,
            "verdict": verdict,
            "formula_bytes": int(formula),
            "read_bytes": read_bytes,
            "measured_read_bytes": int(measured_read),
            "measured_write_bytes": int(measured_write),  # reported, never judged (§4.5)
            "kernels": n_kernels,
            "note": "small workload" if formula < SMALL_WORKLOAD_BYTES else "",
        }
        if verdict == "NO-VERDICT":
            row_out["reason"] = "read half undeclared"
        elif verdict == "ERROR":
            row_out["reason"] = f"declared read half is {read_bytes}"
        else:
            row_out["read_ratio"] = round(measured_read / read_bytes, 4)
        results.append(row_out)
    return results


def check_counters() -> int:
    """Profile a trivial kernel; report whether GPU counters are readable.

    A job that skips this reports an empty audit as a passed one.
    """
    probe = (
        "import torch;"
        "a=torch.zeros(256,256,device='cuda');torch.cuda.synchronize();"
        f"torch.cuda.nvtx.range_push('{NVTX_RANGE}');b=a+1;torch.cuda.nvtx.range_pop();"
        "torch.cuda.synchronize()"
    )
    cmd = [
        "ncu", "--nvtx", f"--nvtx-include={NVTX_RANGE}/",
        "--metrics", "dram__bytes_read.sum", "--csv",
        sys.executable, "-c", probe,
    ]  # fmt: skip
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"ncu unavailable: {exc}")
        return 1
    if proc.returncode == 0 and "dram__bytes_read.sum" in proc.stdout:
        print("GPU performance counters readable.")
        return 0
    print((proc.stdout + proc.stderr).strip()[-800:])
    print(
        "\nGPU performance counters unavailable. Any of these satisfies the audit:\n"
        "  - run as root on the host;\n"
        "  - a rootful container running as root (a rootless daemon cannot, however\n"
        "    the container is privileged: its root maps into a subuid namespace);\n"
        "  - load the driver with NVreg_RestrictProfilingToAdminUsers=0."
    )
    return 1


def _family(entry: dict) -> str:
    return entry.get("family") or ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", help="Audit a single op (default: every implemented op)")
    parser.add_argument("--family", help="Audit one manifest family")
    parser.add_argument("--out", default="roofline_bytes_audit", help="Output directory")
    parser.add_argument(
        "--check-counters",
        action="store_true",
        help="Exit 0 if GPU performance counters are readable, 1 otherwise",
    )
    parser.add_argument("--child", metavar="OP", help=argparse.SUPPRESS)
    parser.add_argument("--row", help=argparse.SUPPRESS)
    parser.add_argument("--dtype", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        run_child(args.child, args.row, args.dtype)
        return
    if args.check_counters:
        sys.exit(check_counters())

    from tileops.manifest import load_manifest

    manifest = load_manifest()
    implemented = {k: v for k, v in manifest.items() if v.get("status") == "implemented"}
    if args.op:
        targets = {args.op: manifest[args.op]}
    elif args.family:
        targets = {k: v for k, v in implemented.items() if _family(v) == args.family}
        if not targets:
            families = sorted({_family(v) for v in implemented.values()})
            parser.error(f"no implemented op in family {args.family!r}; have {families}")
    else:
        targets = implemented
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for op_name, entry in sorted(targets.items()):
        rows = audit_one(op_name, entry, out_dir)
        all_results.extend(rows)
        for r in rows:
            print(
                f"{r['verdict']:10} {r['op']:40} {r.get('workload', '-'):28} "
                f"{r.get('dtype', '-'):9} read_ratio={r.get('read_ratio', '-')} "
                f"{r.get('reason', '')}"
            )

    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))
    counts: dict[str, int] = {}
    for r in all_results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print(f"\nSummary: {counts} → {out_dir}/results.json")
    # A row the audit ran without reaching a verdict is not a passed one.
    # SKIPPED (never run, reason stated) and WARN stay green.
    failed = ("FAIL", "ERROR", "NO-VERDICT")
    sys.exit(1 if any(counts.get(v) for v in failed) else 0)


if __name__ == "__main__":
    main()
