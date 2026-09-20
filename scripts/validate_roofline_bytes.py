#!/usr/bin/env python3
"""Audit manifest ``bytes`` formulas against NCU DRAM counters.

Spec: docs/design/roofline.md §4.5. For each audited op, one ``forward()``
runs under Nsight Compute with cache control on, and ``dram__bytes_read.sum``
over the call's kernels is compared against the formula's read half:

- measured_read < read_bytes × (1 − EPS)  → FAIL  (read-side overestimate), or
                                            EXEMPT where the entry's
                                            ``read_bound_exception`` covers
                                            this row (reported, not judged)
- measured_read > read_bytes × OVER       → WARN  (multi-pass / replay inflation)
- missing metric or empty range            → ERROR (never a verdict)
- a declared read half of zero             → SKIPPED (nothing to judge it by)
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
# Verdicts a green run may contain; see exit_code().
GREEN_VERDICTS = frozenset({"PASS", "WARN", "EXEMPT", "SKIPPED"})
METRICS = "dram__bytes_read.sum,dram__bytes_write.sum"
# Workloads at least this large keep fixed sector/TLB overheads inside EPS.
SMALL_WORKLOAD_BYTES = 32 * 2**20
# The read-side bound holds only under cold-cache replay. Every judged row
# carries it, so a row copied out of results.json cannot be read as an
# unconditional hardware measurement.
COLD_CACHE_PREMISE = "cold-cache replay (ncu --cache-control all)"


def _op_class(op_name: str, entry: dict):
    mod_path = entry["source"]["op"].removesuffix(".py").replace("/", ".")
    return getattr(importlib.import_module(mod_path), op_name)


def _single_input_case(op_name: str, entry: dict, row: dict, dtype):
    """(op, inputs) via the manifest single-tensor-input contract, or None."""

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
    x = _sample(tuple(row[shape_key]), dtype)
    return op, (x,)


def _sample(shape: tuple, dtype) -> "object":
    """A valid input of *dtype*; the counters read traffic, not values."""
    import torch

    if dtype is torch.bool:
        return torch.ones(shape, dtype=torch.bool, device="cuda")
    if not dtype.is_floating_point:
        return torch.ones(shape, dtype=dtype, device="cuda")
    # Positive, away from zero: valid for every unary domain (log, rsqrt, ...).
    return torch.rand(shape, dtype=dtype, device="cuda") + 0.5


def _gemm_case(op_name: str, entry: dict, row: dict, dtype):
    """The row names the layout, and the operands are stored in it."""
    import torch

    trans_a = bool(row.get("trans_a", False))
    trans_b = bool(row.get("trans_b", True))
    op = _op_class(op_name, entry)(trans_a=trans_a, trans_b=trans_b)
    m, n, k = row["m"], row["n"], row["k"]
    a = torch.randn(*((k, m) if trans_a else (m, k)), dtype=dtype, device="cuda")
    b = torch.randn(*((n, k) if trans_b else (k, n)), dtype=dtype, device="cuda")
    return op, (a, b)


def _bmm_case(op_name: str, entry: dict, row: dict, dtype):
    import torch

    op = _op_class(op_name, entry)()
    batch, m, n, k = row["b"], row["m"], row["n"], row["k"]
    a = torch.randn(batch, m, k, dtype=dtype, device="cuda")
    b = torch.randn(batch, k, n, dtype=dtype, device="cuda")
    return op, (a, b)


def _fused_moe_case(op_name: str, entry: dict, row: dict, dtype):
    """Build the manifest's routed-MoE workload."""
    from workloads.moe import FusedMoeWorkload

    params = {
        name: row[name] for name in (entry.get("signature") or {}).get("params", {}) if name in row
    }
    op = _op_class(op_name, entry)(**params)
    workload = FusedMoeWorkload(
        op.num_tokens,
        op.num_experts,
        op.top_k,
        op.hidden_size,
        op.ffn_size,
        op.scoring_func,
        op.renormalize,
        "correction_bias_shape" in row,
        op.routed_scaling_factor,
        dtype,
    )
    hidden, gating, correction_bias, w_gate_up, w_down = workload.gen_inputs()
    return op, (hidden, gating, w_gate_up, w_down, correction_bias)


# Multi-input ops the audit can build. Extend per family; an op absent here
# and outside the single-input contract is SKIPPED, visibly.
INPUT_BUILDERS = {
    "FusedMoeFwdOp": _fused_moe_case,
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
        tuple(sorted((k, v) for k, v in row.items() if isinstance(v, bool))),
        tuple(sorted(k for k in row if k.endswith("_shape"))),
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


def _declared_read_bytes(op) -> int | None:
    """The op's declared read half, or None when it declares none.

    There is no fallback. Summing the call's input tensors is not the read
    half: an op that reads a subset of an input -- a routed MoE reading the
    experts its routing selects -- would be charged the whole of it and fail
    a correct formula.
    """
    declared = getattr(op, "eval_roofline_read_bytes", None)
    if not callable(declared):
        return None
    value = declared()
    return None if value is NotImplemented else int(value)


def run_child(op_name: str, row_json: str, dtype_str: str) -> None:
    """Run one op's forward inside an NVTX range; print the formula values."""
    import torch

    from tileops.manifest import load_manifest
    from tileops.ops.op_base import record_roofline_calls

    # The read half needs the shapes the call carried; ops do not keep them.
    record_roofline_calls()
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
        read_bytes = _declared_read_bytes(op)
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


def read_side_verdict(
    measured_read: float,
    read_bytes: int | None,
    bound: bool = True,
) -> str:
    """§4.5's verdict table. Write traffic never reaches it.

    FAIL says the formula charged reads the implementation did not make. That
    reading holds only where every conforming implementation must fetch what the
    formula charges. Where this call is one the entry's ``read_bound_exception``
    covers, *bound* is false and a shortfall comes back EXEMPT: measured,
    reported, and not a verdict on the formula.
    """
    if read_bytes is None:
        return "NO-VERDICT"
    if read_bytes < 0:
        return "ERROR"  # a negative read half is a broken declaration
    if read_bytes == 0:
        # A call that reads none of its inputs -- dropout at p == 1 writes zeros
        # -- states a read half of zero, and there is no ratio to judge it by.
        return "SKIPPED"
    if measured_read < read_bytes * (1 - EPS):
        return "FAIL" if bound else "EXEMPT"
    if measured_read > read_bytes * OVER:
        return "WARN"
    return "PASS"


def read_bound_exception(entry: dict, row: dict, dtype_str: str | None = None) -> str:
    """The reason this row's read half is not a bound, or ``""``.

    The exception states the condition it holds under, and a row outside that
    condition is judged like any other: a dropout that trains with 0 < p < 1 may
    skip a dropped position's load, and the same op in eval mode reads all of
    its input.
    """
    exception = (entry.get("roofline") or {}).get("read_bound_exception") or {}
    when = (exception.get("when") or "").strip()
    reason = (exception.get("reason") or "").strip()
    if not when or not reason:
        return ""
    names = {
        name: spec.get("default")
        for name, spec in ((entry.get("signature") or {}).get("params") or {}).items()
        if isinstance(spec, dict)
    }
    names.update({k: v for k, v in row.items() if not k.startswith("__")})
    # The row carries the dtype axis; the call runs one element type off it.
    if dtype_str is not None:
        names["dtype"] = dtype_str
    try:
        holds = eval(when, {"__builtins__": {}}, names)  # noqa: S307 - validator limits the form
    except Exception:
        return ""  # a condition this row cannot answer does not waive anything
    return reason if holds else ""


def exit_code(counts: dict[str, int]) -> int:
    """Zero means no unwaived failure, not that every row was judged.

    Three verdicts are green: SKIPPED (never run, reason stated), WARN (more
    traffic than the formula charges, which passed the lower-bound check) and
    EXEMPT (a shortfall the entry's ``read_bound_exception`` covers, measured
    and not judged). Anything else, including a verdict this function does not
    know, is red: a spelling nobody reads is not a pass.
    """
    return 0 if set(counts) <= GREEN_VERDICTS else 1


def fully_waived(results: list[dict]) -> list[str]:
    """Ops whose every judged row came back EXEMPT.

    The exception states the calls whose read half is not a bound, and an op
    whose rows are all such calls leaves that half unchecked. The audit reports
    it: the alternative, failing the run, would push a short-circuit row into
    the release-facing workloads, which the benchmark then measures.
    """
    judged: dict[str, set[str]] = {}
    for row in results:
        if row["verdict"] in ("SKIPPED", "ERROR"):
            continue
        judged.setdefault(row["op"], set()).add(row["verdict"])
    return sorted(op for op, verdicts in judged.items() if verdicts == {"EXEMPT"})


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
        waived = read_bound_exception(entry, row, dtype_str)
        verdict = read_side_verdict(measured_read, read_bytes, bound=not waived)
        row_out = {
            **base,
            "verdict": verdict,
            "formula_bytes": int(formula),
            "read_bytes": read_bytes,
            "measured_read_bytes": int(measured_read),
            "measured_write_bytes": int(measured_write),  # reported, never judged (§4.5)
            "kernels": n_kernels,
            "measured_under": COLD_CACHE_PREMISE,
            "note": "small workload" if formula < SMALL_WORKLOAD_BYTES else "",
        }
        if verdict == "EXEMPT":
            row_out["reason"] = waived
        if verdict == "NO-VERDICT":
            row_out["reason"] = "read half undeclared"
        elif verdict == "SKIPPED":
            row_out["reason"] = "the formula declares no read"
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

    print(f"Read side judged under {COLD_CACHE_PREMISE}; write side reported, never judged.\n")
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
    for op_name in fully_waived(all_results):
        # Every row of this op fell inside its exception, so the run judged the
        # formula's read half nowhere. The rows are still measured and reported.
        print(f"  every audited row of {op_name} is EXEMPT: its read half went unjudged")
    sys.exit(exit_code(counts))


if __name__ == "__main__":
    main()
