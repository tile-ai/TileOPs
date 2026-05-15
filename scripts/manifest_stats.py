#!/usr/bin/env python3
"""Summarize the ops manifest: status, coverage, workloads, roofline.

The manifest at ``tileops/manifest/`` is the project's source of truth.
This script aggregates it into a single snapshot intended for CI status
reporting and project dashboards.

Usage:
    python scripts/manifest_stats.py [--format {text,md,json}] [--output PATH]
    python scripts/manifest_stats.py --diff baseline.json  # compare vs. baseline

Exit code is always 0; this is an informational reporter, not a gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tileops.manifest import load_manifest

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

STATUSES = ("implemented", "spec-only", "deprecated")


def _has_roofline(op: dict[str, Any]) -> bool:
    rf = op.get("roofline") or {}
    # Either a func-based roofline OR explicit flops/bytes formulae.
    return bool(rf.get("func")) or ("flops" in rf and "bytes" in rf)


def _has_kernel_map(op: dict[str, Any]) -> bool:
    km = (op.get("source") or {}).get("kernel_map")
    return isinstance(km, dict) and len(km) > 0


def _has_bench_manifest_driven(op: dict[str, Any]) -> bool:
    return bool((op.get("source") or {}).get("bench_manifest_driven"))


def collect_stats(manifest: dict[str, dict]) -> dict[str, Any]:
    total = len(manifest)
    status_counts: Counter[str] = Counter()
    per_family: dict[str, Counter[str]] = defaultdict(Counter)
    workloads_per_op: dict[str, int] = {}
    families_workloads: dict[str, int] = defaultdict(int)

    roofline_ok = 0
    kernel_map_ok = 0
    bench_manifest_ok = 0
    ref_api_ok = 0
    variant_count = 0

    # Conformance flags: things expected for implemented ops but missing.
    missing_kernel_map: list[str] = []
    missing_roofline: list[str] = []
    workloads_below_two: list[str] = []
    spec_only: list[str] = []

    for name, op in manifest.items():
        status = op.get("status", "unknown")
        family = op.get("family", "unknown")
        status_counts[status] += 1
        per_family[family][status] += 1

        wls = op.get("workloads") or []
        workloads_per_op[name] = len(wls)
        families_workloads[family] += len(wls)

        if _has_roofline(op):
            roofline_ok += 1
        if _has_kernel_map(op):
            kernel_map_ok += 1
        if _has_bench_manifest_driven(op):
            bench_manifest_ok += 1
        if op.get("ref_api"):
            ref_api_ok += 1
        if op.get("variant_of"):
            variant_count += 1

        if status == "implemented":
            if not _has_kernel_map(op):
                missing_kernel_map.append(name)
            if not _has_roofline(op):
                missing_roofline.append(name)
            if len(wls) < 2:
                workloads_below_two.append(name)
        elif status == "spec-only":
            spec_only.append(name)

    families = sorted(per_family)
    family_rows = []
    for fam in families:
        c = per_family[fam]
        fam_total = sum(c.values())
        impl = c.get("implemented", 0)
        family_rows.append(
            {
                "family": fam,
                "total": fam_total,
                "implemented": impl,
                "spec_only": c.get("spec-only", 0),
                "deprecated": c.get("deprecated", 0),
                "pct_implemented": (impl / fam_total) if fam_total else 0.0,
                "workloads": families_workloads[fam],
            }
        )

    implemented_total = status_counts.get("implemented", 0)
    workloads_total = sum(workloads_per_op.values())

    return {
        "total_ops": total,
        "by_status": dict(status_counts),
        "pct_implemented": (implemented_total / total) if total else 0.0,
        "families": family_rows,
        "workloads_total": workloads_total,
        "workloads_avg_per_implemented": (
            workloads_total / implemented_total if implemented_total else 0.0
        ),
        "coverage": {
            "ref_api": ref_api_ok,
            "roofline": roofline_ok,
            "kernel_map": kernel_map_ok,
            "bench_manifest_driven": bench_manifest_ok,
            "variant_of": variant_count,
        },
        "conformance_gaps": {
            "implemented_without_kernel_map": sorted(missing_kernel_map),
            "implemented_without_roofline": sorted(missing_roofline),
            "implemented_with_fewer_than_two_workloads": sorted(workloads_below_two),
            "spec_only_ops": sorted(spec_only),
        },
    }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _bar(pct: float, width: int = 20) -> str:
    filled = int(round(pct * width))
    return "█" * filled + "░" * (width - filled)


def render_text(stats: dict[str, Any]) -> str:
    lines: list[str] = []
    total = stats["total_ops"]
    by_status = stats["by_status"]
    pct = stats["pct_implemented"]
    lines.append("Manifest snapshot")
    lines.append("=" * 60)
    lines.append(f"Total ops:    {total}")
    lines.append(
        f"Implemented:  {by_status.get('implemented', 0):4d}  "
        f"[{_bar(pct)}] {pct * 100:5.1f}%"
    )
    lines.append(f"Spec-only:    {by_status.get('spec-only', 0):4d}")
    if by_status.get("deprecated"):
        lines.append(f"Deprecated:   {by_status['deprecated']:4d}")
    lines.append("")
    lines.append("Per-family coverage")
    lines.append("-" * 60)
    lines.append(f"{'family':<18}{'impl':>6}{'spec':>6}{'total':>7}  pct")
    for row in stats["families"]:
        lines.append(
            f"{row['family']:<18}{row['implemented']:>6}{row['spec_only']:>6}"
            f"{row['total']:>7}  {row['pct_implemented'] * 100:5.1f}%  "
            f"{_bar(row['pct_implemented'], 12)}"
        )
    lines.append("")
    lines.append("Spec coverage (all ops)")
    lines.append("-" * 60)
    cov = stats["coverage"]
    for label, key in [
        ("ref_api declared", "ref_api"),
        ("roofline (func or flops+bytes)", "roofline"),
        ("source.kernel_map", "kernel_map"),
        ("bench_manifest_driven", "bench_manifest_driven"),
    ]:
        n = cov[key]
        p = n / total if total else 0
        lines.append(f"  {label:<32} {n:4d}/{total}  {p * 100:5.1f}%")
    lines.append("")
    lines.append(f"Workloads total: {stats['workloads_total']}  "
                 f"(avg {stats['workloads_avg_per_implemented']:.2f} per implemented op)")
    lines.append("")
    gaps = stats["conformance_gaps"]
    lines.append("Conformance gaps")
    lines.append("-" * 60)
    lines.append(
        f"  implemented without kernel_map: "
        f"{len(gaps['implemented_without_kernel_map'])}"
    )
    lines.append(
        f"  implemented without roofline:   "
        f"{len(gaps['implemented_without_roofline'])}"
    )
    lines.append(
        f"  implemented with <2 workloads:  "
        f"{len(gaps['implemented_with_fewer_than_two_workloads'])}"
    )
    return "\n".join(lines) + "\n"


def render_markdown(stats: dict[str, Any]) -> str:
    total = stats["total_ops"]
    by_status = stats["by_status"]
    impl = by_status.get("implemented", 0)
    spec = by_status.get("spec-only", 0)
    pct = stats["pct_implemented"] * 100
    cov = stats["coverage"]
    gaps = stats["conformance_gaps"]

    lines: list[str] = []
    lines.append("## Manifest status")
    lines.append("")
    lines.append(
        f"![ops](https://img.shields.io/badge/ops-{total}-blue) "
        f"![implemented](https://img.shields.io/badge/implemented-"
        f"{impl}%20%2F%20{total}%20%28{pct:.0f}%25%29-brightgreen) "
        f"![spec--only](https://img.shields.io/badge/spec--only-{spec}-orange)"
    )
    lines.append("")
    lines.append("### Per-family coverage")
    lines.append("")
    lines.append("| Family | Implemented | Spec-only | Total | Progress | Workloads |")
    lines.append("| --- | ---: | ---: | ---: | --- | ---: |")
    for row in stats["families"]:
        p = row["pct_implemented"] * 100
        bar = _bar(row["pct_implemented"], 10)
        lines.append(
            f"| `{row['family']}` | {row['implemented']} | {row['spec_only']} | "
            f"{row['total']} | `{bar}` {p:.0f}% | {row['workloads']} |"
        )
    lines.append("")
    lines.append("### Spec coverage")
    lines.append("")
    lines.append("| Field | Coverage |")
    lines.append("| --- | ---: |")
    for label, key in [
        ("`ref_api`", "ref_api"),
        ("`roofline` (func or flops+bytes)", "roofline"),
        ("`source.kernel_map`", "kernel_map"),
        ("`source.bench_manifest_driven`", "bench_manifest_driven"),
    ]:
        n = cov[key]
        p = n / total * 100 if total else 0
        lines.append(f"| {label} | {n} / {total} ({p:.0f}%) |")
    lines.append("")
    lines.append(
        f"**Workloads:** {stats['workloads_total']} total — "
        f"{stats['workloads_avg_per_implemented']:.2f} per implemented op."
    )
    lines.append("")
    lines.append("### Conformance gaps")
    lines.append("")
    lines.append(
        f"- Implemented ops without `kernel_map`: "
        f"**{len(gaps['implemented_without_kernel_map'])}**"
    )
    lines.append(
        f"- Implemented ops without `roofline`: "
        f"**{len(gaps['implemented_without_roofline'])}**"
    )
    lines.append(
        f"- Implemented ops with fewer than two workloads: "
        f"**{len(gaps['implemented_with_fewer_than_two_workloads'])}**"
    )
    spec_list = gaps["spec_only_ops"]
    if spec_list:
        lines.append("")
        lines.append("<details><summary>Spec-only ops "
                     f"({len(spec_list)})</summary>")
        lines.append("")
        lines.append(", ".join(f"`{n}`" for n in spec_list))
        lines.append("")
        lines.append("</details>")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Diff (vs. baseline JSON snapshot)
# ---------------------------------------------------------------------------


def render_badges(stats: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build shields.io endpoint payloads for the README badges.

    Returns a mapping of badge name -> shields endpoint dict
    (``schemaVersion``/``label``/``message``/``color``).
    """
    total = stats["total_ops"]
    impl = stats["by_status"].get("implemented", 0)
    pct_impl = (impl / total * 100) if total else 0.0

    missing_km = len(
        stats["conformance_gaps"]["implemented_without_kernel_map"]
    )
    impl_with_km = impl - missing_km
    pct_km = (impl_with_km / impl * 100) if impl else 0.0

    def _color(pct: float) -> str:
        if pct >= 90:
            return "brightgreen"
        if pct >= 75:
            return "green"
        if pct >= 50:
            return "yellow"
        if pct >= 25:
            return "orange"
        return "red"

    return {
        "implemented": {
            "schemaVersion": 1,
            "label": "implemented",
            "message": f"{impl}/{total} ({pct_impl:.0f}%)",
            "color": _color(pct_impl),
        },
        "kernel_map": {
            "schemaVersion": 1,
            "label": "kernel_map coverage",
            "message": f"{impl_with_km}/{impl} ({pct_km:.0f}%)",
            "color": _color(pct_km),
        },
    }


def render_diff(current: dict[str, Any], baseline: dict[str, Any]) -> str:
    lines = ["## Manifest diff vs. baseline", ""]
    d_total = current["total_ops"] - baseline["total_ops"]
    d_impl = current["by_status"].get("implemented", 0) - baseline["by_status"].get(
        "implemented", 0
    )
    d_spec = current["by_status"].get("spec-only", 0) - baseline["by_status"].get(
        "spec-only", 0
    )
    lines.append(f"- Total ops: {baseline['total_ops']} → {current['total_ops']} "
                 f"({d_total:+d})")
    lines.append(
        f"- Implemented: {baseline['by_status'].get('implemented', 0)} → "
        f"{current['by_status'].get('implemented', 0)} ({d_impl:+d})"
    )
    lines.append(
        f"- Spec-only: {baseline['by_status'].get('spec-only', 0)} → "
        f"{current['by_status'].get('spec-only', 0)} ({d_spec:+d})"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("text", "md", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write output to this path instead of stdout.",
    )
    parser.add_argument(
        "--diff",
        type=Path,
        help="Compare against a previous JSON snapshot and emit a delta section.",
    )
    parser.add_argument(
        "--badge-output",
        type=Path,
        help=(
            "Write a directory containing shields.io endpoint JSON files "
            "(manifest-implemented.json, manifest-kernel-map.json). "
            "Independent of --format/--output."
        ),
    )
    args = parser.parse_args(argv)

    manifest = load_manifest()
    stats = collect_stats(manifest)

    if args.format == "json":
        out = json.dumps(stats, indent=2, sort_keys=True) + "\n"
    elif args.format == "md":
        out = render_markdown(stats)
    else:
        out = render_text(stats)

    if args.diff:
        baseline = json.loads(args.diff.read_text())
        diff_block = render_diff(stats, baseline)
        out = out + "\n" + diff_block

    if args.output:
        args.output.write_text(out)
    else:
        sys.stdout.write(out)

    if args.badge_output:
        badges = render_badges(stats)
        args.badge_output.mkdir(parents=True, exist_ok=True)
        for slug, payload in badges.items():
            fname = f"manifest-{slug.replace('_', '-')}.json"
            (args.badge_output / fname).write_text(
                json.dumps(payload, indent=2) + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
