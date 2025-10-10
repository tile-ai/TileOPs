#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common performance tools for FA operator sweep runners.

This module centralizes:
- CSV/TSV autodetection and robust reading/writing
- Header normalization + alias mapping
- Typed cell parsing (int/bool)
- Logging setup
- Subprocess execution with full stdout/stderr logging
- Metrics parsing from test scripts stdout
- A generic `run_sweep(...)` to remove duplication in per-op scripts
- Pretty table printing to stdout with a switch
"""

from __future__ import annotations

import csv
import io
import logging
import pathlib
import re
import subprocess
import sys
from typing import Callable, Dict, List, Optional, Tuple


# --------------------------
# Regex for metrics
# --------------------------
# --- Robust float pattern (supports scientific notation) ---
_FLOAT = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"

# --- Line-anchored, case-insensitive patterns (Fwd/Bwd must be at column 0) ---
# Fwd
FWD_LAT_RE        = re.compile(rf"(?im)^fwd\s+latency:\s*{_FLOAT}\s*ms\b")
FWD_TFLOPS_RE     = re.compile(rf"(?im)^fwd\s+flops?:\s*{_FLOAT}\s*t[fF][lL][oO][pP][s]?\b")
FWD_REF_LAT_RE    = re.compile(rf"(?im)^fwd\s+(?:ref|reference)\s+latency:\s*{_FLOAT}\s*ms\b")
FWD_REF_TFLOPS_RE = re.compile(rf"(?im)^fwd\s+(?:ref|reference)\s+flops?:\s*{_FLOAT}\s*t[fF][lL][oO][pP][s]?\b")

# Bwd
BWD_LAT_RE        = re.compile(rf"(?im)^bwd\s+latency:\s*{_FLOAT}\s*ms\b")
BWD_TFLOPS_RE     = re.compile(rf"(?im)^bwd\s+flops?:\s*{_FLOAT}\s*t[fF][lL][oO][pP][s]?\b")
BWD_REF_LAT_RE    = re.compile(rf"(?im)^bwd\s+(?:ref|reference)\s+latency:\s*{_FLOAT}\s*ms\b")
BWD_REF_TFLOPS_RE = re.compile(rf"(?im)^bwd\s+(?:ref|reference)\s+flops?:\s*{_FLOAT}\s*t[fF][lL][oO][pP][s]?\b")

def parse_stdout_metrics(stdout: str) -> Dict[str, str]:
    """Parse Fwd/Bwd (ref and final) latency/FLOPs from stdout.

    Rules:
      * Case-insensitive, but lines must start with 'Fwd' or 'Bwd' at column 0.
      * If multiple matches per metric appear, the last one is taken.
      * Missing metrics return empty strings.
    Returns:
      Dict with keys:
        fwd_latency_ms, fwd_tflops, fwd_ref_latency_ms, fwd_ref_tflops,
        bwd_latency_ms, bwd_tflops, bwd_ref_latency_ms, bwd_ref_tflops
        (plus backward-compat keys if you still need them).
    """
    def _last_float(pat: re.Pattern[str]) -> Optional[float]:
        matches = list(pat.finditer(stdout))
        if not matches:
            return None
        return float(matches[-1].group(1))

    fwd_lat  = _last_float(FWD_LAT_RE)
    fwd_tf   = _last_float(FWD_TFLOPS_RE)
    fwd_rlat = _last_float(FWD_REF_LAT_RE)
    fwd_rtf  = _last_float(FWD_REF_TFLOPS_RE)

    bwd_lat  = _last_float(BWD_LAT_RE)
    bwd_tf   = _last_float(BWD_TFLOPS_RE)
    bwd_rlat = _last_float(BWD_REF_LAT_RE)
    bwd_rtf  = _last_float(BWD_REF_TFLOPS_RE)

    out = {
        "fwd_latency_ms":      f"{fwd_lat:.2f}"  if fwd_lat  is not None else "",
        "fwd_tflops":          f"{fwd_tf:.2f}"   if fwd_tf   is not None else "",
        "fwd_ref_latency_ms":  f"{fwd_rlat:.2f}" if fwd_rlat is not None else "",
        "fwd_ref_tflops":      f"{fwd_rtf:.2f}"  if fwd_rtf  is not None else "",
        "bwd_latency_ms":      f"{bwd_lat:.2f}"  if bwd_lat  is not None else "",
        "bwd_tflops":          f"{bwd_tf:.2f}"   if bwd_tf   is not None else "",
        "bwd_ref_latency_ms":  f"{bwd_rlat:.2f}" if bwd_rlat is not None else "",
        "bwd_ref_tflops":      f"{bwd_rtf:.2f}"  if bwd_rtf  is not None else "",
    }

    # Optional: keep backward-compat fields empty or map to fwd values if desired.
    # Here we keep them empty to avoid confusion; uncomment if you need fallback.
    # out.update({
    #     "ref_latency_ms": out["fwd_ref_latency_ms"],
    #     "ref_tflops": out["fwd_ref_tflops"],
    #     "latency_ms": out["fwd_latency_ms"],
    #     "tflops": out["fwd_tflops"],
    # })

    return out


# --------------------------
# Logging
# --------------------------
def setup_logger(log_path: pathlib.Path) -> logging.Logger:
    """Create a fresh logger; overwrites the log file."""
    logger = logging.getLogger("fa_perf_sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Open in write mode to avoid appending to old logs.
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# --------------------------
# CSV/TSV utilities
# --------------------------
def _detect_dialect(sample: str) -> csv.Dialect:
    """Detect CSV dialect (comma/tab/semicolon/pipe)."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
    except Exception:
        if sample.count("\t") >= max(sample.count(","), sample.count(";"), sample.count("|")):
            class _TSV(csv.Dialect):
                delimiter = "\t"
                quotechar = '"'
                doublequote = True
                escapechar = None
                lineterminator = "\n"
                quoting = csv.QUOTE_MINIMAL
                skipinitialspace = False
            return _TSV()
        return csv.get_dialect("excel")


def read_rows_csv_any(path: pathlib.Path, logger: Optional[logging.Logger] = None) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read CSV/TSV with autodetected delimiter. Returns (rows, fieldnames)."""
    raw_text = path.read_text(encoding="utf-8-sig")
    sample = raw_text[:4096]
    dialect = _detect_dialect(sample)
    if logger:
        logger.info("Detected delimiter: %r", getattr(dialect, "delimiter", ","))

    reader = csv.DictReader(io.StringIO(raw_text), dialect=dialect)
    fieldnames = reader.fieldnames or []
    if logger:
        logger.info("Raw headers: %s", fieldnames)
    rows = list(reader)
    return rows, fieldnames


def _build_out_fieldnames(input_fieldnames: List[str], extra_cols: List[str]) -> List[str]:
    """Compute final CSV header = input headers + extra result columns (dedup)."""
    return input_fieldnames + [c for c in extra_cols if c not in input_fieldnames]


def write_results_csv(
    out_path: pathlib.Path,
    input_fieldnames: List[str],
    results_rows: List[Dict[str, str]],
    extra_cols: List[str],
) -> List[str]:
    """Write output CSV preserving original columns plus extra result columns.

    Returns:
        The final header used (out_fieldnames).
    """
    out_fieldnames = _build_out_fieldnames(input_fieldnames, extra_cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)
    return out_fieldnames


# --------------------------
# Header + value helpers
# --------------------------
def norm_header(s: str) -> str:
    """Normalize header names: strip, lower, unify separators."""
    s = s.strip().lower().replace("\ufeff", "")
    s = s.replace(" ", "_").replace("-", "_")
    return s


def build_header_map(fieldnames: List[str], alias_map: Dict[str, List[str]]) -> Dict[str, str]:
    """Build mapping canonical -> actual column given aliases."""
    norm_to_actual = {norm_header(f): f for f in fieldnames if f is not None}
    mapping: Dict[str, str] = {}
    for canonical, cands in alias_map.items():
        for cand in cands:
            key = norm_header(cand)
            if key in norm_to_actual:
                mapping[canonical] = norm_to_actual[key]
                break
    return mapping


def get_cell(row: Dict[str, str], header_map: Dict[str, str], canonical: str, default: Optional[str]) -> Optional[str]:
    """Safe getter using canonical name with fallback default."""
    actual = header_map.get(canonical)
    if actual is None:
        return default
    return row.get(actual, default)


def to_int(s: Optional[str], default: int) -> int:
    """Robust int parsing with defaults."""
    if s is None:
        return default
    s = s.strip()
    if s == "":
        return default
    try:
        return int(float(s))
    except Exception:
        return default


def to_bool(s: Optional[str], default: bool) -> bool:
    """Truthiness parse for boolean flags."""
    if s is None:
        return default
    v = s.strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    return default


# --------------------------
# Subprocess + metrics
# --------------------------
def run_and_log(cmd: List[str], logger: logging.Logger) -> Tuple[int, str, str]:
    """Run subprocess, return (returncode, stdout, stderr), log full outputs."""
    logger.info("Command: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except FileNotFoundError as e:
        logger.exception("Python or script not found: %s", e)
        return 127, "", str(e)

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if stdout.strip():
        logger.info("===== STDOUT BEGIN =====\n%s\n===== STDOUT END =====", stdout.rstrip("\n"))
    if stderr.strip():
        logger.warning("===== STDERR BEGIN =====\n%s\n===== STDERR END =====", stderr.rstrip("\n"))

    return proc.returncode, stdout, stderr


def tail_line(text: str) -> str:
    """Return the last non-empty line of a multi-line string for concise notes."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# --------------------------
# Pretty table printing
# --------------------------
def _truncate(s: str, limit: int) -> str:
    if limit <= 3 or len(s) <= limit:
        return s[:limit]
    return s[: limit - 3] + "..."


def _compute_col_widths(headers: List[str], rows: List[Dict[str, str]], max_col_width: int) -> List[int]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, h in enumerate(headers):
            val = str(row.get(h, ""))
            widths[i] = min(max(widths[i], len(val)), max_col_width)
    return widths


def _draw_row(cells: List[str], widths: List[int]) -> str:
    parts = []
    for c, w in zip(cells, widths):
        parts.append(" " + c.ljust(w) + " ")
    return "|" + "|".join(parts) + "|"


def _draw_sep(widths: List[int]) -> str:
    parts = []
    for w in widths:
        parts.append("-" * (w + 2))
    return "+" + "+".join(parts) + "+"


def print_table(out_fieldnames: List[str], results_rows: List[Dict[str, str]], max_col_width: int = 32) -> None:
    """Print a pretty ASCII table of the results to stdout."""
    headers = out_fieldnames
    # Prepare truncated rows to avoid super-wide lines.
    trunc_rows: List[Dict[str, str]] = []
    for row in results_rows:
        new_row = {}
        for h in headers:
            new_row[h] = _truncate(str(row.get(h, "")), max_col_width)
        trunc_rows.append(new_row)

    widths = _compute_col_widths(headers, trunc_rows, max_col_width)
    sep = _draw_sep(widths)

    print(sep)
    print(_draw_row(headers, widths))
    print(sep)
    for row in trunc_rows:
        print(_draw_row([str(row.get(h, "")) for h in headers], widths))
    print(sep)


# --------------------------
# Generic sweep driver
# --------------------------
RowCmdBuilder = Callable[[Dict[str, str], Dict[str, str], str, pathlib.Path], Tuple[List[str], str]]

def run_sweep(
    *,
    operator_name: str,
    in_path: pathlib.Path,
    out_path: pathlib.Path,
    script_path: pathlib.Path,
    python_bin: str,
    log_path: pathlib.Path,
    alias_map: Dict[str, List[str]],
    row_cmd_builder: RowCmdBuilder,
    # NEW:
    print_table_enable: bool = True,
    table_max_col_width: int = 32,
    table_columns: Optional[List[str]] = None,
) -> None:
    """Generic sweep runner to minimize duplication.

    After writing results, optionally prints a pretty ASCII table to stdout.

    Args:
        print_table_enable: Whether to print the results table to stdout.
        table_max_col_width: Max width of each column when printing.
        table_columns: If provided, only print these columns (that exist).
    """
    # If the log file exists, announce and remove it before creating the logger.
    if log_path.exists():
        # stdout print so the user sees it even before logger is ready.
        print(f"[INFO] Existing log found, removing: {log_path}")
        try:
            os.remove(log_path)
        except Exception as e:
            # If deletion fails, we still proceed; setup_logger(...) opens with mode='w'.
            print(f"[WARN] Failed to remove existing log: {e}")

    logger = setup_logger(log_path)
    logger.info("[%s] Input:  %s", operator_name, in_path)
    logger.info("[%s] Output: %s", operator_name, out_path)
    logger.info("[%s] Script: %s", operator_name, script_path)
    logger.info("[%s] Python: %s", operator_name, python_bin)
    logger.info("[%s] Log:    %s", operator_name, log_path)

    if not in_path.exists():
        logger.error("Input file not found: %s", in_path)
        sys.exit(2)
    if not script_path.exists():
        logger.error("Test script not found: %s", script_path)
        sys.exit(2)

    rows, fieldnames = read_rows_csv_any(in_path, logger)
    if not rows:
        logger.error("No rows found in input.")
        sys.exit(3)

    header_map = build_header_map(fieldnames, alias_map)
    logger.info("Header mapping (canonical -> actual): %s", header_map)

    results_rows: List[Dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        try:
            cmd, screen_msg = row_cmd_builder(row, header_map, python_bin, script_path)
        except Exception as e:
            logger.exception("Row %d: failed to build command: %s", idx, e)
            out_row = dict(row)
            out_row.update({
                "ref_latency_ms": "", "ref_tflops": "",
                "latency_ms": "", "tflops": "",
                "returncode": "-1",
                "note": f"build_cmd_error={type(e).__name__}:{e}",
            })
            results_rows.append(out_row)
            continue

        print(screen_msg)
        logger.info(screen_msg)

        rc, stdout, stderr = run_and_log(cmd, logger)
        metrics = parse_stdout_metrics(stdout)
        note = ""
        if rc != 0:
            note_parts = [f"rc={rc}"]
            last = tail_line(stderr)
            if last:
                note_parts.append(f"stderr_tail={last}")
            note = ";".join(note_parts)

        out_row = dict(row)
        out_row.update(metrics)
        out_row["returncode"] = str(rc)
        out_row["note"] = note
        results_rows.append(out_row)

        print(f"[DONE {idx}/{len(rows)}] latency_ms={metrics.get('latency_ms','')} TFLOPs={metrics.get('tflops','')}")
        logger.info("[DONE %d/%d] latency_ms=%s TFLOPs=%s",
                    idx, len(rows), metrics.get("latency_ms", ""), metrics.get("tflops", ""))

    # Write to CSV file
    out_fieldnames = write_results_csv(
        out_path,
        fieldnames,
        results_rows,
        extra_cols=["fwd_latency_ms", "fwd_tflops", "fwd_ref_latency_ms", "fwd_ref_tflops", "bwd_latency_ms", "bwd_tflops", "bwd_ref_latency_ms", "bwd_ref_tflops", "returncode", "note"],
    )
    logger.info("[%s] All done. Results saved to: %s", operator_name, out_path)

    # Pretty table (optional)
    if print_table_enable:
        if table_columns:
            # Filter to a subset if requested and exists.
            cols = [c for c in table_columns if c in out_fieldnames]
            if cols:
                filtered_rows = []
                for r in results_rows:
                    filtered_rows.append({c: r.get(c, "") for c in cols})
                print_table(cols, filtered_rows, max_col_width=table_max_col_width)
            else:
                # Fall back to all if provided columns are invalid.
                print_table(out_fieldnames, results_rows, max_col_width=table_max_col_width)
        else:
            print_table(out_fieldnames, results_rows, max_col_width=table_max_col_width)
