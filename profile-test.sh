#!/usr/bin/env bash
# profile-test.sh
# Orchestrates MHA and MHA-Decode sweeps and prints final CSVs as tables.

set -euo pipefail

############################################
# User-editable settings                   #
############################################

# Python interpreter
PYTHON_BIN="python"

# Sweep wrapper scripts
SWEEP_MHA="profile/mha_profile_test.py"
SWEEP_DECODE="profile/mha_decode_profile_test.py"

# Underlying test scripts (passed into the sweep wrappers)
TEST_MHA="tests/test_mha.py"
TEST_DECODE="tests/test_mha_decode.py"

# Input CSVs for each sweep
INPUT_MHA="profile/input_params/mha_params.csv"
INPUT_DECODE="profile/input_params/mha_decode_params.csv"

# Terminal table rendering width
TABLE_MAX_COL_WIDTH=40

############################################
# End of user-editable settings            #
############################################

# ANSI colors
RED=$'\033[31m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
CYAN=$'\033[36m'
BOLD=$'\033[1m'
RESET=$'\033[0m'

# Fatal error helper
die() { echo "${RED}Error:${RESET} $*" >&2; exit 1; }

# Section separator with optional title
sep() {
  local title="${1:-}"
  local line="=============================================================================="
  if [[ -n "$title" ]]; then
    echo "$line"
    echo "${BOLD}${CYAN}$title${RESET}"
    echo "$line"
  else
    echo "$line"
  fi
}

# Pretty-print a CSV file as a fixed-width ASCII table (embedded Python, no extra deps)
print_csv_as_table() {
  local csv_path="$1"
  local max_col_width="$2"
  "$PYTHON_BIN" - "$csv_path" "$max_col_width" <<'PYCODE'
# -*- coding: utf-8 -*-
# Render a CSV as a fixed-width ASCII table (truncates long cells).

import csv, sys, os
from typing import List, Dict

def truncate(s: str, limit: int) -> str:
    if limit <= 3 or len(s) <= limit:
        return s[:limit]
    return s[:limit-3] + "..."

def col_widths(headers: List[str], rows: List[Dict[str,str]], maxw: int) -> List[int]:
    w = [len(h) for h in headers]
    for r in rows:
        for i, h in enumerate(headers):
            val = str(r.get(h, ""))
            w[i] = min(max(w[i], len(val)), maxw)
    return w

def draw_row(cells: List[str], widths: List[int]) -> str:
    parts = []
    for c, w in zip(cells, widths):
        parts.append(" " + c.ljust(w) + " ")
    return "|" + "|".join(parts) + "|"

def draw_sep(widths: List[int]) -> str:
    parts = [ "-" * (w + 2) for w in widths ]
    return "+" + "+".join(parts) + "+"

def main():
    if len(sys.argv) < 3:
        print("Usage: <script> <csv_path> <max_col_width>", file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    maxw = int(sys.argv[2])
    if not os.path.exists(path):
        print(f"[WARN] CSV not found: {path}", file=sys.stderr)
        sys.exit(0)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    trunc_rows = [{ h: truncate(str(r.get(h, "")), maxw) for h in headers } for r in rows]
    widths = col_widths(headers, trunc_rows, maxw)
    sep = draw_sep(widths)

    print(sep)
    print(draw_row(headers, widths))
    print(sep)
    for r in trunc_rows:
        print(draw_row([str(r.get(h, "")) for h in headers], widths))
    print(sep)

if __name__ == "__main__":
    main()
PYCODE
}

# Arg parsing: only --out-dir is supported
OUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
    *)
      die "Unknown or unsupported argument: $1 (only --out-dir is accepted)"
      ;;
  esac
done

[[ -z "$OUT_DIR" ]] && die "--out-dir is required (target directory for logs and CSVs)"

# Validate tools and inputs
[[ -x "$(command -v "$PYTHON_BIN")" ]] || die "Python not found/executable: $PYTHON_BIN"
[[ -f "$SWEEP_MHA" ]]      || die "Missing sweep script: $SWEEP_MHA"
[[ -f "$SWEEP_DECODE" ]]   || die "Missing sweep script: $SWEEP_DECODE"
[[ -f "$TEST_MHA" ]]       || die "Missing test script: $TEST_MHA"
[[ -f "$TEST_DECODE" ]]    || die "Missing test script: $TEST_DECODE"
[[ -f "$INPUT_MHA" ]]      || die "Missing input CSV: $INPUT_MHA"
[[ -f "$INPUT_DECODE" ]]   || die "Missing input CSV: $INPUT_DECODE"

# Prepare output directory (warn and recreate)
if [[ -d "$OUT_DIR" ]]; then
  echo "${YELLOW}[WARN] Output directory exists, removing: ${OUT_DIR}${RESET}"
  rm -rf -- "$OUT_DIR" || die "Failed to remove existing out-dir"
fi
mkdir -p -- "$OUT_DIR" || die "Failed to create out-dir"

# Output file paths
LOG_MHA="${OUT_DIR}/mha_sweep.log"
CSV_MHA="${OUT_DIR}/mha_results.csv"
LOG_DEC="${OUT_DIR}/mha_decode_sweep.log"
CSV_DEC="${OUT_DIR}/mha_decode_results.csv"

# Run MHA sweep
sep "START MHA SWEEP"
set +e
"$PYTHON_BIN" "$SWEEP_MHA" \
  --input "$INPUT_MHA" \
  --output "$CSV_MHA" \
  --script "$TEST_MHA" \
  --log "$LOG_MHA" \
  --python "$PYTHON_BIN" \
  --no-print-table
RC_MHA=$?
set -e
if [[ $RC_MHA -eq 0 ]]; then
  echo "${GREEN}Success${RESET} (MHA sweep completed)"
else
  echo "${RED}Fail${RESET} (MHA sweep rc=$RC_MHA)"
fi
sep

# Run MHA-Decode sweep
sep "START MHA-DECODE SWEEP"
set +e
"$PYTHON_BIN" "$SWEEP_DECODE" \
  --input "$INPUT_DECODE" \
  --output "$CSV_DEC" \
  --script "$TEST_DECODE" \
  --log "$LOG_DEC" \
  --python "$PYTHON_BIN" \
  --no-print-table
RC_DEC=$?
set -e
if [[ $RC_DEC -eq 0 ]]; then
  echo "${GREEN}Success${RESET} (DECODE sweep completed)"
else
  echo "${RED}Fail${RESET} (DECODE sweep rc=$RC_DEC)"
fi
sep

# Print all CSVs in output directory as tables
echo "${BOLD}${CYAN}Final CSV tables in: ${OUT_DIR}${RESET}"
shopt -s nullglob
CSV_LIST=("$OUT_DIR"/*.csv)
if [[ ${#CSV_LIST[@]} -eq 0 ]]; then
  echo "${YELLOW}[WARN] No CSV files found in ${OUT_DIR}${RESET}"
  if [[ $RC_MHA -ne 0 || $RC_DEC -ne 0 ]]; then exit 1; else exit 0; fi
fi

for csv in "${CSV_LIST[@]}"; do
  echo
  sep "TABLE: $(basename "$csv")"
  print_csv_as_table "$csv" "$TABLE_MAX_COL_WIDTH"
done

# Exit code reflects whether any sweep failed
if [[ $RC_MHA -ne 0 || $RC_DEC -ne 0 ]]; then
  exit 1
fi
exit 0
