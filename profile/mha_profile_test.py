#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep runner for MHA using common_tools.run_sweep."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, List, Tuple

from common_tools import (
    RowCmdBuilder,
    get_cell,
    run_sweep,
    to_bool,
    to_int,
)


ALIASES_MHA: Dict[str, List[str]] = {
    "bs": ["bs", "batch", "b"],
    "seq_len": ["seq_len", "sl", "seq", "sequence_len", "sequence_length"],
    "head_num": ["head_num", "heads", "head", "nheads", "h"],
    "dim": ["dim", "hd", "head_dim", "d", "embed_dim"],
    "causal": ["causal", "is_causal", "mask_causal"],
    "tune": ["tune", "autotune", "enable_tune"],
}


def build_cmd_mha(row: Dict[str, str], header_map: Dict[str, str], python_bin: str, script_path: pathlib.Path) -> Tuple[List[str], str]:
    batch = to_int(get_cell(row, header_map, "bs", "8"), 8)
    seq_len = to_int(get_cell(row, header_map, "seq_len", "1024"), 1024)
    heads = to_int(get_cell(row, header_map, "head_num", "32"), 32)
    dim = to_int(get_cell(row, header_map, "dim", "64"), 64)
    causal = to_bool(get_cell(row, header_map, "causal", "False"), False)
    tune = to_bool(get_cell(row, header_map, "tune", "True"), True)

    cmd = [
        python_bin, str(script_path),
        "--batch", str(batch),
        "--seq_len", str(seq_len),
        "--heads", str(heads),
        "--dim", str(dim),
    ]
    if tune:
        cmd.append("--tune")
    if causal:
        cmd.append("--causal")

    msg = f"[RUN MHA] batch={batch}, seq_len={seq_len}, heads={heads}, dim={dim}, causal={causal}, tune={tune}"
    return cmd, msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep MHA from CSV/TSV using common_tools.run_sweep.")
    parser.add_argument("--input", required=True, help="Path to input CSV/TSV.")
    parser.add_argument("--output", required=True, help="Path to output CSV.")
    parser.add_argument("--script", default="test_mha.py", help="Path to test_mha.py.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter.")
    parser.add_argument("--log", default="mha_sweep.log", help="Path to log file.")

    # Default to printing the table, with a switch to disable it.
    parser.add_argument("--no-print-table", action="store_true",
                        help="Disable pretty ASCII table printing (enabled by default).")
    parser.add_argument("--table-max-col-width", type=int, default=32,
                        help="Max width for each printed column.")
    parser.add_argument("--table-columns", type=str, default="",
                        help="Comma-separated subset of columns to print (optional).")

    args = parser.parse_args()
    table_cols = [c.strip() for c in args.table_columns.split(",") if c.strip()] if args.table_columns else None

    run_sweep(
        operator_name="MHA",
        in_path=pathlib.Path(args.input).expanduser().resolve(),
        out_path=pathlib.Path(args.output).expanduser().resolve(),
        script_path=pathlib.Path(args.script).expanduser().resolve(),
        python_bin=args.python,
        log_path=pathlib.Path(args.log).expanduser().resolve(),
        alias_map=ALIASES_MHA,
        row_cmd_builder=build_cmd_mha,
        print_table_enable=not args.no_print_table,
        table_max_col_width=args.table_max_col_width,
        table_columns=table_cols,
    )


if __name__ == "__main__":
    main()
