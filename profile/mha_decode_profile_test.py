#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep runner for MHA-Decode using common_tools.run_sweep."""

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


ALIASES_DECODE: Dict[str, List[str]] = {
    "bs": ["bs", "batch", "b"],
    "seqlen_q": ["seqlen_q", "q_seq_len", "q_len", "seq_len_q"],
    "seqlen_kv": ["seqlen_kv", "kv_seq_len", "kvsl", "kv_len", "seq_len_kv"],
    "head_num": ["head_num", "heads", "head", "nheads", "h"],
    "dim": ["dim", "hd", "head_dim", "d", "embed_dim"],
    "tune": ["tune", "autotune", "enable_tune"],
}


def build_cmd_decode(row: Dict[str, str], header_map: Dict[str, str], python_bin: str, script_path: pathlib.Path) -> Tuple[List[str], str]:
    batch = to_int(get_cell(row, header_map, "bs", "1"), 1)
    seqlen_q = to_int(get_cell(row, header_map, "seqlen_q", "64"), 64)
    seqlen_kv = to_int(get_cell(row, header_map, "seqlen_kv", "8192"), 8192)
    heads = to_int(get_cell(row, header_map, "head_num", "32"), 32)
    dim = to_int(get_cell(row, header_map, "dim", "128"), 128)
    tune = to_bool(get_cell(row, header_map, "tune", "True"), True)

    cmd = [
        python_bin, str(script_path),
        "--batch", str(batch),
        "--seqlen_q", str(seqlen_q),
        "--seqlen_kv", str(seqlen_kv),
        "--heads", str(heads),
        "--dim", str(dim),
    ]
    if tune:
        cmd.append("--tune")

    msg = (f"[RUN DECODE] batch={batch}, seqlen_q={seqlen_q}, "
           f"seqlen_kv={seqlen_kv}, heads={heads}, dim={dim}, tune={tune}")
    return cmd, msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep MHA-Decode from CSV/TSV using common_tools.run_sweep.")
    parser.add_argument("--input", required=True, help="Path to input CSV/TSV.")
    parser.add_argument("--output", required=True, help="Path to output CSV.")
    parser.add_argument("--script", default="test_mha_decode.py", help="Path to test_mha_decode.py.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter.")
    parser.add_argument("--log", default="mha_decode_sweep.log", help="Path to log file.")

    # 改为“默认打印表格”，提供关闭开关
    parser.add_argument("--no-print-table", action="store_true",
                        help="Disable pretty ASCII table printing (enabled by default).")
    parser.add_argument("--table-max-col-width", type=int, default=32,
                        help="Max width for each printed column.")
    parser.add_argument("--table-columns", type=str, default="",
                        help="Comma-separated subset of columns to print (optional).")

    args = parser.parse_args()
    table_cols = [c.strip() for c in args.table_columns.split(",") if c.strip()] if args.table_columns else None

    run_sweep(
        operator_name="DECODE",
        in_path=pathlib.Path(args.input).expanduser().resolve(),
        out_path=pathlib.Path(args.output).expanduser().resolve(),
        script_path=pathlib.Path(args.script).expanduser().resolve(),
        python_bin=args.python,
        log_path=pathlib.Path(args.log).expanduser().resolve(),
        alias_map=ALIASES_DECODE,
        row_cmd_builder=build_cmd_decode,
        print_table_enable=not args.no_print_table,
        table_max_col_width=args.table_max_col_width,
        table_columns=table_cols,
    )


if __name__ == "__main__":
    main()
