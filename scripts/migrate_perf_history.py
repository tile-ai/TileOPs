#!/usr/bin/env python3
"""Migrate perf_history.json op keys from old class names to new manifest keys.

Usage:
    python scripts/migrate_perf_history.py perf_history.json
    python scripts/migrate_perf_history.py perf_history.json -o migrated.json

If no -o is given, the file is overwritten in place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Old class name -> new manifest key (PascalCase with direction suffix).
RENAME_MAP: dict[str, str] = {
    # Norms
    "AdaLayerNormOp": "AdaLayerNormFwdOp",
    "AdaLayerNormZeroOp": "AdaLayerNormZeroFwdOp",
    "FusedAddLayerNormOp": "FusedAddLayerNormFwdOp",
    "FusedAddRmsNormOp": "FusedAddRMSNormFwdOp",
    "GroupNormOp": "GroupNormFwdOp",
    "InstanceNormOp": "InstanceNormFwdOp",
    "LayerNormOp": "LayerNormFwdOp",
    "RmsNormOp": "RMSNormFwdOp",
    # Reduction
    "AllOp": "AllFwdOp",
    "AmaxOp": "AmaxFwdOp",
    "AminOp": "AminFwdOp",
    "AnyOp": "AnyFwdOp",
    "ArgmaxOp": "ArgmaxFwdOp",
    "ArgminOp": "ArgminFwdOp",
    "CountNonzeroOp": "CountNonzeroFwdOp",
    "CumprodOp": "CumprodFwdOp",
    "CumsumOp": "CumsumFwdOp",
    "InfNormOp": "InfNormFwdOp",
    "L1NormOp": "L1NormFwdOp",
    "L2NormOp": "L2NormFwdOp",
    "LogSoftmaxOp": "LogSoftmaxFwdOp",
    "LogSumExpOp": "LogSumExpFwdOp",
    "MeanOp": "MeanFwdOp",
    "ProdOp": "ProdFwdOp",
    "SoftmaxOp": "SoftmaxFwdOp",
    "StdOp": "StdFwdOp",
    "SumOp": "SumFwdOp",
    "VarOp": "VarFwdOp",
    "VarMeanOp": "VarMeanFwdOp",
    # Conv
    "Conv1dOp": "Conv1dFwdOp",
    # Attention
    "GroupQueryAttentionFwdOp": "GqaFwdOp",
    "GroupQueryAttentionBwdOp": "GqaBwdOp",
    "GroupQueryAttentionDecodeWithKVCacheOp": "GqaDecodeFwdOp",
    "GroupQueryAttentionDecodePagedWithKVCacheOp": "GqaDecodePagedFwdOp",
    "MultiHeadAttentionFwdOp": "MhaFwdOp",
    "MultiHeadAttentionBwdOp": "MhaBwdOp",
    "MultiHeadAttentionDecodeWithKVCacheOp": "MhaDecodeFwdOp",
    "MultiHeadAttentionDecodePagedWithKVCacheOp": "MhaDecodePagedFwdOp",
    "DeepSeekSparseAttentionDecodeWithKVCacheOp": "DeepSeekDsaDecodeFwdOp",
    "MultiHeadLatentAttentionDecodeWithKVCacheOp": "DeepSeekMlaDecodeFwdOp",
    # MOE
    "FusedMoeExperts": "MoeFusedExpertsFwdOp",
    "FusedMoeExpertsPadded": "MoeFusedExpertsPaddedFwdOp",
    "MoeGroupedGemmNopadOp": "MoeGroupedGemmNopadFwdOp",
    "MoePermuteAlignOp": "MoePermuteAlignFwdOp",
    "MoePermuteNopadOp": "MoePermuteNopadFwdOp",
    "MoePermutePaddedOp": "MoePermutePaddedFwdOp",
    "MoeUnpermuteOp": "MoeUnpermuteFwdOp",
}


def _rename_op_keys(ops: dict) -> tuple[dict, int]:
    """Rename op keys in an ops dict using RENAME_MAP.

    Returns (new_ops_dict, count_of_renamed_keys).
    """
    migrated: dict = {}
    renamed = 0
    for key, value in ops.items():
        new_key = RENAME_MAP.get(key, key)
        if new_key != key:
            renamed += 1
        if new_key in migrated:
            print(
                f"WARNING: key collision — both '{key}' and another key map to "
                f"'{new_key}'. Keeping the last occurrence.",
                file=sys.stderr,
            )
        migrated[new_key] = value
    return migrated, renamed


def migrate(data: dict) -> tuple[dict, int]:
    """Rename op keys in *data* using RENAME_MAP.

    Supports the real perf_history.json format::

        {"runs": [{"date": "...", "ops": {"OldName": {...}}}]}

    as well as a flat dict format (top-level keys are op names).

    Returns (new_data, count_of_renamed_keys).
    """
    # Nested format: {"runs": [...]}
    if "runs" in data:
        total_renamed = 0
        new_runs = []
        for run in data["runs"]:
            if "ops" not in run:
                new_runs.append(run)
                continue
            new_ops, count = _rename_op_keys(run["ops"])
            total_renamed += count
            new_run = {k: v for k, v in run.items() if k != "ops"}
            new_run["ops"] = new_ops
            new_runs.append(new_run)
        return {"runs": new_runs}, total_renamed

    # Flat format: top-level keys are op names
    return _rename_op_keys(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to perf_history.json")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output path (default: overwrite input in place)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(args.input) as f:
        data = json.load(f)

    migrated, count = migrate(data)

    out_path = args.output or args.input
    with open(out_path, "w") as f:
        json.dump(migrated, f, indent=2)
        f.write("\n")

    print(f"Migrated {count} key(s) in {args.input} -> {out_path}")


if __name__ == "__main__":
    main()
