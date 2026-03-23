#!/usr/bin/env python3
"""Generate gpu_profile.yaml from hardware benchmark CSV results.

Reads CSV files from benchmarks/hardware/results/ and produces a YAML profile
containing measured peaks and calibration factors for use by tileops/perf/.

Usage:
    python scripts/generate_gpu_profile.py [--results-dir DIR] [--output PATH] [--gpu-name NAME]

Example:
    python scripts/generate_gpu_profile.py \
        --results-dir benchmarks/hardware/results \
        --output tileops/perf/profiles/h200.yaml \
        --gpu-name "NVIDIA H200"
"""

import argparse
import csv
import os
from datetime import datetime

import yaml

# Theoretical peak specs (must stay in sync with benchmarks/hardware/utils/env.py)
THEORETICAL_PEAKS = {
    "NVIDIA H200": {
        "hbm_bw_gbs": 4800.0,
        "fp16_tensor_tflops": 989.5,
        "bf16_tensor_tflops": 989.5,
        "fp8_tensor_tflops": 1979.0,
        "tf32_tensor_tflops": 494.7,
        "fp32_tflops": 67.0,
    },
    "NVIDIA H100 80GB HBM3": {
        "hbm_bw_gbs": 3350.0,
        "fp16_tensor_tflops": 989.5,
        "bf16_tensor_tflops": 989.5,
        "fp8_tensor_tflops": 1979.0,
        "tf32_tensor_tflops": 494.7,
        "fp32_tflops": 67.0,
    },
}


def parse_hbm_results(results_dir):
    """Parse HBM peak bandwidth from hbm_peak.csv or bandwidth.csv.

    Returns the best measured copy bandwidth in GB/s.
    """
    best_copy_bw = 0.0

    # Try hbm_peak.csv first (h200_clean format)
    for csv_name in ["hbm_peak.csv", "bandwidth.csv"]:
        csv_path = os.path.join(results_dir, csv_name)
        if not os.path.isfile(csv_path):
            continue

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Identify copy rows: check "op" column (hbm_peak.csv schema),
                # then "notes" column (bandwidth.csv schema from hbm_bandwidth.py
                # which writes benchmark="bandwidth" and puts "copy, ..." in notes)
                op = row.get("op", "")
                notes = row.get("notes", "")
                is_copy = "copy" in op.lower() or notes.lower().startswith("copy")
                if not is_copy:
                    continue
                for field in ["best_gbs", "bandwidth_gbs"]:
                    val = row.get(field, "")
                    if val:
                        try:
                            bw = float(val)
                            best_copy_bw = max(best_copy_bw, bw)
                        except (ValueError, TypeError):
                            pass
        if best_copy_bw > 0:
            break

    return best_copy_bw


def parse_gemm_results(results_dir):
    """Parse GEMM throughput from gemm_throughput.csv.

    Returns dict of {dtype: best_measured_tflops} for cuBLAS results.
    """
    best_tflops = {}

    csv_path = os.path.join(results_dir, "gemm_throughput.csv")
    if not os.path.isfile(csv_path):
        return best_tflops

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            backend = row.get("backend", "")
            if backend != "cublas":
                continue

            dtype = row.get("dtype", "")
            tflops_str = row.get("tflops", "")
            if not tflops_str:
                continue
            try:
                tflops = float(tflops_str)
            except (ValueError, TypeError):
                continue

            if dtype not in best_tflops or tflops > best_tflops[dtype]:
                best_tflops[dtype] = tflops

    return best_tflops


def generate_profile(results_dir, output_path, gpu_name):
    """Generate a GPU profile YAML from benchmark results.

    Args:
        results_dir: Path to directory containing CSV benchmark results
        output_path: Path to write the output YAML file
        gpu_name: GPU model name (e.g., "NVIDIA H200")
    """
    theoretical = THEORETICAL_PEAKS.get(gpu_name, {})

    # Parse HBM bandwidth
    measured_hbm_bw = parse_hbm_results(results_dir)
    theo_hbm_bw = theoretical.get("hbm_bw_gbs", 0)
    hbm_cal = round(measured_hbm_bw / theo_hbm_bw, 4) if theo_hbm_bw > 0 else 0

    # Parse GEMM throughput
    gemm_tflops = parse_gemm_results(results_dir)

    # Build tensor core section
    tensor_core = {}
    for dtype in ["fp16", "bf16"]:
        measured = gemm_tflops.get(dtype, 0)
        theo_key = f"{dtype}_tensor_tflops"
        theo = theoretical.get(theo_key, 0)
        cal = round(measured / theo, 4) if theo > 0 else 0
        tensor_core[dtype] = {
            "measured": round(measured, 2),
            "theoretical": theo,
            "calibration_factor": cal,
        }

    # Build profile
    profile = {
        "gpu_name": gpu_name,
        "generated_at": datetime.now().isoformat(),
        "hbm_bandwidth_gbs": {
            "measured": round(measured_hbm_bw, 2),
            "theoretical": theo_hbm_bw,
            "calibration_factor": hbm_cal,
        },
        "tensor_core_tflops": tensor_core,
    }

    # Add any extra theoretical peaks
    if theoretical:
        profile["theoretical_peaks"] = theoretical

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(profile, f, default_flow_style=False, sort_keys=False)

    print(f"GPU profile written to: {output_path}")
    print(f"  GPU: {gpu_name}")
    print(f"  HBM BW: {measured_hbm_bw:.1f} GB/s "
          f"({hbm_cal * 100:.1f}% of {theo_hbm_bw} GB/s)")
    for dtype in ["fp16", "bf16"]:
        tc = tensor_core.get(dtype, {})
        print(f"  {dtype} TC: {tc.get('measured', 0):.1f} TFLOPS "
              f"({tc.get('calibration_factor', 0) * 100:.1f}% of "
              f"{tc.get('theoretical', 0)} TFLOPS)")

    return profile


def main():
    parser = argparse.ArgumentParser(
        description="Generate GPU profile YAML from benchmark results",
    )
    parser.add_argument(
        "--results-dir",
        default="benchmarks/hardware/results",
        help="Directory containing CSV benchmark results",
    )
    parser.add_argument(
        "--output",
        default="tileops/perf/profiles/gpu_profile.yaml",
        help="Output YAML file path",
    )
    parser.add_argument(
        "--gpu-name",
        default="NVIDIA H200",
        help="GPU model name for theoretical peak lookup",
    )
    args = parser.parse_args()

    generate_profile(args.results_dir, args.output, args.gpu_name)


if __name__ == "__main__":
    main()
