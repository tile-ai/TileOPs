"""HBM Bandwidth Benchmark — Python wrapper for hbm_saturation.cu.

Compiles and runs the CUDA microbenchmark, parses output, and prints
the calibration factor for tileops/perf/profiles/.

Calibration is derived from the STREAM Triad kernel (a = b + s*c, 2 reads +
1 write).  Triad is the industry standard for roofline bandwidth calibration:

    McCalpin, J.D., 1995. "Memory Bandwidth and Machine Balance in Current
    High Performance Computers." IEEE TCCA Newsletter.
    https://www.cs.virginia.edu/stream/

    Williams, S., Waterman, A. & Patterson, D., 2009. "Roofline: An Insightful
    Visual Performance Model for Multicore Architectures." CACM 52(4).

Usage:
    python benchmarks/hardware/memory/hbm_bandwidth.py [--profile h200] [--size-mb 2048]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from tileops.perf import load_profile

_CU_SRC = Path(__file__).parent / "hbm_saturation.cu"


def _compile(cu_path, binary_path, arch="sm_90"):
    """Compile the CUDA source. Raises on failure."""
    cmd = [
        "nvcc", "-O3", f"-arch={arch}",
        "-Wno-deprecated-gpu-targets",
        "-o", str(binary_path), str(cu_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"nvcc compilation failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


def _run(binary_path, size_mb, theo_peak_gbs):
    """Run the benchmark binary and return stdout lines."""
    cmd = [str(binary_path), str(size_mb), str(theo_peak_gbs)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"Benchmark failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip().splitlines()


def _parse_triad_peak(lines):
    """Extract the best Triad bandwidth (GB/s) from CSV output.

    Only considers lines starting with 'triad,' — STREAM Triad (2 reads +
    1 write) is the standard calibration kernel for roofline analysis.
    Its 2:1 read:write ratio is closer to real compute kernels than pure
    copy (1:1), which suffers worst-case HBM bus turnaround overhead.
    """
    best_gbs = 0.0
    for line in lines:
        if not line.startswith("triad,"):
            continue
        parts = line.split(",")
        if len(parts) >= 6:
            try:
                gbs = float(parts[5])  # best_gbs column
                best_gbs = max(best_gbs, gbs)
            except ValueError:
                continue
    return best_gbs


def main():
    parser = argparse.ArgumentParser(description="HBM bandwidth microbenchmark")
    parser.add_argument("--profile", default="h200", help="GPU profile name")
    parser.add_argument("--size-mb", type=int, default=2048, help="Working set size in MB")
    parser.add_argument("--arch", default="sm_90", help="CUDA architecture")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    theo_peak_gbs = profile["hbm"]["theoretical"] / 1e9

    print(f"Profile: {args.profile}")
    print(f"Theoretical HBM BW: {theo_peak_gbs:.1f} GB/s")
    print(f"Working set: {args.size_mb} MB")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        binary = Path(tmpdir) / "hbm_saturation"

        print("Compiling hbm_saturation.cu ...")
        _compile(_CU_SRC, binary, arch=args.arch)

        print("Running benchmark (5 runs x 200 reps, this may take a few minutes) ...\n")
        lines = _run(binary, args.size_mb, theo_peak_gbs)

    # Print raw output
    for line in lines:
        print(line)

    # Extract calibration from STREAM Triad results
    measured_peak = _parse_triad_peak(lines)
    if measured_peak > 0 and theo_peak_gbs > 0:
        calibration = measured_peak / theo_peak_gbs
        print(f"\n{'='*60}")
        print(f"Measured peak (triad vec4): {measured_peak:.2f} GB/s")
        print(f"Theoretical:               {theo_peak_gbs:.1f} GB/s")
        print(f"Calibration:               {calibration:.4f}")
        print(f"\nUpdate tileops/perf/profiles/{args.profile}.yaml:")
        print(f"  hbm.calibration: {calibration:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
