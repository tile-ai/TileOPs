"""CUDA-Core FMA Throughput Benchmark — Python wrapper for fma_saturation.cu.

Compiles and runs the CUDA microbenchmark, parses its output, and prints the
calibration factor for src/tileops/perf/profiles/.  The CUDA-core counterpart
of memory/hbm_bandwidth.py: the two together fix both roofline axes.

Usage:
    python benchmarks/hardware/compute/fma_throughput.py [--profile h200] [--iters 20000]
"""

import argparse
import statistics
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from tileops.perf import load_profile

_CU_SRC = Path(__file__).parent / "fma_saturation.cu"

# One boost bin on sm_90; more clock movement than this is not a locked-clock run.
_CLOCK_STEADY_MHZ = 15.0

# fp32 FMA lanes per SM per clock on compute capability 9.0, from the CUDA C
# Programming Guide's native arithmetic throughput table.
_FMA_LANES_PER_SM = 128


def _compile(cu_path, binary_path, arch="sm_90"):
    """Compile the CUDA source. Raises on failure."""
    cmd = [
        "nvcc",
        "-O3",
        f"-arch={arch}",
        "-Wno-deprecated-gpu-targets",
        "-o",
        str(binary_path),
        str(cu_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"nvcc compilation failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


class _ClockSampler(threading.Thread):
    """Sample SM clock, power and temperature while the benchmark runs.

    A calibration is only meaningful alongside the clocks it was taken at;
    recording them turns an unreproducible number into a conditional one.
    """

    def __init__(self, gpu_index, interval_ms=200):
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self.samples = []  # (sm_mhz, watts, celsius)
        self._stopped = threading.Event()

    def run(self):
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=clocks.sm,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            f"-lms={self.interval_ms}",
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
        except OSError:
            return  # no nvidia-smi: telemetry is best-effort
        self._proc = proc
        for line in proc.stdout:
            if self._stopped.is_set():
                break
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                self.samples.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
        proc.terminate()

    def stop(self):
        self._stopped.set()
        proc = getattr(self, "_proc", None)
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)  # reap; terminate alone leaves a zombie
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def summary(self):
        """Return (sm_min, sm_median, sm_max, watts_max, temp_max) or None."""
        if not self.samples:
            return None
        sm = sorted(s[0] for s in self.samples)
        return (
            sm[0],
            statistics.median(sm),
            sm[-1],
            max(s[1] for s in self.samples),
            max(s[2] for s in self.samples),
        )


def _run(binary_path, iters, theo_peak_tflops, gpu_index):
    """Run the benchmark binary and return (stdout lines, clock sampler)."""
    cmd = [str(binary_path), str(iters), str(theo_peak_tflops)]
    sampler = _ClockSampler(gpu_index)
    sampler.start()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    finally:
        sampler.stop()
        sampler.join(timeout=5)
    if result.returncode != 0:
        print(f"Benchmark failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip().splitlines(), sampler


# CSV layout emitted by fma_saturation.cu
_COL_ILP, _COL_STDDEV, _COL_MEDIAN_RATE = 1, 6, 8


def _parse_peak(lines, op):
    """Return (median_rate, ilp, stddev_pct) for the best config of one op.

    Best config across the ILP sweep, but each config judged by its median of
    five runs — one lucky run is not a sustained rate.
    """
    best = (0.0, None, None)
    for line in lines:
        if not line.startswith(f"{op},"):
            continue
        parts = line.split(",")
        if len(parts) <= _COL_MEDIAN_RATE:
            continue
        try:
            rate = float(parts[_COL_MEDIAN_RATE])
            ilp = int(parts[_COL_ILP])
            stddev = float(parts[_COL_STDDEV])
        except ValueError:
            continue
        if rate > best[0]:
            best = (rate, ilp, stddev)
    return best


def _parse_device(lines):
    """Pull (sm_count, max_clock_ghz) out of the benchmark's banner line."""
    for line in lines:
        if not line.startswith("GPU:"):
            continue
        sm_count = max_ghz = None
        for field in line.split("|"):
            field = field.strip()
            if field.startswith("SMs:"):
                sm_count = int(field.split(":")[1])
            elif field.startswith("max SM clock:"):
                max_ghz = float(field.split(":")[1].strip().split()[0])
        return sm_count, max_ghz
    return None, None


def _parse_derived(lines, key):
    """Extract a `key,value` line from the '# derived' block."""
    for line in lines:
        if line.startswith(f"{key},"):
            try:
                return float(line.split(",")[1])
            except (IndexError, ValueError):
                return None
    return None


def main():
    parser = argparse.ArgumentParser(description="CUDA-core fp32 FMA microbenchmark")
    parser.add_argument("--profile", default="h200", help="GPU profile name")
    parser.add_argument("--iters", type=int, default=20000, help="FMA iterations per chain")
    parser.add_argument("--arch", default="sm_90", help="CUDA architecture")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU to sample telemetry from")
    parser.add_argument(
        "--allow-unlocked-clocks",
        action="store_true",
        help="Emit a calibration factor even though the SM clock moved during the run",
    )
    args = parser.parse_args()

    profile = load_profile(args.profile)
    cuda_core = profile.get("cuda_core", {}).get("fp32")
    if cuda_core is None:
        print(
            f"Profile '{args.profile}' has no cuda_core.fp32 section. "
            "Add it with the datasheet fp32 (non-tensor) peak before calibrating.",
            file=sys.stderr,
        )
        sys.exit(1)
    theo_peak_tflops = cuda_core["theoretical"] / 1e12

    print(f"Profile: {args.profile}")
    print(f"Theoretical fp32 FMA: {theo_peak_tflops:.1f} TFLOP/s")
    print(f"Iterations per chain: {args.iters}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        binary = Path(tmpdir) / "fma_saturation"

        print("Compiling fma_saturation.cu ...")
        _compile(_CU_SRC, binary, arch=args.arch)

        print("Running benchmark (5 runs x 50 reps per ILP, this takes a few minutes) ...\n")
        lines, sampler = _run(binary, args.iters, theo_peak_tflops, args.gpu_index)

    for line in lines:
        print(line)

    measured_peak, fma_ilp, fma_stddev = _parse_peak(lines, "fma")
    mufu_peak, mufu_ilp, _ = _parse_peak(lines, "mufu")
    implied_ghz = _parse_derived(lines, "implied_sm_clock_ghz")
    mufu_ratio = _parse_derived(lines, "fma_to_mufu_ratio")
    telemetry = sampler.summary()

    if measured_peak <= 0 or theo_peak_tflops <= 0:
        return

    calibration = measured_peak / theo_peak_tflops
    print(f"\n{'=' * 60}")
    print(
        f"Measured peak (fp32 FMA):  {measured_peak:.2f} TFLOP/s"
        f"  (median of 5, ILP={fma_ilp}, spread {fma_stddev:.2f}%)"
    )
    print(f"Theoretical:               {theo_peak_tflops:.1f} TFLOP/s")
    print(f"Calibration:               {calibration:.4f}")
    if implied_ghz is not None:
        print(
            f"Implied SM clock:          {implied_ghz:.3f} GHz"
            f"  (assumes all {_FMA_LANES_PER_SM} lanes busy)"
        )
    if mufu_peak > 0:
        print(f"MUFU (rsqrt.approx.ftz):   {mufu_peak:.2f} Gop/s (ILP={mufu_ilp})", end="")
        print(f"  [{mufu_ratio:.1f} FMA per MUFU result]" if mufu_ratio else "")

    # An unlocked clock makes the calibration conditional, so the update line is
    # withheld until the caller locks the clock or passes --allow-unlocked-clocks.
    clocks_steady = False
    if telemetry is None:
        print("\nSM clock:                  not sampled (nvidia-smi unavailable)")
    else:
        sm_min, sm_med, sm_max, watts, temp = telemetry
        print(
            f"\nSM clock during run:       {sm_min:.0f} / {sm_med:.0f} / {sm_max:.0f} MHz"
            f"  (min / median / max)"
        )
        print(f"Peak power, temperature:   {watts:.0f} W, {temp:.0f} C")
        clocks_steady = (sm_max - sm_min) <= _CLOCK_STEADY_MHZ

        # calibration = lane utilisation * clock headroom:
        #   lane utilisation = measured / (2 * lanes * SMs * clock)  — hardware + kernel
        #   clock headroom   = held clock / max clock                — the site's policy
        sm_count, max_ghz = _parse_device(lines)
        if sm_count and sm_med > 0:
            ceiling = 2.0 * _FMA_LANES_PER_SM * sm_count * (sm_med * 1e6) / 1e12
            lane_util = measured_peak / ceiling
            print(
                f"Ceiling at {sm_med:.0f} MHz:        {ceiling:.2f} TFLOP/s"
                f"  ({_FMA_LANES_PER_SM} lanes x {sm_count} SMs)"
            )
            print(f"Lane utilisation:          {lane_util * 100:.1f}%")
            if max_ghz and sm_med * 1e6 < max_ghz * 1e9 * 0.99:
                headroom = (sm_med * 1e6) / (max_ghz * 1e9)
                print(
                    f"Clock headroom:            {headroom * 100:.1f}%"
                    f" of the {max_ghz:.2f} GHz maximum — this board is not at boost."
                )
                print(
                    f"  calibration {calibration:.4f}"
                    f" = lane utilisation {lane_util:.4f} x clock headroom {headroom:.4f}"
                )

    if clocks_steady or args.allow_unlocked_clocks:
        if not clocks_steady:
            print("\nWARNING: the SM clock moved during this run; the calibration below is")
            print("         conditional on the clocks recorded above, not on locked ones.")
        print(f"\nUpdate src/tileops/perf/profiles/{args.profile}.yaml:")
        print(f"  cuda_core.fp32.calibration: {calibration:.4f}")
    elif telemetry is None:
        print("\nNot emitting a calibration factor: the SM clock could not be sampled,")
        print("so a locked clock cannot be verified. Pass --allow-unlocked-clocks to")
        print("accept a calibration taken at unknown clocks.")
    else:
        print(
            f"\nNot emitting a calibration factor: the SM clock varied by"
            f" {telemetry[2] - telemetry[0]:.0f} MHz during the run."
        )
        print("Lock the clocks (see benchmarks/hardware/README.md) and re-run, or pass")
        print("--allow-unlocked-clocks to accept a calibration taken at the clocks above.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
