"""Tensor-core GEMM throughput via cuBLAS — calibrates tensor_core.* in profiles.

Times torch.matmul (cuBLAS) for fp16/bf16/tf32 and torch._scaled_mm (cuBLASLt)
for fp8, sweeping square sizes, and prints two factors per dtype for
src/tileops/perf/profiles/: `calibration` (sustained) and `calibration_burst`.
The tensor-core counterpart of fma_throughput.py.

Unlike the FMA benchmark, clock locking cannot hold here: a saturating GEMM
drives an H200 into its power cap, so the SM clock is set by the cap, not the
lock.  Each dtype gets its own telemetry window, and the sustained factor is
reported next to the clocks it was taken at; burst is the pre-cap rate.

Usage:
    python benchmarks/hardware/compute/gemm_throughput.py [--profile h200]
"""

import argparse
import statistics
import subprocess
import sys
import time

import torch
from fma_throughput import _ClockSampler

from tileops.perf import load_profile

# Square GEMMs large enough to saturate the tensor cores; the sweep exists
# because the cuBLAS-peak size moves with dtype and architecture.
_SIZES = (4096, 8192, 16384)

_RUNS = 5
# Long enough to average over several power-cap clock oscillations; short runs
# sample one slice of the cycle and spread by >10% run to run.
_TARGET_RUN_MS = 4000.0

# Burst: rate over the first short slice of load after _COOLDOWN_S of idle,
# before the power cap engages.  Reps are sized from the sustained per-launch
# time, so at burst clocks the timed slice is somewhat shorter than the
# nominal window.  The idle is under the board's clock policy as-is; no
# power or temperature floor is verified.
_BURST_WINDOW_MS = 200.0
_BURST_ATTEMPTS = 3
_COOLDOWN_S = 5.0

# Warmup ends when two consecutive windows agree on the SM clock to within one
# sm_90 boost bin (15 MHz) — a fixed duration only rides the power-cap ramp.
_SETTLE_WINDOW_MS = 2000.0
_SETTLE_TOL_MHZ = 15.0
_SETTLE_CAP_MS = 30000.0


def _telemetry_index():
    """Map the torch device to its nvidia-smi index via the device UUID."""
    uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        index, _, smi_uuid = line.partition(",")
        if uuid in smi_uuid:
            return int(index)
    return None


def _make_gemm(dtype_name, n, device):
    """Return (launch, flops_per_launch) for one dtype at size n."""
    if dtype_name in ("fp16", "bf16", "tf32"):
        dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "tf32": torch.float32}[dtype_name]
        a = torch.randn(n, n, device=device, dtype=dt)
        b = torch.randn(n, n, device=device, dtype=dt)
        out = torch.empty(n, n, device=device, dtype=dt)
        return lambda: torch.matmul(a, b, out=out), 2.0 * n**3

    if dtype_name == "fp8":
        a = torch.randn(n, n, device=device).to(torch.float8_e4m3fn)
        # cuBLASLt wants the second operand column-major.
        b = torch.randn(n, n, device=device).to(torch.float8_e4m3fn).t().contiguous().t()
        scale = torch.ones((), device=device)
        out = torch.empty(n, n, device=device, dtype=torch.bfloat16)
        return (
            lambda: torch._scaled_mm(
                a, b, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16, out=out
            ),
            2.0 * n**3,
        )

    raise ValueError(f"unknown dtype {dtype_name}")


def _settle(launch, sampler):
    """Run under load until the SM clock holds steady; return the settled MHz.

    The clocks come from *sampler*, whose thread reads nvidia-smi while the
    launch loop keeps the GPU busy — a one-shot read here would see the clock
    after the queue drained, not the clock under load.
    """
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    elapsed, window_start, prev_mhz, mhz = 0.0, 0.0, None, None
    mark = len(sampler.samples) if sampler else 0
    while elapsed < _SETTLE_CAP_MS:
        for _ in range(5):
            launch()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        if elapsed - window_start < _SETTLE_WINDOW_MS:
            continue
        window_start = elapsed
        if sampler is None:
            if elapsed >= 5000.0:  # no telemetry: fall back to a fixed warmup
                return None
            continue
        window = [s[0] for s in sampler.samples[mark:]]
        mark = len(sampler.samples)
        if not window:
            continue
        mhz = statistics.median(window)
        if prev_mhz is not None and abs(mhz - prev_mhz) <= _SETTLE_TOL_MHZ:
            return mhz
        prev_mhz = mhz
    return mhz


def _time_gemm(launch, flops):
    """Median TFLOP/s over _RUNS timed runs, plus the run-to-run spread."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(3):
        launch()
    end.record()
    end.synchronize()
    reps = max(1, int(_TARGET_RUN_MS / max(start.elapsed_time(end) / 3, 1e-3)))

    ms_per_launch = []
    for _ in range(_RUNS):
        start.record()
        for _ in range(reps):
            launch()
        end.record()
        end.synchronize()
        ms_per_launch.append(start.elapsed_time(end) / reps)

    tflops = sorted(flops / (ms * 1e-3) / 1e12 for ms in ms_per_launch)
    spread = (tflops[-1] - tflops[0]) / tflops[len(tflops) // 2] * 100.0
    return tflops[len(tflops) // 2], spread, statistics.median(ms_per_launch)


def _burst_gemm(launch, flops, ms_per_launch):
    """Median burst TFLOP/s: a short window timed from a cooled-down clock.

    Sustained rates sit at the power-cap clock; a kernel that starts with power
    headroom (after idle, or between memory-bound phases) runs at boost clocks
    until the cap engages.  Each attempt drains the queue, idles the board, and
    times only the first _BURST_WINDOW_MS of load.
    """
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    reps = max(1, int(_BURST_WINDOW_MS / max(ms_per_launch, 1e-3)))
    rates = []
    for _ in range(_BURST_ATTEMPTS):
        torch.cuda.synchronize()
        time.sleep(_COOLDOWN_S)
        start.record()
        for _ in range(reps):
            launch()
        end.record()
        end.synchronize()
        rates.append(flops * reps / (start.elapsed_time(end) * 1e-3) / 1e12)
    return statistics.median(rates)


def _enable_tf32():
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    else:
        torch.backends.cuda.matmul.allow_tf32 = True


def main():
    parser = argparse.ArgumentParser(description="Tensor-core GEMM throughput via cuBLAS")
    parser.add_argument("--profile", default="h200", help="GPU profile name")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    tensor_core = profile.get("tensor_core", {})
    if not tensor_core:
        print(f"Profile '{args.profile}' has no tensor_core section.", file=sys.stderr)
        sys.exit(1)
    fp32_ceiling_tflops = profile.get("cuda_core", {}).get("fp32", {}).get("theoretical", 0) / 1e12
    if "tf32" in tensor_core and not fp32_ceiling_tflops:
        print(
            f"Profile '{args.profile}' has tensor_core.tf32 but no cuda_core.fp32.theoretical;"
            " the tf32 guard needs that ceiling. Add it before calibrating.",
            file=sys.stderr,
        )
        sys.exit(1)

    _enable_tf32()
    device = torch.device("cuda")
    gpu_index = _telemetry_index()
    print(
        f"Profile: {args.profile} | GPU: {torch.cuda.get_device_name(device)} (smi index {gpu_index})"
    )
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"Each config: settle to steady clock, then {_RUNS} runs; calibration uses the median\n")
    print("dtype,n,median_tflops,spread_pct,burst_tflops,settled_mhz,pct_of_theo")

    results = {}
    for dtype_name in ("fp16", "bf16", "tf32", "fp8"):
        section = tensor_core.get(dtype_name)
        if section is None:
            continue
        theo_tflops = section["theoretical"] / 1e12

        # Probe once: fp8 in particular needs hardware and torch support the
        # profile alone cannot promise.
        try:
            probe, _ = _make_gemm(dtype_name, 256, device)
            probe()
            torch.cuda.synchronize()
            del probe
        except (RuntimeError, NotImplementedError, AttributeError) as exc:
            print(f"{dtype_name}: skipped, not supported here ({exc})", flush=True)
            continue

        best = (0.0, None, None)  # (median_tflops, n, spread)
        configs = []  # (n, flops, ms_per_launch, med, spread, settled)
        sampler = _ClockSampler(gpu_index) if gpu_index is not None else None
        if sampler:
            sampler.start()
        try:
            for n in _SIZES:
                launch, flops = _make_gemm(dtype_name, n, device)
                settled = _settle(launch, sampler)
                med, spread, ms_per_launch = _time_gemm(launch, flops)
                configs.append((n, flops, ms_per_launch, med, spread, settled))
                if med > best[0]:
                    best = (med, n, spread)
                del launch
                torch.cuda.empty_cache()
        finally:
            # Stop before the bursts: their idle gaps would drag the dtype's
            # clock summary below what the sustained measurement ran at.
            if sampler:
                sampler.stop()
                sampler.join(timeout=5)

        best_burst = 0.0
        for n, flops, ms_per_launch, med, spread, settled in configs:
            # Rebuilt per size so only one size's tensors are alive at a time.
            launch, _ = _make_gemm(dtype_name, n, device)
            burst = _burst_gemm(launch, flops, ms_per_launch)
            del launch
            torch.cuda.empty_cache()
            best_burst = max(best_burst, burst)
            settled_s = f"{settled:.0f}" if settled is not None else "-"
            print(
                f"{dtype_name},{n},{med:.1f},{spread:.2f},{burst:.1f},{settled_s},"
                f"{med / theo_tflops * 100:.1f}",
                flush=True,
            )
        results[dtype_name] = (
            best,
            best_burst,
            theo_tflops,
            sampler.summary() if sampler else None,
        )

    if not results:
        print(
            f"Profile '{args.profile}' tensor_core section has none of the dtypes"
            " this benchmark measures (fp16/bf16/tf32/fp8).",
            file=sys.stderr,
        )
        sys.exit(1)

    # A tf32 rate at or below the non-tensor fp32 ceiling means TF32 never
    # engaged and the row is IEEE fp32 GEMM against a TF32 theoretical.
    if "tf32" in results and results["tf32"][0][0] <= fp32_ceiling_tflops:
        print(
            f"\nERROR: tf32 measured {results['tf32'][0][0]:.1f} TFLOP/s, at or below the"
            f" {fp32_ceiling_tflops:.1f} TFLOP/s non-tensor fp32 ceiling — TF32 is not"
            " enabled; not emitting calibrations.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n{'=' * 60}")
    for dtype_name, ((med, n, spread), burst, _theo, telemetry) in results.items():
        clocks = ""
        if telemetry:
            sm_min, sm_med, sm_max, watts, _ = telemetry
            clocks = f"  clock {sm_min:.0f}/{sm_med:.0f}/{sm_max:.0f} MHz, {watts:.0f} W peak"
        print(
            f"{dtype_name}: {med:8.1f} TFLOP/s sustained at n={n} (spread {spread:.2f}%),"
            f" {burst:8.1f} burst{clocks}"
        )

    print(f"\nUpdate src/tileops/perf/profiles/{args.profile}.yaml:")
    for dtype_name, ((med, _, _), burst, theo, _) in results.items():
        print(f"  tensor_core.{dtype_name}.calibration: {med / theo:.4f}")
        print(f"  tensor_core.{dtype_name}.calibration_burst: {burst / theo:.4f}")
    print("Sustained rates are power-cap limited; record the clocks above with the numbers.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
