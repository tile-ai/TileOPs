# Hardware Microbenchmarks

GPU hardware characterization benchmarks that produce calibration factors for `src/tileops/perf/profiles/`.

## Prerequisites

- NVIDIA GPU with CUDA toolkit (`nvcc` in PATH)
- TileOPs installed (`pip install -e .` from project root)
- Root/sudo access for clock locking (recommended)

## HBM Bandwidth

Measures peak HBM bandwidth using vectorized CUDA kernels (float4 load/store) with cudaEvent timing. The calibration factor is derived from the **STREAM Triad** kernel (`a[i] = b[i] + s*c[i]`, 2 reads + 1 write), the industry-standard pattern for roofline bandwidth calibration.

Triad's 2:1 read:write ratio is closer to real compute kernels than pure copy (1:1), which suffers worst-case HBM bus turnaround overhead. Copy, read-only, and write-only results are included as reference measurements.

### Lock GPU clocks (recommended)

GPU boost clocks fluctuate during benchmarks. Lock memory and SM clocks to their maximum for stable, reproducible results:

```bash
# Lock the SM clock to a range whose top is the card maximum (requires root/sudo).
# Pass a range, not a single value: a lone value that is not a valid clock bin is
# accepted without effect, and the card stays wherever it was.
sudo nvidia-smi -i <gpu> -lgc <idle_clock>,$(nvidia-smi -i <gpu> --query-gpu=clocks.max.sm --format=csv,noheader,nounits)
```

Memory clocks need no locking on HBM3e parts — `clocks.mem` already sits at
`clocks.max.memory` at all times. Recent drivers (595.x) reject `-lmc` outright
and point at `--lock-memory-clocks-deferred`, which only takes effect after a GPU
reset and is therefore useless mid-session.

Verify the lock took: on an idle card `clocks.sm` should jump to the bottom of
the range you set. Locking does not guarantee the top of the range is reached —
an H200 holds 1830 MHz under a saturating FMA load and does not approach its
1980 MHz boost ceiling even with power to spare. Record what it holds, not what
the datasheet implies.

Unlock when the measurement is done. A locked card idles at the bottom of its
range: measured on an idle H200, 1830 MHz and 124 W locked against 345 MHz and
77 W unlocked. On a shared machine the setting is global to that GPU and
outlives the process that set it.

After benchmarking, reset to default:

```bash
sudo nvidia-smi -i <gpu> -rgc
```

### Run

Run from project root:

```bash
python benchmarks/hardware/memory/hbm_bandwidth.py --profile h200 --arch sm_90
```

Options:

| Flag        | Default | Description                                                                 |
| ----------- | ------- | --------------------------------------------------------------------------- |
| `--profile` | `h200`  | GPU profile name (reads theoretical peak from `src/tileops/perf/profiles/`) |
| `--arch`    | `sm_90` | CUDA compute capability for nvcc                                            |
| `--size-mb` | `2048`  | Working set size in MB                                                      |

### Output

```
Measured peak (triad vec4): 4070.44 GB/s
Theoretical:               4800.0 GB/s
Calibration:               0.8480

Update src/tileops/perf/profiles/h200.yaml:
  hbm.calibration: 0.8480
```

### Methodology

- **Calibration kernel:** STREAM Triad `a[i] = b[i] + s*c[i]` (2 reads + 1 write, `float4` vectorized)
- **Reference kernels:** Copy (1:1 read:write), Read-only, Write-only
- **Timing:** `cudaEvent` (GPU-side, no host overhead)
- **Warmup:** 100 iterations per config (ensures boost clocks stabilize)
- **Measurement:** 200 iterations × 5 runs, report best and median
- **Working set:** 2 GB default (>> L2 cache, ensures HBM is measured)
- **Calibration source:** best Triad bandwidth across block size sweep (128/256/512)

## Adding a new GPU profile

1. Create `src/tileops/perf/profiles/<gpu>.yaml` with theoretical specs from the datasheet
1. Lock GPU clocks (see above)
1. Run `python benchmarks/hardware/memory/hbm_bandwidth.py --profile <gpu> --arch <sm_XX>`
1. Update `<gpu>.yaml` with the measured calibration factor
1. Reset GPU clocks

## fp32 FMA Throughput (CUDA cores)

Measures the sustained fp32 FMA issue rate of the CUDA cores — the compute axis
of the roofline, and the counterpart to the HBM benchmark above. Every operand
stays in registers; memory is touched once per thread at the end. A MUFU rate
(`rsqrt.approx.ftz.f32`) is reported alongside as a reference measurement.

### Run

```bash
python benchmarks/hardware/compute/fma_throughput.py --profile h200 --arch sm_90
```

| Flag                      | Default | Description                                           |
| ------------------------- | ------- | ----------------------------------------------------- |
| `--profile`               | `h200`  | GPU profile name (reads `cuda_core.fp32.theoretical`) |
| `--arch`                  | `sm_90` | CUDA compute capability for nvcc                      |
| `--iters`                 | `20000` | FMA iterations per dependency chain                   |
| `--gpu-index`             | `0`     | Which GPU to sample clock and power telemetry from    |
| `--allow-unlocked-clocks` | off     | Emit a calibration factor even if the SM clock moved  |

### Methodology

- **Calibration kernel:** ILP independent FMA chains per thread, swept over ILP
  1/2/4/8/16. One chain per thread is latency-bound; the sweep finds where the
  rate saturates.
- **Selection:** the *median* of five runs within a config, then the best config
  across the sweep. Best-of-best reports a rate the hardware does not sustain.
- **Telemetry:** SM clock, power and temperature are sampled throughout. If the
  clock moves by more than one boost bin, no calibration factor is printed
  unless `--allow-unlocked-clocks` is passed.
- **Decomposition:** the factor is reported as `lane utilisation x clock headroom`. Only the first is a property of the silicon; the second is the
  site's clock policy. On H200, lane utilisation measured 92.6% / 92.4% / 92.6%
  across three different clock configurations.

### Two checks before trusting a number from here

Both were needed to catch real defects in this benchmark:

1. **Read the SASS.** `cuobjdump -sass` on the kernel must show a loop body
   containing nothing but the instruction under test. Without `.ftz`,
   `rcp.approx.f32` expands into a denormal guard of 2 FSEL + 2 FMUL + 2 FSETP
   per MUFU, and a MUFU benchmark becomes a mostly-ALU one.
1. **Multiply `--iters` by ten and confirm the time scales with it.** SASS is
   necessary but not sufficient: a loop the compiler has collapsed can still
   disassemble into a plausible-looking loop body. A chain of reciprocals is
   periodic (`1/(1/x) = x`) and gets folded away; `rsqrt` has no such identity.
