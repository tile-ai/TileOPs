# Hardware Microbenchmarks

GPU hardware characterization benchmarks that produce calibration factors for `src/tileops/perf/profiles/`.

## Prerequisites

- NVIDIA GPU with CUDA toolkit (`nvcc` in PATH)
- TileOPs installed (`pip install -e .` from project root)
- Root/sudo access for clock locking (recommended)

## Lock GPU clocks (recommended)

Boost clocks fluctuate during benchmarks; lock the SM clock for reproducible results.

```bash
# Pass a range, not a single value: a lone value that is not a valid clock bin
# is accepted without effect.
sudo nvidia-smi -i <gpu> -lgc <idle_clock>,$(nvidia-smi -i <gpu> --query-gpu=clocks.max.sm --format=csv,noheader,nounits)
```

- Memory clocks need no locking on HBM3e parts; recent drivers (595.x) reject `-lmc` outright.
- Verify the lock took: an idle card jumps to the bottom of the range. Locking does not guarantee the top is reached — record the clock the board holds under load, not the datasheet boost.
- Unlock after measuring: a locked card idles at the bottom of its range, and the setting is global to the GPU and outlives the process.

```bash
sudo nvidia-smi -i <gpu> -rgc
```

## HBM Bandwidth

Peak HBM bandwidth from `float4`-vectorized kernels with `cudaEvent` timing. The calibration factor comes from **STREAM Triad** (`a[i] = b[i] + s*c[i]`, 2 reads + 1 write), whose read:write ratio is closer to real compute kernels than pure copy; copy / read-only / write-only are reported as references.

```bash
python benchmarks/hardware/memory/hbm_bandwidth.py --profile h200 --arch sm_90
```

| Flag        | Default | Description                                                                 |
| ----------- | ------- | --------------------------------------------------------------------------- |
| `--profile` | `h200`  | GPU profile name (reads theoretical peak from `src/tileops/perf/profiles/`) |
| `--arch`    | `sm_90` | CUDA compute capability for nvcc                                            |
| `--size-mb` | `2048`  | Working set size in MB (>> L2, so HBM is what is measured)                  |

Method: warmup 100 iterations, then 200 iterations × 5 runs per block size (128/256/512); the calibration is the best Triad bandwidth. The output ends with the `hbm.calibration` line to copy into the profile.

## fp32 FMA Throughput (CUDA cores)

Sustained fp32 FMA issue rate of the CUDA cores — the compute axis of the roofline. Every operand stays in registers; memory is touched once per thread at the end. A MUFU rate (`rsqrt.approx.ftz.f32`) is reported as a reference.

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

Method: ILP independent FMA chains per thread, swept 1/2/4/8/16; each config judged by the median of 5 runs, the calibration by the best config. SM clock, power and temperature are sampled throughout, and the calibration line is withheld when the clock moves more than one boost bin unless `--allow-unlocked-clocks` is passed. The factor is decomposed as `lane utilisation × clock headroom`; only the first is a property of the silicon.

When modifying the kernel, verify it two ways: `cuobjdump -sass` must show a loop body of nothing but the instruction under test, and the runtime must scale linearly with `--iters` (a collapsed loop can still disassemble into a plausible-looking body).

## Tensor-core GEMM Throughput

Sustained cuBLAS GEMM rate per tensor-core dtype (fp16/bf16/tf32/fp8), calibrating `tensor_core.*` in the profile. Runs `torch.matmul` (`torch._scaled_mm` for fp8) over square sizes 4096/8192/16384.

```bash
CUDA_VISIBLE_DEVICES=<gpu> python benchmarks/hardware/compute/gemm_throughput.py --profile h200
```

| Flag        | Default | Description                                          |
| ----------- | ------- | ---------------------------------------------------- |
| `--profile` | `h200`  | GPU profile name (reads `tensor_core.*.theoretical`) |

Method: per config, warm up until the SM clock holds steady across two consecutive 2 s windows, then take the median of 5 timed runs of ~4 s each — long runs average over the power-cap clock oscillation, which puts run-to-run spread below 1%. The telemetry GPU is resolved from the device UUID, so `CUDA_VISIBLE_DEVICES` is honored.

Clock locking does not apply here: a saturating GEMM drives the board into its power cap and the cap, not the lock, sets the clock. Calibrations are therefore power-cap-limited sustained rates; the output prints the per-dtype clock range to record next to the numbers. A tf32 result at or below the non-tensor fp32 ceiling aborts the run — it means TF32 never engaged.

## Adding a new GPU profile

1. Create `src/tileops/perf/profiles/<gpu>.yaml` with theoretical specs from the datasheet
1. Lock GPU clocks (see above)
1. Run both benchmarks with `--profile <gpu> --arch <sm_XX>`
1. Update `<gpu>.yaml` with the measured calibration factors
1. Reset GPU clocks
