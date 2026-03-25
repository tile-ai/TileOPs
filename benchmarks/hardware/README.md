# Hardware Microbenchmarks

GPU hardware characterization benchmarks that produce calibration factors for `tileops/perf/profiles/`.

## Prerequisites

- NVIDIA GPU with CUDA toolkit (`nvcc` in PATH)
- TileOPs installed (`pip install -e .` from project root)
- Root/sudo access for clock locking (recommended)

## HBM Bandwidth

Measures peak HBM bandwidth using vectorized CUDA copy kernels (float4 load/store) with cudaEvent timing. The calibration factor is derived from copy bandwidth only, which exercises both read and write paths — matching roofline `bytes_moved` semantics.

### Lock GPU clocks (recommended)

GPU boost clocks fluctuate during benchmarks. Lock memory and SM clocks to their maximum for stable, reproducible results:

```bash
# Lock clocks (requires root/sudo)
sudo nvidia-smi -lgc $(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits)
sudo nvidia-smi -lmc $(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits)
```

After benchmarking, reset to default:

```bash
sudo nvidia-smi -rgc
sudo nvidia-smi -rmc
```

### Run

```bash
# Run from project root
python benchmarks/hardware/memory/hbm_bandwidth.py --profile h200 --arch sm_90
```

Options:

| Flag        | Default | Description                                                             |
| ----------- | ------- | ----------------------------------------------------------------------- |
| `--profile` | `h200`  | GPU profile name (reads theoretical peak from `tileops/perf/profiles/`) |
| `--arch`    | `sm_90` | CUDA compute capability for nvcc                                        |
| `--size-mb` | `2048`  | Working set size in MB                                                  |

### Output

```
Measured peak (copy vec4): 4512.00 GB/s
Theoretical:              4800.0 GB/s
Calibration:              0.9400

Update tileops/perf/profiles/h200.yaml:
  hbm.calibration: 0.9400
```

### Methodology

- **Kernel:** `float4` vectorized copy (16 bytes per thread per iteration)
- **Timing:** `cudaEvent` (GPU-side, no host overhead)
- **Warmup:** 100 iterations (ensures boost clocks stabilize)
- **Measurement:** 200 iterations × 5 runs, report best and median
- **Working set:** 2 GB default (>> L2 cache, ensures HBM is measured)
- **Calibration source:** best copy bandwidth across block size sweep (128/256/512)

## Adding a new GPU profile

1. Create `tileops/perf/profiles/<gpu>.yaml` with theoretical specs from the datasheet
1. Lock GPU clocks (see above)
1. Run `python benchmarks/hardware/memory/hbm_bandwidth.py --profile <gpu> --arch <sm_XX>`
1. Update `<gpu>.yaml` with the measured calibration factor
1. Reset GPU clocks
