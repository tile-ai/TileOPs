# Hardware Microbenchmarks

GPU hardware characterization benchmarks that produce calibration factors for `tileops/perf/profiles/`.

## Prerequisites

- NVIDIA GPU with CUDA toolkit (`nvcc` in PATH)
- TileOPs installed (`pip install -e .` from project root)

## HBM Bandwidth

Measures peak HBM read/write/copy bandwidth using vectorized CUDA kernels (float4 load/store) with cudaEvent timing.

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

Output includes per-kernel bandwidth measurements and a calibration factor:

```
Measured peak:  4512.00 GB/s
Theoretical:    4800.0 GB/s
Calibration:    0.9400

Update tileops/perf/profiles/h200.yaml:
  hbm.calibration: 0.9400
```

## Adding a new GPU profile

1. Create `tileops/perf/profiles/<gpu>.yaml` with theoretical specs from the datasheet
1. Run `python benchmarks/hardware/memory/hbm_bandwidth.py --profile <gpu> --arch <sm_XX>`
1. Update `<gpu>.yaml` with the measured calibration factor
