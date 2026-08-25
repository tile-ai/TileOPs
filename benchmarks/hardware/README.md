# Hardware Microbenchmarks

Measure the calibration factors recorded in `src/tileops/perf/profiles/` (`effective = theoretical × calibration`).

## Run

```bash
# Lock the SM clock (pass a range — a lone off-bin value is silently ignored):
sudo nvidia-smi -i <gpu> -lgc <idle_clock>,$(nvidia-smi -i <gpu> --query-gpu=clocks.max.sm --format=csv,noheader,nounits)

python benchmarks/hardware/memory/hbm_bandwidth.py --profile <gpu> --arch <sm_XX>
python benchmarks/hardware/compute/fma_throughput.py --profile <gpu> --arch <sm_XX>
CUDA_VISIBLE_DEVICES=<n> python benchmarks/hardware/compute/gemm_throughput.py --profile <gpu>

# Unlock — the setting is global to the GPU and outlives the process:
sudo nvidia-smi -i <gpu> -rgc
```

Each script ends with the `*.calibration` lines to copy into `<gpu>.yaml`; for a new GPU, create the yaml with datasheet numbers first. `--help` lists the remaining flags. Memory clocks need no locking: HBM parts hold max, and 595.x drivers reject `-lmc`.

## Method

| Benchmark         | Calibrates                         | Kernel                                                   | Selection                                    |
| ----------------- | ---------------------------------- | -------------------------------------------------------- | -------------------------------------------- |
| `hbm_bandwidth`   | `hbm`                              | STREAM Triad `a = b + s*c`, `float4`                     | best block size (128/256/512), 200 iters × 5 |
| `fma_throughput`  | `cuda_core.fp32`                   | register-only FMA chains on the CUDA cores               | best ILP (1..16), median of 5 runs each      |
| `gemm_throughput` | `tensor_core.{fp16,bf16,tf32,fp8}` | cuBLAS GEMM (`torch.matmul`; `torch._scaled_mm` for fp8) | best n (4096/8192/16384), median of 5 × ~4 s |

- `fma_throughput` withholds the calibration when the SM clock moved more than one boost bin (`--allow-unlocked-clocks` overrides). After editing the kernel, confirm `cuobjdump -sass` shows only the instruction under test and that runtime scales linearly with `--iters`.
- `gemm_throughput` is power-cap limited — the cap, not the lock, sets the clock. It warms up to a steady clock, then records `calibration` (sustained; what `effective` is computed from) and `calibration_burst` (first ~200 ms of load after 5 s idle, before the cap engages). A tf32 rate at or below the non-tensor fp32 ceiling aborts the run: TF32 never engaged.
