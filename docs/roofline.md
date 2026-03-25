# Roofline Evaluation

TileOPs evaluates kernel performance against hardware-theoretical Speed-of-Light (SOL) bounds, not relative to PyTorch baselines.

## Efficiency Ratio

```
efficiency = sol_bound / actual_time
```

Where `sol_bound` is the theoretical minimum execution time:

```
memory_time  = bytes_moved / hbm_bandwidth
compute_time = total_flops / peak_flops
sol_bound    = max(memory_time, compute_time)
```

An efficiency of 90% means the kernel runs within 10% of the hardware theoretical limit.

## Bound Type

Whichever of `memory_time` or `compute_time` is larger determines the bound type (memory-bound vs compute-bound). This is **not** a static property of an op — it varies by shape (e.g., a small GEMM may be memory-bound while a large GEMM is compute-bound). Bound type is computed per-workload by the roofline tool and displayed in auto-generated documentation. It is **not** declared in the manifest.

## GPU Profile

Hardware parameters use theoretical values with calibration factors from one-time microbenchmark measurements. YAML files store only `theoretical` and `calibration`; `effective = theoretical × calibration` is computed by `load_profile()`:

```yaml
# tileops/perf/profiles/h200.yaml
hbm:
  theoretical: 4800e9       # bytes/s, from spec sheet
  calibration: 0.94         # from microbench
tensor_core:
  fp16:
    theoretical: 989.5e12   # FLOPS, from spec sheet
    calibration: 0.75       # from microbench (cuBLAS peak)
```

Profiles are stored in `tileops/perf/profiles/`. Microbenchmarks for calibration live in `benchmarks/hardware/`.

## Benchmark / Roofline Decoupling

Benchmark (M4) produces raw time (JSON/CSV). Roofline (M5) is a separate tool that reads raw time + manifest formulas + GPU profile to compute efficiency. This separation enables:

- Re-analyzing historical data when GPU profiles are updated
- Multiple consumers of raw benchmark data (roofline, regression detection, dashboards)
- Benchmark module has no third-party dependencies beyond the project itself

## Roofline Formulas

Defined per-op in `ops_manifest.yaml` (see [manifest.md](manifest.md)). Simple ops use inline expressions; complex ops reference functions in `tileops/perf/formulas.py` that return `{"flops": int, "bytes": int}`.
