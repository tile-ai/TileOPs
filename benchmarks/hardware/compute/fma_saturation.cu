// CUDA-Core Saturation Benchmark
// Peak fp32 FMA throughput on the CUDA cores, plus a MUFU (special-function
// unit) reference rate.
//
// Every operand stays in registers and memory is touched once at the end, so
// the kernels measure the issue rate of the arithmetic pipelines — the compute
// axis of the roofline (Williams et al., CACM 52(4), 2009).  ILP (independent
// chains per thread) is swept because one chain is latency-bound and where the
// rate saturates moves with the architecture; the grid fills every SM to its
// 2048-thread limit.
//
// Compile: nvcc -O3 -arch=sm_90 -Wno-deprecated-gpu-targets -o fma_saturation fma_saturation.cu
// Usage: ./fma_saturation [iters] [theo_peak_tflops]  (defaults: 20000, 67.0)

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_LAST() CHECK_CUDA(cudaGetLastError())

// ============================================================
// Kernels
// ============================================================

// rsqrt.approx.ftz.f32 issues on the MUFU (same unit as ex2/lg2/rcp).  Not
// __expf — its multiply by log2(e) would show in the rate — and not rcp, whose
// chain is periodic (1/(1/x) = x) and gets folded to one iteration.
//
// Both qualifiers are load-bearing: without .ftz ptxas guards denormals
// (2 FSEL + 2 FMUL + 2 FSETP per MUFU on sm_90) and the benchmark turns
// mostly-ALU; without volatile the asm() may be hoisted out of the loop.
// Verify both ways: `cuobjdump -sass` must show bare MUFU.RSQ in the loop,
// and the time must scale linearly with `iters` — a collapsed loop can still
// disassemble into a plausible-looking body.
__device__ __forceinline__ float rsqrt_approx(float x) {
    float r;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// Each thread runs ILP independent FMA chains.  `iters` is a runtime argument
// so the outer loop survives constant folding; the inner loop is unrolled so
// the ILP chains interleave in the issue window.
template <int ILP>
__global__ void k_fma_fp32(float* __restrict__ sink, int iters) {
    float acc[ILP];
#pragma unroll
    for (int i = 0; i < ILP; ++i) acc[i] = 1.0f + 1e-3f * (float)(threadIdx.x + i);

    const float m = 0.99999994f;
    const float b = 1e-7f;

    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < ILP; ++i) acc[i] = fmaf(acc[i], m, b);
    }

    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < ILP; ++i) s += acc[i];

    // Unconditional store: nothing for the optimizer to prove dead.  One STG
    // per thread against tens of thousands of FMAs, paid identically by every
    // config.
    sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// MUFU reference, same shape as the FMA kernel.
template <int ILP>
__global__ void k_rsqrt_fp32(float* __restrict__ sink, int iters) {
    float acc[ILP];
#pragma unroll
    for (int i = 0; i < ILP; ++i) acc[i] = 1.5f + 1e-3f * (float)(threadIdx.x + i);

    // `asm volatile` keeps this outer loop rolled (the FMA one unrolls ~30x).
    // No bias: saturating the MUFU takes one warp instruction per 2 clocks
    // against 4 issue slots, so the loop bookkeeping fits in the slack.
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < ILP; ++i) acc[i] = rsqrt_approx(acc[i]);
    }

    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < ILP; ++i) s += acc[i];

    sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// ============================================================
// Benchmark helper
// ============================================================

struct BenchResult {
    float best_ms;
    float median_ms;
    float worst_ms;
    float stddev_pct;   // run-to-run spread, relative to the mean
    double best_ops;    // arithmetic results per second
    double median_ops;
    double worst_ops;
};

BenchResult run_bench(std::function<void()> launch, double ops_per_launch,
                      int warmup = 20, int reps = 50) {
    std::vector<float> latencies;

    for (int run = 0; run < 5; run++) {
        // Warmup doubles as the clock-settling window: a saturating load drops
        // the SM clock within the first few milliseconds.
        for (int i = 0; i < warmup; i++) launch();
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t t0, t1;
        CHECK_CUDA(cudaEventCreate(&t0));
        CHECK_CUDA(cudaEventCreate(&t1));
        CHECK_CUDA(cudaEventRecord(t0));
        for (int i = 0; i < reps; i++) launch();
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
        latencies.push_back(ms / reps);

        CHECK_CUDA(cudaEventDestroy(t0));
        CHECK_CUDA(cudaEventDestroy(t1));
    }

    double mean = 0.0;
    for (float v : latencies) mean += v;
    mean /= latencies.size();
    double var = 0.0;
    for (float v : latencies) var += (v - mean) * (v - mean);
    var /= latencies.size();
    float stddev_pct = (mean > 0) ? (float)(sqrt(var) / mean * 100.0) : 0.0f;

    std::sort(latencies.begin(), latencies.end());
    float best = latencies[0];
    float median = latencies[latencies.size() / 2];
    float worst = latencies.back();

    auto to_ops = [&](float ms) -> double {
        return (ms > 0) ? ops_per_launch / ((double)ms * 1e-3) : 0.0;
    };

    return {best, median, worst, stddev_pct,
            to_ops(best), to_ops(median), to_ops(worst)};
}

// ============================================================
// Sweeps
// ============================================================

// fp32 FMA lanes per SM per clock on compute capability 9.0, from the CUDA C
// Programming Guide's native arithmetic throughput table.
static const int FMA_LANES_PER_SM = 128;

struct Grid {
    int nblocks;
    int block;
    int iters;
    double threads;
    float* sink;
};

// A __global__ function cannot be launched through a function pointer, so ILP
// is a template argument.
template <int ILP>
double sweep_fma(const Grid& g, double theo_peak_tflops) {
    double fmas = g.threads * (double)ILP * (double)g.iters;
    auto launch = [&]() { k_fma_fp32<ILP><<<g.nblocks, g.block>>>(g.sink, g.iters); CHECK_LAST(); };
    BenchResult r = run_bench(launch, fmas);
    double best_tflops = r.best_ops * 2.0 / 1e12;
    double median_tflops = r.median_ops * 2.0 / 1e12;
    double worst_tflops = r.worst_ops * 2.0 / 1e12;
    printf("fma,%d,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,TFLOP/s,%.1f%%\n",
           ILP, g.block, r.best_ms, r.median_ms, r.worst_ms, r.stddev_pct,
           best_tflops, median_tflops, worst_tflops,
           median_tflops / theo_peak_tflops * 100.0);
    // Median, not best: one lucky run out of five is not a sustained rate.
    return median_tflops;
}

template <int ILP>
double sweep_mufu(const Grid& g) {
    double ops = g.threads * (double)ILP * (double)g.iters;
    auto launch = [&]() { k_rsqrt_fp32<ILP><<<g.nblocks, g.block>>>(g.sink, g.iters); CHECK_LAST(); };
    BenchResult r = run_bench(launch, ops);
    double best_gops = r.best_ops / 1e9;
    double median_gops = r.median_ops / 1e9;
    double worst_gops = r.worst_ops / 1e9;
    printf("mufu,%d,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,Gop/s,-\n",
           ILP, g.block, r.best_ms, r.median_ms, r.worst_ms, r.stddev_pct,
           best_gops, median_gops, worst_gops);
    return median_gops;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    int iters = 20000;
    double theo_peak_tflops = 67.0;
    if (argc >= 2) iters = atoi(argv[1]);
    if (argc >= 3) theo_peak_tflops = atof(argv[2]);

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    const int sm_count = prop.multiProcessorCount;
    const int block = 256;
    const int nblocks = sm_count * (2048 / block);   // fill every SM to its thread limit
    const double threads = (double)nblocks * block;

    // cudaDeviceProp::clockRate was removed in CUDA 13; the attribute query is
    // the portable spelling.
    int clock_khz = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));

    printf("GPU: %s | SMs: %d | max SM clock: %.2f GHz\n",
           prop.name, sm_count, clock_khz / 1e6);
    printf("Grid: %d blocks x %d threads = %.0f threads (2048 per SM)\n",
           nblocks, block, threads);
    printf("Each config: 5 runs x 50 reps, warmup 20; calibration uses the median\n");
    printf("iters=%d  theo_peak_tflops=%.1f\n\n", iters, theo_peak_tflops);

    // One slot per thread: every thread stores its accumulator.
    float* d_sink;
    const size_t sink_bytes = (size_t)nblocks * block * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_sink, sink_bytes));
    CHECK_CUDA(cudaMemset(d_sink, 0, sink_bytes));

    printf("op,ilp,block_size,best_ms,median_ms,worst_ms,stddev_pct,best_rate,median_rate,worst_rate,unit,pct_of_theo\n");

    // ============================================================
    // 1. fp32 FMA — primary calibration
    //    Rate reported in TFLOP/s, counting an FMA as 2 FLOP.
    // ============================================================
    printf("# fp32 FMA (CUDA core) — primary calibration\n");
    Grid g{nblocks, block, iters, threads, d_sink};
    double peak_fma_tflops = 0.0;   // best median across the ILP sweep
    peak_fma_tflops = std::max(peak_fma_tflops, sweep_fma<1>(g, theo_peak_tflops));
    peak_fma_tflops = std::max(peak_fma_tflops, sweep_fma<2>(g, theo_peak_tflops));
    peak_fma_tflops = std::max(peak_fma_tflops, sweep_fma<4>(g, theo_peak_tflops));
    peak_fma_tflops = std::max(peak_fma_tflops, sweep_fma<8>(g, theo_peak_tflops));
    peak_fma_tflops = std::max(peak_fma_tflops, sweep_fma<16>(g, theo_peak_tflops));

    // ============================================================
    // 2. MUFU (rsqrt.approx.ftz.f32) — reference, not used for calibration
    //    Rate reported in Gop/s, counting one result per instruction.
    // ============================================================
    printf("# MUFU rsqrt.approx.ftz.f32 (special function unit) — reference\n");
    double peak_mufu_gops = 0.0;
    peak_mufu_gops = std::max(peak_mufu_gops, sweep_mufu<1>(g));
    peak_mufu_gops = std::max(peak_mufu_gops, sweep_mufu<2>(g));
    peak_mufu_gops = std::max(peak_mufu_gops, sweep_mufu<4>(g));
    peak_mufu_gops = std::max(peak_mufu_gops, sweep_mufu<8>(g));

    // ============================================================
    // Derived quantities
    // ============================================================
    // measured rate / (lanes * SMs) = the clock the GPU actually held; the gap
    // to the boost ceiling is the power cap at work.
    double fma_per_s = peak_fma_tflops * 1e12 / 2.0;
    double implied_ghz = fma_per_s / ((double)FMA_LANES_PER_SM * sm_count) / 1e9;
    double mufu_ratio = (peak_mufu_gops > 0) ? fma_per_s / (peak_mufu_gops * 1e9) : 0.0;

    printf("\n# derived\n");
    printf("implied_sm_clock_ghz,%.3f\n", implied_ghz);
    printf("fma_to_mufu_ratio,%.2f\n", mufu_ratio);

    CHECK_CUDA(cudaFree(d_sink));
    return 0;
}
