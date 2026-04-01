// HBM Saturation Benchmark
// Measures peak HBM bandwidth with vectorized CUDA kernels.
// Uses STREAM Triad [1] as the primary calibration pattern for roofline
// analysis [2], with Copy/Read/Write as reference measurements.
//
// Triad (a[i] = b[i] + s*c[i]) has a 2:1 read:write ratio — closer to real
// compute kernels than pure Copy (1:1), which suffers worst-case HBM bus
// turnaround overhead.  BabelStream [3] establishes this methodology for GPUs.
//
// References:
//   [1] McCalpin, J.D., 1995. "Memory Bandwidth and Machine Balance in Current
//       High Performance Computers." IEEE TCCA Newsletter, pp.19-25.
//       https://www.cs.virginia.edu/stream/
//   [2] Williams, S., Waterman, A. & Patterson, D., 2009. "Roofline: An
//       Insightful Visual Performance Model for Multicore Architectures."
//       Communications of the ACM, 52(4), pp.65-76.
//   [3] Deakin, T. et al., 2018. "Evaluating attainable memory bandwidth of
//       parallel programming models via BabelStream." International Journal of
//       Computational Science and Engineering, 17(3), pp.247-262.
//       https://github.com/UoB-HPC/BabelStream
//
// Compile: nvcc -O3 -arch=sm_90 -Wno-deprecated-gpu-targets -o hbm_saturation hbm_saturation.cu
// Usage: ./hbm_saturation [size_mb] [theo_peak_gbs]  (defaults: 2048, 4800)

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <functional>

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

// Read kernels: use volatile to prevent compiler from optimizing away loads.
// Accumulate into a register and write once to prevent the reduction overhead
// from dominating measurement at small sizes.
__global__ void k_read_vec4(const float4* __restrict__ data, volatile float* __restrict__ out, long long n) {
    float sum = 0.0f;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < n; i += stride) {
        float4 v = data[i];
        sum += v.x + v.y + v.z + v.w;
    }
    // Single atomic per warp — negligible overhead vs memory-bound loop above
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (threadIdx.x % 32 == 0) atomicAdd((float*)out, sum);
}

// Write kernels
__global__ void k_write_vec4(float4* __restrict__ data, long long n, float val) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    float4 v = make_float4(val, val, val, val);
    for (long long i = idx; i < n; i += stride) data[i] = v;
}

// Copy kernels (read + write, 1:1 ratio)
__global__ void k_copy_vec4(const float4* __restrict__ src, float4* __restrict__ dst, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < n; i += stride) dst[i] = src[i];
}

// STREAM Triad: a[i] = b[i] + scalar * c[i]  (2 reads + 1 write, 2:1 ratio)
// Industry-standard pattern for roofline calibration (closer to real kernel
// memory access patterns than pure copy).
__global__ void k_triad_vec4(const float4* __restrict__ b, const float4* __restrict__ c,
                             float4* __restrict__ a, long long n, float scalar) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < n; i += stride) {
        float4 vb = b[i];
        float4 vc = c[i];
        a[i] = make_float4(vb.x + scalar * vc.x, vb.y + scalar * vc.y,
                            vb.z + scalar * vc.z, vb.w + scalar * vc.w);
    }
}

// ============================================================
// Benchmark helper
// ============================================================

struct BenchResult {
    float best_ms;
    float median_ms;
    float best_gbs;
    float median_gbs;
};

BenchResult run_bench(std::function<void()> launch, long long total_bytes,
                      int warmup = 100, int reps = 200) {
    std::vector<float> latencies;

    for (int run = 0; run < 5; run++) {
        // Warmup: enough iterations for GPU clocks to stabilize
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

    std::sort(latencies.begin(), latencies.end());
    float best = latencies[0];
    float median = latencies[2]; // middle of 5

    auto to_gbs = [&](float ms) -> float {
        return (ms > 0) ? (float)total_bytes / (ms * 1e6f) : 0.0f;
    };

    return {best, median, to_gbs(best), to_gbs(median)};
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    long long size_mb = 2048;
    float theo_peak_gbs = 4800.0f;
    if (argc >= 2) size_mb = atoll(argv[1]);
    if (argc >= 3) theo_peak_gbs = atof(argv[2]);

    long long n_floats = size_mb * 1024LL * 1024LL / sizeof(float);
    long long nbytes = n_floats * sizeof(float);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;

    printf("GPU: %s | SMs: %d | L2: %d MB\n", prop.name, sm_count,
           prop.l2CacheSize / (1024*1024));
    printf("Working set: %lld MB (%lld float32 elements)\n", size_mb, n_floats);
    printf("Each config: 5 runs x 200 reps, warmup 100, reporting best and median\n");
    printf("theo_peak_gbs=%.1f\n\n", theo_peak_gbs);

    // Allocate with cudaMalloc (guaranteed 256-byte aligned)
    // d_src, d_dst used by copy/read/write; d_src2 is the second input for triad
    float *d_src, *d_src2, *d_dst;
    volatile float *d_out;
    CHECK_CUDA(cudaMalloc(&d_src, nbytes));
    CHECK_CUDA(cudaMalloc(&d_src2, nbytes));
    CHECK_CUDA(cudaMalloc(&d_dst, nbytes));
    CHECK_CUDA(cudaMalloc((void**)&d_out, sizeof(float)));

    // Initialize with a valid float pattern (not byte-fill).
    // Non-zero values prevent hardware memory compression from inflating results.
    {
        std::vector<float> host_data(n_floats, 1.0f);
        CHECK_CUDA(cudaMemcpy(d_src, host_data.data(), nbytes, cudaMemcpyHostToDevice));
        std::fill(host_data.begin(), host_data.end(), 2.0f);
        CHECK_CUDA(cudaMemcpy(d_src2, host_data.data(), nbytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_dst, 0, nbytes));
        CHECK_CUDA(cudaMemset((void*)d_out, 0, sizeof(float)));
    }

    // CSV header
    printf("op,vec_width,block_size,best_ms,median_ms,best_gbs,median_gbs,pct_of_theo\n");

    auto print_row = [theo_peak_gbs](const char* op, const char* vec, int bs, BenchResult r) {
        printf("%s,%s,%d,%.4f,%.4f,%.2f,%.2f,%.1f%%\n",
               op, vec, bs,
               r.best_ms, r.median_ms, r.best_gbs, r.median_gbs,
               r.best_gbs / theo_peak_gbs * 100.0f);
    };

    // Block sizes to sweep
    int bss[] = {128, 256, 512};

    // ============================================================
    // 1. Copy (read + write, 1:1 ratio) — reference
    //    Bytes reported = 2 * nbytes (read src + write dst)
    //    Note: 1:1 read:write creates worst-case bus turnaround pressure.
    //    Real kernels are typically read-heavy and achieve higher bandwidth.
    // ============================================================
    printf("# Copy (read+write, 1:1 ratio) — reference\n");
    for (int bi = 0; bi < 3; bi++) {
        int bs = bss[bi];
        int nblocks = sm_count * (2048 / bs);
        long long n = n_floats / 4;
        auto launch = [&]() { k_copy_vec4<<<nblocks, bs>>>((float4*)d_src, (float4*)d_dst, n); CHECK_LAST(); };
        auto r = run_bench(launch, 2 * nbytes);
        print_row("copy", "vec4", bs, r);
    }

    // ============================================================
    // 2. STREAM Triad: a[i] = b[i] + s*c[i] (2 reads + 1 write)
    //    Bytes reported = 3 * nbytes (read b + read c + write a)
    //    2:1 read:write ratio — closer to real compute kernels than copy.
    //    Used as the primary calibration measurement.
    // ============================================================
    printf("# Triad (2 reads + 1 write) — primary calibration\n");
    for (int bi = 0; bi < 3; bi++) {
        int bs = bss[bi];
        int nblocks = sm_count * (2048 / bs);
        long long n = n_floats / 4;
        auto launch = [&]() {
            k_triad_vec4<<<nblocks, bs>>>((float4*)d_src, (float4*)d_src2,
                                          (float4*)d_dst, n, 3.14f);
            CHECK_LAST();
        };
        auto r = run_bench(launch, 3 * nbytes);
        print_row("triad", "vec4", bs, r);
    }

    // ============================================================
    // 3. Read-only (for reference, not used for calibration)
    // ============================================================
    printf("# Read-only — reference\n");
    for (int bi = 0; bi < 3; bi++) {
        int bs = bss[bi];
        int nblocks = sm_count * (2048 / bs);
        long long n = n_floats / 4;
        auto launch = [&]() { k_read_vec4<<<nblocks, bs>>>((float4*)d_src, d_out, n); CHECK_LAST(); };
        auto r = run_bench(launch, nbytes);
        print_row("read", "vec4", bs, r);
    }

    // ============================================================
    // 4. Write-only (for reference, not used for calibration)
    // ============================================================
    printf("# Write-only — reference\n");
    for (int bi = 0; bi < 3; bi++) {
        int bs = bss[bi];
        int nblocks = sm_count * (2048 / bs);
        long long n = n_floats / 4;
        auto launch = [&]() { k_write_vec4<<<nblocks, bs>>>((float4*)d_dst, n, 1.0f); CHECK_LAST(); };
        auto r = run_bench(launch, nbytes);
        print_row("write", "vec4", bs, r);
    }

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_src2));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFree((void*)d_out));

    return 0;
}
