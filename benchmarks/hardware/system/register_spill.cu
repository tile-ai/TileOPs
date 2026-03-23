// Register Pressure / Spill Cookbook
// Measures how register pressure and spilling affect performance.
//
// Compile: nvcc -O2 -arch=sm_90 -o register_spill register_spill.cu
// Usage:   ./register_spill
//
// Tests:
// 1. __launch_bounds__ to force max register counts → observe spill and perf
// 2. Increasing live variable count → natural register growth
// 3. Memory-bound kernel: does register spill matter?
// 4. Compute-bound kernel: when does spill become fatal?
//
// Key question: at what point does register spill actually hurt throughput?

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Compute kernels with increasing register pressure
// Each uses N independent FMA accumulator chains.
// More chains = more registers = more ILP.
// ============================================================

// The key insight: we use arrays of accumulators.
// The compiler tries to keep them in registers, but with too many
// it spills to local memory (stack in HBM, possibly cached in L1).

#define MAKE_COMPUTE_KERNEL(NAME, N_ACCUM) \
__global__ void NAME(float* out, int n_iters) { \
    float a = 1.0f + 0.0001f * threadIdx.x; \
    float acc[N_ACCUM]; \
    _Pragma("unroll") \
    for (int k = 0; k < N_ACCUM; k++) acc[k] = 1.0f + 0.001f * k; \
    _Pragma("unroll 1") \
    for (int i = 0; i < n_iters; i++) { \
        _Pragma("unroll") \
        for (int k = 0; k < N_ACCUM; k++) \
            acc[k] = fmaf(a, acc[k], a); \
    } \
    float total = 0; \
    _Pragma("unroll") \
    for (int k = 0; k < N_ACCUM; k++) total += acc[k]; \
    if (total == -999.0f) *out = total; \
}

MAKE_COMPUTE_KERNEL(k_compute_4,   4)
MAKE_COMPUTE_KERNEL(k_compute_8,   8)
MAKE_COMPUTE_KERNEL(k_compute_16,  16)
MAKE_COMPUTE_KERNEL(k_compute_32,  32)
MAKE_COMPUTE_KERNEL(k_compute_64,  64)
MAKE_COMPUTE_KERNEL(k_compute_128, 128)
MAKE_COMPUTE_KERNEL(k_compute_256, 256)

// ============================================================
// Same kernels but with __launch_bounds__ to force register cap
// This forces spilling when the kernel needs more regs than allowed
// ============================================================

#define MAKE_COMPUTE_KERNEL_BOUNDED(NAME, N_ACCUM, MAX_THREADS, MIN_BLOCKS) \
__global__ void __launch_bounds__(MAX_THREADS, MIN_BLOCKS) \
NAME(float* out, int n_iters) { \
    float a = 1.0f + 0.0001f * threadIdx.x; \
    float acc[N_ACCUM]; \
    _Pragma("unroll") \
    for (int k = 0; k < N_ACCUM; k++) acc[k] = 1.0f + 0.001f * k; \
    _Pragma("unroll 1") \
    for (int i = 0; i < n_iters; i++) { \
        _Pragma("unroll") \
        for (int k = 0; k < N_ACCUM; k++) \
            acc[k] = fmaf(a, acc[k], a); \
    } \
    float total = 0; \
    _Pragma("unroll") \
    for (int k = 0; k < N_ACCUM; k++) total += acc[k]; \
    if (total == -999.0f) *out = total; \
}

// 32 accumulators — naturally wants ~40 regs
// Force to 32 regs → some spill
MAKE_COMPUTE_KERNEL_BOUNDED(k_compute_32_bounded_high, 32, 256, 8)  // ~32 regs
// Force to high occupancy → more spill
MAKE_COMPUTE_KERNEL_BOUNDED(k_compute_32_bounded_max,  32, 256, 16) // ~16 regs forced

// 64 accumulators — naturally wants ~70+ regs
MAKE_COMPUTE_KERNEL_BOUNDED(k_compute_64_bounded_high, 64, 256, 8)
MAKE_COMPUTE_KERNEL_BOUNDED(k_compute_64_bounded_max,  64, 256, 16)

// ============================================================
// Memory-bound kernel with varying register pressure
// Tests whether spill hurts memory-bound workloads
// ============================================================

#define MAKE_MEMBW_KERNEL(NAME, N_ACCUM) \
__global__ void NAME(const float4* __restrict__ src, float* __restrict__ dst, long long n) { \
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    long long stride = (long long)gridDim.x * blockDim.x; \
    float acc[N_ACCUM]; \
    _Pragma("unroll") \
    for (int k = 0; k < N_ACCUM; k++) acc[k] = 0.0f; \
    for (long long i = idx; i < n; i += stride) { \
        float4 v = src[i]; \
        acc[0] += v.x + v.y + v.z + v.w; \
        _Pragma("unroll") \
        for (int k = 1; k < N_ACCUM; k++) \
            acc[k] = fmaf(acc[k-1], 1.0001f, v.x); \
    } \
    float total = 0; \
    _Pragma("unroll") \
    for (int k = 0; k < N_ACCUM; k++) total += acc[k]; \
    if (total != 0.0f) dst[idx] = total; \
}

MAKE_MEMBW_KERNEL(k_membw_4,   4)
MAKE_MEMBW_KERNEL(k_membw_16,  16)
MAKE_MEMBW_KERNEL(k_membw_32,  32)
MAKE_MEMBW_KERNEL(k_membw_64,  64)
MAKE_MEMBW_KERNEL(k_membw_128, 128)

// ============================================================
// Helpers
// ============================================================

void print_kernel_info(const char* name, void* kernel, int block_size) {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);

    int max_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, block_size, 0);
    int warps_per_sm = max_blocks * ((block_size + 31) / 32);
    float occ = warps_per_sm / 64.0f * 100.0f;

    printf("  %s: regs=%d, spill_load=%zuB, spill_store=%zuB, max_blocks/SM=%d, occ=%.0f%%\n",
           name, attr.numRegs, attr.localSizeBytes, attr.localSizeBytes,
           max_blocks, occ);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;

    int sm_clock_khz;
    cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, 0);
    float sm_clock_mhz = sm_clock_khz / 1000.0f;

    printf("GPU: %s | SMs: %d | SM Clock: %.0f MHz\n", prop.name, sm_count, sm_clock_mhz);
    printf("Registers/SM: %d | Max warps/SM: %d\n\n", prop.regsPerMultiprocessor,
           prop.maxThreadsPerMultiProcessor / 32);

    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));

    // Memory kernel data: 256 MB
    long long n_bytes = 256LL * 1024 * 1024;
    long long n_float4 = n_bytes / sizeof(float4);
    float4* d_src;
    float* d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, n_bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, n_bytes));
    CHECK_CUDA(cudaMemset(d_src, 1, n_bytes));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int bs = 256;
    int n_iters = 100000;

    // ============================================================
    // 1. Kernel info: registers and spill
    // ============================================================
    printf("=== Kernel info (block=%d) ===\n", bs);

    printf("Compute kernels (natural register usage):\n");
    print_kernel_info("4_accum",   (void*)k_compute_4,   bs);
    print_kernel_info("8_accum",   (void*)k_compute_8,   bs);
    print_kernel_info("16_accum",  (void*)k_compute_16,  bs);
    print_kernel_info("32_accum",  (void*)k_compute_32,  bs);
    print_kernel_info("64_accum",  (void*)k_compute_64,  bs);
    print_kernel_info("128_accum", (void*)k_compute_128, bs);
    print_kernel_info("256_accum", (void*)k_compute_256, bs);

    printf("\nBounded compute kernels (forced register cap):\n");
    print_kernel_info("32_bounded_high", (void*)k_compute_32_bounded_high, bs);
    print_kernel_info("32_bounded_max",  (void*)k_compute_32_bounded_max,  bs);
    print_kernel_info("64_bounded_high", (void*)k_compute_64_bounded_high, bs);
    print_kernel_info("64_bounded_max",  (void*)k_compute_64_bounded_max,  bs);

    printf("\nMemory-bound kernels:\n");
    print_kernel_info("membw_4",   (void*)k_membw_4,   bs);
    print_kernel_info("membw_16",  (void*)k_membw_16,  bs);
    print_kernel_info("membw_32",  (void*)k_membw_32,  bs);
    print_kernel_info("membw_64",  (void*)k_membw_64,  bs);
    print_kernel_info("membw_128", (void*)k_membw_128, bs);
    printf("\n");

    // ============================================================
    // 2. Compute-bound: GFLOPS vs register pressure
    // ============================================================
    printf("=== Compute-bound: GFLOPS vs accumulator count ===\n");
    printf("accumulators,regs,occupancy_pct,gflops,pct_fp32_peak\n");

    double fp32_peak_gflops = (double)sm_count * sm_clock_mhz * 128 * 2 / 1000.0;

    typedef void (*compute_fn)(float*, int);
    compute_fn comp_kernels[] = {k_compute_4, k_compute_8, k_compute_16,
                                  k_compute_32, k_compute_64, k_compute_128, k_compute_256};
    int accum_counts[] = {4, 8, 16, 32, 64, 128, 256};
    int n_comp = 7;

    for (int ki = 0; ki < n_comp; ki++) {
        int grid = sm_count * 4;

        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (void*)comp_kernels[ki]);

        int max_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, (void*)comp_kernels[ki], bs, 0);
        int warps_per_sm = max_blocks * ((bs + 31) / 32);
        float occ = warps_per_sm / 64.0f * 100.0f;

        // Warmup
        for (int w = 0; w < 3; w++)
            comp_kernels[ki]<<<grid, bs>>>(d_out, n_iters);
        CHECK_CUDA(cudaDeviceSynchronize());

        double times[5];
        for (int r = 0; r < 5; r++) {
            cudaEventRecord(start);
            comp_kernels[ki]<<<grid, bs>>>(d_out, n_iters);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            float ms;
            cudaEventElapsedTime(&ms, start, end);
            times[r] = ms;
        }
        for (int i = 0; i < 4; i++)
            for (int j = i+1; j < 5; j++)
                if (times[i] > times[j]) { double t = times[i]; times[i] = times[j]; times[j] = t; }

        double median_ms = times[2];
        double total_flops = (double)grid * bs * n_iters * accum_counts[ki] * 2;
        double gflops = total_flops / (median_ms * 1e-3) / 1e9;
        double pct = gflops / fp32_peak_gflops * 100.0;

        printf("%d,%d,%.0f,%.0f,%.1f\n", accum_counts[ki], attr.numRegs, occ, gflops, pct);
    }
    printf("\n");

    // ============================================================
    // 3. Launch bounds effect: same workload, different register caps
    // ============================================================
    printf("=== Launch bounds effect: 32 accumulators ===\n");
    printf("variant,regs,spill_bytes,occupancy_pct,gflops,pct_fp32_peak\n");

    compute_fn bounded_32[] = {k_compute_32, k_compute_32_bounded_high, k_compute_32_bounded_max};
    const char* bounded_32_names[] = {"natural", "bounded_high", "bounded_max"};

    for (int ki = 0; ki < 3; ki++) {
        int grid = sm_count * 4;
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (void*)bounded_32[ki]);
        int max_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, (void*)bounded_32[ki], bs, 0);
        float occ = max_blocks * ((bs + 31) / 32) / 64.0f * 100.0f;

        for (int w = 0; w < 3; w++)
            bounded_32[ki]<<<grid, bs>>>(d_out, n_iters);
        CHECK_CUDA(cudaDeviceSynchronize());

        double times[5];
        for (int r = 0; r < 5; r++) {
            cudaEventRecord(start);
            bounded_32[ki]<<<grid, bs>>>(d_out, n_iters);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            float ms;
            cudaEventElapsedTime(&ms, start, end);
            times[r] = ms;
        }
        for (int i = 0; i < 4; i++)
            for (int j = i+1; j < 5; j++)
                if (times[i] > times[j]) { double t = times[i]; times[i] = times[j]; times[j] = t; }

        double median_ms = times[2];
        double gflops = (double)grid * bs * n_iters * 32 * 2 / (median_ms * 1e-3) / 1e9;
        double pct = gflops / fp32_peak_gflops * 100.0;
        printf("%s,%d,%zu,%.0f,%.0f,%.1f\n", bounded_32_names[ki], attr.numRegs,
               attr.localSizeBytes, occ, gflops, pct);
    }
    printf("\n");

    printf("=== Launch bounds effect: 64 accumulators ===\n");
    printf("variant,regs,spill_bytes,occupancy_pct,gflops,pct_fp32_peak\n");

    compute_fn bounded_64[] = {k_compute_64, k_compute_64_bounded_high, k_compute_64_bounded_max};
    const char* bounded_64_names[] = {"natural", "bounded_high", "bounded_max"};

    for (int ki = 0; ki < 3; ki++) {
        int grid = sm_count * 4;
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (void*)bounded_64[ki]);
        int max_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, (void*)bounded_64[ki], bs, 0);
        float occ = max_blocks * ((bs + 31) / 32) / 64.0f * 100.0f;

        for (int w = 0; w < 3; w++)
            bounded_64[ki]<<<grid, bs>>>(d_out, n_iters);
        CHECK_CUDA(cudaDeviceSynchronize());

        double times[5];
        for (int r = 0; r < 5; r++) {
            cudaEventRecord(start);
            bounded_64[ki]<<<grid, bs>>>(d_out, n_iters);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            float ms;
            cudaEventElapsedTime(&ms, start, end);
            times[r] = ms;
        }
        for (int i = 0; i < 4; i++)
            for (int j = i+1; j < 5; j++)
                if (times[i] > times[j]) { double t = times[i]; times[i] = times[j]; times[j] = t; }

        double median_ms = times[2];
        double gflops = (double)grid * bs * n_iters * 64 * 2 / (median_ms * 1e-3) / 1e9;
        double pct = gflops / fp32_peak_gflops * 100.0;
        printf("%s,%d,%zu,%.0f,%.0f,%.1f\n", bounded_64_names[ki], attr.numRegs,
               attr.localSizeBytes, occ, gflops, pct);
    }
    printf("\n");

    // ============================================================
    // 4. Memory-bound: BW vs register pressure
    // ============================================================
    printf("=== Memory-bound: BW vs register pressure ===\n");
    printf("accumulators,regs,occupancy_pct,bw_gbs,pct_peak\n");

    typedef void (*membw_fn)(const float4*, float*, long long);
    membw_fn mem_kernels[] = {k_membw_4, k_membw_16, k_membw_32, k_membw_64, k_membw_128};
    int mem_accums[] = {4, 16, 32, 64, 128};
    int n_mem = 5;

    for (int ki = 0; ki < n_mem; ki++) {
        int grid = sm_count * 8;
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (void*)mem_kernels[ki]);
        int max_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, (void*)mem_kernels[ki], bs, 0);
        float occ = max_blocks * ((bs + 31) / 32) / 64.0f * 100.0f;

        for (int w = 0; w < 3; w++)
            mem_kernels[ki]<<<grid, bs>>>(d_src, d_dst, n_float4);
        CHECK_CUDA(cudaDeviceSynchronize());

        double times[5];
        for (int r = 0; r < 5; r++) {
            cudaEventRecord(start);
            mem_kernels[ki]<<<grid, bs>>>(d_src, d_dst, n_float4);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            float ms;
            cudaEventElapsedTime(&ms, start, end);
            times[r] = ms;
        }
        for (int i = 0; i < 4; i++)
            for (int j = i+1; j < 5; j++)
                if (times[i] > times[j]) { double t = times[i]; times[i] = times[j]; times[j] = t; }

        double median_ms = times[2];
        double bw = (double)n_bytes / median_ms * 1e-6;
        double pct = bw / 4800.0 * 100.0;
        printf("%d,%d,%.0f,%.0f,%.1f\n", mem_accums[ki], attr.numRegs, occ, bw, pct);
    }
    printf("\n");

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    printf("Done.\n");
    return 0;
}
