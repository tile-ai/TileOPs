// Occupancy vs Latency-Hiding Cookbook
// Measures how occupancy (warps/SM) affects throughput for different workloads.
//
// Compile: nvcc -O2 -arch=sm_90 -o occupancy_latency occupancy_latency.cu
// Usage:   ./occupancy_latency
//
// Tests:
// 1. Memory-bound kernel: vary block size + register pressure → observe BW
// 2. Compute-bound kernel: vary block size + register pressure → observe TFLOPS
// 3. Explicit register limiting via __launch_bounds__ + manual spill
// 4. Shared memory occupancy limiter
//
// Key question: how much occupancy do you actually need to hide latency?

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
// Memory-bound kernels: streaming read with varying register pressure
// ============================================================

// Minimal registers — high occupancy
__global__ void k_membw_light(const float4* __restrict__ src, float* __restrict__ dst, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (long long i = idx; i < n; i += stride) {
        float4 v = src[i];
        sum += v.x + v.y + v.z + v.w;
    }
    if (sum != 0.0f) dst[idx] = sum;  // prevent DCE, rarely writes
}

// Medium registers — use extra local vars to increase register pressure
__global__ void k_membw_medium(const float4* __restrict__ src, float* __restrict__ dst, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    float s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    for (long long i = idx; i < n; i += stride) {
        float4 v = src[i];
        s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
        s4 += v.x * v.y; s5 += v.z * v.w; s6 += v.x * v.z; s7 += v.y * v.w;
    }
    float total = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    if (total != 0.0f) dst[idx] = total;
}

// Heavy registers — many accumulators to force register spill
__global__ void k_membw_heavy(const float4* __restrict__ src, float* __restrict__ dst, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    float s[16];
    #pragma unroll
    for (int k = 0; k < 16; k++) s[k] = 0.0f;
    for (long long i = idx; i < n; i += stride) {
        float4 v = src[i];
        s[0] += v.x; s[1] += v.y; s[2] += v.z; s[3] += v.w;
        s[4] += v.x * 2.0f; s[5] += v.y * 2.0f; s[6] += v.z * 2.0f; s[7] += v.w * 2.0f;
        s[8] += v.x * v.x; s[9] += v.y * v.y; s[10] += v.z * v.z; s[11] += v.w * v.w;
        s[12] += v.x + v.w; s[13] += v.y + v.z; s[14] += v.x - v.y; s[15] += v.z - v.w;
    }
    float total = 0;
    #pragma unroll
    for (int k = 0; k < 16; k++) total += s[k];
    if (total != 0.0f) dst[idx] = total;
}

// ============================================================
// Compute-bound kernels: FMA-heavy with varying register pressure
// ============================================================

__global__ void k_compute_light(float* out, int n_iters) {
    float a = 1.0f + 0.0001f * threadIdx.x;
    float b = 1.0f;
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        b = fmaf(a, b, a);  // 1 FMA = 2 FLOPS
    }
    if (b == -999.0f) *out = b;  // prevent DCE
}

__global__ void k_compute_medium(float* out, int n_iters) {
    float a = 1.0f + 0.0001f * threadIdx.x;
    float b0 = 1.0f, b1 = 1.0f, b2 = 1.0f, b3 = 1.0f;
    float b4 = 1.0f, b5 = 1.0f, b6 = 1.0f, b7 = 1.0f;
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        b0 = fmaf(a, b0, a);
        b1 = fmaf(a, b1, a);
        b2 = fmaf(a, b2, a);
        b3 = fmaf(a, b3, a);
        b4 = fmaf(a, b4, a);
        b5 = fmaf(a, b5, a);
        b6 = fmaf(a, b6, a);
        b7 = fmaf(a, b7, a);
    }
    float total = b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7;
    if (total == -999.0f) *out = total;
}

__global__ void k_compute_heavy(float* out, int n_iters) {
    float a = 1.0f + 0.0001f * threadIdx.x;
    float b[16];
    #pragma unroll
    for (int k = 0; k < 16; k++) b[k] = 1.0f;
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        #pragma unroll
        for (int k = 0; k < 16; k++)
            b[k] = fmaf(a, b[k], a);
    }
    float total = 0;
    #pragma unroll
    for (int k = 0; k < 16; k++) total += b[k];
    if (total == -999.0f) *out = total;
}

// ============================================================
// Shared memory occupancy limiter
// ============================================================

// Uses dynamic smem to limit occupancy
__global__ void k_membw_smem_limit(const float4* __restrict__ src, float* __restrict__ dst,
                                     long long n) {
    extern __shared__ char smem[];
    // Touch smem
    if (threadIdx.x == 0) smem[0] = 1;
    __syncthreads();

    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (long long i = idx; i < n; i += stride) {
        float4 v = src[i];
        sum += v.x + v.y + v.z + v.w;
    }
    if (sum != 0.0f) dst[idx] = sum;
}

// ============================================================
// Benchmark helpers
// ============================================================

struct BenchResult {
    double time_ms;
    double bw_gbs;     // for memory kernels
    double gflops;     // for compute kernels
};

void print_occupancy(const char* name, void* kernel, int block_size, int smem_bytes) {
    int max_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, block_size, smem_bytes);

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);

    int warps_per_block = (block_size + 31) / 32;
    int warps_per_sm = max_blocks * warps_per_block;
    // H200 max warps per SM = 64 (2048 threads / 32)
    float occupancy = warps_per_sm / 64.0f * 100.0f;

    printf("  %s: block=%d, regs=%d, smem_static=%zuB, max_blocks/SM=%d, warps/SM=%d, occupancy=%.0f%%\n",
           name, block_size, attr.numRegs, attr.sharedSizeBytes, max_blocks, warps_per_sm, occupancy);
}

// ============================================================
// Main
// ============================================================

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;

    int sm_clock_khz;
    cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, 0);
    float sm_clock_mhz = sm_clock_khz / 1000.0f;

    printf("GPU: %s | SMs: %d | SM Clock: %.0f MHz\n", prop.name, sm_count, sm_clock_mhz);
    printf("Max threads/SM: %d | Max warps/SM: %d | Registers/SM: %d\n",
           prop.maxThreadsPerMultiProcessor,
           prop.maxThreadsPerMultiProcessor / 32,
           prop.regsPerMultiprocessor);
    printf("Max shared mem/SM: %zu KB\n\n", prop.sharedMemPerMultiprocessor / 1024);

    // Allocate data for memory kernels: 256 MB
    long long n_bytes = 256LL * 1024 * 1024;
    long long n_float4 = n_bytes / sizeof(float4);
    float4* d_src;
    float* d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, n_bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, n_bytes));  // oversized but safe
    CHECK_CUDA(cudaMemset(d_src, 1, n_bytes));

    float* d_compute_out;
    CHECK_CUDA(cudaMalloc(&d_compute_out, sizeof(float)));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // ============================================================
    // 1. Occupancy info for all kernels
    // ============================================================
    printf("=== Occupancy info ===\n");
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int n_bs = 6;

    for (int bi = 0; bi < n_bs; bi++) {
        int bs = block_sizes[bi];
        printf("Block size = %d:\n", bs);
        print_occupancy("membw_light", (void*)k_membw_light, bs, 0);
        print_occupancy("membw_medium", (void*)k_membw_medium, bs, 0);
        print_occupancy("membw_heavy", (void*)k_membw_heavy, bs, 0);
        print_occupancy("compute_light", (void*)k_compute_light, bs, 0);
        print_occupancy("compute_medium", (void*)k_compute_medium, bs, 0);
        print_occupancy("compute_heavy", (void*)k_compute_heavy, bs, 0);
    }
    printf("\n");

    // ============================================================
    // 2. Memory-bound: BW vs occupancy
    // ============================================================
    printf("=== Memory-bound: BW vs block size / register pressure ===\n");
    printf("kernel,block_size,occupancy_pct,bw_gbs,pct_peak\n");

    typedef void (*membw_fn)(const float4*, float*, long long);
    membw_fn mem_kernels[] = {k_membw_light, k_membw_medium, k_membw_heavy};
    const char* mem_names[] = {"light", "medium", "heavy"};

    for (int ki = 0; ki < 3; ki++) {
        for (int bi = 0; bi < n_bs; bi++) {
            int bs = block_sizes[bi];
            int grid = sm_count * 8;  // enough blocks to fill GPU

            // Get occupancy
            int max_blocks;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, (void*)mem_kernels[ki], bs, 0);
            int warps_per_sm = max_blocks * ((bs + 31) / 32);
            float occ = warps_per_sm / 64.0f * 100.0f;

            // Warmup
            for (int w = 0; w < 3; w++)
                mem_kernels[ki]<<<grid, bs>>>(d_src, d_dst, n_float4);
            CHECK_CUDA(cudaDeviceSynchronize());

            // Measure: 5 runs, median
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
            // Sort for median
            for (int i = 0; i < 4; i++)
                for (int j = i+1; j < 5; j++)
                    if (times[i] > times[j]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
            double median_ms = times[2];
            double bw = (double)n_bytes / median_ms * 1e-6;  // bytes / ms * 1e-6 = GB/s
            double pct = bw / 4800.0 * 100.0;

            printf("%s,%d,%.0f,%.0f,%.1f\n", mem_names[ki], bs, occ, bw, pct);
        }
    }
    printf("\n");

    // ============================================================
    // 3. Compute-bound: GFLOPS vs occupancy
    // ============================================================
    printf("=== Compute-bound: GFLOPS vs block size / register pressure ===\n");
    printf("kernel,block_size,occupancy_pct,gflops,pct_fp32_peak\n");

    // FP32 peak: 132 SMs * 1980 MHz * 128 FMA units * 2 = ~66.8 TFLOPS
    // Actually SM 9.0 has 128 FP32 units per SM
    double fp32_peak_gflops = (double)sm_count * sm_clock_mhz * 128 * 2 / 1000.0;  // GFLOPS

    typedef void (*compute_fn)(float*, int);
    compute_fn comp_kernels[] = {k_compute_light, k_compute_medium, k_compute_heavy};
    const char* comp_names[] = {"light_1fma", "medium_8fma", "heavy_16fma"};
    int fma_per_iter[] = {1, 8, 16};

    int n_iters = 100000;

    for (int ki = 0; ki < 3; ki++) {
        for (int bi = 0; bi < n_bs; bi++) {
            int bs = block_sizes[bi];
            int grid = sm_count * 4;

            int max_blocks;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, (void*)comp_kernels[ki], bs, 0);
            int warps_per_sm = max_blocks * ((bs + 31) / 32);
            float occ = warps_per_sm / 64.0f * 100.0f;

            // Warmup
            for (int w = 0; w < 3; w++)
                comp_kernels[ki]<<<grid, bs>>>(d_compute_out, n_iters);
            CHECK_CUDA(cudaDeviceSynchronize());

            double times[5];
            for (int r = 0; r < 5; r++) {
                cudaEventRecord(start);
                comp_kernels[ki]<<<grid, bs>>>(d_compute_out, n_iters);
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

            // FLOPS: grid * bs threads * n_iters * fma_per_iter * 2 (FMA = 2 FLOPS)
            double total_flops = (double)grid * bs * n_iters * fma_per_iter[ki] * 2;
            double gflops = total_flops / (median_ms * 1e-3) / 1e9;
            double pct = gflops / fp32_peak_gflops * 100.0;

            printf("%s,%d,%.0f,%.0f,%.1f\n", comp_names[ki], bs, occ, gflops, pct);
        }
    }
    printf("\n");

    // ============================================================
    // 4. Shared memory as occupancy limiter
    // ============================================================
    printf("=== Shared memory occupancy limiter (membw_light kernel + dynamic smem) ===\n");
    printf("smem_kb,block_size,occupancy_pct,bw_gbs,pct_peak\n");

    int smem_sizes[] = {1024, 16*1024, 32*1024, 48*1024, 64*1024, 96*1024, 128*1024};
    int n_smem = 7;

    for (int si = 0; si < n_smem; si++) {
        int smem = smem_sizes[si];
        int bs = 256;

        if (smem > 48 * 1024) {
            cudaFuncSetAttribute(k_membw_smem_limit,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        }

        int max_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
            (void*)k_membw_smem_limit, bs, smem);
        int warps_per_sm = max_blocks * ((bs + 31) / 32);
        float occ = warps_per_sm / 64.0f * 100.0f;

        int grid = sm_count * (max_blocks > 0 ? max_blocks : 1);

        if (max_blocks == 0) {
            printf("%d,%d,0,0,0.0\n", smem / 1024, bs);
            continue;
        }

        // Warmup
        for (int w = 0; w < 3; w++)
            k_membw_smem_limit<<<grid, bs, smem>>>(d_src, d_dst, n_float4);
        CHECK_CUDA(cudaDeviceSynchronize());

        double times[5];
        for (int r = 0; r < 5; r++) {
            cudaEventRecord(start);
            k_membw_smem_limit<<<grid, bs, smem>>>(d_src, d_dst, n_float4);
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

        printf("%d,%d,%.0f,%.0f,%.1f\n", smem / 1024, bs, occ, bw, pct);
    }
    printf("\n");

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFree(d_compute_out));

    printf("Done.\n");
    return 0;
}
