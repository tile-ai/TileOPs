// TMA / cp.async / Async Copy Cookbook
// Measures cost and throughput of different global→smem copy methods on H200.
//
// Compile: nvcc -O2 -arch=sm_90 -o async_copy async_copy.cu
// Usage:   ./async_copy
//
// Methods compared:
// 1. Explicit: LDG + STS (load from global, store to shared)
// 2. cp.async via inline PTX (hardware-accelerated async global→smem)
// 3. cp.async with multi-stage pipeline

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// cp.async inline PTX helpers
__device__ __forceinline__ void cp_async_f4(void* smem_ptr, const void* global_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(global_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

__device__ __forceinline__ void cp_async_wait_group(int n) {
    // Wait until at most n groups are pending
    // We use template-like approach with if-else
    if (n == 0) asm volatile("cp.async.wait_group 0;\n");
    else if (n == 1) asm volatile("cp.async.wait_group 1;\n");
    else if (n == 2) asm volatile("cp.async.wait_group 2;\n");
    else asm volatile("cp.async.wait_all;\n");
}

// ============================================================
// Method 1: Explicit LDG + STS
// ============================================================

__global__ void k_explicit_g2s(const float4* __restrict__ src,
                                int n_elements, long long* out_cycles) {
    extern __shared__ float4 smem[];
    int tid = threadIdx.x;
    int n_per_block = blockDim.x;

    long long start = clock64();
    for (int offset = 0; offset < n_elements; offset += n_per_block) {
        int idx = offset + tid;
        if (idx < n_elements) {
            smem[tid] = src[idx];  // LDG → register → STS
        }
        __syncthreads();
    }
    long long end = clock64();

    if (tid == 0) out_cycles[0] = end - start;
    // Prevent DCE
    if (tid == 0 && smem[0].x == -999.0f) out_cycles[1] = __float_as_int(smem[0].x);
}

// ============================================================
// Method 2: cp.async (bypass register file)
// ============================================================

__global__ void k_cpasync_g2s(const float4* __restrict__ src,
                               int n_elements, long long* out_cycles) {
    extern __shared__ float4 smem[];
    int tid = threadIdx.x;
    int n_per_block = blockDim.x;

    long long start = clock64();
    for (int offset = 0; offset < n_elements; offset += n_per_block) {
        int idx = offset + tid;
        if (idx < n_elements) {
            cp_async_f4(&smem[tid], &src[idx]);
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
    }
    long long end = clock64();

    if (tid == 0) out_cycles[0] = end - start;
    if (tid == 0 && smem[0].x == -999.0f) out_cycles[1] = __float_as_int(smem[0].x);
}

// ============================================================
// Method 3: cp.async with double-buffer pipeline
// ============================================================

__global__ void k_cpasync_pipeline(const float4* __restrict__ src,
                                    int n_elements, long long* out_cycles) {
    extern __shared__ float4 smem_all[];
    float4* buf0 = smem_all;
    float4* buf1 = smem_all + blockDim.x;

    int tid = threadIdx.x;
    int n_per_block = blockDim.x;
    int n_iters = (n_elements + n_per_block - 1) / n_per_block;

    long long start = clock64();

    // Prologue: kick off first tile
    if (tid < n_elements) {
        cp_async_f4(&buf0[tid], &src[tid]);
    }
    cp_async_commit();

    for (int iter = 0; iter < n_iters; iter++) {
        int cur = iter & 1;
        float4* cur_buf = cur ? buf1 : buf0;
        float4* nxt_buf = cur ? buf0 : buf1;

        // Kick off next tile (if exists)
        int next_offset = (iter + 1) * n_per_block;
        if (iter + 1 < n_iters) {
            int next_idx = next_offset + tid;
            if (next_idx < n_elements) {
                cp_async_f4(&nxt_buf[tid], &src[next_idx]);
            }
            cp_async_commit();
        }

        // Wait for current tile
        if (iter + 1 < n_iters)
            cp_async_wait_group(1);
        else
            cp_async_wait_all();
        __syncthreads();

        // "Use" current tile (just read to prevent DCE)
        if (tid == 0) {
            float tmp = cur_buf[0].x;
            asm volatile("" :: "f"(tmp));
        }
        __syncthreads();
    }

    long long end = clock64();
    if (tid == 0) out_cycles[0] = end - start;
    if (tid == 0 && smem_all[0].x == -999.0f) out_cycles[1] = __float_as_int(smem_all[0].x);
}

// ============================================================
// Round-trip: global→smem→global (for BW measurement)
// ============================================================

__global__ void k_explicit_roundtrip(const float4* __restrict__ src, float4* dst,
                                      int n_elements, long long* out_cycles) {
    extern __shared__ float4 smem[];
    int tid = threadIdx.x;
    int n_per_block = blockDim.x;

    long long start = clock64();
    for (int offset = 0; offset < n_elements; offset += n_per_block) {
        int idx = offset + tid;
        if (idx < n_elements) {
            smem[tid] = src[idx];
        }
        __syncthreads();
        if (idx < n_elements) {
            dst[idx] = smem[tid];
        }
        __syncthreads();
    }
    long long end = clock64();
    if (tid == 0) out_cycles[0] = end - start;
}

__global__ void k_cpasync_roundtrip(const float4* __restrict__ src, float4* dst,
                                     int n_elements, long long* out_cycles) {
    extern __shared__ float4 smem[];
    int tid = threadIdx.x;
    int n_per_block = blockDim.x;

    long long start = clock64();
    for (int offset = 0; offset < n_elements; offset += n_per_block) {
        int idx = offset + tid;
        if (idx < n_elements) {
            cp_async_f4(&smem[tid], &src[idx]);
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
        if (idx < n_elements) {
            dst[idx] = smem[tid];
        }
        __syncthreads();
    }
    long long end = clock64();
    if (tid == 0) out_cycles[0] = end - start;
}

// ============================================================
// Benchmark helper
// ============================================================

struct CopyResult {
    double cycles;
    double time_us;
    double bw_gbs;
};

template <typename LaunchFn>
CopyResult bench_copy(LaunchFn launch_fn, long long* d_cycles,
                       long long total_bytes, float sm_clock_mhz) {
    for (int w = 0; w < 5; w++) launch_fn();
    CHECK_CUDA(cudaDeviceSynchronize());

    double results[5];
    long long h;
    for (int r = 0; r < 5; r++) {
        CHECK_CUDA(cudaMemset(d_cycles, 0, 2 * sizeof(long long)));
        launch_fn();
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&h, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost));
        results[r] = (double)h;
    }
    for (int i = 0; i < 4; i++)
        for (int j = i+1; j < 5; j++)
            if (results[i] > results[j]) { double t = results[i]; results[i] = results[j]; results[j] = t; }

    double median = results[2];
    double time_us = median / sm_clock_mhz;
    double bw = (double)total_bytes / time_us * 1e-3;
    return {median, time_us, bw};
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int sm_clock_khz;
    cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, 0);
    float sm_clock_mhz = sm_clock_khz / 1000.0f;

    printf("GPU: %s | SMs: %d | SM Clock: %.0f MHz\n", prop.name, prop.multiProcessorCount, sm_clock_mhz);
    printf("Shared memory per SM: %zu KB\n\n", prop.sharedMemPerMultiprocessor / 1024);

    int threads = 256;
    int smem_bytes = threads * sizeof(float4);
    int smem_bytes_double = smem_bytes * 2;

    long long* d_cycles;
    CHECK_CUDA(cudaMalloc(&d_cycles, 2 * sizeof(long long)));

    // ============================================================
    // 1. Global→smem only: explicit vs cp.async vs pipeline
    // ============================================================
    int copy_sizes[] = {1024, 4096, 16384, 65536};
    const char* copy_labels[] = {"16KB", "64KB", "256KB", "1MB"};
    int n_sizes = 4;

    for (int si = 0; si < n_sizes; si++) {
        int n_elem = copy_sizes[si];
        long long total_bytes = (long long)n_elem * sizeof(float4);

        float4* d_src;
        CHECK_CUDA(cudaMalloc(&d_src, total_bytes));
        CHECK_CUDA(cudaMemset(d_src, 1, total_bytes));

        printf("=== Global→smem: %s (%lld bytes), 1 CTA, %d threads ===\n",
               copy_labels[si], total_bytes, threads);
        printf("method,cycles,time_us,bw_gbs\n");

        auto r1 = bench_copy([&]() {
            k_explicit_g2s<<<1, threads, smem_bytes>>>(d_src, n_elem, d_cycles);
        }, d_cycles, total_bytes, sm_clock_mhz);
        printf("explicit,%.0f,%.2f,%.1f\n", r1.cycles, r1.time_us, r1.bw_gbs);

        auto r2 = bench_copy([&]() {
            k_cpasync_g2s<<<1, threads, smem_bytes>>>(d_src, n_elem, d_cycles);
        }, d_cycles, total_bytes, sm_clock_mhz);
        printf("cp_async,%.0f,%.2f,%.1f\n", r2.cycles, r2.time_us, r2.bw_gbs);

        auto r3 = bench_copy([&]() {
            k_cpasync_pipeline<<<1, threads, smem_bytes_double>>>(d_src, n_elem, d_cycles);
        }, d_cycles, total_bytes, sm_clock_mhz);
        printf("cp_async_pipeline,%.0f,%.2f,%.1f\n", r3.cycles, r3.time_us, r3.bw_gbs);

        printf("cp_async speedup vs explicit: %.2fx\n", r1.cycles / r2.cycles);
        printf("pipeline speedup vs explicit: %.2fx\n", r1.cycles / r3.cycles);
        printf("\n");

        CHECK_CUDA(cudaFree(d_src));
    }

    // ============================================================
    // 2. Round-trip: global→smem→global
    // ============================================================
    printf("=== Round-trip: global→smem→global, 256KB ===\n");
    printf("method,cycles,time_us,bw_gbs\n");
    {
        int n_elem = 16384;
        long long total = (long long)n_elem * sizeof(float4) * 2;  // read + write

        float4* d_src;
        float4* d_dst;
        CHECK_CUDA(cudaMalloc(&d_src, n_elem * sizeof(float4)));
        CHECK_CUDA(cudaMalloc(&d_dst, n_elem * sizeof(float4)));
        CHECK_CUDA(cudaMemset(d_src, 1, n_elem * sizeof(float4)));

        auto r1 = bench_copy([&]() {
            k_explicit_roundtrip<<<1, threads, smem_bytes>>>(d_src, d_dst, n_elem, d_cycles);
        }, d_cycles, total, sm_clock_mhz);
        printf("explicit_roundtrip,%.0f,%.2f,%.1f\n", r1.cycles, r1.time_us, r1.bw_gbs);

        auto r2 = bench_copy([&]() {
            k_cpasync_roundtrip<<<1, threads, smem_bytes>>>(d_src, d_dst, n_elem, d_cycles);
        }, d_cycles, total, sm_clock_mhz);
        printf("cp_async_roundtrip,%.0f,%.2f,%.1f\n", r2.cycles, r2.time_us, r2.bw_gbs);

        printf("cp_async speedup: %.2fx\n\n", r1.cycles / r2.cycles);

        CHECK_CUDA(cudaFree(d_src));
        CHECK_CUDA(cudaFree(d_dst));
    }

    // ============================================================
    // 3. Thread count sweep
    // ============================================================
    printf("=== Thread count sweep: 256KB global→smem ===\n");
    printf("threads,method,cycles,bw_gbs\n");

    int thread_counts[] = {32, 64, 128, 256, 512};
    int n_elem_fixed = 16384;
    long long bytes_fixed = (long long)n_elem_fixed * sizeof(float4);

    float4* d_src_fixed;
    CHECK_CUDA(cudaMalloc(&d_src_fixed, bytes_fixed));
    CHECK_CUDA(cudaMemset(d_src_fixed, 1, bytes_fixed));

    for (int ti = 0; ti < 5; ti++) {
        int t = thread_counts[ti];
        int smem_t = t * sizeof(float4);

        auto r1 = bench_copy([&]() {
            k_explicit_g2s<<<1, t, smem_t>>>(d_src_fixed, n_elem_fixed, d_cycles);
        }, d_cycles, bytes_fixed, sm_clock_mhz);
        printf("%d,explicit,%.0f,%.1f\n", t, r1.cycles, r1.bw_gbs);

        auto r2 = bench_copy([&]() {
            k_cpasync_g2s<<<1, t, smem_t>>>(d_src_fixed, n_elem_fixed, d_cycles);
        }, d_cycles, bytes_fixed, sm_clock_mhz);
        printf("%d,cp_async,%.0f,%.1f\n", t, r2.cycles, r2.bw_gbs);
    }
    printf("\n");

    CHECK_CUDA(cudaFree(d_src_fixed));
    CHECK_CUDA(cudaFree(d_cycles));

    printf("Done.\n");
    return 0;
}
