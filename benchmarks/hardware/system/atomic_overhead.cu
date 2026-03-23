// Atomic Overhead Cookbook
// Measures cost of atomic operations under varying contention levels.
//
// Compile: nvcc -O2 -arch=sm_90 -o atomic_overhead atomic_overhead.cu
// Usage:   ./atomic_overhead
//
// Tests:
// 1. Atomic op type: atomicAdd, atomicCAS, atomicExch, atomicMin (int32 + fp32)
// 2. Memory space: shared memory vs global memory
// 3. Contention level: single address (max contention) vs per-thread address (no contention)
// 4. Scaling: 1 warp → full CTA → multi-CTA
//
// Method: Each kernel does N_OPS atomic ops in a tight loop, measures
// total cycles with clock64(), subtracts baseline.

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

#define N_OPS 1000  // Number of atomic ops per measurement

// ============================================================
// Global memory atomics — single address (max contention)
// ============================================================

__global__ void k_global_atomicAdd_contended(long long* out_cycles, int* target, int n_ops) {
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        atomicAdd(target, 1);
    }
    long long end = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out_cycles[0] = end - start;
    }
}

__global__ void k_global_atomicAdd_fp32_contended(long long* out_cycles, float* target, int n_ops) {
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        atomicAdd(target, 1.0f);
    }
    long long end = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out_cycles[0] = end - start;
    }
}

__global__ void k_global_atomicCAS_contended(long long* out_cycles, int* target, int n_ops) {
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        atomicCAS(target, 0, 1);
    }
    long long end = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out_cycles[0] = end - start;
    }
}

__global__ void k_global_atomicExch_contended(long long* out_cycles, int* target, int n_ops) {
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        atomicExch(target, threadIdx.x);
    }
    long long end = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out_cycles[0] = end - start;
    }
}

__global__ void k_global_atomicMin_contended(long long* out_cycles, int* target, int n_ops) {
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        atomicMin(target, threadIdx.x);
    }
    long long end = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out_cycles[0] = end - start;
    }
}

// ============================================================
// Global memory atomics — per-thread address (no contention)
// ============================================================

__global__ void k_global_atomicAdd_uncontended(long long* out_cycles, int* target, int n_ops) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        atomicAdd(&target[idx], 1);
    }
    long long end = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out_cycles[0] = end - start;
    }
}

__global__ void k_global_atomicAdd_fp32_uncontended(long long* out_cycles, float* target, int n_ops) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        atomicAdd(&target[idx], 1.0f);
    }
    long long end = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out_cycles[0] = end - start;
    }
}

// ============================================================
// Shared memory atomics — single address (max contention)
// ============================================================

__global__ void k_smem_atomicAdd_contended(long long* out_cycles, int n_ops) {
    __shared__ int s_target;
    if (threadIdx.x == 0) s_target = 0;
    __syncthreads();

    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        atomicAdd(&s_target, 1);
    }
    long long end = clock64();
    if (threadIdx.x == 0) {
        out_cycles[0] = end - start;
    }
}

__global__ void k_smem_atomicAdd_fp32_contended(long long* out_cycles, int n_ops) {
    __shared__ float s_target;
    if (threadIdx.x == 0) s_target = 0.0f;
    __syncthreads();

    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        atomicAdd(&s_target, 1.0f);
    }
    long long end = clock64();
    if (threadIdx.x == 0) {
        out_cycles[0] = end - start;
    }
}

__global__ void k_smem_atomicCAS_contended(long long* out_cycles, int n_ops) {
    __shared__ int s_target;
    if (threadIdx.x == 0) s_target = 0;
    __syncthreads();

    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        atomicCAS(&s_target, 0, 1);
    }
    long long end = clock64();
    if (threadIdx.x == 0) {
        out_cycles[0] = end - start;
    }
}

// ============================================================
// Shared memory atomics — per-thread address (no contention)
// ============================================================

__global__ void k_smem_atomicAdd_uncontended(long long* out_cycles, int n_ops) {
    extern __shared__ int s_arr[];
    s_arr[threadIdx.x] = 0;
    __syncthreads();

    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        atomicAdd(&s_arr[threadIdx.x], 1);
    }
    long long end = clock64();
    if (threadIdx.x == 0) {
        out_cycles[0] = end - start;
    }
}

// ============================================================
// Baseline: non-atomic global write (to calibrate)
// ============================================================

__global__ void k_global_plain_write(long long* out_cycles, int* target, int n_ops) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        target[idx] = i;
        asm volatile("" ::: "memory");  // prevent optimization
    }
    long long end = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out_cycles[0] = end - start;
    }
}

// ============================================================
// Benchmark helper
// ============================================================

struct AtomicResult {
    double total_cycles;   // thread 0 total
    double per_op_cycles;
    double per_op_ns;
};

// Generic launcher type
typedef void (*kernel_fn_global)(long long*, int*, int);
typedef void (*kernel_fn_global_fp32)(long long*, float*, int);
typedef void (*kernel_fn_smem)(long long*, int);

AtomicResult run_bench_global(kernel_fn_global fn, int* d_target, long long* d_cycles,
                               int n_ops, int threads, int blocks, float sm_clock_mhz) {
    // Warmup
    for (int i = 0; i < 3; i++) {
        cudaMemset(d_target, 0, threads * blocks * sizeof(int));
        fn<<<blocks, threads>>>(d_cycles, d_target, n_ops);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    double results[3];
    long long h;
    for (int run = 0; run < 3; run++) {
        cudaMemset(d_target, 0, threads * blocks * sizeof(int));
        cudaMemset(d_cycles, 0, sizeof(long long));
        fn<<<blocks, threads>>>(d_cycles, d_target, n_ops);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&h, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost));
        results[run] = (double)h;
    }

    // Median
    for (int i = 0; i < 2; i++)
        for (int j = i+1; j < 3; j++)
            if (results[i] > results[j]) { double t = results[i]; results[i] = results[j]; results[j] = t; }

    double median = results[1];
    double per_op = median / n_ops;
    double ns = per_op / (sm_clock_mhz * 1e-3);
    return {median, per_op, ns};
}

AtomicResult run_bench_global_fp32(kernel_fn_global_fp32 fn, float* d_target, long long* d_cycles,
                                    int n_ops, int threads, int blocks, float sm_clock_mhz) {
    for (int i = 0; i < 3; i++) {
        cudaMemset(d_target, 0, threads * blocks * sizeof(float));
        fn<<<blocks, threads>>>(d_cycles, d_target, n_ops);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    double results[3];
    long long h;
    for (int run = 0; run < 3; run++) {
        cudaMemset(d_target, 0, threads * blocks * sizeof(float));
        cudaMemset(d_cycles, 0, sizeof(long long));
        fn<<<blocks, threads>>>(d_cycles, d_target, n_ops);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&h, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost));
        results[run] = (double)h;
    }

    for (int i = 0; i < 2; i++)
        for (int j = i+1; j < 3; j++)
            if (results[i] > results[j]) { double t = results[i]; results[i] = results[j]; results[j] = t; }

    double median = results[1];
    double per_op = median / n_ops;
    double ns = per_op / (sm_clock_mhz * 1e-3);
    return {median, per_op, ns};
}

AtomicResult run_bench_smem(kernel_fn_smem fn, long long* d_cycles,
                             int n_ops, int threads, int smem_bytes, float sm_clock_mhz) {
    for (int i = 0; i < 3; i++)
        fn<<<1, threads, smem_bytes>>>(d_cycles, n_ops);
    CHECK_CUDA(cudaDeviceSynchronize());

    double results[3];
    long long h;
    for (int run = 0; run < 3; run++) {
        cudaMemset(d_cycles, 0, sizeof(long long));
        fn<<<1, threads, smem_bytes>>>(d_cycles, n_ops);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&h, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost));
        results[run] = (double)h;
    }

    for (int i = 0; i < 2; i++)
        for (int j = i+1; j < 3; j++)
            if (results[i] > results[j]) { double t = results[i]; results[i] = results[j]; results[j] = t; }

    double median = results[1];
    double per_op = median / n_ops;
    double ns = per_op / (sm_clock_mhz * 1e-3);
    return {median, per_op, ns};
}

// ============================================================
// Main
// ============================================================

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int sm_clock_khz;
    cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, 0);
    float sm_clock_mhz = sm_clock_khz / 1000.0f;

    printf("GPU: %s | SMs: %d | SM Clock: %.0f MHz\n", prop.name, prop.multiProcessorCount, sm_clock_mhz);
    printf("N_OPS = %d per measurement, 3 runs median\n\n", N_OPS);

    long long* d_cycles;
    CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(long long)));

    int max_threads = 1024;
    int max_elements = max_threads * 528; // for multi-CTA uncontended
    int* d_int_target;
    float* d_float_target;
    CHECK_CUDA(cudaMalloc(&d_int_target, max_elements * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_float_target, max_elements * sizeof(float)));

    int thread_counts[] = {32, 64, 128, 256, 512, 1024};
    int n_tc = 6;

    // ============================================================
    // 1. Baseline: plain write (no atomic)
    // ============================================================
    printf("=== Baseline: plain global write (per-thread addr) ===\n");
    printf("threads,per_op_cycles,per_op_ns\n");
    for (int ti = 0; ti < n_tc; ti++) {
        int t = thread_counts[ti];
        auto r = run_bench_global(k_global_plain_write, d_int_target, d_cycles, N_OPS, t, 1, sm_clock_mhz);
        printf("%d,%.1f,%.2f\n", t, r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    // ============================================================
    // 2. Global atomicAdd — contended (single addr) vs uncontended
    // ============================================================
    printf("=== Global atomicAdd int32 — contended (single addr) ===\n");
    printf("threads,blocks,per_op_cycles,per_op_ns\n");
    for (int ti = 0; ti < n_tc; ti++) {
        int t = thread_counts[ti];
        auto r = run_bench_global(k_global_atomicAdd_contended, d_int_target, d_cycles, N_OPS, t, 1, sm_clock_mhz);
        printf("%d,1,%.1f,%.2f\n", t, r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    printf("=== Global atomicAdd int32 — uncontended (per-thread addr) ===\n");
    printf("threads,blocks,per_op_cycles,per_op_ns\n");
    for (int ti = 0; ti < n_tc; ti++) {
        int t = thread_counts[ti];
        auto r = run_bench_global(k_global_atomicAdd_uncontended, d_int_target, d_cycles, N_OPS, t, 1, sm_clock_mhz);
        printf("%d,1,%.1f,%.2f\n", t, r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    // ============================================================
    // 3. Global atomicAdd fp32 — contended vs uncontended
    // ============================================================
    printf("=== Global atomicAdd fp32 — contended ===\n");
    printf("threads,blocks,per_op_cycles,per_op_ns\n");
    for (int ti = 0; ti < n_tc; ti++) {
        int t = thread_counts[ti];
        auto r = run_bench_global_fp32(k_global_atomicAdd_fp32_contended, d_float_target, d_cycles, N_OPS, t, 1, sm_clock_mhz);
        printf("%d,1,%.1f,%.2f\n", t, r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    printf("=== Global atomicAdd fp32 — uncontended ===\n");
    printf("threads,blocks,per_op_cycles,per_op_ns\n");
    for (int ti = 0; ti < n_tc; ti++) {
        int t = thread_counts[ti];
        auto r = run_bench_global_fp32(k_global_atomicAdd_fp32_uncontended, d_float_target, d_cycles, N_OPS, t, 1, sm_clock_mhz);
        printf("%d,1,%.1f,%.2f\n", t, r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    // ============================================================
    // 4. Different atomic ops — contended, 256 threads
    // ============================================================
    printf("=== Different atomic ops — global, contended, 256 threads ===\n");
    printf("op,per_op_cycles,per_op_ns\n");
    {
        int t = 256;
        auto r1 = run_bench_global(k_global_atomicAdd_contended, d_int_target, d_cycles, N_OPS, t, 1, sm_clock_mhz);
        printf("atomicAdd_i32,%.1f,%.2f\n", r1.per_op_cycles, r1.per_op_ns);

        auto r2 = run_bench_global_fp32(k_global_atomicAdd_fp32_contended, d_float_target, d_cycles, N_OPS, t, 1, sm_clock_mhz);
        printf("atomicAdd_fp32,%.1f,%.2f\n", r2.per_op_cycles, r2.per_op_ns);

        auto r3 = run_bench_global(k_global_atomicCAS_contended, d_int_target, d_cycles, N_OPS, t, 1, sm_clock_mhz);
        printf("atomicCAS_i32,%.1f,%.2f\n", r3.per_op_cycles, r3.per_op_ns);

        auto r4 = run_bench_global(k_global_atomicExch_contended, d_int_target, d_cycles, N_OPS, t, 1, sm_clock_mhz);
        printf("atomicExch_i32,%.1f,%.2f\n", r4.per_op_cycles, r4.per_op_ns);

        auto r5 = run_bench_global(k_global_atomicMin_contended, d_int_target, d_cycles, N_OPS, t, 1, sm_clock_mhz);
        printf("atomicMin_i32,%.1f,%.2f\n", r5.per_op_cycles, r5.per_op_ns);
    }
    printf("\n");

    // ============================================================
    // 5. Shared memory atomics — contended vs uncontended
    // ============================================================
    printf("=== Shared mem atomicAdd int32 — contended ===\n");
    printf("threads,per_op_cycles,per_op_ns\n");
    for (int ti = 0; ti < n_tc; ti++) {
        int t = thread_counts[ti];
        auto r = run_bench_smem(k_smem_atomicAdd_contended, d_cycles, N_OPS, t, 0, sm_clock_mhz);
        printf("%d,%.1f,%.2f\n", t, r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    printf("=== Shared mem atomicAdd fp32 — contended ===\n");
    printf("threads,per_op_cycles,per_op_ns\n");
    for (int ti = 0; ti < n_tc; ti++) {
        int t = thread_counts[ti];
        auto r = run_bench_smem(k_smem_atomicAdd_fp32_contended, d_cycles, N_OPS, t, 0, sm_clock_mhz);
        printf("%d,%.1f,%.2f\n", t, r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    printf("=== Shared mem atomicCAS int32 — contended ===\n");
    printf("threads,per_op_cycles,per_op_ns\n");
    for (int ti = 0; ti < n_tc; ti++) {
        int t = thread_counts[ti];
        auto r = run_bench_smem(k_smem_atomicCAS_contended, d_cycles, N_OPS, t, 0, sm_clock_mhz);
        printf("%d,%.1f,%.2f\n", t, r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    printf("=== Shared mem atomicAdd int32 — uncontended ===\n");
    printf("threads,per_op_cycles,per_op_ns\n");
    for (int ti = 0; ti < n_tc; ti++) {
        int t = thread_counts[ti];
        auto r = run_bench_smem(k_smem_atomicAdd_uncontended, d_cycles, N_OPS, t, t * sizeof(int), sm_clock_mhz);
        printf("%d,%.1f,%.2f\n", t, r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    // ============================================================
    // 6. Shared vs Global comparison (256 threads, contended)
    // ============================================================
    printf("=== Shared vs Global atomicAdd int32 — 256t contended ===\n");
    printf("memory,per_op_cycles,per_op_ns\n");
    {
        int t = 256;
        auto rg = run_bench_global(k_global_atomicAdd_contended, d_int_target, d_cycles, N_OPS, t, 1, sm_clock_mhz);
        printf("global,%.1f,%.2f\n", rg.per_op_cycles, rg.per_op_ns);
        auto rs = run_bench_smem(k_smem_atomicAdd_contended, d_cycles, N_OPS, t, 0, sm_clock_mhz);
        printf("shared,%.1f,%.2f\n", rs.per_op_cycles, rs.per_op_ns);
    }
    printf("\n");

    // ============================================================
    // 7. Multi-CTA contention scaling (global atomicAdd, 256 threads)
    // ============================================================
    printf("=== Multi-CTA global atomicAdd int32 — contended, 256 threads ===\n");
    printf("num_ctas,per_op_cycles,per_op_ns\n");
    int cta_counts[] = {1, 4, 16, 64, 132, 264, 528};
    for (int ci = 0; ci < 7; ci++) {
        int nblocks = cta_counts[ci];
        auto r = run_bench_global(k_global_atomicAdd_contended, d_int_target, d_cycles, N_OPS, 256, nblocks, sm_clock_mhz);
        printf("%d,%.1f,%.2f\n", nblocks, r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    CHECK_CUDA(cudaFree(d_cycles));
    CHECK_CUDA(cudaFree(d_int_target));
    CHECK_CUDA(cudaFree(d_float_target));

    printf("Done.\n");
    return 0;
}
