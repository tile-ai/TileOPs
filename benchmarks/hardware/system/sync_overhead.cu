// Sync Overhead Cookbook
// Measures __syncthreads() and __syncwarp() cost using clock64().
//
// Compile: nvcc -O2 -arch=sm_90 -o sync_overhead sync_overhead.cu
// Usage:   ./sync_overhead
//
// Method: Each kernel does N_SYNC sync calls in a tight loop, measures
// total cycles with clock64(), subtracts baseline (empty loop).
// Single CTA to isolate sync cost from scheduling effects.

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

#define N_SYNC 1000  // Number of sync calls per measurement

// ============================================================
// Kernels
// ============================================================

// Baseline: empty loop (measure loop overhead)
__global__ void k_baseline(long long* out_cycles, int n_sync) {
    if (threadIdx.x == 0) {
        long long start = clock64();
        #pragma unroll 1
        for (int i = 0; i < n_sync; i++) {
            // empty — just loop overhead
            asm volatile("" ::: "memory");
        }
        long long end = clock64();
        out_cycles[0] = end - start;
    }
}

// __syncthreads() in a loop
__global__ void k_syncthreads(long long* out_cycles, int n_sync) {
    if (threadIdx.x == 0) {
        long long start = clock64();
    }
    __syncthreads(); // initial barrier to ensure all threads ready

    long long start;
    if (threadIdx.x == 0) start = clock64();

    #pragma unroll 1
    for (int i = 0; i < n_sync; i++) {
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        long long end = clock64();
        out_cycles[0] = end - start;
    }
}

// __syncwarp() in a loop (only first warp participates)
__global__ void k_syncwarp(long long* out_cycles, int n_sync) {
    if (threadIdx.x < 32) {
        __syncwarp();
        long long start = clock64();
        #pragma unroll 1
        for (int i = 0; i < n_sync; i++) {
            __syncwarp();
        }
        long long end = clock64();
        if (threadIdx.x == 0) {
            out_cycles[0] = end - start;
        }
    }
}

// __syncthreads() with shared memory allocation (to test occupancy effect)
__global__ void k_syncthreads_smem(long long* out_cycles, int n_sync) {
    // Dynamic shared memory — size controlled at launch
    extern __shared__ char smem[];

    // Touch shared memory to ensure it's allocated
    if (threadIdx.x == 0) smem[0] = 1;
    __syncthreads();

    long long start;
    if (threadIdx.x == 0) start = clock64();

    #pragma unroll 1
    for (int i = 0; i < n_sync; i++) {
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        long long end = clock64();
        out_cycles[0] = end - start;
    }
}

// __syncthreads() with actual shared memory work between syncs
__global__ void k_syncthreads_with_work(long long* out_cycles, int n_sync) {
    extern __shared__ float smem_f[];
    int tid = threadIdx.x;

    // Init smem
    smem_f[tid] = (float)tid;
    __syncthreads();

    long long start;
    if (tid == 0) start = clock64();

    #pragma unroll 1
    for (int i = 0; i < n_sync; i++) {
        smem_f[tid] += 1.0f;
        __syncthreads();
    }

    if (tid == 0) {
        long long end = clock64();
        out_cycles[0] = end - start;
    }

    // Prevent DCE
    if (tid == 0 && smem_f[0] < -1e30f) out_cycles[1] = (long long)smem_f[0];
}

// ============================================================
// Benchmark helper
// ============================================================

struct SyncResult {
    double total_cycles;
    double per_sync_cycles;
    double per_sync_ns;
};

SyncResult bench_sync(void (*launch_fn)(long long*, int, int, int), int threads, int smem_bytes,
                       int n_sync, float sm_clock_mhz) {
    long long* d_cycles;
    long long h_cycles;
    CHECK_CUDA(cudaMalloc(&d_cycles, 2 * sizeof(long long)));

    // Warmup
    for (int i = 0; i < 5; i++) {
        launch_fn(d_cycles, n_sync, threads, smem_bytes);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Measure: 3 runs, take median
    double results[3];
    for (int run = 0; run < 3; run++) {
        CHECK_CUDA(cudaMemset(d_cycles, 0, 2 * sizeof(long long)));
        launch_fn(d_cycles, n_sync, threads, smem_bytes);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&h_cycles, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost));
        results[run] = (double)h_cycles;
    }

    // Sort for median
    for (int i = 0; i < 2; i++)
        for (int j = i+1; j < 3; j++)
            if (results[i] > results[j]) { double t = results[i]; results[i] = results[j]; results[j] = t; }

    double median = results[1];
    double per_sync = median / n_sync;
    double ns = per_sync / (sm_clock_mhz * 1e-3);  // cycles / (MHz * 1e-3) = cycles / (GHz) = ns

    CHECK_CUDA(cudaFree(d_cycles));

    return {median, per_sync, ns};
}

// Launch wrappers
void launch_baseline(long long* d, int n, int threads, int smem) {
    k_baseline<<<1, threads>>>(d, n);
}
void launch_syncthreads(long long* d, int n, int threads, int smem) {
    k_syncthreads<<<1, threads>>>(d, n);
}
void launch_syncwarp(long long* d, int n, int threads, int smem) {
    k_syncwarp<<<1, threads>>>(d, n);
}
void launch_syncthreads_smem(long long* d, int n, int threads, int smem) {
    if (smem > 48*1024) {
        cudaFuncSetAttribute(k_syncthreads_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    k_syncthreads_smem<<<1, threads, smem>>>(d, n);
}
void launch_syncthreads_work(long long* d, int n, int threads, int smem) {
    k_syncthreads_with_work<<<1, threads, threads * sizeof(float)>>>(d, n);
}

// ============================================================
// Main
// ============================================================

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Get actual SM clock
    int sm_clock_khz;
    cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, 0);
    float sm_clock_mhz = sm_clock_khz / 1000.0f;

    printf("GPU: %s | SMs: %d | SM Clock: %.0f MHz\n", prop.name, prop.multiProcessorCount, sm_clock_mhz);
    printf("Shared memory per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("N_SYNC = %d per measurement, 3 runs median\n\n", N_SYNC);

    int thread_counts[] = {32, 64, 128, 256, 512, 1024};
    int n_threads = sizeof(thread_counts) / sizeof(thread_counts[0]);

    // ============================================================
    // 1. Baseline (empty loop)
    // ============================================================
    printf("=== Baseline (empty loop) ===\n");
    printf("threads,total_cycles,per_iter_cycles,per_iter_ns\n");
    for (int ti = 0; ti < n_threads; ti++) {
        int t = thread_counts[ti];
        auto r = bench_sync(launch_baseline, t, 0, N_SYNC, sm_clock_mhz);
        printf("%d,%.0f,%.1f,%.2f\n", t, r.total_cycles, r.per_sync_cycles, r.per_sync_ns);
    }
    printf("\n");

    // ============================================================
    // 2. __syncthreads() vs __syncwarp()
    // ============================================================
    printf("=== __syncthreads() cost (1 CTA, no smem) ===\n");
    printf("threads,total_cycles,per_sync_cycles,per_sync_ns\n");
    for (int ti = 0; ti < n_threads; ti++) {
        int t = thread_counts[ti];
        auto r = bench_sync(launch_syncthreads, t, 0, N_SYNC, sm_clock_mhz);
        printf("%d,%.0f,%.1f,%.2f\n", t, r.total_cycles, r.per_sync_cycles, r.per_sync_ns);
    }
    printf("\n");

    printf("=== __syncwarp() cost (first warp only) ===\n");
    printf("threads,total_cycles,per_sync_cycles,per_sync_ns\n");
    for (int ti = 0; ti < n_threads; ti++) {
        int t = thread_counts[ti];
        auto r = bench_sync(launch_syncwarp, t, 0, N_SYNC, sm_clock_mhz);
        printf("%d,%.0f,%.1f,%.2f\n", t, r.total_cycles, r.per_sync_cycles, r.per_sync_ns);
    }
    printf("\n");

    // ============================================================
    // 3. __syncthreads() with varying shared memory
    // ============================================================
    printf("=== __syncthreads() + shared memory allocation ===\n");
    printf("threads,smem_kb,total_cycles,per_sync_cycles,per_sync_ns\n");

    int smem_sizes[] = {1024, 16*1024, 32*1024, 64*1024};
    const char* smem_labels[] = {"1", "16", "32", "64"};

    for (int ti = 0; ti < n_threads; ti++) {
        int t = thread_counts[ti];
        for (int si = 0; si < 4; si++) {
            int smem = smem_sizes[si];
            auto r = bench_sync(launch_syncthreads_smem, t, smem, N_SYNC, sm_clock_mhz);
            printf("%d,%s,%.0f,%.1f,%.2f\n", t, smem_labels[si], r.total_cycles, r.per_sync_cycles, r.per_sync_ns);
        }
    }
    printf("\n");

    // ============================================================
    // 4. __syncthreads() with actual smem work between syncs
    // ============================================================
    printf("=== __syncthreads() + smem work (write + sync) ===\n");
    printf("threads,total_cycles,per_iter_cycles,per_iter_ns,note\n");
    for (int ti = 0; ti < n_threads; ti++) {
        int t = thread_counts[ti];
        // Sync only
        auto r_sync = bench_sync(launch_syncthreads, t, 0, N_SYNC, sm_clock_mhz);
        // Sync + work
        auto r_work = bench_sync(launch_syncthreads_work, t, t * 4, N_SYNC, sm_clock_mhz);
        double work_overhead = r_work.per_sync_cycles - r_sync.per_sync_cycles;
        printf("%d,%.0f,%.1f,%.2f,sync_only=%.1f work_delta=%.1f cycles\n",
               t, r_work.total_cycles, r_work.per_sync_cycles, r_work.per_sync_ns,
               r_sync.per_sync_cycles, work_overhead);
    }
    printf("\n");

    // ============================================================
    // 5. Multi-CTA: does concurrent CTAs change sync cost?
    // ============================================================
    printf("=== Multi-CTA __syncthreads() (256 threads) ===\n");
    printf("num_ctas,per_sync_cycles,per_sync_ns\n");
    int cta_counts[] = {1, 4, 16, 64, 132, 264, 528};
    for (int ci = 0; ci < 7; ci++) {
        int nblocks = cta_counts[ci];
        long long* d_cycles;
        CHECK_CUDA(cudaMalloc(&d_cycles, 2 * sizeof(long long)));

        // Warmup
        for (int i = 0; i < 5; i++)
            k_syncthreads<<<nblocks, 256>>>(d_cycles, N_SYNC);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Measure
        double results[3];
        long long h;
        for (int run = 0; run < 3; run++) {
            CHECK_CUDA(cudaMemset(d_cycles, 0, 2 * sizeof(long long)));
            k_syncthreads<<<nblocks, 256>>>(d_cycles, N_SYNC);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(&h, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost));
            results[run] = (double)h;
        }
        for (int i = 0; i < 2; i++)
            for (int j = i+1; j < 3; j++)
                if (results[i] > results[j]) { double t = results[i]; results[i] = results[j]; results[j] = t; }
        double per_sync = results[1] / N_SYNC;
        double ns = per_sync / (sm_clock_mhz * 1e-3);
        printf("%d,%.1f,%.2f\n", nblocks, per_sync, ns);

        CHECK_CUDA(cudaFree(d_cycles));
    }
    printf("\n");

    printf("Done.\n");
    return 0;
}
