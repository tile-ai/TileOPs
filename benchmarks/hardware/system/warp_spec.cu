// Warp Specialization Cookbook
// Measures: unified CTA vs producer/consumer warp splitting
// with and without cp.async pipeline overlap.
//
// Compile: nvcc -O2 -arch=sm_90 -o warp_spec warp_spec.cu
// Usage:   ./warp_spec
//
// Key insight: compute results MUST be stored to global memory
// to prevent ptxas from eliminating the FMA loops entirely.

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

// ============================================================
// Compute: FMA chain that MUST store result to global memory
// so ptxas cannot eliminate it.
// ============================================================

// Inline FMA chain macro — writes result to dst[tid]
// The global store is the only reliable way to prevent ptxas
// from eliminating the entire FMA loop.
#define COMPUTE_TILE(buf_ptr, tile_elems, compute_depth, n_threads, my_id, dst) \
    do { \
        float _acc = 0.0f; \
        for (int _i = my_id; _i < tile_elems; _i += n_threads) { \
            float _val = (buf_ptr)[_i].x; \
            float _chain = _val; \
            for (int _d = 0; _d < compute_depth; _d++) { \
                _chain = fmaf(_chain, 1.0001f, _val); \
            } \
            _acc += _chain; \
        } \
        (dst)[my_id] = _acc; \
    } while(0)

// ============================================================
// Kernel 1: Unified, no pipeline
// ============================================================

__global__ void k_unified_nopipe(const float4* __restrict__ src, float* dst,
                                  int n_tiles, int tile_elems, int compute_depth,
                                  long long* out_cycles) {
    extern __shared__ float4 smem[];
    int tid = threadIdx.x;
    int bs = blockDim.x;

    long long start = clock64();

    for (int t = 0; t < n_tiles; t++) {
        int base = t * tile_elems;
        for (int i = tid; i < tile_elems; i += bs) {
            smem[i] = src[base + i];
        }
        __syncthreads();

        COMPUTE_TILE(smem, tile_elems, compute_depth, bs, tid, dst);
        __syncthreads();
    }

    long long end = clock64();
    if (tid == 0) out_cycles[0] = end - start;
}

// ============================================================
// Kernel 2: Unified, cp.async pipeline (double-buffer)
// ============================================================

__global__ void k_unified_pipe(const float4* __restrict__ src, float* dst,
                                int n_tiles, int tile_elems, int compute_depth,
                                long long* out_cycles) {
    extern __shared__ float4 smem_all[];
    float4* buf0 = smem_all;
    float4* buf1 = smem_all + tile_elems;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    long long start = clock64();

    // Prologue: load tile 0
    for (int i = tid; i < tile_elems; i += bs) {
        cp_async_f4(&buf0[i], &src[i]);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    for (int t = 0; t < n_tiles; t++) {
        float4* cur_buf = (t & 1) ? buf1 : buf0;
        float4* nxt_buf = (t & 1) ? buf0 : buf1;

        if (t + 1 < n_tiles) {
            int next_base = (t + 1) * tile_elems;
            for (int i = tid; i < tile_elems; i += bs) {
                cp_async_f4(&nxt_buf[i], &src[next_base + i]);
            }
            cp_async_commit();
        }

        COMPUTE_TILE(cur_buf, tile_elems, compute_depth, bs, tid, dst);

        if (t + 1 < n_tiles) {
            cp_async_wait_all();
        }
        __syncthreads();
    }

    long long end = clock64();
    if (tid == 0) out_cycles[0] = end - start;
}

// ============================================================
// Kernel 3: Specialized, no pipeline
// ============================================================

__global__ void k_specialized_nopipe(const float4* __restrict__ src, float* dst,
                                      int n_tiles, int tile_elems, int compute_depth,
                                      int producer_warps, long long* out_cycles) {
    extern __shared__ float4 smem[];
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int n_producers = producer_warps * 32;
    int n_consumers = blockDim.x - n_producers;
    bool is_producer = warp_id < producer_warps;

    long long start = clock64();

    for (int t = 0; t < n_tiles; t++) {
        int base = t * tile_elems;

        if (is_producer) {
            for (int i = tid; i < tile_elems; i += n_producers) {
                smem[i] = src[base + i];
            }
        }
        __syncthreads();

        if (!is_producer) {
            int cid = tid - n_producers;
            COMPUTE_TILE(smem, tile_elems, compute_depth, n_consumers, cid, dst);
        }
        __syncthreads();
    }

    long long end = clock64();
    if (tid == 0) out_cycles[0] = end - start;
}

// ============================================================
// Kernel 4: Specialized, cp.async pipeline (double-buffer)
// ============================================================

__global__ void k_specialized_pipe(const float4* __restrict__ src, float* dst,
                                    int n_tiles, int tile_elems, int compute_depth,
                                    int producer_warps, long long* out_cycles) {
    extern __shared__ float4 smem_all[];
    float4* buf0 = smem_all;
    float4* buf1 = smem_all + tile_elems;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int n_producers = producer_warps * 32;
    int n_consumers = blockDim.x - n_producers;
    bool is_producer = warp_id < producer_warps;

    long long start = clock64();

    // Prologue
    if (is_producer) {
        for (int i = tid; i < tile_elems; i += n_producers) {
            cp_async_f4(&buf0[i], &src[i]);
        }
        cp_async_commit();
        cp_async_wait_all();
    }
    __syncthreads();

    for (int t = 0; t < n_tiles; t++) {
        float4* cur_buf = (t & 1) ? buf1 : buf0;
        float4* nxt_buf = (t & 1) ? buf0 : buf1;

        if (is_producer && t + 1 < n_tiles) {
            int next_base = (t + 1) * tile_elems;
            for (int i = tid; i < tile_elems; i += n_producers) {
                cp_async_f4(&nxt_buf[i], &src[next_base + i]);
            }
            cp_async_commit();
        }

        if (!is_producer) {
            int cid = tid - n_producers;
            COMPUTE_TILE(cur_buf, tile_elems, compute_depth, n_consumers, cid, dst);
        }

        __syncthreads();

        if (is_producer && t + 1 < n_tiles) {
            cp_async_wait_all();
        }
        __syncthreads();
    }

    long long end = clock64();
    if (tid == 0) out_cycles[0] = end - start;
}

// ============================================================
// Init kernel
// ============================================================

__global__ void k_init_src(float4* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = 1.0f + (float)(i & 255) * 0.001f;
        dst[i] = make_float4(v, v, v, v);
    }
}

// ============================================================
// Benchmark helper
// ============================================================

struct BenchResult {
    double cycles;
    double time_us;
};

template <typename LaunchFn>
BenchResult bench_kernel(LaunchFn launch_fn, long long* d_cycles, float sm_clock_mhz) {
    for (int w = 0; w < 5; w++) launch_fn();
    CHECK_CUDA(cudaDeviceSynchronize());

    double results[5];
    long long h;
    for (int r = 0; r < 5; r++) {
        CHECK_CUDA(cudaMemset(d_cycles, 0, sizeof(long long)));
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
    return {median, time_us};
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
    printf("Shared memory per SM: %zu KB\n\n", prop.sharedMemPerMultiprocessor / 1024);

    int threads = 256;
    long long* d_cycles;
    CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(long long)));

    // Dst buffer for compute results (prevents DCE)
    float* d_dst;
    CHECK_CUDA(cudaMalloc(&d_dst, threads * sizeof(float)));

    int tile_elems_arr[]  = {256,  1024,  4096};
    const char* tile_labels[] = {"4KB", "16KB", "64KB"};
    int n_tile_sizes = 3;

    int compute_depths[] = {8, 64, 512, 2048};
    const char* compute_labels[] = {"light(8)", "med(64)", "heavy(512)", "extreme(2048)"};
    int n_computes = 4;

    int producer_warps_arr[] = {1, 2, 4};
    int n_pw = 3;

    int n_tiles = 64;

    printf("tile_size,compute,method,producer_warps,cycles,time_us,speedup_vs_unified_nopipe\n");

    for (int ti = 0; ti < n_tile_sizes; ti++) {
        int tile_elems = tile_elems_arr[ti];
        long long tile_bytes = (long long)tile_elems * sizeof(float4);
        long long total_bytes = (long long)n_tiles * tile_bytes;

        float4* d_src;
        CHECK_CUDA(cudaMalloc(&d_src, total_bytes));
        int total_elems = n_tiles * tile_elems;
        k_init_src<<<(total_elems + 255) / 256, 256>>>(d_src, total_elems);
        CHECK_CUDA(cudaDeviceSynchronize());

        int smem_single = tile_elems * sizeof(float4);
        int smem_double = smem_single * 2;

        for (int ci = 0; ci < n_computes; ci++) {
            int cdepth = compute_depths[ci];

            if (smem_double > 48 * 1024) {
                CHECK_CUDA(cudaFuncSetAttribute(k_unified_pipe,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_double));
                CHECK_CUDA(cudaFuncSetAttribute(k_specialized_pipe,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_double));
            }
            if (smem_single > 48 * 1024) {
                CHECK_CUDA(cudaFuncSetAttribute(k_unified_nopipe,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_single));
                CHECK_CUDA(cudaFuncSetAttribute(k_specialized_nopipe,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_single));
            }

            auto r_base = bench_kernel([&]() {
                k_unified_nopipe<<<1, threads, smem_single>>>(d_src, d_dst, n_tiles, tile_elems, cdepth, d_cycles);
            }, d_cycles, sm_clock_mhz);
            printf("%s,%s,unified_nopipe,0,%.0f,%.2f,1.00\n",
                   tile_labels[ti], compute_labels[ci], r_base.cycles, r_base.time_us);

            auto r_upipe = bench_kernel([&]() {
                k_unified_pipe<<<1, threads, smem_double>>>(d_src, d_dst, n_tiles, tile_elems, cdepth, d_cycles);
            }, d_cycles, sm_clock_mhz);
            printf("%s,%s,unified_pipe,0,%.0f,%.2f,%.2f\n",
                   tile_labels[ti], compute_labels[ci], r_upipe.cycles, r_upipe.time_us,
                   r_base.cycles / r_upipe.cycles);

            for (int pi = 0; pi < n_pw; pi++) {
                int pw = producer_warps_arr[pi];

                auto r_snp = bench_kernel([&]() {
                    k_specialized_nopipe<<<1, threads, smem_single>>>(
                        d_src, d_dst, n_tiles, tile_elems, cdepth, pw, d_cycles);
                }, d_cycles, sm_clock_mhz);
                printf("%s,%s,spec_nopipe,%d,%.0f,%.2f,%.2f\n",
                       tile_labels[ti], compute_labels[ci], pw, r_snp.cycles, r_snp.time_us,
                       r_base.cycles / r_snp.cycles);

                auto r_sp = bench_kernel([&]() {
                    k_specialized_pipe<<<1, threads, smem_double>>>(
                        d_src, d_dst, n_tiles, tile_elems, cdepth, pw, d_cycles);
                }, d_cycles, sm_clock_mhz);
                printf("%s,%s,spec_pipe,%d,%.0f,%.2f,%.2f\n",
                       tile_labels[ti], compute_labels[ci], pw, r_sp.cycles, r_sp.time_us,
                       r_base.cycles / r_sp.cycles);
            }
        }

        CHECK_CUDA(cudaFree(d_src));
    }

    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFree(d_cycles));
    printf("\nDone.\n");
    return 0;
}
