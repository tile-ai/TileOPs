// Bank Conflict Cookbook v2
// More aggressive measurement: uses shared memory read result as index
// to create true data dependency, preventing any optimization.
//
// Also measures actual throughput (ops/cycle) not just latency.

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

#define N_OPS 10000

// ============================================================
// Throughput-oriented: all threads do N_OPS accesses, measure total time
// Using pointer chasing to create true dependency
// ============================================================

// Read with stride — use dependent read (result feeds back)
template <int STRIDE>
__global__ void k_read_dep(long long* out_cycles, int n_ops) {
    __shared__ int smem[1024];
    // Init: create a pattern where reading creates bank conflicts
    for (int i = threadIdx.x; i < 1024; i += blockDim.x)
        smem[i] = (threadIdx.x * STRIDE) & 1023;
    __syncthreads();

    int idx = (threadIdx.x * STRIDE) & 1023;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        idx = smem[idx];  // true data dependency
        idx = (idx + threadIdx.x * STRIDE) & 1023;  // maintain stride pattern
    }
    long long end = clock64();

    if (threadIdx.x == 0) out_cycles[0] = end - start;
    // Prevent DCE
    if (idx == -1) out_cycles[1] = idx;
}

// Independent reads — measure throughput not latency
// All 32 threads do independent reads, no data dependency between iterations
template <int STRIDE>
__global__ void k_read_throughput(long long* out_cycles, int n_ops) {
    __shared__ float smem[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x)
        smem[i] = (float)i;
    __syncthreads();

    float sum = 0.0f;
    int base = (threadIdx.x * STRIDE) & 1023;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        // Read with stride pattern, vary index each iteration
        int idx = (base + i) & 1023;
        // But keep the STRIDE relationship between threads
        // Thread t reads: (t*STRIDE + i) & 1023
        sum += smem[(threadIdx.x * STRIDE + i) & 1023];
    }
    long long end = clock64();

    if (threadIdx.x == 0) out_cycles[0] = end - start;
    if (sum == -999.0f) out_cycles[1] = __float_as_int(sum);
}

// Write throughput
template <int STRIDE>
__global__ void k_write_throughput(long long* out_cycles, int n_ops) {
    __shared__ float smem[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x)
        smem[i] = 0.0f;
    __syncthreads();

    float val = (float)threadIdx.x;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        smem[(threadIdx.x * STRIDE + i) & 1023] = val;
        val += 0.001f;
        // Force write to complete before next iteration
        __threadfence_block();
    }
    long long end = clock64();

    if (threadIdx.x == 0) out_cycles[0] = end - start;
}

// ============================================================
// Tile patterns with real work
// ============================================================

// 32x32 tile, row-major, each thread reads a column
// This creates 32-way bank conflict at every access
__global__ void k_tile_conflict(long long* out_cycles, int n_ops) {
    __shared__ float tile[32][32];
    int tid = threadIdx.x;
    for (int r = 0; r < 32; r++) tile[r][tid] = (float)(r * 32 + tid);
    __syncthreads();

    float sum = 0.0f;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        // Each thread reads column tid, all rows
        // tile[r][tid] = address (r*32 + tid) * 4
        // For different r, same tid: bank = (r*32 + tid) % 32 = tid
        // All threads read from bank=tid → no conflict!
        // Wait, this is actually conflict-free. Let me fix:
        // For conflict: all threads should read SAME column
        int col = i & 31;
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            sum += tile[r][col];  // all threads read same column → same banks
        }
    }
    long long end = clock64();

    if (tid == 0) out_cycles[0] = end - start;
    if (sum == -999.0f) out_cycles[1] = __float_as_int(sum);
}

// Same but padded
__global__ void k_tile_conflict_padded(long long* out_cycles, int n_ops) {
    __shared__ float tile[32][33];  // +1 padding
    int tid = threadIdx.x;
    for (int r = 0; r < 32; r++) tile[r][tid] = (float)(r * 32 + tid);
    __syncthreads();

    float sum = 0.0f;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        int col = i & 31;
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            sum += tile[r][col];
        }
    }
    long long end = clock64();

    if (tid == 0) out_cycles[0] = end - start;
    if (sum == -999.0f) out_cycles[1] = __float_as_int(sum);
}

// Each thread reads different column (conflict-free baseline)
__global__ void k_tile_no_conflict(long long* out_cycles, int n_ops) {
    __shared__ float tile[32][32];
    int tid = threadIdx.x;
    for (int r = 0; r < 32; r++) tile[r][tid] = (float)(r * 32 + tid);
    __syncthreads();

    float sum = 0.0f;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            sum += tile[r][tid];  // each thread reads its own column → different banks
        }
    }
    long long end = clock64();

    if (tid == 0) out_cycles[0] = end - start;
    if (sum == -999.0f) out_cycles[1] = __float_as_int(sum);
}

// ============================================================
// Helpers
// ============================================================

typedef void (*kern_fn)(long long*, int);

struct BenchResult {
    double per_op_cycles;
    double per_op_ns;
};

BenchResult bench_kernel(kern_fn fn, long long* d_cycles, int n_ops, int threads, float sm_clock_mhz) {
    for (int w = 0; w < 3; w++) fn<<<1, threads>>>(d_cycles, n_ops);
    CHECK_CUDA(cudaDeviceSynchronize());

    double results[3];
    long long h;
    for (int r = 0; r < 3; r++) {
        CHECK_CUDA(cudaMemset(d_cycles, 0, 2 * sizeof(long long)));
        fn<<<1, threads>>>(d_cycles, n_ops);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&h, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost));
        results[r] = (double)h;
    }
    for (int i = 0; i < 2; i++)
        for (int j = i+1; j < 3; j++)
            if (results[i] > results[j]) { double t = results[i]; results[i] = results[j]; results[j] = t; }

    double median = results[1];
    double per_op = median / n_ops;
    double ns = per_op / (sm_clock_mhz * 1e-3);
    return {per_op, ns};
}

#define MAKE_LAUNCHER_READ(S) \
void launch_read_dep_##S(long long* d, int n) { k_read_dep<S><<<1, 32>>>(d, n); }

#define MAKE_LAUNCHER_RTHRU(S) \
void launch_read_thru_##S(long long* d, int n) { k_read_throughput<S><<<1, 32>>>(d, n); }

#define MAKE_LAUNCHER_WTHRU(S) \
void launch_write_thru_##S(long long* d, int n) { k_write_throughput<S><<<1, 32>>>(d, n); }

MAKE_LAUNCHER_READ(1) MAKE_LAUNCHER_READ(2) MAKE_LAUNCHER_READ(4)
MAKE_LAUNCHER_READ(8) MAKE_LAUNCHER_READ(16) MAKE_LAUNCHER_READ(32)

MAKE_LAUNCHER_RTHRU(1) MAKE_LAUNCHER_RTHRU(2) MAKE_LAUNCHER_RTHRU(3)
MAKE_LAUNCHER_RTHRU(4) MAKE_LAUNCHER_RTHRU(5) MAKE_LAUNCHER_RTHRU(7)
MAKE_LAUNCHER_RTHRU(8) MAKE_LAUNCHER_RTHRU(16) MAKE_LAUNCHER_RTHRU(32)

MAKE_LAUNCHER_WTHRU(1) MAKE_LAUNCHER_WTHRU(2) MAKE_LAUNCHER_WTHRU(4)
MAKE_LAUNCHER_WTHRU(8) MAKE_LAUNCHER_WTHRU(16) MAKE_LAUNCHER_WTHRU(32)

void launch_tile_conflict(long long* d, int n) { k_tile_conflict<<<1, 32>>>(d, n); }
void launch_tile_padded(long long* d, int n) { k_tile_conflict_padded<<<1, 32>>>(d, n); }
void launch_tile_no_conflict(long long* d, int n) { k_tile_no_conflict<<<1, 32>>>(d, n); }

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_clock_khz;
    cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, 0);
    float sm_clock_mhz = sm_clock_khz / 1000.0f;

    printf("GPU: %s | SM Clock: %.0f MHz\n", prop.name, sm_clock_mhz);
    printf("N_OPS = %d, 1 warp (32 threads)\n\n", N_OPS);

    long long* d_cycles;
    CHECK_CUDA(cudaMalloc(&d_cycles, 2 * sizeof(long long)));

    // ============================================================
    // 1. Dependent read (latency): stride sweep
    // ============================================================
    printf("=== Dependent smem read (latency): stride sweep ===\n");
    printf("stride,per_op_cycles,per_op_ns\n");
    struct { kern_fn fn; int s; } dep_tests[] = {
        {launch_read_dep_1,1},{launch_read_dep_2,2},{launch_read_dep_4,4},
        {launch_read_dep_8,8},{launch_read_dep_16,16},{launch_read_dep_32,32}
    };
    for (auto& t : dep_tests) {
        auto r = bench_kernel(t.fn, d_cycles, N_OPS, 32, sm_clock_mhz);
        printf("%d,%.1f,%.2f\n", t.s, r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    // ============================================================
    // 2. Independent read (throughput): stride sweep
    // ============================================================
    printf("=== Independent smem read (throughput): stride sweep ===\n");
    printf("stride,per_op_cycles,per_op_ns\n");
    struct { kern_fn fn; int s; } thru_tests[] = {
        {launch_read_thru_1,1},{launch_read_thru_2,2},{launch_read_thru_3,3},
        {launch_read_thru_4,4},{launch_read_thru_5,5},{launch_read_thru_7,7},
        {launch_read_thru_8,8},{launch_read_thru_16,16},{launch_read_thru_32,32}
    };
    for (auto& t : thru_tests) {
        auto r = bench_kernel(t.fn, d_cycles, N_OPS, 32, sm_clock_mhz);
        printf("%d,%.1f,%.2f\n", t.s, r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    // ============================================================
    // 3. Write throughput: stride sweep
    // ============================================================
    printf("=== Smem write (with threadfence_block): stride sweep ===\n");
    printf("stride,per_op_cycles,per_op_ns\n");
    struct { kern_fn fn; int s; } write_tests[] = {
        {launch_write_thru_1,1},{launch_write_thru_2,2},{launch_write_thru_4,4},
        {launch_write_thru_8,8},{launch_write_thru_16,16},{launch_write_thru_32,32}
    };
    for (auto& t : write_tests) {
        auto r = bench_kernel(t.fn, d_cycles, N_OPS, 32, sm_clock_mhz);
        printf("%d,%.1f,%.2f\n", t.s, r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    // ============================================================
    // 4. Tile patterns: conflict vs padded vs no-conflict
    // ============================================================
    printf("=== 32x32 tile: all threads read same col vs own col ===\n");
    printf("pattern,per_iter_cycles,per_iter_ns\n");
    {
        auto r1 = bench_kernel(launch_tile_no_conflict, d_cycles, N_OPS, 32, sm_clock_mhz);
        printf("no_conflict (each thread own col),%.1f,%.2f\n", r1.per_op_cycles, r1.per_op_ns);

        auto r2 = bench_kernel(launch_tile_conflict, d_cycles, N_OPS, 32, sm_clock_mhz);
        printf("conflict (all threads same col),%.1f,%.2f\n", r2.per_op_cycles, r2.per_op_ns);

        auto r3 = bench_kernel(launch_tile_padded, d_cycles, N_OPS, 32, sm_clock_mhz);
        printf("conflict_padded (33-wide, same col),%.1f,%.2f\n", r3.per_op_cycles, r3.per_op_ns);
    }
    printf("\n");

    CHECK_CUDA(cudaFree(d_cycles));
    printf("Done.\n");
    return 0;
}
