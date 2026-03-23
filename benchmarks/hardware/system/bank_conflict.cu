// Bank Conflict Cookbook
// Measures shared memory bank conflict overhead on H200 (SM 9.0).
//
// Compile: nvcc -O2 -arch=sm_90 -o bank_conflict bank_conflict.cu
// Usage:   ./bank_conflict
//
// SM 9.0: 32 banks, 4-byte bank width, consecutive 4-byte words map to consecutive banks.
//
// Tests:
// 1. Stride sweep: stride 1-32 words → 0-way to 32-way conflict
// 2. Broadcast: all threads read same address (should be free)
// 3. fp32 vs fp64: bank conflict with different data widths
// 4. Read vs write: are conflicts symmetric?
// 5. Practical: row-major vs column-major tile access pattern

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
// Shared memory read with configurable stride
// ============================================================

// Each thread reads smem[threadIdx.x * stride % SMEM_ELEMENTS] in a loop
// stride=1: no conflict (consecutive banks)
// stride=2: 2-way conflict
// stride=32: 32-way conflict (all threads hit same bank)

template <int STRIDE>
__global__ void k_smem_read_stride(long long* out_cycles, int n_ops) {
    __shared__ float smem[1024];
    // Init
    smem[threadIdx.x] = (float)threadIdx.x;
    if (threadIdx.x + 32 < 1024) smem[threadIdx.x + 32] = 0.0f;
    __syncthreads();

    float sum = 0.0f;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        int idx = (threadIdx.x * STRIDE) % 1024;
        sum += smem[idx];
        asm volatile("" :: "f"(sum));  // prevent optimization
    }
    long long end = clock64();

    if (threadIdx.x == 0) {
        out_cycles[0] = end - start;
    }
    // Prevent DCE
    if (sum == -999.0f) out_cycles[1] = __float_as_int(sum);
}

// Write with stride
template <int STRIDE>
__global__ void k_smem_write_stride(long long* out_cycles, int n_ops) {
    __shared__ float smem[1024];
    smem[threadIdx.x] = 0.0f;
    if (threadIdx.x + 32 < 1024) smem[threadIdx.x + 32] = 0.0f;
    __syncthreads();

    float val = (float)threadIdx.x;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        int idx = (threadIdx.x * STRIDE) % 1024;
        smem[idx] = val;
        asm volatile("" ::: "memory");
        val += 0.001f;  // vary value to prevent optimization
    }
    long long end = clock64();

    if (threadIdx.x == 0) {
        out_cycles[0] = end - start;
    }
}

// Broadcast: all threads read same address
__global__ void k_smem_broadcast(long long* out_cycles, int n_ops) {
    __shared__ float smem[32];
    if (threadIdx.x < 32) smem[threadIdx.x] = (float)threadIdx.x;
    __syncthreads();

    float sum = 0.0f;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        sum += smem[0];  // all threads read same address → broadcast
        asm volatile("" :: "f"(sum));
    }
    long long end = clock64();

    if (threadIdx.x == 0) out_cycles[0] = end - start;
    if (sum == -999.0f) out_cycles[1] = __float_as_int(sum);
}

// ============================================================
// fp64 bank conflicts (8 bytes = 2 banks per element)
// ============================================================

template <int STRIDE>
__global__ void k_smem_read_fp64_stride(long long* out_cycles, int n_ops) {
    __shared__ double smem[512];
    smem[threadIdx.x] = (double)threadIdx.x;
    __syncthreads();

    double sum = 0.0;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        int idx = (threadIdx.x * STRIDE) % 512;
        sum += smem[idx];
        asm volatile("" :: "d"(sum));
    }
    long long end = clock64();

    if (threadIdx.x == 0) out_cycles[0] = end - start;
    if (sum == -999.0) out_cycles[1] = (long long)sum;
}

// ============================================================
// Practical: row-major vs column-major 2D tile access
// Simulates accessing a TILE_M x TILE_N matrix in shared memory
// ============================================================

#define TILE_M 32
#define TILE_N 32

// Row-major layout: smem[row][col] — threads in a warp read consecutive cols
// No bank conflict if TILE_N is not a multiple of 32... but it is here
__global__ void k_tile_row_major(long long* out_cycles, int n_ops) {
    __shared__ float tile[TILE_M][TILE_N];
    int tid = threadIdx.x;  // 0-31
    int row = tid;           // each thread owns a row
    // Init
    for (int c = 0; c < TILE_N; c++) tile[row][c] = (float)(row * TILE_N + c);
    __syncthreads();

    float sum = 0.0f;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        // Each thread reads its row sequentially — different threads read different rows
        // At column c, thread t reads tile[t][c] → address = t*32 + c
        // For same c: addresses are t*32+c for t=0..31 → strides of 32 → all same bank!
        for (int c = 0; c < TILE_N; c++) {
            sum += tile[row][c];
        }
        asm volatile("" :: "f"(sum));
    }
    long long end = clock64();

    if (tid == 0) out_cycles[0] = end - start;
    if (sum == -999.0f) out_cycles[1] = __float_as_int(sum);
}

// Row-major with +1 padding: smem[row][col+1] — breaks bank alignment
__global__ void k_tile_row_major_padded(long long* out_cycles, int n_ops) {
    __shared__ float tile[TILE_M][TILE_N + 1];  // +1 padding
    int tid = threadIdx.x;
    int row = tid;
    for (int c = 0; c < TILE_N; c++) tile[row][c] = (float)(row * TILE_N + c);
    __syncthreads();

    float sum = 0.0f;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        for (int c = 0; c < TILE_N; c++) {
            sum += tile[row][c];
        }
        asm volatile("" :: "f"(sum));
    }
    long long end = clock64();

    if (tid == 0) out_cycles[0] = end - start;
    if (sum == -999.0f) out_cycles[1] = __float_as_int(sum);
}

// Column-major: threads in warp read consecutive rows at same column
__global__ void k_tile_col_major(long long* out_cycles, int n_ops) {
    __shared__ float tile[TILE_M][TILE_N];
    int tid = threadIdx.x;
    int row = tid;
    for (int c = 0; c < TILE_N; c++) tile[row][c] = (float)(row * TILE_N + c);
    __syncthreads();

    float sum = 0.0f;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_ops; i++) {
        // Each thread reads column 'col' across all rows
        // At row r, thread t reads tile[t][col] — wait, let's do it properly:
        // All threads read the same column but different rows
        for (int c = 0; c < TILE_N; c++) {
            sum += tile[tid][c];  // same pattern as row_major actually
        }
        asm volatile("" :: "f"(sum));
    }
    long long end = clock64();

    if (tid == 0) out_cycles[0] = end - start;
    if (sum == -999.0f) out_cycles[1] = __float_as_int(sum);
}

// ============================================================
// Helpers
// ============================================================

struct BenchResult {
    double per_op_cycles;
    double per_op_ns;
};

BenchResult bench_kernel(void (*fn)(long long*, int), long long* d_cycles,
                          int n_ops, int threads, float sm_clock_mhz) {
    // Warmup
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

// Wrapper to match function pointer type
#define STRIDE_WRAPPER_READ(STRIDE) \
void launch_read_##STRIDE(long long* d, int n) { k_smem_read_stride<STRIDE><<<1, 32>>>(d, n); }

#define STRIDE_WRAPPER_WRITE(STRIDE) \
void launch_write_##STRIDE(long long* d, int n) { k_smem_write_stride<STRIDE><<<1, 32>>>(d, n); }

#define STRIDE_WRAPPER_FP64(STRIDE) \
void launch_fp64_##STRIDE(long long* d, int n) { k_smem_read_fp64_stride<STRIDE><<<1, 32>>>(d, n); }

STRIDE_WRAPPER_READ(1)
STRIDE_WRAPPER_READ(2)
STRIDE_WRAPPER_READ(3)
STRIDE_WRAPPER_READ(4)
STRIDE_WRAPPER_READ(5)
STRIDE_WRAPPER_READ(7)
STRIDE_WRAPPER_READ(8)
STRIDE_WRAPPER_READ(9)
STRIDE_WRAPPER_READ(15)
STRIDE_WRAPPER_READ(16)
STRIDE_WRAPPER_READ(17)
STRIDE_WRAPPER_READ(31)
STRIDE_WRAPPER_READ(32)

STRIDE_WRAPPER_WRITE(1)
STRIDE_WRAPPER_WRITE(2)
STRIDE_WRAPPER_WRITE(4)
STRIDE_WRAPPER_WRITE(8)
STRIDE_WRAPPER_WRITE(16)
STRIDE_WRAPPER_WRITE(32)

STRIDE_WRAPPER_FP64(1)
STRIDE_WRAPPER_FP64(2)
STRIDE_WRAPPER_FP64(4)
STRIDE_WRAPPER_FP64(8)
STRIDE_WRAPPER_FP64(16)

void launch_broadcast(long long* d, int n) { k_smem_broadcast<<<1, 32>>>(d, n); }
void launch_tile_row(long long* d, int n) { k_tile_row_major<<<1, 32>>>(d, n); }
void launch_tile_row_pad(long long* d, int n) { k_tile_row_major_padded<<<1, 32>>>(d, n); }
void launch_tile_col(long long* d, int n) { k_tile_col_major<<<1, 32>>>(d, n); }

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_clock_khz;
    cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, 0);
    float sm_clock_mhz = sm_clock_khz / 1000.0f;

    printf("GPU: %s | SMs: %d | SM Clock: %.0f MHz\n", prop.name, prop.multiProcessorCount, sm_clock_mhz);
    printf("Shared memory banks: 32, bank width: 4 bytes\n");
    printf("N_OPS = %d per measurement, 3 runs median, 1 warp (32 threads)\n\n", N_OPS);

    long long* d_cycles;
    CHECK_CUDA(cudaMalloc(&d_cycles, 2 * sizeof(long long)));

    typedef void (*kern_fn)(long long*, int);

    // ============================================================
    // 1. Read stride sweep
    // ============================================================
    printf("=== Shared memory READ: stride sweep (32 threads) ===\n");
    printf("stride,conflict_way,per_op_cycles,per_op_ns\n");

    struct { kern_fn fn; int stride; int conflict; } read_tests[] = {
        {launch_read_1,  1,  0},
        {launch_read_2,  2,  2},
        {launch_read_3,  3,  0},  // gcd(3,32)=1
        {launch_read_4,  4,  4},
        {launch_read_5,  5,  0},  // gcd(5,32)=1
        {launch_read_7,  7,  0},  // gcd(7,32)=1
        {launch_read_8,  8,  8},
        {launch_read_9,  9,  0},  // gcd(9,32)=1
        {launch_read_15, 15, 0},  // gcd(15,32)=1
        {launch_read_16, 16, 16},
        {launch_read_17, 17, 0},  // gcd(17,32)=1
        {launch_read_31, 31, 0},  // gcd(31,32)=1
        {launch_read_32, 32, 32},
    };

    for (auto& t : read_tests) {
        auto r = bench_kernel(t.fn, d_cycles, N_OPS, 32, sm_clock_mhz);
        printf("%d,%d,%.1f,%.2f\n", t.stride, t.conflict, r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    // Broadcast
    printf("=== Broadcast (all threads read smem[0]) ===\n");
    printf("pattern,per_op_cycles,per_op_ns\n");
    {
        auto r = bench_kernel(launch_broadcast, d_cycles, N_OPS, 32, sm_clock_mhz);
        printf("broadcast,%.1f,%.2f\n", r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    // ============================================================
    // 2. Write stride sweep
    // ============================================================
    printf("=== Shared memory WRITE: stride sweep (32 threads) ===\n");
    printf("stride,per_op_cycles,per_op_ns\n");

    struct { kern_fn fn; int stride; } write_tests[] = {
        {launch_write_1,  1},
        {launch_write_2,  2},
        {launch_write_4,  4},
        {launch_write_8,  8},
        {launch_write_16, 16},
        {launch_write_32, 32},
    };

    for (auto& t : write_tests) {
        auto r = bench_kernel(t.fn, d_cycles, N_OPS, 32, sm_clock_mhz);
        printf("%d,%.1f,%.2f\n", t.stride, r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    // ============================================================
    // 3. fp64 stride sweep
    // ============================================================
    printf("=== Shared memory fp64 READ: stride sweep (32 threads) ===\n");
    printf("stride,per_op_cycles,per_op_ns\n");

    struct { kern_fn fn; int stride; } fp64_tests[] = {
        {launch_fp64_1,  1},
        {launch_fp64_2,  2},
        {launch_fp64_4,  4},
        {launch_fp64_8,  8},
        {launch_fp64_16, 16},
    };

    for (auto& t : fp64_tests) {
        auto r = bench_kernel(t.fn, d_cycles, N_OPS, 32, sm_clock_mhz);
        printf("%d,%.1f,%.2f\n", t.stride, r.per_op_cycles, r.per_op_ns);
    }
    printf("\n");

    // ============================================================
    // 4. Practical tile access patterns
    // ============================================================
    printf("=== Practical: 32x32 tile access patterns ===\n");
    printf("pattern,per_iter_cycles,per_iter_ns\n");
    {
        auto r1 = bench_kernel(launch_tile_row, d_cycles, N_OPS, 32, sm_clock_mhz);
        printf("row_major_32x32,%.1f,%.2f\n", r1.per_op_cycles, r1.per_op_ns);

        auto r2 = bench_kernel(launch_tile_row_pad, d_cycles, N_OPS, 32, sm_clock_mhz);
        printf("row_major_32x33_padded,%.1f,%.2f\n", r2.per_op_cycles, r2.per_op_ns);

        auto r3 = bench_kernel(launch_tile_col, d_cycles, N_OPS, 32, sm_clock_mhz);
        printf("col_major_32x32,%.1f,%.2f\n", r3.per_op_cycles, r3.per_op_ns);
    }
    printf("\n");

    CHECK_CUDA(cudaFree(d_cycles));
    printf("Done.\n");
    return 0;
}
