// HBM Saturation Cookbook
// Systematic experiment: what maximizes HBM bandwidth?
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

// --- Read ---
__global__ void k_read_scalar(const float* __restrict__ data, float* __restrict__ out, long long n) {
    float sum = 0.0f;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < n; i += stride) sum += data[i];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (threadIdx.x % 32 == 0) atomicAdd(out, sum);
}

__global__ void k_read_vec2(const float2* __restrict__ data, float* __restrict__ out, long long n) {
    float sum = 0.0f;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < n; i += stride) { float2 v = data[i]; sum += v.x + v.y; }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (threadIdx.x % 32 == 0) atomicAdd(out, sum);
}

__global__ void k_read_vec4(const float4* __restrict__ data, float* __restrict__ out, long long n) {
    float sum = 0.0f;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < n; i += stride) { float4 v = data[i]; sum += v.x + v.y + v.z + v.w; }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (threadIdx.x % 32 == 0) atomicAdd(out, sum);
}

// --- Write ---
__global__ void k_write_scalar(float* __restrict__ data, long long n, float val) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < n; i += stride) data[i] = val;
}

__global__ void k_write_vec4(float4* __restrict__ data, long long n, float val) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    float4 v = make_float4(val, val, val, val);
    for (long long i = idx; i < n; i += stride) data[i] = v;
}

// --- Copy ---
__global__ void k_copy_scalar(const float* __restrict__ src, float* __restrict__ dst, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < n; i += stride) dst[i] = src[i];
}

__global__ void k_copy_vec4(const float4* __restrict__ src, float4* __restrict__ dst, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < n; i += stride) dst[i] = src[i];
}

// --- Strided read ---
__global__ void k_read_strided(const float* __restrict__ data, float* __restrict__ out,
                               long long n, int stride_elems) {
    float sum = 0.0f;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long grid_stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; (long long)i * stride_elems < n; i += grid_stride)
        sum += data[i * stride_elems];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (threadIdx.x % 32 == 0) atomicAdd(out, sum);
}

// --- Misaligned read (scalar, offset by N bytes) ---
__global__ void k_read_offset(const float* __restrict__ data, float* __restrict__ out,
                              long long n, int offset_elems) {
    float sum = 0.0f;
    const float* ptr = data + offset_elems;
    long long actual_n = n - offset_elems;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < actual_n; i += stride) sum += ptr[i];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (threadIdx.x % 32 == 0) atomicAdd(out, sum);
}

// ============================================================
// Benchmark helper: 3 runs, return best and median
// ============================================================

struct BenchResult {
    float best_ms;
    float median_ms;
    float best_gbs;
    float median_gbs;
};

BenchResult bench3(std::function<void()> launch, long long total_bytes, int warmup = 10, int reps = 50) {
    std::vector<float> latencies;

    for (int run = 0; run < 3; run++) {
        // Warmup
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
    float median = latencies[1]; // middle of 3

    auto to_gbs = [&](float ms) -> float {
        return (ms > 0) ? (float)total_bytes / (ms * 1e6f) : 0.0f;
    };

    return {best, median, to_gbs(best), to_gbs(median)};
}

// ============================================================
// Environment snapshot
// ============================================================

void print_env_snapshot(const char* label) {
    printf("\n--- nvidia-smi snapshot [%s] ---\n", label);
    fflush(stdout);
    system("nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,"
           "clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,"
           "memory.used,memory.total --format=csv,noheader --id=0 2>/dev/null");
    // Also check for other processes
    printf("GPU processes: ");
    fflush(stdout);
    system("nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader --id=0 2>/dev/null");
    printf("---\n\n");
}

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
    printf("Each config: 3 runs x 50 reps, reporting best and median\n");

    print_env_snapshot("BEFORE");

    float *d_src, *d_dst, *d_out;
    CHECK_CUDA(cudaMalloc(&d_src, nbytes + 256));
    CHECK_CUDA(cudaMalloc(&d_dst, nbytes + 256));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_src, 1, nbytes));
    CHECK_CUDA(cudaMemset(d_dst, 0, nbytes));
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));

    // CSV header
    printf("theo_peak_gbs=%.1f\n", theo_peak_gbs);
    printf("op,vec_width,block_size,access,align,best_ms,median_ms,best_gbs,median_gbs,pct_of_theo_best\n");

    auto print_row = [theo_peak_gbs](const char* op, const char* vec, int bs, const char* access,
                        const char* align, BenchResult r) {
        printf("%s,%s,%d,%s,%s,%.4f,%.4f,%.2f,%.2f,%.1f%%\n",
               op, vec, bs, access, align,
               r.best_ms, r.median_ms, r.best_gbs, r.median_gbs,
               r.best_gbs / theo_peak_gbs * 100.0f);
    };

    // ============================================================
    // 1. Contiguous read: scalar vs vec2 vs vec4, block_size sweep
    // ============================================================
    int bss[] = {128, 256, 512};

    for (int bi = 0; bi < 3; bi++) {
        int bs = bss[bi];
        int nblocks = sm_count * (2048 / bs);

        // Read scalar
        {
            long long n = n_floats;
            auto launch = [&]() { k_read_scalar<<<nblocks, bs>>>(d_src, d_out, n); CHECK_LAST(); };
            auto r = bench3(launch, nbytes);
            print_row("read", "scalar", bs, "contiguous", "aligned", r);
        }
        // Read vec2
        {
            long long n = n_floats / 2;
            auto launch = [&]() { k_read_vec2<<<nblocks, bs>>>((float2*)d_src, d_out, n); CHECK_LAST(); };
            auto r = bench3(launch, nbytes);
            print_row("read", "vec2", bs, "contiguous", "aligned", r);
        }
        // Read vec4
        {
            long long n = n_floats / 4;
            auto launch = [&]() { k_read_vec4<<<nblocks, bs>>>((float4*)d_src, d_out, n); CHECK_LAST(); };
            auto r = bench3(launch, nbytes);
            print_row("read", "vec4", bs, "contiguous", "aligned", r);
        }
    }

    // ============================================================
    // 2. Contiguous write: scalar vs vec4, block_size sweep
    // ============================================================
    for (int bi = 0; bi < 3; bi++) {
        int bs = bss[bi];
        int nblocks = sm_count * (2048 / bs);

        {
            long long n = n_floats;
            auto launch = [&]() { k_write_scalar<<<nblocks, bs>>>(d_dst, n, 1.0f); CHECK_LAST(); };
            auto r = bench3(launch, nbytes);
            print_row("write", "scalar", bs, "contiguous", "aligned", r);
        }
        {
            long long n = n_floats / 4;
            auto launch = [&]() { k_write_vec4<<<nblocks, bs>>>((float4*)d_dst, n, 1.0f); CHECK_LAST(); };
            auto r = bench3(launch, nbytes);
            print_row("write", "vec4", bs, "contiguous", "aligned", r);
        }
    }

    // ============================================================
    // 3. Contiguous copy: scalar vs vec4, block_size sweep
    // ============================================================
    for (int bi = 0; bi < 3; bi++) {
        int bs = bss[bi];
        int nblocks = sm_count * (2048 / bs);

        {
            long long n = n_floats;
            auto launch = [&]() { k_copy_scalar<<<nblocks, bs>>>(d_src, d_dst, n); CHECK_LAST(); };
            auto r = bench3(launch, 2 * nbytes);
            print_row("copy", "scalar", bs, "contiguous", "aligned", r);
        }
        {
            long long n = n_floats / 4;
            auto launch = [&]() { k_copy_vec4<<<nblocks, bs>>>((float4*)d_src, (float4*)d_dst, n); CHECK_LAST(); };
            auto r = bench3(launch, 2 * nbytes);
            print_row("copy", "vec4", bs, "contiguous", "aligned", r);
        }
    }

    // ============================================================
    // 4. Strided read (scalar, 256 threads)
    // ============================================================
    {
        int bs = 256;
        int nblocks = sm_count * (2048 / bs);
        int stride_elems[] = {1, 8, 32, 128};
        const char* stride_names[] = {"contiguous", "stride-32B", "stride-128B", "stride-512B"};

        for (int si = 0; si < 4; si++) {
            long long n = n_floats;
            int se = stride_elems[si];
            long long accessed_bytes = (n / se) * 4;
            auto launch = [&]() { k_read_strided<<<nblocks, bs>>>(d_src, d_out, n, se); CHECK_LAST(); };
            auto r = bench3(launch, accessed_bytes);
            print_row("read", "scalar", bs, stride_names[si], "aligned", r);
        }
    }

    // ============================================================
    // 5. Misalignment test: scalar read, offset by 1/2/3 elements
    // ============================================================
    {
        int bs = 256;
        int nblocks = sm_count * (2048 / bs);
        long long n = n_floats;

        // Aligned baseline
        {
            auto launch = [&]() { k_read_offset<<<nblocks, bs>>>(d_src, d_out, n, 0); CHECK_LAST(); };
            auto r = bench3(launch, nbytes);
            print_row("read", "scalar", bs, "contiguous", "aligned", r);
        }
        // Offset by 1 float (4B)
        {
            long long accessed = (n - 1) * 4;
            auto launch = [&]() { k_read_offset<<<nblocks, bs>>>(d_src, d_out, n, 1); CHECK_LAST(); };
            auto r = bench3(launch, accessed);
            print_row("read", "scalar", bs, "contiguous", "misalign_4B", r);
        }
        // Offset by 3 floats (12B) — worst case for 16B alignment
        {
            long long accessed = (n - 3) * 4;
            auto launch = [&]() { k_read_offset<<<nblocks, bs>>>(d_src, d_out, n, 3); CHECK_LAST(); };
            auto r = bench3(launch, accessed);
            print_row("read", "scalar", bs, "contiguous", "misalign_12B", r);
        }
    }

    print_env_snapshot("AFTER");

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
