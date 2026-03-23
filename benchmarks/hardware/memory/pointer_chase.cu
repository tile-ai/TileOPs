// Pointer-chase kernel for memory latency measurement.
// Single-thread dependent load chain: next = data[next]
// Compile: nvcc -O0 -o pointer_chase pointer_chase.cu
// Usage: ./pointer_chase <working_set_KB> [num_chases]
// If num_chases is 0 or omitted, uses n_elements (full cycle traversal)

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void pointer_chase_kernel(
    const int* __restrict__ data,
    int* __restrict__ output,
    int num_chases
) {
    // Single thread to serialize all memory accesses
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx = 0;
        #pragma unroll 1
        for (int i = 0; i < num_chases; i++) {
            idx = __ldg(&data[idx]);
        }
        output[0] = idx;  // prevent dead code elimination
    }
}

// Create a random cyclic permutation
void make_chase_array(int* host_data, int n) {
    // Initialize as identity
    for (int i = 0; i < n; i++) host_data[i] = i;

    // Fisher-Yates shuffle
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = host_data[i];
        host_data[i] = host_data[j];
        host_data[j] = tmp;
    }

    // Convert permutation to single cycle
    // perm[i] -> perm[(i+1) % n]
    int* cycle = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        cycle[host_data[i]] = host_data[(i + 1) % n];
    }
    memcpy(host_data, cycle, n * sizeof(int));
    free(cycle);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <working_set_KB> [num_chases]\n", argv[0]);
        return 1;
    }

    int ws_kb = atoi(argv[1]);
    int n_elements = (ws_kb * 1024) / sizeof(int);
    if (n_elements < 256) n_elements = 256;
    // Default: traverse full cycle to exercise entire working set
    int num_chases = (argc >= 3) ? atoi(argv[2]) : n_elements;
    if (num_chases <= 0) num_chases = n_elements;

    // Setup
    int* h_data = (int*)malloc(n_elements * sizeof(int));
    make_chase_array(h_data, n_elements);

    int* d_data;
    int* d_output;
    cudaMalloc(&d_data, n_elements * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));
    cudaMemcpy(d_data, h_data, n_elements * sizeof(int), cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 10; i++) {
        pointer_chase_kernel<<<1, 1>>>(d_data, d_output, num_chases);
    }
    cudaDeviceSynchronize();

    // Benchmark with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int reps = 50;
    cudaEventRecord(start);
    for (int i = 0; i < reps; i++) {
        pointer_chase_kernel<<<1, 1>>>(d_data, d_output, num_chases);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / reps;
    float ns_per_load = (avg_ms * 1e6f) / num_chases;

    // Output: ws_kb,num_chases,avg_ms,ns_per_load
    printf("%d,%d,%.6f,%.2f\n", ws_kb, num_chases, avg_ms, ns_per_load);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_output);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
