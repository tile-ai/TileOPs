"""System-level CUDA microbenchmarks.

These benchmarks require nvcc with -arch=sm_90 (or appropriate target).
Compile each .cu file individually:

    nvcc -O2 -arch=sm_90 -o <name> <name>.cu

See individual file headers for specific compile and usage instructions.
"""
