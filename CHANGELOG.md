# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Flash Attention v2 kernels for Ampere GPUs (SM_80/SM_86)
- Flash Attention v3 kernels for Hopper GPUs (SM_90)
- DeepSeek Multi-Head Latent Attention (MLA) decode kernel
- DeepSeek Sparse Attention (DSA) decode kernel
- DeepSeek Native Sparse Attention (NSA) kernels (forward, top-k, window sliding, mean pooling)
- Multi-Head Attention (MHA) and Group Query Attention (GQA) forward/backward/decode ops
- Paged KV-cache support for MHA/GQA decode
- MatMul (GEMM/GEMV) kernel with auto-tuning support
- Grouped GEMM kernel
- 1D C2C FFT kernels (radix and LUT variants)
- Manifold-Constrained Hyper-Connection (MHC) pre/post kernels
- FP8 quantization and lighting indexer kernels
- Top-k selector kernel
- 2-layer hierarchical API: Kernel → Op
- Auto-tuning infrastructure for kernel parameter search
- Test framework (`TestBase`, `FixtureBase`) and benchmark framework (`BenchmarkBase`, `BenchmarkReport`)
- CI with pre-commit linting, packaging, and GPU-based test runs
