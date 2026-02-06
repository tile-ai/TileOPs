# TileOPs Architecture Refactoring RFC

**Feature Name:** TileOPs Architecture Refactoring
**Start Date:** 2025-02-06
**RFC PR:** 
**GitHub Issue:** 

## 1. Summary

This proposal outlines a comprehensive refactoring plan for TileOPs to reduce development barriers, optimize architecture design, and improve development efficiency. The refactoring aims to enable contributors of different backgrounds and skill levels to add new operators more quickly while maintaining code quality and performance.

| Metric | Current State | Target State |
|--------|---------------|--------------|
| File Count | 50+ | 15-20 |
| Files Required for New Operator | 7 | 2-3 |
| Test Execution Time | Long (CI blocker) | Fast (80% reduction) |
| Code Redundancy | ~60% | ~10% |

## 2. Motivation

### 2.1 Four-Layer Abstraction Redundancy

The current architecture suffers from excessive abstraction layers that add complexity without providing business value:

```
Current Architecture: Layer → Function → Op → Kernel (Only Kernel has business value)
```

| Layer | File Location | Avg Lines of Code | Usage Frequency |
|-------|---------------|-------------------|------------------|
| Layer | top/layers/*.py | ~60 lines | Very Low (pure forwarding) |
| Function | top/functions/*.py | ~120 lines | Low (glue code) |
| Op | top/ops/*.py | ~40 lines | Medium (thin wrapper) |
| Kernel | top/kernels/**/*.py | ~3000 lines/dir | High (core value) |

This four-layer abstraction creates significant overhead:
- **Increased cognitive load**: Developers must understand multiple abstraction levels
- **Code duplication**: Similar patterns repeated across layers
- **Maintenance burden**: Changes require updates in multiple files
- **Slow onboarding**: New contributors face steep learning curves

### 2.2 Test and Benchmark Coupling

The current Benchmark class violates the Single Responsibility Principle by handling six distinct responsibilities:

1. Correctness verification
2. Performance profiling
3. Input data generation
4. Reference implementation
5. Test configuration
6. Result reporting

This coupling leads to:
- Tests that run benchmarks (incorrect responsibility)
- Slow test execution times
- Complex and hard-to-maintain test code
- Difficulty in separating performance testing from correctness testing

## 3. Refactor Overview

### 3.1 Core Goals

The refactoring aims to achieve three primary objectives:

1. **Reduce Development Barriers**: Simplify the process of adding new operators
2. **Optimize Architecture**: Eliminate unnecessary abstraction layers
3. **Improve Development Efficiency**: Faster development cycles and CI times

### 3.2 Proposed File Structure

```
Before → After

top/                          top/
├── layers/  (delete)          ops/                   
├── functions/ (delete)          │   ├── __init__.py
├── ops/                         │   ├── attention.py
└── kernels/ (keep)              │   ├── deepseek_mla.py
                                 │   ├── deepseek_nsa.py
                                 │   ├── gemm.py
                                 │   └── quantization.py
                                 ├── kernels/ (unchanged)
                                 └── experimental/ (new)
                                  
tests/                        tests/
├── ops/ (refactor)            ├── base.py
├── functions/ (refactor)      ├── deepseek_mla/
└── layers/ (refactor)         ├── deepseek_nsa/
                               └── experimental/ (new)
                              
benchmarks/                   benchmarks/
├── ... (separate)            ├── base.py
                              ├── deepseek_mla/
                              └── microbenchmarks/ (new)
```

### 3.3 Directory Description

| Directory | Responsibility | Status |
|-----------|----------------|--------|
| `ops/` | Stable operator implementations | Core |
| `kernels/` | High-performance kernel implementations | Core |
| `experimental/` | Experimental operator verification | New |
| `tests/` | Correctness tests | Core |
| `tests/experimental/` | Experimental operator tests | New |
| `benchmarks/` | Performance benchmarks | Core |
| `benchmarks/microbenchmarks/` | GPU micro-benchmarks | New |

### 3.4 Experimental Directory

Following the TensorFlow `tf.contrib` design philosophy, this directory stores:

**Design Goals:**
- Validate immature kernel ideas
- Rapid prototyping
- Low-risk experimentation
- Gradual promotion to `ops/` upon maturation

**Directory Structure:**
```
top/experimental/
├── __init__.py
├── attention/
│   ├── flash_attention_v1.py  # v1 version validation
│   └── ring_attention.py      # Ring attention experiment
├── reduction/
│   ├── tree_reduction.py      # Tree-based reduction
│   └── warp_reduction.py      # Warp-level reduction
└── utils/
    ├── prototype_utils.py     # Prototyping utilities
    └── memory_patterns.py       # Memory pattern analysis
```

**Design Principles:**
1. **No API stability guarantee**: APIs in experimental may change at any time
2. **Risk documentation**: Each module must clearly indicate experimental nature
3. **Independent test suite**: Tests in `tests/experimental/` run separately
4. **CI pipeline separation**: Experimental tests run independently without blocking main CI
5. **Promotion mechanism**: Mature experimental code can be promoted to `ops/`

### 3.5 Microbenchmarks Directory

GPU micro-benchmarks measuring memory bandwidth, compute throughput, Tensor Core performance, kernel launch overhead, and other hardware characteristics to support performance optimization and auto-tuning.

```
benchmarks/microbenchmarks/
├── memory/           # Bandwidth and latency tests
├── compute/         # Compute throughput tests
├── tensor_core/     # Tensor Core performance tests
├── kernel/          # Launch overhead and occupancy tests
└── cuda/            # Stream and overlap tests
```

## 4. Reforactor Details

### 4.1 Test Framework Design

**Design Principles:**
- **Tests**: Only verify correctness, do not execute benchmarks
- **Benchmarks**: Only perform performance testing, do not include correctness verification
- **Data Generation**: Independent reusable components

#### 4.1.1 Before Refactoring

```python
# tests/ops/test_deepseek_mla_decode.py
# Problem: Test includes benchmark logic

from benchmarks import MultiHeadLatentAttentionDecodeBenchmark
from top.ops import MultiHeadLatentAttentionDecodeWithKVCacheOp

def test_mla_decode():
    op = MultiHeadLatentAttentionDecodeWithKVCacheOp(...)
    benchmark = MultiHeadLatentAttentionDecodeBenchmark(...)
    
    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs, atol=3e-4, rtol=1e-5)  # ✓ Correct
    benchmark.profile(op, *inputs)  # ✗ Should not execute in tests
```

#### 4.1.2 After Refactoring (Separated Design)

```python
# tests/base.py

"""
Test base class providing unified interface for all operator tests.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple


class TestBase(ABC):
    @abstractmethod
    def gen_inputs(self) -> Tuple[Any, ...]:
        pass
    
    @abstractmethod
    def ref_program(self, *inputs) -> Any:
        pass


# tests/test_deepseek_mla.py

"""
MLA Decode correctness test.
"""
import torch
from top.ops import mla_decode_with_kvcache
from tests.base import TestBase


class MLADecodeTest(TestBase):
    def __init__(
        self,
        batch: int = 32,
        heads: int = 128,
        kv_head_num: int = 1,
        seqlen_kv: int = 8192,
        dim: int = 512,
        pe_dim: int = 64,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        self.batch = batch
        self.heads = heads
        self.kv_head_num = kv_head_num
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.pe_dim = pe_dim
        self.dtype = dtype
        self.device = device
    
    def gen_inputs(self) -> tuple:
        """Generate test input data"""
        Q = torch.randn(
            self.batch, self.heads, self.dim,
            device=self.device, dtype=self.dtype
        )
        Q_pe = torch.randn(
            self.batch, self.heads, self.pe_dim,
            device=self.device, dtype=self.dtype
        )
        K = torch.randn(
            self.batch, self.seqlen_kv, self.kv_head_num, self.dim,
            device=self.device, dtype=self.dtype
        )
        K_pe = torch.randn(
            self.batch, self.seqlen_kv, self.kv_head_num, self.pe_dim,
            device=self.device, dtype=self.dtype
        )
        return (Q, Q_pe, K, K_pe)
    
    def ref_program(
        self,
        Q: torch.Tensor,
        Q_pe: torch.Tensor,
        K: torch.Tensor,
        K_pe: torch.Tensor
    ) -> torch.Tensor:
        """
        PyTorch reference implementation.
        """
        from einops import rearrange
        from torch.nn import functional as F
        from torch import einsum
        
        dim = Q.shape[-1]
        pe_dim = Q_pe.shape[-1]
        num_head_groups = Q.shape[1] // K.shape[2]
        scale = (dim + pe_dim)**0.5
        
        Q = rearrange(Q, 'b (h g) d -> b g h d', g=num_head_groups)
        Q_pe = rearrange(Q_pe, 'b (h g) d -> b g h d', g=num_head_groups)
        KV = rearrange(K, 'b n h d -> b h n d')
        K_pe = rearrange(K_pe, 'b n h d -> b h n d')
        
        query = torch.concat([Q, Q_pe], dim=-1)
        key = torch.concat([KV, K_pe], dim=-1)
        
        scores = einsum(query, key, 'b g h d, b h s d -> b g h s')
        attention = F.softmax(scores / scale, dim=-1)
        
        out = einsum(attention, KV, 'b g h s, b h s d -> b g h d')
        
        return rearrange(out, 'b g h d -> b (h g) d')


def test_mla_decode(
    batch: int = 32,
    heads: int = 128,
    kv_head_num: int = 1,
    seqlen_kv: int = 8192,
    dim: int = 512,
    pe_dim: int = 64,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    atol: float = 3e-4,
    rtol: float = 1e-5
):
    """MLA Decode correctness test"""
    test = MLADecodeTest(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, dtype, device)
    inputs = test.gen_inputs()
    
    output = mla_decode_with_kvcache(*inputs)
    ref_output = test.ref_program(*inputs)
    
    assert output.shape == ref_output.shape
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol)
```

#### 4.1.3 Independent Benchmark Design (After Refactoring)

```python
# benchmarks/profile/base.py

"""
Benchmark base class providing unified interface for all operator performance tests.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
from tilelang.profiler import do_bench
from benchmarks.utils import calculate_tflops, calculate_bandwidth


class BenchmarkBase(ABC):
    def __init__(self, op, test_instance, warmup: int = 100, rep: int = 100):
        self.op = op
        self.test_instance = test_instance
        self.warmup = warmup
        self.rep = rep
    
    @abstractmethod
    def calculate_flops(self, *inputs) -> float:
        pass
    
    @abstractmethod
    def calculate_memory(self, *inputs) -> float:
        pass
    
    def profile(self, backend: str = "tilelang") -> Dict[str, float]:
        # 1. Generate input data
        inputs = self.test_instance.gen_inputs()
        
        # 2. Warmup
        for _ in range(self.warmup):
            _ = self.op(*inputs)
        
        # 3. Synchronize
        torch.cuda.synchronize()
        
        # 4. Timing
        latency_ms = do_bench(
            lambda: self.op(*inputs),
            warmup=10,  # internal warmup
            rep=self.rep,
            backend=backend
        )
        
        # 5. Synchronize
        torch.cuda.synchronize()
        
        # 6. Calculate performance metrics
        flops = self.calculate_flops(*inputs)
        memory = self.calculate_memory(*inputs)
        
        tflops = calculate_tflops(flops, latency_ms)
        bandwidth = calculate_bandwidth(memory, latency_ms)
        
        return {
            "latency_ms": latency_ms,
            "tflops": tflops,
            "bandwidth_GB_s": bandwidth
        }


# benchmarks/profile/profile_deepseek_mla.py

"""
MLA Decode performance test.

Responsibility: Only perform performance testing, do not include correctness verification.
"""

import pytest
from typing import Dict, Any
import torch

from top.ops import mla_decode_with_kvcache
from benchmarks.profile.base import BenchmarkBase
from tests.test_correctness.test_deepseek_mla import MLADecodeTest


class MLADecodeProfiler(BenchmarkBase):
    """
    MLA Decode performance test class.
    """
    
    def calculate_flops(self, *inputs) -> float:
        """Calculate FLOPs for MLA decode"""
        Q, Q_pe, K, K_pe = inputs
        batch, heads, dim = Q.shape
        _, seqlen_kv, kv_head_num, _ = K.shape
        pe_dim = Q_pe.shape[-1]
        
        qk_flops = 2 * batch * heads * seqlen_kv * (dim + pe_dim)
        pv_flops = 2 * batch * heads * seqlen_kv * dim
        return qk_flops + pv_flops
    
    def calculate_memory(self, *inputs) -> float:
        """Calculate memory access"""
        Q, Q_pe, K, K_pe = inputs
        batch, heads, dim = Q.shape
        _, seqlen_kv, kv_head_num, _ = K.shape
        pe_dim = Q_pe.shape[-1]
        dtype_size = Q.element_size()
        
        q_size = batch * heads * dim * dtype_size
        q_pe_size = batch * heads * pe_dim * dtype_size
        k_size = batch * seqlen_kv * kv_head_num * dim * dtype_size
        k_pe_size = batch * seqlen_kv * kv_head_num * pe_dim * dtype_size
        
        return q_size + q_pe_size + k_size + k_pe_size


# pytest parameterized configuration
@pytest.mark.parametrize("batch", [16, 32, 64])
@pytest.mark.parametrize("heads", [64, 128, 256])
@pytest.mark.parametrize("seqlen_kv", [4096, 8192])
@pytest.mark.parametrize("dim", [256, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mla_decode_benchmark(batch: int, heads: int, seqlen_kv: int, dim: int, dtype: torch.dtype):
    """
    MLA Decode performance test.
    
    Use pytest parameterized tests for different configurations.
    """
    # Create test instance
    test = MLADecodeTest(
        batch=batch,
        heads=heads,
        kv_head_num=1,
        seqlen_kv=seqlen_kv,
        dim=dim,
        pe_dim=64,
        dtype=dtype
    )
    
    # Create profiler
    profiler = MLADecodeProfiler(
        op=mla_decode_with_kvcache,
        test_instance=test
    )
    
    # Run performance test
    result = profiler.profile()
    
    # Output performance results
    print(f"\nMLA Decode Performance ({batch=}, {heads=}, {seqlen_kv=}, {dim=}, {dtype=}):")
    print(f"  Latency: {result['latency_ms']:.2f} ms")
    print(f"  TFlops:  {result['tflops']:.2f} TF/s")
    print(f"  Bandwidth: {result['bandwidth_GB_s']:.2f} GB/s")


# Keep command line running for independent execution
if __name__ == "__main__":
    pytest.main([__file__])
```

