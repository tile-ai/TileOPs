# MoeGroupedGemm Persistent Kernel 设计文档

**日期**: 2026-05-01  
**状态**: 设计阶段  
**目标**: 将两阶段 nopad GEMM（独立 tile scheduler kernel + GEMM kernel）合并为单个 persistent kernel，消除额外的 kernel launch 开销，提升 decode/prefill 性能。

---

## 背景与动机

### 当前两阶段方案的问题

```
Phase 1: [tile_scheduler_kernel]  →  tile_expert_ids[], tile_row_offsets[], total_tiles
Phase 2: [gemm_kernel]            →  C[numel, N]
```

**开销来源**：
1. **双 kernel launch**：每次 forward 需要两次 `cudaLaunch`，单次约 5-10μs
2. **Thread-0 串行扫描**：scheduler kernel 中 thread-0 顺序计算 prefix sum，O(E) 串行
3. **Dead CTA 浪费**：GEMM kernel grid = `max_tiles × N_tiles`，约 33% CTA 是 dead CTA（无效后立即 exit）

基准测试表明：nopad vs padded 的性能差距主要来自 tile scheduler 开销，而非 GEMM 本身。

---

## 设计决策

| 问题 | 选择 | 理由 |
|------|------|------|
| Tile 分配策略 | Dynamic Atomics | MoE token 分布高度不均匀，dynamic 自然 load balance |
| cum_tiles 计算 | In-kernel 并行 warp scan | 无额外 kernel，CUTLASS 风格，O(log E) |
| Grid size | Grid = SM count（真正的 Persistent Kernel） | 消除 dead CTA，最大化 SM 利用率 |

---

## 架构设计

### 核心思路

```
[persistent_moe_gemm_kernel]  →  C[numel, N]
  - Grid = SM_count (H200: 132)
  - 每个 CTA 持续 while loop 领取 tile，直到所有 tile 完成
  - tile 分配通过 atomicAdd 全局计数器实现
  - cum_tiles 通过 in-kernel warp scan 计算
```

### 数据流

```
输入:
  A[numel, K]            - tight permuted hidden states
  B[E, N, K]             - expert weight matrices
  true_sizes[E]          - 每个 expert 的 token count
  true_offsets[E]        - 每个 expert 在 A 中的起始偏移

内部（SMEM）:
  s_cum[E+1]             - prefix sum of ceildiv(sizes[e], block_m)
  tile_counter[1]        - 全局 atomicAdd 计数器（GMEM）

输出:
  C[numel, N]            - GEMM 结果
```

### Kernel 内部执行流程

```
每个 CTA 的执行流程:

1. [初始化阶段] in-kernel warp scan 计算 cum_tiles（每个 CTA 独立计算自己的私有 SMEM）
   - Thread 0 首先写入 s_cum[0] = 0（前缀和边界条件）
   - Warp 0（thread 0-31）负责：
     - 每个 thread i 处理 expert[r*32+i]，读取 ceildiv(sizes[e], block_m)
     - 越界 expert（e >= E）按 size=0 处理（OOB 安全）
     - 通过 __shfl_up_sync warp scan 计算 prefix sum
     - 存入 SMEM: s_cum[r*32 + i + 1] = val[i] + carry
   - __syncthreads()：等待同一 CTA 内 warp 0 完成全部 SMEM 写入（包括 s_cum[0]=0 的初始化及所有轮次的 scan 写入），确保其他 warp 读取 s_cum 时的 happens-before 关系成立
     （注意：__syncthreads() 仅同步同一 CTA 内的线程，每个 CTA 各自独立完成此步骤）
   - total_tiles = s_cum[E] × N_tiles  （thread-local 读取）

2. [Tile 领取循环] while True:
   flat_tile_id = atomicAdd(tile_counter_gmem, 1)
   if flat_tile_id >= total_tiles: break

3. [解码 tile 坐标 + Binary Search] O(log E)
   m_tile_id = flat_tile_id // N_tiles   ← 先提取 M 方向 tile index
   n_tile_id = flat_tile_id % N_tiles    ← 再提取 N 方向 tile index
   expert_id, row_in_expert = binary_search(s_cum, m_tile_id)

4. [GEMM Tile]
   m_start = true_offsets[expert_id] + row_in_expert * block_m
   n_start = n_tile_id * block_n
   执行 T.Pipelined K-loop with T.copy (TMA) + T.gemm + C 写回

5. 回到步骤 2，领取下一个 tile
```

---

## 关键技术细节

### In-kernel Warp Scan（cum_tiles 计算）

参考 CUTLASS `GroupedProblemVisitor` 中的 warp scan 实现，关键在于**正确处理轮间进位（carry）**：

```
E = 256 个 expert
每个 warp = 32 threads（Warp 0 负责 scan）
需要 ceil(256/32) = 8 轮 warp scan

carry = 0  ← 轮间进位，初始为 0

Round r (处理 expert[r*32 .. r*32+31]):
  1. thread i 读取 ceildiv(sizes[r*32 + i], block_m) → val[i]
  2. warp-level inclusive prefix sum（__shfl_up_sync）:
       for shift in [1, 2, 4, 8, 16]:
           v = __shfl_up_sync(val, shift)
           if lane_idx >= shift: val += v
  3. s_cum[r*32 + i + 1] = val[i] + carry    （写入 SMEM）
  4. carry = __shfl_sync(val, 31) + carry      （thread 31 的结果即为本轮总和，传入下一轮）

最终 s_cum[E] = total_tiles（即 carry 的最终值）
```

**关键**：`carry` 变量在每轮结束时从 thread 31 广播并累加，确保 `s_cum` 是全局正确的前缀和。

总计算步骤：8 rounds × 5 步 shuffle + 进位传播 = ~50 次 shuffle 操作，开销约 ~1μs。

### GMEM tile_counter 初始化

需要在 kernel 启动前将 `tile_counter` 置 0。

**采用方案 A**：Python `forward()` 中 `tile_counter.fill_(0)`，开销 ~1μs，最简单。

> **stream 约束**：`tile_counter.fill_(0)` 必须与 kernel launch 在**同一 CUDA stream** 上排队（PyTorch 默认流即满足此条件），确保 fill 在 kernel 启动前完成。禁止将两者提交到不同 stream，否则仍存在竞争条件。

> **方案 B（kernel 内重置）已排除**：若 thread 0 在 warp scan 后写 `tile_counter=0`，其他 CTA 可能在写入完成前已读取计数器并开始领取 tile，产生竞争条件（race condition）。正确修复需要全局 barrier（grid-level sync），在 TileLang 中不可行。方案 A 完全避免此问题。

### TMA 路径保留

和当前 nopad 一样，K 对齐时走 `T.copy`（TMA-eligible），K 不对齐时走 `T.if_then_else`。Persistent kernel 不影响这个判断。

### N 维度处理

当前 nopad 方案的 grid 是 2D：`(max_tiles, N_tiles)`，每个 (bx, by) CTA 固定处理一个 (M-tile, N-tile) 对。

Persistent kernel 的 grid 是 1D：`SM_count`。每个 CTA 在 while loop 内：
- 领取一个 `flat_tile_id`
- 将其映射到 `(m_tile_id, n_tile_id)` = `(flat_tile_id // N_tiles, flat_tile_id % N_tiles)`

这样 total_tiles = `s_cum[E] × N_tiles`，N 维度自然融入 flat tile 空间。

---

## 实现计划

### 新文件

`tileops/kernels/moe/moe_grouped_gemm_persistent.py`

保持与 `moe_grouped_gemm_nopad.py` 相同的外部接口：
```python
class MoeGroupedGemmPersistentKernel(Kernel):
    def forward(self, A, B, true_sizes, true_offsets) -> torch.Tensor
```

### 接口变化

- 新增参数：`sm_count: int`（可从 `torch.cuda.get_device_properties` 获取）
- 去掉：独立的 `_tile_scheduler_kernel`
- 新增：`tile_counter` tensor（persistent across loop，每次 forward 前 fill_(0)）

### Kernel 参数

```python
def _persistent_moe_gemm_kernel(numel, num_experts, N, K, dtype, sm_count):
    ...
    @T.prim_func
    def _gemm_main(
        A, B, C,
        true_sizes,     # [E] int32
        true_offsets,   # [E] int32
        tile_counter,   # [1] int32  (GMEM atomicAdd target)
    ): ...
```

---

## 性能预期

| 场景 | 当前 nopad | Persistent 预期 | 改善来源 |
|------|-----------|----------------|---------|
| E=128 decode | 1.959ms | ~1.7-1.8ms | -scheduler kernel, -dead CTA |
| E=128 prefill | 5.346ms | ~4.8-5.0ms | -scheduler kernel, better LB |
| E=256 decode | 3.660ms | ~3.2-3.4ms | -scheduler kernel, -dead CTA |
| E=256 prefill | 6.336ms | ~5.5-6.0ms | -scheduler kernel, better LB |

目标：消除当前 nopad 相对 padded 的 2-6% 差距，并在 skewed 分布下（实际推理）优于 padded。

---

## 风险与注意事项

1. **TileLang `atomicAdd` 支持**：需要确认 TileLang 是否有 `T.atomic_add` 或需要用 `T.call_extern`
2. **Warp scan in TileLang**：`__shfl_up_sync` 可能需要 `T.call_extern` 调用
3. **SMEM layout 分区**：`s_cum[E+1]` 需要在整个 kernel 生命周期内保持有效（binary search 阶段仍需读取），不能被 A/B pipeline buffer 覆盖。具体布局：
   - SMEM 低地址：`s_cum[E+1]`，大小 = `(E+1) × 4` 字节（E=256 时约 1028 B）
   - SMEM 高地址：A/B double buffer，大小 = `num_stages × (block_m + block_n) × block_k × sizeof(dtype)`
   - 两者不重叠；warp scan 完成后 `s_cum` 保持只读，无并发写冲突
   - 实现时需通过 `T.alloc_shared` 声明顺序或 offset 控制确保两者不重叠
4. **Backward 兼容**：新 kernel 作为独立类，不替换现有 `MoeGroupedGemmNopadKernel`，通过 Kernel 参数选择

---

## 参考

- CUTLASS `GroupedProblemVisitor` (`grouped_problem_visitor.h`)
- CUTLASS `GroupScheduleMode::kDeviceOnly` warp scan 实现
- 当前 `tileops/kernels/moe/moe_grouped_gemm_nopad.py`
- vllm MoE grouped GEMM persistent kernel
