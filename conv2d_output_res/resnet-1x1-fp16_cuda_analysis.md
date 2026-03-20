# resnet-1x1-fp16 CUDA 代码分析

## 导出文件

- CUDA 源码：[resnet-1x1-fp16_tileops_kernel.cu](/home/lyc/Project/TileOps-workspace2/conv2d_output_res/resnet-1x1-fp16_tileops_kernel.cu)

## 结论

当前生成的 CUDA 代码已经使用了 **TMA + WGMMA**，不再是之前的 `cp.async + WGMMA` 路线。

## 判断依据

### 1. 存在 TMA descriptor 预取

- [resnet-1x1-fp16_tileops_kernel.cu](/home/lyc/Project/TileOps-workspace2/conv2d_output_res/resnet-1x1-fp16_tileops_kernel.cu#L22)
- [resnet-1x1-fp16_tileops_kernel.cu](/home/lyc/Project/TileOps-workspace2/conv2d_output_res/resnet-1x1-fp16_tileops_kernel.cu#L23)
- [resnet-1x1-fp16_tileops_kernel.cu](/home/lyc/Project/TileOps-workspace2/conv2d_output_res/resnet-1x1-fp16_tileops_kernel.cu#L24)

源码中出现了 `prefetch_tma_descriptor(...)`，这是 TMA lowering 的直接信号。

### 2. 存在 TMA load / store

- [resnet-1x1-fp16_tileops_kernel.cu](/home/lyc/Project/TileOps-workspace2/conv2d_output_res/resnet-1x1-fp16_tileops_kernel.cu#L39)
- [resnet-1x1-fp16_tileops_kernel.cu](/home/lyc/Project/TileOps-workspace2/conv2d_output_res/resnet-1x1-fp16_tileops_kernel.cu#L42)
- [resnet-1x1-fp16_tileops_kernel.cu](/home/lyc/Project/TileOps-workspace2/conv2d_output_res/resnet-1x1-fp16_tileops_kernel.cu#L83)

源码中明确出现了：

- `tl::tma_load(weight_desc, ...)`
- `tl::tma_load(x_desc, ...)`
- `tl::tma_store(out_desc, ...)`

这说明 `weight_shared`、`data_shared` 和输出写回都已经通过 TMA 完成。

### 3. 存在 mbarrier 协调

- [resnet-1x1-fp16_tileops_kernel.cu](/home/lyc/Project/TileOps-workspace2/conv2d_output_res/resnet-1x1-fp16_tileops_kernel.cu#L19)
- [resnet-1x1-fp16_tileops_kernel.cu](/home/lyc/Project/TileOps-workspace2/conv2d_output_res/resnet-1x1-fp16_tileops_kernel.cu#L35)
- [resnet-1x1-fp16_tileops_kernel.cu](/home/lyc/Project/TileOps-workspace2/conv2d_output_res/resnet-1x1-fp16_tileops_kernel.cu#L53)

源码中使用了 `mbarrier`、`expect_transaction`、`arrive`、`wait`，这也是 Hopper 上 TMA 搬运的典型同步方式。

### 4. 计算仍然是 WGMMA

当前修改只改变了搬运路径，主计算仍然使用 Hopper Tensor Core 路径，因此整体路线是 **TMA + WGMMA**。

## 为什么这次能 lower 到 TMA

这次 `conv2d_1x1_main` 做了两件关键修改：

- `weight_shared` 和 `data_shared` 从手写逐元素加载改成了 `T.copy`
- 输入/权重/输出改成了更适合张量搬运的扁平视图：
  - `x: [n, c_in, hw]`
  - `weight: [c_out, c_in]`
  - `out: [n, c_out, hw]`

这让 TileLang 可以把全局内存到 shared memory 的规则搬运识别成 TMA 模式。

## 还有哪些优化空间

### 1. 单 K tile 的 1x1 特化仍然值得做

`resnet-1x1-fp16` 的 `c_in=64`，而当前最优配置的 `block_k=64`，所以实际只有一次 K 分块。  
这种场景里，TMA 已经把搬运路径做对了，但 pipeline 和同步的固定开销仍然存在。

下一步可以继续尝试：

- 当 `c_in == block_k` 时做单 K tile 专用 kernel
- 压缩 pipeline 深度和同步开销
- 评估是否能进一步缩短 epilogue 路径

### 2. 输出写回还可以继续观察

当前已经把 epilogue 写到 `out_shared` 再做 `T.copy`。这一步能换来 TMA store，但也引入了一次 shared staging。  
对这种小 K、强算子特化的场景，可以继续比较两类路线：

- 保留 `out_shared + TMA store`
- 直接从寄存器 epilogue 写回

是否值得继续保留这一步，需要以 benchmark 为准。

### 3. autotune 搜索空间可以更聚焦 1x1

这次 1x1 case 的最优配置已经明显偏向更大的 `block_m/block_n`。  
后续可以继续缩小搜索空间，把重点放在：

- `block_k=64` 或 `128`
- 更大的 `block_m/block_n`
- 适合单 K tile 场景的较浅 `num_stages`

## 简短总结

当前 `resnet-1x1-fp16` 已经成功 lower 到 **TMA + WGMMA**。  
对 1x1 路径来说，这次修改方向是正确的，后续最值得继续挖的是单 K tile 专用化，以及对 epilogue / store 路径的进一步取舍。
