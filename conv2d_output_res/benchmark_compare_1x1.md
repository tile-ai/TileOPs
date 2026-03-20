# 1x1 Conv Benchmark 对比

## 数据来源

- 旧结果：[profile_run.log](/home/lyc/Project/TileOps-workspace2/conv2d_output_res/profile_run.log)
- 新结果：[profile_run_latest.log](/home/lyc/Project/TileOps-workspace2/conv2d_output_res/profile_run_latest.log)

## 对比范围

只关注 1x1 测试用例，主要看 `TFLOPS`。

## 分 Case 对比

| Case | DType | TileOps 旧 TFLOPS | TileOps 新 TFLOPS | 提升倍数 | Torch 新 TFLOPS | 当前相对 Torch |
| --- | --- | --- | --- | --- | --- | --- |
| resnet-1x1-fp16 | `fp16` | 18.14 | 47.75 | `2.63x` | 12.71 | `3.76x` |
| bottleneck-expand-1x1-fp16 | `fp16` | 7.18 | 34.34 | `4.78x` | 17.16 | `2.00x` |
| bottleneck-reduce-1x1-fp16 | `fp16` | 6.49 | 26.80 | `4.13x` | 17.36 | `1.54x` |
| late-stage-1x1-fp16 | `fp16` | 11.70 | 11.64 | `0.99x` | 5.98 | `1.95x` |
| classifier-1x1-fp16 | `fp16` | 1.71 | 4.99 | `2.92x` | 4.76 | `1.05x` |
| resnet-1x1-bf16 | `bf16` | 6.59 | 38.06 | `5.78x` | 12.73 | `2.99x` |

## 结论

- 这次改动对大多数 1x1 case 都有明显收益，尤其是中大规模 `fp16/bf16` shape。
- 提升最明显的是 `resnet-1x1-bf16`、`bottleneck-expand-1x1-fp16`、`bottleneck-reduce-1x1-fp16`，都超过了 `4x`。
- `resnet-1x1-fp16` 当前达到 `47.75 TFLOPS`，相对 Torch 是 `3.76x`。
- `late-stage-1x1-fp16` 基本持平，说明这类较小空间尺寸的 case 还没有从新搬运路径中获得明显增益。
- `classifier-1x1-fp16` 也有提升，但领先 Torch 的幅度已经很小，说明在极小 `hw` 场景下固定开销仍然比较显眼。

## 小结

把 `weight_shared` / `data_shared` 改成 `T.copy` 后，1x1 kernel 已经稳定 lower 到 **TMA + WGMMA**，并且在大多数 1x1 case 上转化成了实际性能收益。  
当前最值得继续优化的方向，不是再追求“是否上 TMA”，而是针对 **单 K tile、小 hw 场景** 继续压缩同步和 epilogue 固定开销。
