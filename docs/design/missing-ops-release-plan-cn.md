# TileOPs 缺失功能发布计划

本文面向 TileOPs 当前 manifest 驱动架构，整理相对早期 operator library 计划仍需补齐的发布路径。

执行口径上，本文把 Wave 0 / Wave 1 视为进入正式功能补齐前的 **baseline reset**：

- Wave 0 先把已有代码面中成熟的 op 补入 manifest。
- Wave 1 再把当前 manifest 中的 spec-only entry 转成 implemented，或明确 defer。
- 从 Wave 2 开始，后续计划默认 Wave 0 / Wave 1 已经合入；仍未完成的项目只计作 manifest debt / spec-only debt，不重复计入功能缺口。

覆盖口径上分两层：

1. **Manifest debt**：代码或测试已经存在，但还没有进入 `tileops/manifest/*.yaml`，因此不能算正式发布面。
1. **功能缺口**：即使把现有代码都补进 manifest，相对早期计划仍然缺失或明显弱化的能力。

当前设计以 manifest 为准。若早期计划的命名、接口或 family 切分与当前 manifest 架构冲突，发布时应优先采用当前 manifest 的接口风格。

## 当前执行前基线

基线 commit：`upstream/main@c49ffd0d`。

当前 manifest 按 `PYTHONPATH=. python scripts/manifest_stats.py` 统计已有 142 个 op 条目，其中 130 个 `implemented`，12 个 `spec-only`。

| Family | Implemented | Spec-only | 备注 |
| --- | ---: | ---: | --- |
| elementwise | 66 | 5 | 主体完成，fused gated / generative position ops 未完成 |
| reduction | 19 | 0 | 基本完整 |
| scan | 2 | 0 | `cumsum` / `cumprod` 从 reduction 拆出 |
| normalization | 12 | 0 | 强于早期计划，但缺 `qk_norm` |
| convolution | 6 | 0 | 只有基础 conv 和 bias variants |
| pool | 3 | 6 | AvgPool implemented，MaxPool spec-only |
| attention | 14 | 0 | GQA/MHA serving 面较强，MLA/NSA/DSA 矩阵不完整 |
| moe | 7 | 0 | Generic/routed MoE 面强于早期抽象 |
| gemm | 1 | 0 | 只有统一 `GemmOp` |
| linear_attention | 0 | 1 | 只有 `GatedDeltaNetPrefillFwdOp` spec-only |
| quantize | 0 | 0 | 无正式 manifest family |
| sampling | 0 | 0 | 无正式 manifest family |
| ssm | 0 | 0 | 无正式 manifest family |

## 清账后基线假设

后续 Wave 2+ 的功能缺口分析默认以下前置事项已经完成：

- 已有成熟代码面的 op 已按当前 manifest 风格发布，例如 RoPE、Dropout、narrow FP8 quant、Grouped GEMM、TopK selector、已有 Linear Attention、Mamba2 / SSD 等。
- 当前 spec-only entry 已经转正或被明确 defer，例如 fused gated elementwise、Alibi、Sinusoidal、MaxPool variants、`GatedDeltaNetPrefillFwdOp`。
- #1648 的 input-inferred API policy 已用于新补 manifest entry：可由 forward input tensor metadata 推导的 shape-only 参数不再作为新的 release-facing 必填构造参数。

因此，后文只总结 Wave 0 / Wave 1 清账之后仍然存在的功能性缺口。

## 发布原则

1. **Manifest first**：任何 release-facing op 必须先有 manifest 条目，包含 signature、dtype、shape rules、workloads、roofline、source。
1. **代码存在不等于发布**：已有 `tileops/ops/*`、benchmark 或 test 的能力，必须补齐 manifest 后才进入正式覆盖统计。
1. **先收敛接口，再优化性能**：新增 family 先做到 correctness + benchmark + roofline，再进入性能专项。
1. **优先补真实 serving / training 路径**：GQA decode、MoE、Mamba2/SSD、Quant、Sampling 这类直接影响 LLM serving 的路径优先于长尾 CNN ops。
1. **避免复刻旧命名矩阵**：例如 GEMM 可以继续采用统一 `GemmOp` + dtype/layout/workload 参数化，而不必硬拆成旧计划的每个名字。
1. **聚合 release-facing op 名称**：旧计划里的 nominal op 若只是在 dtype、rank、granularity、kernel schedule、decode/prefill mode 或 workload regime 上不同，优先聚合为同一个 op surface，通过输入 tensor metadata、semantic params、workload descriptor 或 kernel-mode 分发。只有输入/输出 contract、随机性 contract、state mutation 语义或外部可见 layout 明显不同，才拆成独立 op。
1. **区分普通量化与低精度 compute core**：Quantize family 和 elementwise fp8 不涉及特殊 swizzle，可较早推进；FP8/lowbit GEMM、FP8 attention Tensor Core path、packed weight、scale swizzle 这类 compute-core 问题放到最后集中处理。

## 命名聚合口径

后续 Wave 2+ 的 op 命名默认遵循以下聚合规则：

这套规则主要约束新增或尚未固化的发布面。已经进入 manifest 的 implemented / spec-only entry 仍以当前 manifest 为准；若要把既有 spec-only 名称合并成参数化 op，需要作为 manifest contract change 单独 review。

这里需要明确区分两件事：

- **功能补齐**：早期计划中列出的能力仍需要逐项覆盖、测试和 benchmark。比如 grouped conv、top-p、RetNet recurrence、FP8 block-scaled GEMM 仍然都是需要验收的功能点。
- **op 层接口聚合**：这些功能点不一定要一一变成独立 release-facing op 名。若它们共享同一输入/输出 contract，可以通过参数、tensor metadata、workload descriptor、mode 或 kernel dispatch 聚合到同一个 op surface。

因此，后文表格里的“早期能力”用于追踪功能覆盖，“推荐发布形态”用于描述 manifest/API 层的聚合后接口。功能没有因为接口聚合而消失；只是验收单位和对外 op 名称不必一一相同。

| 领域 | 聚合原则 | 例外 |
| --- | --- | --- |
| Conv / Pool | grouped、depthwise、dilated 优先作为 `Conv2dFwdOp` 的参数/workload；rank-specific 名称是否保留跟随当前 manifest 风格 | ConvTranspose、AdaptivePool 若输入/输出 rank contract 分离明显，可保留 1d/2d/3d 名称 |
| Quantize | per-tensor / per-channel / per-block 由 scale tensor shape 或 layout descriptor 表达；dtype 由 input/output dtype combo 表达 | SmoothQuant、cast-transpose、packed NF4 若输出 contract 明显不同，可单独 op |
| Sampling | temperature、top-k、top-p、min-p 作为同一 logits-processing op 的参数组合 | 随机采样和 speculative verification 有独立随机性/验证 contract，可单独 op |
| Attention | MQA 由 GQA `H_kv == 1` 表达；varlen/chunked/sliding-window 优先作为 prefill/decode mode 或 metadata contract | Paged KV、MLA latent cache、NSA/DSA sparse metadata 输入不同，可保留独立 op family |
| Linear Attention / SSM | 算法名是主要 op surface；chunkwise/recurrence/prefill/decode 优先作为 mode 或 state contract | in-place state mutation 或 backward contract 明显不同，可拆独立 op |
| GEMM | 参考 #1653：gemv、small-batch、scale granularity 是 kernel-mode / scale-shape contract | BMM、outer、low-bit packed、2:4 sparse 输入 contract 不同，应拆 sibling sub-issue |

## 前置 Wave 0：Manifest Debt 清账

目标：把已有代码面中成熟度足够的 op 变成正式 manifest 发布面。这个 wave 不主张写大量新 kernel，重点是补契约、workload、roofline、测试和 benchmark。

接口要求：新增 manifest entry 应对齐 [#1648](https://github.com/tile-ai/TileOPs/issues/1648) 的 input-inferred API 方向。凡是可以从 forward 输入 tensor metadata 推导的 shape-only 参数，例如 `batch`、`seq_len`、`heads`、`dim`、`N_total`、`M/N`、空间尺寸等，优先在 `forward()` 推导并纳入 lazy kernel cache key；不要作为新的必填构造参数固化到 release-facing contract。语义参数、调度参数和无法不读 CUDA tensor 值就推导的 host-visible metadata 仍保持显式。

候选范围：

| Area | 候选代码面 | 发布动作 |
| --- | --- | --- |
| RoPE | `tileops/ops/rope.py` | 增加 `rope.yaml` 或并入 elementwise/attention manifest，明确 neox、non-neox、llama31、yarn、longrope variants |
| Dropout | `tileops/ops/dropout.py` | 增加 generative/random op manifest，明确 seed/offset/training/p 语义 |
| FP8 quant | `tileops/ops/fp8_quant.py` | 若定位是 KV/attention 辅助量化，先作为 narrow FP8 quant entry；不要宣称覆盖完整 quant family |
| Grouped GEMM | `tileops/ops/grouped_gemm.py` | 补 `GroupedGemmOp` manifest，区分通用 grouped GEMM 与 MoE-specific grouped GEMM |
| TopK selector | `tileops/ops/topk_selector.py` | 作为 selector / sampling 前置能力发布，不等同完整 sampling |
| Linear attention code | GLA / DeltaNet / GatedDeltaNet | 把已有 chunkwise / recurrence 代码按当前成熟度拆入 `linear_attention.yaml` |
| Mamba2 / SSD | `mamba2_fwd.py`、`ssd_*` | 增加 `ssm.yaml`，先发布 Mamba2 SSD prefill/e2e/decode 已有面 |
| Engram / FFT / MHC | 相关 ops/workloads | 单独评估是否属于早期计划外扩展，不阻塞旧计划缺口收敛 |

验收标准：

- 每个新增 manifest entry 通过 `scripts/validate_manifest.py`。
- 每个 entry 有至少一个 smoke workload 和一个代表性 benchmark workload。
- `source.test`、`source.bench`、`source.kernel_map` 完整。
- 对已有 op，不改变用户可见接口；如必须改变，以 manifest contract 作为 breaking-change 记录。
- Manifest 的 `signature.params`、`static_dims`、`shape_rules`、workloads 和 roofline 写法与 #1648 的 input-inferred API policy 对齐。

## 前置 Wave 1：Spec-only 收尾

目标：把当前 manifest 中已经有契约但未实现的条目全部推进到 `implemented`。

范围：

| Family | Op |
| --- | --- |
| elementwise | `SiluAndMulFwdOp` |
| elementwise | `GeluAndMulFwdOp` |
| elementwise | `GeluTanhAndMulFwdOp` |
| elementwise | `AlibiFwdOp` |
| elementwise | `SinusoidalFwdOp` |
| pool | `MaxPool1dFwdOp` |
| pool | `MaxPool1dIndicesFwdOp` |
| pool | `MaxPool2dFwdOp` |
| pool | `MaxPool2dIndicesFwdOp` |
| pool | `MaxPool3dFwdOp` |
| pool | `MaxPool3dIndicesFwdOp` |
| linear_attention | `GatedDeltaNetPrefillFwdOp` |

验收标准：

- 全部 spec-only entry 改为 `implemented`。
- Fused gated elementwise 覆盖 fp16/bf16/fp32 的 correctness；fp8 dtype 支持可作为 elementwise 后续增强，不阻塞低精度 compute-core wave。
- MaxPool variants 对齐 PyTorch 输出语义，indices variant 明确 index dtype 和 flatten 规则。
- `GatedDeltaNetPrefillFwdOp` 需要明确它是 prefill/chunkwise 还是兼容旧计划的 chunkwise 命名。
- 转正时同步审查 constructor 中的冗余 shape-only 参数；若可由输入 tensor metadata 推导，应按 #1648 改成 optional compatibility path 或 preferred input-inferred API。

## Upstream Issue 提案：前置 Wave 0 / Wave 1

建议先向 upstream 提一个 umbrella issue，争取 maintainer 对 manifest debt 和 spec-only 清账的共识，再按 family 拆子 issue / PR。这个 issue 是正式功能补齐计划的前置条件，不代表后续 Wave 2+ 的功能缺口已经关闭。

建议标题：

```text
[Tracking][Manifest] Promote existing op surface and spec-only entries to release-facing manifest coverage
```

建议正文：

```md
## Summary

This issue tracks the baseline-reset work that should land before the later functional gap-closure waves under the manifest-driven design.

The goal is not to introduce risky new low-level compute-core contracts or to close every historical functional gap. Instead, this issue focuses on:

1. promoting mature code-present ops into `tileops/manifest/*.yaml`;
2. turning existing `status: spec-only` manifest entries into `status: implemented`;
3. aligning promoted or implemented ops with the input-inferred API direction from #1648 where applicable.

## Scope

### Wave 0: Manifest debt cleanup

Code, tests, or benchmarks already exist, but the ops are not yet part of the release-facing manifest surface.

Candidate areas:

- RoPE variants
- Dropout
- narrow FP8 quant / KV helper quant
- Grouped GEMM
- TopK selector
- existing Linear Attention ops
- Mamba2 / SSD ops
- other mature code-present ops such as Engram / FFT / MHC, if maintainers agree they should become release-facing

Each promoted op should include:

- manifest entry
- source metadata
- workloads
- roofline metadata
- tests
- manifest-driven benchmark coverage

### Wave 1: Spec-only cleanup

Current manifest entries that should either become implemented or be explicitly deferred:

- `SiluAndMulFwdOp`
- `GeluAndMulFwdOp`
- `GeluTanhAndMulFwdOp`
- `AlibiFwdOp`
- `SinusoidalFwdOp`
- `MaxPool1dFwdOp`
- `MaxPool1dIndicesFwdOp`
- `MaxPool2dFwdOp`
- `MaxPool2dIndicesFwdOp`
- `MaxPool3dFwdOp`
- `MaxPool3dIndicesFwdOp`
- `GatedDeltaNetPrefillFwdOp`

## API policy

This work should align with #1648:

- infer shape-only metadata from forward input tensors when possible;
- keep semantic/config parameters explicit, such as `top_k`, `dim`, `eps`, routing policy, layout flags, and similar choices;
- do not infer scheduling metadata by reading CUDA tensor values;
- update manifest `signature.params`, `static_dims`, `shape_rules`, workloads, roofline bindings, tests, benchmarks, and in-repo call sites together.

## Non-goals

This issue does not attempt to finalize high-risk low-precision compute-core contracts, including:

- FP8 GEMM
- block-scaled GEMM
- low-bit GEMM
- special swizzled layouts
- FP8 attention Tensor Core paths

Quantize/dequantize and elementwise fp8 can be tracked separately because they do not require special swizzle contracts.

## Acceptance criteria

- All promoted ops pass `scripts/validate_manifest.py`.
- Each implemented manifest entry has correctness tests and benchmark coverage.
- Existing `spec-only` entries are either implemented or explicitly deferred with rationale.
- The manifest remains the source of truth for release-facing operator coverage.
- New or promoted public APIs avoid required constructor shape parameters when the values are derivable from forward tensor metadata.
```

## Wave 2：低风险功能补全

目标：在 Wave 0 / Wave 1 已经合入的清账后基线上，补齐早期计划中接口清晰、实现风险中等以下的功能缺口。

### Normalization

缺口：

- `qk_norm`

聚合后的发布面：

- `QKNormFwdOp`
- 如实际模型需要，可拆 `QKNormRMSFwdOp` / `QKNormLayerFwdOp`，但优先保持一个 manifest entry，用参数区分 normalization type。

验收标准：

- 覆盖常见 attention head layout：`(B, S, H, D)` 和必要的 packed varlen 形态。
- 与 PyTorch reference 或内部 decomposed reference 对齐。

### Conv & Pooling

缺口：

- `conv_transpose1d`
- `conv_transpose2d`
- `depthwise_conv2d`
- `grouped_conv2d`
- `dilated_conv2d`
- `adaptive_avg_pool2d`
- `adaptive_max_pool2d`

聚合后的发布面：

下表只描述 op 层接口如何聚合；左侧每个早期能力仍是独立功能验收点，需要对应 workload / correctness case。

| 早期能力 | 推荐发布形态 | 说明 |
| --- | --- | --- |
| `conv_transpose1d` / `conv_transpose2d` | `ConvTranspose1dFwdOp` / `ConvTranspose2dFwdOp`，或若 manifest 后续支持 rank-polymorphic contract，则聚合为 `ConvTransposeFwdOp` | rank-specific 命名可跟随现有 Conv / Pool manifest 风格 |
| `depthwise_conv2d` | `Conv2dFwdOp` 的 `groups == in_channels == out_channels` workload / kernel-mode | 不建议新增 `DepthwiseConv2dFwdOp` release-facing 名称 |
| `grouped_conv2d` | `Conv2dFwdOp` 的 `groups > 1` 参数组合 | 不建议新增 `GroupedConv2dFwdOp` release-facing 名称 |
| `dilated_conv2d` | `Conv2dFwdOp` 的 `dilation > 1` 参数组合 | 不建议新增 `DilatedConv2dFwdOp` release-facing 名称 |
| `adaptive_avg_pool2d` / `adaptive_max_pool2d` | `AdaptiveAvgPool2dFwdOp` / `AdaptiveMaxPool2dFwdOp`，或 rank-polymorphic `AdaptiveAvgPoolFwdOp` / `AdaptiveMaxPoolFwdOp` | 首版只要求 2d；rank 聚合可后续评估 |

验收标准：

- 明确 NCHW / NDHWC 支持策略；首版建议只发布一种 layout。
- bias、stride、padding、dilation、groups 的 manifest params 与 PyTorch 对齐。
- adaptive pool 的 output size 作为 static dim 或 init param 固定。
- Conv2d 的 grouped/depthwise/dilated 需要在 workloads 中显式覆盖，避免功能存在但 release 统计不可见。

## Wave 3：Quantize 与 Sampling

目标：补齐清账后仍不完整、但 serving 价值高的 Quantize 与 Sampling family。Wave 0 中 narrow FP8 quant 或 TopK selector 即使已发布，也只视为局部能力，不等同完整 family。

### Quantize

说明：这里的 Quantize family 只处理量化/反量化本身，包括 scale 计算、rounding、clamp、packing 或简单 cast-transpose contract。它不负责 FP8 GEMM / lowbit GEMM 的 Tensor Core 数据布局，也不承担特殊 swizzle 优化。因此它可以早于低精度 compute-core wave 发布。

早期目标：

- INT8：per-tensor、per-channel、per-block、smooth_quant
- INT4：per-channel、per-block、nf4
- FP8：per-tensor、per-block、cast_transpose
- 每个 quantize op 隐含或配套 dequantize 能力

聚合后的发布面：

下表中的 per-tensor、per-channel、per-block、NF4 等仍是功能覆盖项；接口层优先通过 dtype combo、scale shape 和 layout descriptor 聚合。

| Release | 推荐发布形态 | 覆盖能力 | 说明 |
| --- | --- | --- | --- |
| Q1 | `QuantizeFwdOp` / `DequantizeFwdOp` | FP8 per-tensor、rowwise/per-channel | 先建立统一 scale contract、dtype combo 和 error tolerance |
| Q2 | `QuantizeFwdOp` / `DequantizeFwdOp` | FP8 per-block | scale tensor shape 表达 granularity，不新增 `FP8QuantPerBlockFwdOp` |
| Q3 | `QuantizedCastTransposeFwdOp` | FP8 cast-transpose | 输出 layout contract 与普通 quantize 不同，允许单独 op |
| Q4 | `QuantizeFwdOp` / `DequantizeFwdOp` | INT8 per-tensor、per-channel、per-block | INT8 主线复用同一 op surface |
| Q5 | `SmoothQuantFwdOp` | activation smoothing + scale adjustment | 语义不同于普通 quantize，单独 op |
| Q6 | `QuantizeFwdOp` / `DequantizeFwdOp`，必要时加 `NF4QuantizeFwdOp` | INT4 per-channel、per-block、NF4 | 若 NF4 packed output contract 与普通 int4 差异过大，可单独拆 |

验收标准：

- Manifest 明确 scale tensor shape、zero point 是否支持、symmetric/asymmetric 策略。
- scale granularity 由 scale tensor shape、axis/layout descriptor 或 dtype combo 表达，不用 per-tensor/per-channel/per-block 拆 release-facing op 名。
- Round-trip correctness 用误差阈值而非 bit-exact。
- 与 GEMM lowbit/FP8 最终 wave 的 dtype、scale tensor、packed tensor 基本约定保持一致；若 GEMM 需要特殊 swizzle，应作为 GEMM 私有 layout，不反向污染通用 quantize op。

### Sampling

早期目标：

- `top_k`
- `top_p`
- `min_p`
- `top_k_top_p`
- `temperature_scale`
- `sampling_from_probs`
- `chain_speculative_sampling`

聚合后的发布面：

下表中的 top-k、top-p、min-p、temperature、sampling、speculative verification 仍分别作为功能验收点；接口层只把共享 logits-processing contract 的部分聚合。

| 早期能力 | 推荐发布形态 | 说明 |
| --- | --- | --- |
| `temperature_scale` | `LogitsProcessorFwdOp` | temperature 是 logits transform 参数 |
| `top_k` / `top_p` / `min_p` / `top_k_top_p` | `LogitsProcessorFwdOp` | top-k/top-p/min-p 可组合，避免拆多个 filter op |
| `sampling_from_probs` | `SamplingFromProbsFwdOp` | 随机性 contract 独立，保留单独 op |
| `chain_speculative_sampling` | `SpeculativeSamplingVerifyFwdOp` | 验证语义独立于 logits processing 和 categorical sampling |

验收标准：

- 支持大 vocab，例如 32K、64K、128K。
- 明确 logits 输入、概率输入、mask 输出、sampled token 输出的 contract。
- 随机采样必须有 seed/offset 或 generator contract，保证可复现测试。
- `TopKSelectorOp` 可作为实现依赖，但不能替代完整 sampling family。
- `LogitsProcessorFwdOp` 的禁用项使用 sentinel 参数表达，例如 `top_k=None`、`top_p=None`、`min_p=None`、`temperature=1.0`，不能把每种组合拆成新 op 名。

## Wave 4：Attention 矩阵补齐

目标：在当前 GQA/MHA serving 面基础上，补齐 MLA/NSA/DSA/MQA 以及 varlen/chunked 维度。

说明：本 wave 优先补 fp16/bf16 和已有稳定路径。FP8 attention Tensor Core path 若涉及特殊 swizzle、scale layout 或 FA3-style packed contract，可延后到最终低精度 compute-core wave，不阻塞 MLA/NSA/DSA 的基础功能发布。类似 batch=1 decode fast path、Hopper TMA/WGMMA、long-context split policy 这类工作属于性能专项；除非缺少基础 op contract，否则不计入功能缺口。

当前强项：

- MHA/GQA fwd/bwd
- MHA/GQA decode with KV cache
- MHA/GQA paged decode
- GQA prefill、paged prefill、sliding-window、sliding-window varlen
- MLA decode
- DSA decode

真实缺口：

- MQA 显式发布面
- Flash attention `prefill_varlen_bwd`
- Flash attention `decode_varlen_fwd`
- Flash attention `chunked_prefill_fwd`
- MLA `prefill_fwd`
- MLA `prefill_bwd`
- MLA `decode_paged_fwd`
- NSA `prefill_fwd`
- NSA `decode_fwd`
- DSA `prefill_fwd`

聚合后的发布面：

下表中的 MQA、varlen、chunked、MLA、NSA、DSA 仍是独立功能补齐项；接口层只在输入/输出 contract 足够一致时聚合到现有 attention op。

| 早期能力 | 推荐发布形态 | 说明 |
| --- | --- | --- |
| MQA | `GroupedQueryAttention*Op`，`H_kv == 1` | 不建议新增 `MultiQueryAttention*Op`，但必须在 workloads / docs 中显式标注 MQA 覆盖 |
| `decode_varlen_fwd` | `GroupedQueryAttentionDecodeWithKVCacheFwdOp` / `MultiHeadAttentionDecodeWithKVCacheFwdOp` 的 varlen metadata mode，或若输入 contract 差异过大则拆 `*DecodeVarlen*Op` | 优先聚合到 decode op；paged KV 仍可独立 |
| `chunked_prefill_fwd` | `GroupedQueryAttentionPrefillFwdOp` / `MultiHeadAttentionFwdOp` 的 chunked mode | chunk size 是调度/semantic param，不应单独成为 op 名 |
| `prefill_varlen_bwd` | `GroupedQueryAttentionBwdOp` / `MultiHeadAttentionBwdOp` 的 varlen metadata mode | 若 lse/cuseq/packed grad contract 不同，可单独拆 varlen bwd |
| MLA prefill/decode/bwd | `MultiHeadLatentAttentionPrefillFwdOp`、`MultiHeadLatentAttentionDecodeWithKVCacheFwdOp`、`MultiHeadLatentAttentionDecodePagedWithKVCacheFwdOp`、`MultiHeadLatentAttentionBwdOp` | MLA latent cache contract 与 GQA/MHA 不同，保留 family 独立 |
| NSA prefill/decode | `NativeSparseAttentionFwdOp`，或当前命名风格下的 `DeepSeekSparseAttention*Op` | prefill/decode 优先作为 mode；若 decode KV-cache 输入不同，可拆 decode op |
| DSA prefill/decode | `DeepSeekSparseAttention*Op` 的 DSA variant / mode | 若 DSA 与 NSA sparse metadata 不同，保留 DSA-specific entry |

验收标准：

- 所有 KV-cache op 明确 dense/paged/varlen metadata contract。
- Paged variants 必须定义 block table、page size、seq len tensor、cache layout。
- Sparse attention 必须定义 mask/indices/compressed path 输入的 ownership。
- Benchmarks 覆盖 Llama、DeepSeek、长上下文 decode/prefill 场景。
- 如果新增 op 名只是为了表达 MQA、chunked、varlen、small-batch decode 等 workload regime，应改为已有 attention op 的 workload / mode / kernel-map dispatch。

## Wave 5：Linear Attention 与 SSM

目标：在已有 Linear Attention / Mamba2 / SSD 代码面完成 Wave 0 manifest 化之后，补齐早期计划中仍缺的算法或变体，并确认 stateful API contract。

### Linear Attention

清账后假设 GatedDeltaNet、DeltaNet、GLA 的已有 chunkwise / recurrence 能力已经进入 manifest。后续真实缺口主要是 RetNet，以及已发布算法在 chunkwise / recurrence / state contract 上是否完整。

早期目标矩阵：

| Algorithm | Chunkwise | Recurrence |
| --- | --- | --- |
| gated_deltanet | 清账后应已发布或明确 defer | 清账后应已发布或明确 defer |
| deltanet | 清账后应已发布或明确 defer | 清账后应已发布或明确 defer |
| gla | 清账后应已发布或明确 defer | 清账后应已发布或明确 defer |
| retnet | 缺口 | 缺口 |

聚合后的发布面：

下表中的 chunkwise / recurrence 仍是功能覆盖维度；接口层按算法名聚合，mode/state contract 负责区分执行路径。

| 算法 | 推荐发布形态 | 说明 |
| --- | --- | --- |
| GatedDeltaNet | `GatedDeltaNetFwdOp` | chunkwise / recurrence / prefill 作为 mode 或 state contract 表达 |
| DeltaNet | `DeltaNetFwdOp` | chunkwise / recurrence 作为 mode 或 state contract 表达 |
| GLA | `GLAFwdOp` | chunkwise / recurrence 作为 mode 或 state contract 表达 |
| RetNet | `RetNetFwdOp` | 清账后仍是真缺口；chunkwise / recurrence 不拆成两个 release-facing 名称 |

建议顺序：

1. 审计 `GatedDeltaNetFwdOp`、`DeltaNetFwdOp`、`GLAFwdOp` manifest 是否覆盖 chunkwise / recurrence，以及 initial/final state 语义。
1. `RetNetFwdOp` chunkwise mode。
1. `RetNetFwdOp` recurrence mode。

验收标准：

- 明确 state tensor shape、initial state、final state 输出语义。
- Chunkwise 和 recurrence 的 numerics 应有同一 reference 路径或交叉验证。
- Backward 是否纳入首发需单独决策；早期计划只要求 chunkwise/recurrence，不强制 bwd。
- 若 recurrence 需要 in-place state mutation，必须在 manifest inputs/outputs 中表达；只有 state mutation contract 明显不同，才考虑拆 `*DecodeFwdOp`。

### SSM

早期目标：

- Mamba1 selective scan
- Mamba2 SSD

清账后假设 Mamba2 / SSD 已有代码面已经进入 manifest。真实缺口主要是 Mamba1 selective scan；Mamba2 只需要确认 prefill / decode / e2e 接口是否完整。

聚合后的发布面：

下表中的 prefill / decode / e2e / selective scan 仍是功能覆盖维度；接口层优先通过 mode 和 state tensor contract 聚合。

| 早期能力 | 推荐发布形态 | 说明 |
| --- | --- | --- |
| Mamba2 SSD prefill/decode/e2e | `Mamba2FwdOp` | prefill/decode/e2e 优先作为 mode、state input/output 或 workload 表达 |
| Mamba1 selective scan | `MambaSelectiveScanFwdOp` 或 `Mamba1FwdOp` | 真实缺口；先发布 scan/prefill contract |
| Mamba1 decode | `Mamba1FwdOp` decode mode，或 `Mamba1DecodeFwdOp` | 若 decode 原地更新 state，则可单独拆 decode op |

建议顺序：

1. 审计 `Mamba2FwdOp` manifest 是否表达 prefill / decode / e2e path。
1. `MambaSelectiveScanFwdOp` 或 `Mamba1FwdOp` 新增实现。
1. 视 state mutation 语义补 Mamba1 decode mode 或 `Mamba1DecodeFwdOp`。

验收标准：

- 对齐 `mamba_ssm` reference。
- 明确 chunk size、state dtype、dt softplus、initial/final state contract。
- Decode op 必须说明 state 是否 in-place mutation。

## Wave 6：GEMM 与低精度 Compute Core

目标：在统一 `GemmOp` 保持当前接口优势、且 Wave 0 已发布通用 `GroupedGemmOp` 的前提下，最后集中处理早期计划中的 GEMM 扩展，尤其是 FP8、lowbit、packed weight、scale layout、Tensor Core 数据搬运和特殊 swizzle。

Dense GEMM 子方向已有 upstream 设计 issue：[tile-ai/TileOPs#1653](https://github.com/tile-ai/TileOPs/issues/1653)。该 issue 将早期 dense GEMM 的 7 个 nominal ops 收敛为当前 TileOPs 风格下的 2 个 op surface 加若干 kernel-mode：

注意：这里的收敛是接口层收敛，不是功能范围缩减。`small_batch_fp16`、`gemm_fp8_block_scaled`、`gemv_fp8` 等仍然分别作为 workload、kernel-mode 和 correctness / benchmark 覆盖项验收。

| 早期 nominal op | TileOPs 发布形态 | 计划状态 |
| --- | --- | --- |
| `gemm_fp16` | `GemmOp`，fp16 / bf16 dtype combo | 已有 |
| `gemv_fp16` | `GemmOp` 的 gemv kernel-mode，`m == 1` 或 `n == 1` | 已有 |
| `small_batch_fp16` | `GemmOp` 的 small-M kernel-mode | #1653 |
| `gemm_fp8` | `GemmFp8Op`，epilogue-scale path | #1653 |
| `gemm_fp8_block_scaled` | `GemmFp8Op`，block-scaled mainloop path | #1653 |
| `gemv_fp8` | `GemmFp8Op` 的 gemv kernel-mode | #1653 |
| `small_batch_fp8` | `GemmFp8Op` 的 small-M kernel-mode | #1653 |

因此，Dense GEMM 的新增 op surface 应优先采用 `GemmFp8Op`，而不是按 per-tensor / rowwise / blockwise 或 gemv / small-batch 拆成多个 release-facing op。

为什么放到最后：

- FP8 / lowbit GEMM 的难点不只是 dtype，而是 scale granularity、packed layout、Tensor Core operand layout、swizzle、metadata 和 benchmark shape 共同决定接口。
- 这些 contract 一旦进入 manifest 就会成为 release-facing API，过早固化风险高。
- Quantize family 可以先定义普通 scale/rounding/packing 语义；GEMM 若需要特殊 layout，应在本 wave 作为 compute-core private 或 explicit packed format 处理。

缺口：

- FP8 GEMM
- FP8 block-scaled GEMM
- BMM fp16/fp8
- Outer product
- Low-bit GEMM：w4a16、w8a8、w8a8_int8、weight-only int4、fp4
- 2:4 sparse GEMM：fp16/fp8

### Dense GEMM：#1653 执行口径

`GemmFp8Op` 的接口应参考 `torch._scaled_mm`，保持 fully input-inferred：

- `M/N/K` 由 `a`、`b` 推导。
- scale granularity 由 `scale_a`、`scale_b` 的 shape 推导，不作为构造参数。
- `scale_a.shape[-1] == 1` 走 epilogue rescale path；`scale_a.shape[-1] > 1` 走 per-K-chunk mainloop dequant path。
- `out_dtype`、`use_fast_accum` 是 forward-time semantic params，对齐 `torch._scaled_mm`。
- gemv 和 small-batch 是 kernel-mode，不是独立 op；dispatch 仍发生在 `forward()`。

建议 manifest contract：

- `GemmFp8Op` 使用 NT layout：`a.shape == (M, K)`，`b.shape == (N, K)`，`d.shape == (M, N)`。
- `a` / `b` dtype 支持 `float8_e4m3fn | float8_e5m2`。
- `scale_a` / `scale_b` dtype 为 `float32`，shape 编码 granularity。
- `bias` 可作为 optional input；`out_dtype` / `use_fast_accum` 对齐 `torch._scaled_mm`，但不参与 shape dispatch。
- shape rules 使用整除关系表达 per-tensor、rowwise、blockwise，不固定 block size 到构造参数。
- 首版 workload 覆盖 epilogue、block-scaled mainloop、gemv、small-batch 四类 path；block-mode workload 保持 128-aligned。
- `scale_mode` 只能作为 workload 生成 scale tensor 的描述符，不能进入 manifest signature params。

建议首批 workload：

| Op | label | M | N | K | scale/path |
| --- | --- | ---: | ---: | ---: | --- |
| `GemmFp8Op` | `rowwise-decode-down` | 128 | 7168 | 2048 | rowwise / epilogue |
| `GemmFp8Op` | `rowwise-prefill-down` | 4096 | 7168 | 2048 | rowwise / epilogue |
| `GemmFp8Op` | `block-prefill-attn-proj` | 4096 | 4096 | 7168 | block128 / mainloop |
| `GemmFp8Op` | `block-k-dominant` | 4096 | 7168 | 16384 | block128 / mainloop |
| `GemmFp8Op` | `block-wide-n` | 4096 | 24576 | 1536 | block128 / mainloop |
| `GemmFp8Op` | `small-batch-down-m8` | 8 | 7168 | 2048 | rowwise / small-batch |
| `GemmFp8Op` | `small-batch-gate-up-m8` | 8 | 2112 | 7168 | rowwise / small-batch |
| `GemmFp8Op` | `gemv-down-m1` | 1 | 7168 | 2048 | rowwise / gemv |
| `GemmOp` | `small-batch-down-m8` | 8 | 7168 | 2048 | bf16 / small-batch |
| `GemmOp` | `small-batch-attn-proj-m16` | 16 | 4096 | 7168 | bf16 / small-batch |
| `GemmOp` | `gemv-down-m1` | 1 | 7168 | 2048 | fp16,bf16 / gemv |

推荐顺序：

1. `GemmFp8Op` manifest-only PR，`status: spec-only`，包含 signature、workloads、roofline、source、kernel_map。
1. Rowwise / per-tensor FP8 epilogue kernel。
1. Block-scaled FP8 mainloop kernel。
1. `GemmFp8Op` 的 gemv kernel-mode。
1. `GemmOp` 和 `GemmFp8Op` 的 small-batch kernel-mode，以及对应 manifest workloads。
1. Correctness 对齐 `torch._scaled_mm`，覆盖 per-tensor、rowwise、blockwise、gemv、small-batch。
1. Benchmark 对齐 cuBLAS / DeepGEMM，并使用 manifest-driven benchmark。

### 其他 GEMM 子方向

Dense GEMM 之外的能力继续作为 #400 的 sibling sub-issues，不由 #1653 一次性关闭。

建议设计：

| 能力 | 推荐 manifest 形态 |
| --- | --- |
| FP8 GEMM | `GemmFp8Op`，与 `torch._scaled_mm` 对齐；per-tensor / rowwise / blockwise 由 scale shape 推导 |
| FP8 block-scaled GEMM | `GemmFp8Op` 的 mainloop path，不单独拆 `FP8BlockScaledGemmOp` |
| BMM | `BmmOp`，支持 batch dim 和 trans flags；fp16/bf16 可以先实现，fp8 跟随低精度 contract |
| Grouped GEMM | Wave 0 应已作为通用 GEMM family entry 发布；本 wave 只处理与低精度或特殊 layout 绑定的 grouped variants |
| Outer | `OuterFwdOp`，也可归入 elementwise/reduction，但按旧计划放 GEMM family |
| Low-bit | 单独 op，显式 weight layout、scale layout、pack format |
| 2:4 sparse | 单独 sparse manifest，明确 metadata tensor contract |

验收标准：

- dtype_combos exhaustive，不能用模糊 dtype union 表达 scale/weight/input 复杂组合。
- 每个低精度 GEMM 必须有对应 quant/dequant 或 packed weight 生成路径。
- 明确普通 quantized tensor 与 GEMM-special swizzled/packed tensor 的边界。
- Benchmarks 覆盖 decode 小 M、prefill 大 M、wide-N、K-dominant 等 regimes。
- 对 dense FP8 GEMM，`GemmFp8Op` 必须保持 input-inferred；scale granularity、block count、gemv / small-batch path 不应成为 release-facing 构造参数。

## 量化相关范围表

| Area | 是否涉及量化/低精度 | 是否需要特殊 swizzle | 处理策略 |
| --- | --- | --- | --- |
| Elementwise fp8 | 是 | 否 | 可作为 elementwise dtype 增强早做 |
| Quantize / Dequantize | 是 | 通常否 | Wave 3 发布通用 scale/rounding/packing contract |
| FP8 KV / attention helper quant | 是 | 通常否 | 可先作为 narrow quant entry；不宣称覆盖 GEMM layout |
| Attention FP8 Tensor Core path | 是 | 可能是 | 基础 attention 先发布；特殊 FP8 path 延后到 compute-core wave |
| GEMM FP8 / block-scaled | 是 | 高概率是 | Wave 6 按 #1653 由统一 `GemmFp8Op` 处理，scale shape 决定 epilogue / mainloop path |
| Low-bit GEMM | 是 | 可能是 | Wave 6 统一处理 packed weight / scale layout |
| 2:4 sparse GEMM | 不是量化本身，但常与低精度 TC 路线耦合 | 可能是 | Wave 6 与 GEMM metadata contract 一起评审 |

## 建议里程碑

M0 / M1 是 baseline reset；正式功能补齐从 M2 开始，并默认 M0 / M1 已合入或已明确 defer 未完成项。

| Milestone | 目标 | 结果 |
| --- | --- | --- |
| M0 | Preflight: Manifest debt 清账 | 当前代码面中的成熟 op 全部进入正式统计 |
| M1 | Preflight: Spec-only 清零 | manifest 无 spec-only，或 spec-only 只保留明确未来项 |
| M2 | Deterministic gaps | qk_norm、conv/pool 扩展完成 |
| M3 | Serving utility | Quantize + Sampling family 首发 |
| M4 | Attention completion | MLA/NSA/DSA/MQA/varlen/chunked 能力补齐 |
| M5 | Sequence alternatives | Linear Attention + SSM release-facing 收敛 |
| M6 | Compute core | Dense GEMM 按 #1653 发布 `GemmFp8Op` 与 small-batch modes；BMM/lowbit/sparse 继续拆 sibling sub-issues |

## 每个 PR 的最低交付要求

每个新增或转正 op PR 至少包含：

1. Manifest entry。
1. Op 和 Kernel source，或明确复用现有 source。
1. Correctness tests。
1. Manifest-driven benchmark。
1. Roofline metadata。
1. Workloads 至少覆盖一个 smoke shape 和一个代表性 production shape。
1. `scripts/validate_manifest.py` 通过。

## 风险与开放问题

- **接口过早固化**：Sampling、Sparse Attention、Low-bit GEMM 的 layout contract 一旦发布就难改，应先做 spec-only review；Dense FP8 GEMM 已由 #1653 收敛为 `GemmFp8Op`，但仍需通过 manifest-only PR 固化 scale shape contract。Quantize 和 elementwise fp8 因不涉及特殊 swizzle，风险主要在 scale/rounding/tolerance contract。
- **旧计划与现设计不完全同构**：例如 MQA 可由 GQA `H_kv=1` 表达，是否需要独立 op 名需要 release owner 决策。
- **代码成熟度不一**：已有代码不代表已满足 manifest trust model，Wave 0 可能暴露接口或测试债。
- **性能期望**：GEMM、Quant、Sampling、Attention 的首发应区分 correctness release 与 performance release，避免把性能优化阻塞接口收敛。
- **模型特化 vs generic op**：MoE、MLA、NSA 等 family 需要决定哪些能力作为 generic op 参数，哪些作为 model-specific op。
