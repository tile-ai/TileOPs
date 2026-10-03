## Manifest status

![ops](https://img.shields.io/badge/ops-205-blue) ![implemented](https://img.shields.io/badge/implemented-186%20%2F%20205%20%2891%25%29-brightgreen) ![spec--only](https://img.shields.io/badge/spec--only-19-orange)

### Per-family coverage

| Family | Implemented | Spec-only | Total | Progress | Workloads |
| --- | ---: | ---: | ---: | --- | ---: |
| `attention` | 11 | 9 | 20 | `██████░░░░` 55% | 113 |
| `convolution` | 3 | 0 | 3 | `██████████` 100% | 43 |
| `elementwise` | 70 | 0 | 70 | `██████████` 100% | 229 |
| `fft` | 1 | 0 | 1 | `██████████` 100% | 10 |
| `gemm` | 6 | 2 | 8 | `████████░░` 75% | 95 |
| `linear_attention` | 9 | 1 | 10 | `█████████░` 90% | 79 |
| `mamba` | 7 | 3 | 10 | `███████░░░` 70% | 44 |
| `moe` | 11 | 1 | 12 | `█████████░` 92% | 84 |
| `norm` | 10 | 2 | 12 | `████████░░` 83% | 75 |
| `pool` | 13 | 0 | 13 | `██████████` 100% | 44 |
| `quantization` | 10 | 0 | 10 | `██████████` 100% | 57 |
| `reduction` | 19 | 0 | 19 | `██████████` 100% | 94 |
| `rope` | 5 | 0 | 5 | `██████████` 100% | 19 |
| `sampling` | 6 | 0 | 6 | `██████████` 100% | 40 |
| `sequence_modeling` | 5 | 0 | 5 | `██████████` 100% | 17 |
| `transform` | 0 | 1 | 1 | `░░░░░░░░░░` 0% | 6 |

### Spec coverage

| Field | Coverage |
| --- | ---: |
| `ref_api` | 108 / 205 (53%) |
| `roofline` (func or flops) | 205 / 205 (100%) |

**Workloads:** 1049 total — 4.98 per implemented op.

### Conformance gaps

- Implemented ops without `roofline`: **0**
- Implemented ops without `workloads`: **0**
- Implemented ops with fewer than two workloads: **0**

<details><summary>Spec-only ops (19)</summary>

| | | |
| --- | --- | --- |
| `CausalConv1dFwdOp` | `CausalConv1dUpdateFwdOp` | `DeepSeekSparseAttentionPagedFwdOp` |
| `FusedAddRMSNormPerTokenQuantFwdOp` | `FusedQKNormRopeFwdOp` | `GemmINT8W8A8FwdOp` |
| `GemmW4A8FwdOp` | `GroupedQueryAttentionPagedFwdOp` | `GroupedQueryAttentionVarlenFwdOp` |
| `HadamardTransformFwdOp` | `KimiDeltaAttentionFwdOp` | `MergeAttentionStatesFwdOp` |
| `MoEGroupedGemmFP8FwdOp` | `MultiHeadLatentAttentionKVCacheWriteFwdOp` | `MultiHeadLatentAttentionPagedFwdOp` |
| `MultiHeadLatentAttentionVarlenFwdOp` | `PagedKVCacheGatherFwdOp` | `PagedKVCacheWriteFwdOp` |
| `SelectiveScanFwdOp` |  |  |

</details>
