## Manifest status

![ops](https://img.shields.io/badge/ops-205-blue) ![implemented](https://img.shields.io/badge/implemented-188%20%2F%20205%20%2892%25%29-brightgreen) ![spec--only](https://img.shields.io/badge/spec--only-17-orange)

### Per-family coverage

| Family | Implemented | Spec-only | Total | Progress | Workloads |
| --- | ---: | ---: | ---: | --- | ---: |
| `attention` | 12 | 8 | 20 | `██████░░░░` 60% | 172 |
| `convolution` | 3 | 0 | 3 | `██████████` 100% | 43 |
| `elementwise` | 70 | 0 | 70 | `██████████` 100% | 230 |
| `fft` | 1 | 0 | 1 | `██████████` 100% | 10 |
| `gemm` | 6 | 2 | 8 | `████████░░` 75% | 115 |
| `linear_attention` | 10 | 0 | 10 | `██████████` 100% | 236 |
| `mamba` | 7 | 3 | 10 | `███████░░░` 70% | 45 |
| `moe` | 11 | 1 | 12 | `█████████░` 92% | 92 |
| `norm` | 10 | 2 | 12 | `████████░░` 83% | 87 |
| `pool` | 13 | 0 | 13 | `██████████` 100% | 44 |
| `quantization` | 10 | 0 | 10 | `██████████` 100% | 60 |
| `reduction` | 19 | 0 | 19 | `██████████` 100% | 106 |
| `rope` | 5 | 0 | 5 | `██████████` 100% | 20 |
| `sampling` | 6 | 0 | 6 | `██████████` 100% | 40 |
| `sequence_modeling` | 5 | 0 | 5 | `██████████` 100% | 17 |
| `transform` | 0 | 1 | 1 | `░░░░░░░░░░` 0% | 6 |

### Spec coverage

| Field | Coverage |
| --- | ---: |
| `ref_api` | 201 / 205 (98%) |
| `roofline` (func or flops) | 205 / 205 (100%) |

**Workloads:** 1323 total — 6.49 per implemented op.

### Conformance gaps

- Implemented ops without `roofline`: **0**
- Implemented ops without `workloads`: **0**
- Implemented ops with fewer than two workloads: **0**

<details><summary>Spec-only ops (17)</summary>

| | | |
| --- | --- | --- |
| `CausalConv1dFwdOp` | `CausalConv1dUpdateFwdOp` | `DSAPagedFwdOp` |
| `FusedAddRMSNormPerTokenQuantFwdOp` | `FusedQKNormRoPEFwdOp` | `GQAPagedFwdOp` |
| `GemmINT8W8A8FwdOp` | `GemmW4A8FwdOp` | `HadamardTransformFwdOp` |
| `MLAKVCacheWriteFwdOp` | `MLAPagedFwdOp` | `MLAVarlenFwdOp` |
| `MergeAttentionStatesFwdOp` | `MoEGroupedGemmFP8FwdOp` | `PagedKVCacheGatherFwdOp` |
| `PagedKVCacheWriteFwdOp` | `SelectiveScanFwdOp` |  |

</details>
