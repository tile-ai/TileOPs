## Manifest status

![ops](https://img.shields.io/badge/ops-180-blue) ![implemented](https://img.shields.io/badge/implemented-175%20%2F%20180%20%2897%25%29-brightgreen) ![spec--only](https://img.shields.io/badge/spec--only-5-orange)

### Per-family coverage

| Family | Implemented | Spec-only | Total | Progress | Workloads |
| --- | ---: | ---: | ---: | --- | ---: |
| `attention` | 15 | 2 | 17 | `█████████░` 88% | 76 |
| `convolution` | 3 | 0 | 3 | `██████████` 100% | 43 |
| `elementwise` | 70 | 0 | 70 | `██████████` 100% | 148 |
| `fft` | 1 | 0 | 1 | `██████████` 100% | 10 |
| `gemm` | 6 | 0 | 6 | `██████████` 100% | 70 |
| `linear_attention` | 7 | 3 | 10 | `███████░░░` 70% | 47 |
| `mamba` | 7 | 0 | 7 | `██████████` 100% | 29 |
| `moe` | 10 | 0 | 10 | `██████████` 100% | 75 |
| `norm` | 10 | 0 | 10 | `██████████` 100% | 60 |
| `pool` | 13 | 0 | 13 | `██████████` 100% | 44 |
| `quantization` | 1 | 0 | 1 | `██████████` 100% | 2 |
| `reduction` | 21 | 0 | 21 | `██████████` 100% | 94 |
| `rope` | 6 | 0 | 6 | `██████████` 100% | 19 |
| `sequence_modeling` | 5 | 0 | 5 | `██████████` 100% | 17 |

### Spec coverage

| Field | Coverage |
| --- | ---: |
| `ref_api` | 128 / 180 (71%) |
| `roofline` (func or flops+bytes) | 28 / 180 (16%) |
| `source.kernel_map` | 179 / 180 (99%) |
| `source.bench_manifest_driven` | 178 / 180 (99%) |

**Workloads:** 734 total — 4.08 per implemented op.

### Conformance gaps

- Implemented ops without `kernel_map`: **0**
- Implemented ops without `roofline`: **149**
- Implemented ops without `source.bench_manifest_driven`: **0**
- Implemented ops with fewer than two workloads: **0**

<details><summary>Spec-only ops (5)</summary>

| | | |
| --- | --- | --- |
| `DeltaNetInferenceFwdOp` | `GLAInferenceFwdOp` | `GatedDeltaNetFwdOp` |
| `GroupedQueryAttentionPagedFwdOp` | `GroupedQueryAttentionVarlenFwdOp` |  |

</details>
