## Manifest status

![ops](https://img.shields.io/badge/ops-183-blue) ![implemented](https://img.shields.io/badge/implemented-180%20%2F%20183%20%2898%25%29-brightgreen) ![spec--only](https://img.shields.io/badge/spec--only-3-orange)

### Per-family coverage

| Family | Implemented | Spec-only | Total | Progress | Workloads |
| --- | ---: | ---: | ---: | --- | ---: |
| `attention` | 14 | 1 | 15 | `█████████░` 93% | 54 |
| `convolution` | 3 | 0 | 3 | `██████████` 100% | 43 |
| `elementwise` | 70 | 0 | 70 | `██████████` 100% | 148 |
| `gemm` | 7 | 0 | 7 | `██████████` 100% | 58 |
| `linear_attention` | 14 | 0 | 14 | `██████████` 100% | 73 |
| `mamba` | 7 | 0 | 7 | `██████████` 100% | 29 |
| `moe` | 8 | 2 | 10 | `████████░░` 80% | 39 |
| `normalization` | 10 | 0 | 10 | `██████████` 100% | 50 |
| `pool` | 13 | 0 | 13 | `██████████` 100% | 41 |
| `position_encoding` | 6 | 0 | 6 | `██████████` 100% | 13 |
| `quantization` | 1 | 0 | 1 | `██████████` 100% | 3 |
| `reduction` | 19 | 0 | 19 | `██████████` 100% | 62 |
| `scan` | 2 | 0 | 2 | `██████████` 100% | 4 |
| `sequence_modeling` | 5 | 0 | 5 | `██████████` 100% | 15 |
| `spectral` | 1 | 0 | 1 | `██████████` 100% | 3 |

### Spec coverage

| Field | Coverage |
| --- | ---: |
| `ref_api` | 183 / 183 (100%) |
| `roofline` (func or flops+bytes) | 183 / 183 (100%) |
| `source.kernel_map` | 181 / 183 (99%) |
| `source.bench_manifest_driven` | 180 / 183 (98%) |

**Workloads:** 635 total — 3.53 per implemented op.

### Conformance gaps

- Implemented ops without `kernel_map`: **0**
- Implemented ops without `roofline`: **0**
- Implemented ops without `source.bench_manifest_driven`: **0**
- Implemented ops with fewer than two workloads: **0**

<details><summary>Spec-only ops (3)</summary>

| | | |
| --- | --- | --- |
| `GroupedQueryAttentionDenseFwdOp` | `MoeExpertMLPFwdOp` | `MoeGroupedGemmFwdOp` |

</details>
