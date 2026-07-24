## Manifest status

![ops](https://img.shields.io/badge/ops-171-blue) ![implemented](https://img.shields.io/badge/implemented-165%20%2F%20171%20%2896%25%29-brightgreen) ![spec--only](https://img.shields.io/badge/spec--only-6-orange)

### Per-family coverage

| Family | Implemented | Spec-only | Total | Progress | Workloads |
| --- | ---: | ---: | ---: | --- | ---: |
| `attention` | 14 | 0 | 14 | `██████████` 100% | 78 |
| `attention_indexing` | 2 | 0 | 2 | `██████████` 100% | 4 |
| `bmm` | 2 | 0 | 2 | `██████████` 100% | 14 |
| `convolution` | 6 | 0 | 6 | `██████████` 100% | 40 |
| `elementwise` | 71 | 0 | 71 | `██████████` 100% | 146 |
| `gemm` | 3 | 0 | 3 | `██████████` 100% | 30 |
| `linear_attention` | 4 | 6 | 10 | `████░░░░░░` 40% | 22 |
| `moe` | 7 | 0 | 7 | `██████████` 100% | 50 |
| `normalization` | 12 | 0 | 12 | `██████████` 100% | 50 |
| `pool` | 9 | 0 | 9 | `██████████` 100% | 27 |
| `position_encoding` | 6 | 0 | 6 | `██████████` 100% | 13 |
| `quantization` | 1 | 0 | 1 | `██████████` 100% | 3 |
| `reduction` | 19 | 0 | 19 | `██████████` 100% | 47 |
| `regularization` | 1 | 0 | 1 | `██████████` 100% | 3 |
| `scan` | 2 | 0 | 2 | `██████████` 100% | 4 |
| `sequence_modeling` | 5 | 0 | 5 | `██████████` 100% | 15 |
| `spectral` | 1 | 0 | 1 | `██████████` 100% | 3 |

### Spec coverage

| Field | Coverage |
| --- | ---: |
| `ref_api` | 171 / 171 (100%) |
| `roofline` (func or flops+bytes) | 171 / 171 (100%) |
| `source.kernel_map` | 131 / 171 (77%) |
| `source.bench_manifest_driven` | 134 / 171 (78%) |

**Workloads:** 549 total — 3.33 per implemented op.

### Conformance gaps

- Implemented ops without `kernel_map`: **34**
- Implemented ops without `roofline`: **0**
- Implemented ops without `source.bench_manifest_driven`: **31**
- Implemented ops with fewer than two workloads: **0**

<details><summary>Spec-only ops (6)</summary>

| | | |
| --- | --- | --- |
| `DeltaNetBwdOp` | `DeltaNetFwdOp` | `GLABwdOp` |
| `GLAFwdOp` | `GatedDeltaNetBwdOp` | `GatedDeltaNetFwdOp` |

</details>
