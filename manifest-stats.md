## Manifest status

![ops](https://img.shields.io/badge/ops-142-blue) ![implemented](https://img.shields.io/badge/implemented-130%20%2F%20142%20%2892%25%29-brightgreen) ![spec--only](https://img.shields.io/badge/spec--only-12-orange)

### Per-family coverage

| Family | Implemented | Spec-only | Total | Progress | Workloads |
| --- | ---: | ---: | ---: | --- | ---: |
| `attention` | 14 | 0 | 14 | `██████████` 100% | 70 |
| `convolution` | 6 | 0 | 6 | `██████████` 100% | 40 |
| `elementwise` | 66 | 5 | 71 | `█████████░` 93% | 142 |
| `gemm` | 1 | 0 | 1 | `██████████` 100% | 8 |
| `linear_attention` | 0 | 1 | 1 | `░░░░░░░░░░` 0% | 4 |
| `moe` | 7 | 0 | 7 | `██████████` 100% | 50 |
| `normalization` | 12 | 0 | 12 | `██████████` 100% | 50 |
| `pool` | 3 | 6 | 9 | `███░░░░░░░` 33% | 27 |
| `reduction` | 19 | 0 | 19 | `██████████` 100% | 47 |
| `scan` | 2 | 0 | 2 | `██████████` 100% | 4 |

### Spec coverage

| Field | Coverage |
| --- | ---: |
| `ref_api` | 142 / 142 (100%) |
| `roofline` (func or flops+bytes) | 142 / 142 (100%) |
| `source.kernel_map` | 102 / 142 (72%) |
| `source.bench_manifest_driven` | 122 / 142 (86%) |

**Workloads:** 442 total — 3.15 per implemented op.

### Conformance gaps

- Implemented ops without `kernel_map`: **34**
- Implemented ops without `roofline`: **0**
- Implemented ops without `source.bench_manifest_driven`: **9**
- Implemented ops with fewer than two workloads: **0**

<details><summary>Spec-only ops (12)</summary>

| | | |
| --- | --- | --- |
| `AlibiFwdOp` | `GatedDeltaNetPrefillFwdOp` | `GeluAndMulFwdOp` |
| `GeluTanhAndMulFwdOp` | `MaxPool1dFwdOp` | `MaxPool1dIndicesFwdOp` |
| `MaxPool2dFwdOp` | `MaxPool2dIndicesFwdOp` | `MaxPool3dFwdOp` |
| `MaxPool3dIndicesFwdOp` | `SiluAndMulFwdOp` | `SinusoidalFwdOp` |

</details>
