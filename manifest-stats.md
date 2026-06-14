## Manifest status

![ops](https://img.shields.io/badge/ops-137-blue) ![implemented](https://img.shields.io/badge/implemented-125%20%2F%20137%20%2891%25%29-brightgreen) ![spec--only](https://img.shields.io/badge/spec--only-12-orange)

### Per-family coverage

| Family | Implemented | Spec-only | Total | Progress | Workloads |
| --- | ---: | ---: | ---: | --- | ---: |
| `attention` | 17 | 0 | 17 | `██████████` 100% | 70 |
| `convolution` | 2 | 4 | 6 | `███░░░░░░░` 33% | 40 |
| `elementwise` | 66 | 5 | 71 | `█████████░` 93% | 142 |
| `moe` | 7 | 0 | 7 | `██████████` 100% | 50 |
| `normalization` | 12 | 0 | 12 | `██████████` 100% | 50 |
| `pool` | 0 | 3 | 3 | `░░░░░░░░░░` 0% | 9 |
| `reduction` | 19 | 0 | 19 | `██████████` 100% | 47 |
| `scan` | 2 | 0 | 2 | `██████████` 100% | 4 |

### Spec coverage

| Field | Coverage |
| --- | ---: |
| `ref_api` | 137 / 137 (100%) |
| `roofline` (func or flops+bytes) | 137 / 137 (100%) |
| `source.kernel_map` | 103 / 137 (75%) |
| `source.bench_manifest_driven` | 105 / 137 (77%) |

**Workloads:** 412 total — 2.89 per implemented op.

### Conformance gaps

- Implemented ops without `kernel_map`: **34**
- Implemented ops without `roofline`: **0**
- Implemented ops without `source.bench_manifest_driven`: **20**
- Implemented ops with fewer than two workloads: **0**

<details><summary>Spec-only ops (12)</summary>

| | | |
| --- | --- | --- |
| `AlibiFwdOp` | `AvgPool1dFwdOp` | `AvgPool2dFwdOp` |
| `AvgPool3dFwdOp` | `Conv2dBiasFwdOp` | `Conv2dFwdOp` |
| `Conv3dBiasFwdOp` | `Conv3dFwdOp` | `GeluAndMulFwdOp` |
| `GeluTanhAndMulFwdOp` | `SiluAndMulFwdOp` | `SinusoidalFwdOp` |

</details>
