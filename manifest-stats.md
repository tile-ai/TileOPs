## Manifest status

![ops](https://img.shields.io/badge/ops-206-blue) ![implemented](https://img.shields.io/badge/implemented-206%20%2F%20206%20%28100%25%29-brightgreen) ![spec--only](https://img.shields.io/badge/spec--only-0-orange)

### Per-family coverage

| Family | Implemented | Spec-only | Total | Progress | Workloads |
| --- | ---: | ---: | ---: | --- | ---: |
| `attention` | 22 | 0 | 22 | `██████████` 100% | 216 |
| `convolution` | 3 | 0 | 3 | `██████████` 100% | 43 |
| `elementwise` | 69 | 0 | 69 | `██████████` 100% | 230 |
| `fft` | 1 | 0 | 1 | `██████████` 100% | 10 |
| `gemm` | 8 | 0 | 8 | `██████████` 100% | 229 |
| `linear_attention` | 10 | 0 | 10 | `██████████` 100% | 265 |
| `mamba` | 10 | 0 | 10 | `██████████` 100% | 47 |
| `moe` | 12 | 0 | 12 | `██████████` 100% | 119 |
| `norm` | 12 | 0 | 12 | `██████████` 100% | 89 |
| `pool` | 13 | 0 | 13 | `██████████` 100% | 44 |
| `quantization` | 10 | 0 | 10 | `██████████` 100% | 60 |
| `reduction` | 19 | 0 | 19 | `██████████` 100% | 109 |
| `rope` | 5 | 0 | 5 | `██████████` 100% | 20 |
| `sampling` | 6 | 0 | 6 | `██████████` 100% | 40 |
| `sequence_modeling` | 5 | 0 | 5 | `██████████` 100% | 17 |
| `transform` | 1 | 0 | 1 | `██████████` 100% | 6 |

### Spec coverage

| Field | Coverage |
| --- | ---: |
| `ref_api` | 202 / 206 (98%) |
| `roofline` (func or flops) | 206 / 206 (100%) |

**Workloads:** 1544 total — 7.50 per implemented op.

### Conformance gaps

- Implemented ops without `roofline`: **0**
- Implemented ops without `workloads`: **0**
- Implemented ops with fewer than two workloads: **0**
