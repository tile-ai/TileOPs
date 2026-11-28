## Manifest status

![ops](https://img.shields.io/badge/ops-213-blue) ![implemented](https://img.shields.io/badge/implemented-213%20%2F%20213%20%28100%25%29-brightgreen) ![spec--only](https://img.shields.io/badge/spec--only-0-orange)

### Per-family coverage

| Family | Implemented | Spec-only | Total | Progress | Workloads |
| --- | ---: | ---: | ---: | --- | ---: |
| `attention` | 23 | 0 | 23 | `██████████` 100% | 277 |
| `convolution` | 3 | 0 | 3 | `██████████` 100% | 43 |
| `elementwise` | 66 | 0 | 66 | `██████████` 100% | 247 |
| `fft` | 1 | 0 | 1 | `██████████` 100% | 10 |
| `gemm` | 9 | 0 | 9 | `██████████` 100% | 313 |
| `indexing` | 3 | 0 | 3 | `██████████` 100% | 24 |
| `linear_attention` | 10 | 0 | 10 | `██████████` 100% | 197 |
| `mamba` | 10 | 0 | 10 | `██████████` 100% | 62 |
| `moe` | 13 | 0 | 13 | `██████████` 100% | 146 |
| `norm` | 13 | 0 | 13 | `██████████` 100% | 120 |
| `pool` | 13 | 0 | 13 | `██████████` 100% | 44 |
| `quantization` | 11 | 0 | 11 | `██████████` 100% | 77 |
| `reduction` | 19 | 0 | 19 | `██████████` 100% | 110 |
| `rope` | 5 | 0 | 5 | `██████████` 100% | 27 |
| `sampling` | 6 | 0 | 6 | `██████████` 100% | 40 |
| `sequence_modeling` | 7 | 0 | 7 | `██████████` 100% | 27 |
| `transform` | 1 | 0 | 1 | `██████████` 100% | 6 |

### Spec coverage

| Field | Coverage |
| --- | ---: |
| `ref_api` | 209 / 213 (98%) |
| `roofline` (func or flops) | 213 / 213 (100%) |

**Workloads:** 1770 total — 8.31 per implemented op.

### Conformance gaps

- Implemented ops without `roofline`: **0**
- Implemented ops without `workloads`: **0**
- Implemented ops with fewer than two workloads: **0**
