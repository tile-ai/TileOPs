## Manifest status

![ops](https://img.shields.io/badge/ops-178-blue) ![implemented](https://img.shields.io/badge/implemented-178%20%2F%20178%20%28100%25%29-brightgreen) ![spec--only](https://img.shields.io/badge/spec--only-0-orange)

### Per-family coverage

| Family | Implemented | Spec-only | Total | Progress | Workloads |
| --- | ---: | ---: | ---: | --- | ---: |
| `attention` | 14 | 0 | 14 | `██████████` 100% | 78 |
| `attention_indexing` | 2 | 0 | 2 | `██████████` 100% | 5 |
| `convolution` | 3 | 0 | 3 | `██████████` 100% | 43 |
| `elementwise` | 69 | 0 | 69 | `██████████` 100% | 146 |
| `gemm` | 7 | 0 | 7 | `██████████` 100% | 56 |
| `linear_attention` | 12 | 0 | 12 | `██████████` 100% | 69 |
| `mamba` | 7 | 0 | 7 | `██████████` 100% | 29 |
| `moe` | 7 | 0 | 7 | `██████████` 100% | 58 |
| `normalization` | 10 | 0 | 10 | `██████████` 100% | 50 |
| `pool` | 12 | 0 | 12 | `██████████` 100% | 36 |
| `position_encoding` | 6 | 0 | 6 | `██████████` 100% | 13 |
| `quantization` | 1 | 0 | 1 | `██████████` 100% | 3 |
| `reduction` | 19 | 0 | 19 | `██████████` 100% | 62 |
| `regularization` | 1 | 0 | 1 | `██████████` 100% | 2 |
| `scan` | 2 | 0 | 2 | `██████████` 100% | 4 |
| `sequence_modeling` | 5 | 0 | 5 | `██████████` 100% | 15 |
| `spectral` | 1 | 0 | 1 | `██████████` 100% | 3 |

### Spec coverage

| Field | Coverage |
| --- | ---: |
| `ref_api` | 178 / 178 (100%) |
| `roofline` (func or flops+bytes) | 178 / 178 (100%) |
| `source.kernel_map` | 178 / 178 (100%) |
| `source.bench_manifest_driven` | 178 / 178 (100%) |

**Workloads:** 672 total — 3.78 per implemented op.

### Conformance gaps

- Implemented ops without `kernel_map`: **0**
- Implemented ops without `roofline`: **0**
- Implemented ops without `source.bench_manifest_driven`: **0**
- Implemented ops with fewer than two workloads: **0**
