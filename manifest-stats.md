## Manifest status

![ops](https://img.shields.io/badge/ops-188-blue) ![implemented](https://img.shields.io/badge/implemented-169%20%2F%20188%20%2890%25%29-brightgreen) ![spec--only](https://img.shields.io/badge/spec--only-19-orange)

### Per-family coverage

| Family | Implemented | Spec-only | Total | Progress | Workloads |
| --- | ---: | ---: | ---: | --- | ---: |
| `attention` | 14 | 0 | 14 | `██████████` 100% | 78 |
| `attention_indexing` | 2 | 0 | 2 | `██████████` 100% | 4 |
| `bmm` | 2 | 0 | 2 | `██████████` 100% | 14 |
| `convolution` | 6 | 0 | 6 | `██████████` 100% | 45 |
| `elementwise` | 71 | 0 | 71 | `██████████` 100% | 146 |
| `gemm` | 4 | 0 | 4 | `██████████` 100% | 36 |
| `linear_attention` | 4 | 6 | 10 | `████░░░░░░` 40% | 37 |
| `mamba` | 0 | 13 | 13 | `░░░░░░░░░░` 0% | 29 |
| `moe` | 7 | 0 | 7 | `██████████` 100% | 50 |
| `normalization` | 12 | 0 | 12 | `██████████` 100% | 50 |
| `pool` | 12 | 0 | 12 | `██████████` 100% | 36 |
| `position_encoding` | 6 | 0 | 6 | `██████████` 100% | 13 |
| `quantization` | 1 | 0 | 1 | `██████████` 100% | 3 |
| `reduction` | 19 | 0 | 19 | `██████████` 100% | 62 |
| `regularization` | 1 | 0 | 1 | `██████████` 100% | 3 |
| `scan` | 2 | 0 | 2 | `██████████` 100% | 4 |
| `sequence_modeling` | 5 | 0 | 5 | `██████████` 100% | 15 |
| `spectral` | 1 | 0 | 1 | `██████████` 100% | 3 |

### Spec coverage

| Field | Coverage |
| --- | ---: |
| `ref_api` | 188 / 188 (100%) |
| `roofline` (func or flops+bytes) | 188 / 188 (100%) |
| `source.kernel_map` | 178 / 188 (95%) |
| `source.bench_manifest_driven` | 169 / 188 (90%) |

**Workloads:** 628 total — 3.54 per implemented op.

### Conformance gaps

- Implemented ops without `kernel_map`: **0**
- Implemented ops without `roofline`: **0**
- Implemented ops without `source.bench_manifest_driven`: **0**
- Implemented ops with fewer than two workloads: **0**

<details><summary>Spec-only ops (19)</summary>

| | | |
| --- | --- | --- |
| `CBProducerOp` | `DaCumsumBiasFwdOp` | `DaCumsumFwdOp` |
| `DeltaNetBwdOp` | `DeltaNetFwdOp` | `GLABwdOp` |
| `GLAFwdOp` | `GatedDeltaNetBwdOp` | `GatedDeltaNetFwdOp` |
| `Mamba2BiasFwdOp` | `Mamba2BiasInitStatesFwdOp` | `Mamba2FwdOp` |
| `Mamba2InitStatesFwdOp` | `SSDChunkScanFwdOp` | `SSDChunkStateFwdOp` |
| `SSDChunkStateSeqIdxFwdOp` | `SSDDecodeOp` | `SSDStatePassingFwdOp` |
| `SSDStatePassingInitStatesFwdOp` |  |  |

</details>
