# Naming Convention

Class naming rules for `tileops/ops/` and `tileops/kernels/`.

## Problem

TileOPs class names currently mix three styles:

| Style              | Example                                              | Where                            |
| ------------------ | ---------------------------------------------------- | -------------------------------- |
| Verbose full name  | `GroupQueryAttentionFwdOp`                           | Op layer main files              |
| PascalCase abbrev. | `GqaSlidingWindowFwdOp`, `Fp8QuantOp`                | Op variants, most kernels        |
| ALL-CAPS abbrev.   | `GLAFwdOp`, `NSATopkVarlenOp`, `FFTC2COp`            | Some ops and kernels             |

In addition, the kernel layer contains snake_case class names (`gqa_decode_kernel`, `mhc_pre_kernel`) that violate Python class naming conventions.

## Rules

1. **Abbreviations from the vocabulary list are written in ALL CAPS.** `GQA`, not `Gqa`. `MLA`, not `Mla`. `FP8`, not `Fp8`. `RMS`, not `Rms`.
2. **Non-abbreviation words use PascalCase.** `SlidingWindow`, `Decode`, `Paged`, `Varlen`. `MoE` is a literature-driven exception and stays mixed-case.
3. **Suffix is mandatory.** Op layer ends with `Op`; kernel layer ends with `Kernel`.
4. **No snake_case class names.** Anywhere.
5. **File names stay snake_case.** Only class names follow this convention.

## Mapping rule

```
file (snake_case)  →  class name
gqa_fwd.py         →  GQAFwdOp / GQAFwdKernel
mla_decode.py      →  MLADecodeOp / MLADecodeKernel
rms_norm.py        →  RMSNormOp / RMSNormKernel
deltanet_fwd.py    →  DeltaNetFwdOp / DeltaNetFwdKernel  (no abbrev hit)
fft_c2c.py         →  FFTC2COp / FFTC2CKernel
```

For each underscore-separated segment in the file name: if the segment is a registered abbreviation, write it ALL CAPS; otherwise capitalize the first letter (PascalCase). Append `Op` or `Kernel`.

## Vocabulary

Domain abbreviations recognized by this rule. Only terms in this list are written ALL CAPS — anything else is PascalCase.

> **Status: TBD.** The list below is a working draft. The final vocabulary is decided at the end of the renaming initiative (see tile-ai/TileOPs#850). New entries require:
> - Term is widely used in published literature or hardware documentation
> - At least one operator in TileOPs uses it
> - Justification in the PR that introduces it

| Abbrev | Expansion                                  | Source            |
| ------ | ------------------------------------------ | ----------------- |
| MHA    | Multi-Head Attention                       | Vaswani 2017      |
| GQA    | Grouped Query Attention                    | Ainslie 2023      |
| MLA    | Multi-Head Latent Attention                | DeepSeek-V2 2024  |
| DSA    | DeepSeek Sparse Attention                  | DeepSeek          |
| MHC    | Manifold-Constrained Hyper-Connection      | Chen 2024         |
| GLA    | Gated Linear Attention                     | Yang 2024         |
| NSA    | Native Sparse Attention                    | DeepSeek 2025     |
| FFT    | Fast Fourier Transform                     | —                 |
| RMS    | Root Mean Square                           | —                 |
| FP8    | 8-bit Floating Point                       | —                 |
| FP16   | 16-bit Floating Point                      | —                 |
| FP32   | 32-bit Floating Point                      | IEEE 754          |
| BF16   | bfloat16                                   | Google Brain      |
| GEMM   | General Matrix Multiply                    | BLAS              |
| GEMV   | General Matrix–Vector Multiply             | BLAS              |
| DA     | Discretized A (= dt × A in Mamba SSM)      | Mamba-2 (Dao 2024)|
| KV     | Key–Value (cache)                          | —                 |
| WGMMA  | Warp-Group Matrix Multiply-Accumulate      | NVIDIA Hopper ISA |
| WS     | Warp Specialization                        | NVIDIA Hopper     |
| MoE    | Mixture of Experts (literature exception, **not** ALL CAPS) | Shazeer 2017 |

## Multi-abbreviation boundaries

Adjacent abbreviations are concatenated and remain ALL CAPS:

| Composition     | Class name      |
| --------------- | --------------- |
| FFT + C2C       | `FFTC2COp`      |
| GQA + KV-cache  | `GQAKVCacheOp`  |
| FP8 + GEMM      | `FP8GEMMOp`     |

If concatenation produces something genuinely unreadable, allow a non-abbreviation word as a visual separator (e.g. `FP8GemmOp` where `Gemm` is intentionally not in the vocabulary). The boundary rule is: **inside the vocabulary → ALL CAPS; outside → PascalCase**.

## Enforcement

A CI lint (`scripts/check_naming.py`, added in the closing phase of the renaming initiative) verifies:

- No snake_case class definitions in `tileops/ops/` or `tileops/kernels/`
- Every all-caps segment in a class name is registered in the vocabulary
- Every Op class ends with `Op`; every Kernel class ends with `Kernel`

Until the lint lands, the convention is enforced by review.

## Rationale

- **PEP 8** explicitly recommends ALL CAPS for acronyms in CapWords (`HTTPServerError`, not `HttpServerError`).
- **Python stdlib** follows the same rule (`HTTPServer`, `XMLParser`).
- **TileOPs is a domain-specific library**; its users are LLM kernel engineers who already read GQA, MHA, MLA in papers. Expanding them to `GroupQueryAttention` adds length without adding clarity.
- Verbose class names produce IDE/grep/traceback noise at the 186-operator scale.

See also: [ops-design.md](ops-design.md) for the Op/Kernel two-layer interface, [trust-model.md](trust-model.md) for the manifest/test/impl/bench pipeline.
