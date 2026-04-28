# Performance Guides

Empirical performance lessons per op category. Each category has:

- **Checklist** — heuristic rules for audit (lightweight, always load)
- **Evidence** — measured data and reasoning (load on demand)

## Test Environment

All conclusions are scoped to this configuration. Re-validate when any component changes.

| Component     | Value                                                                        |
| ------------- | ---------------------------------------------------------------------------- |
| GPU           | NVIDIA H200 (HBM3e, 4.8 TB/s peak, SM_90a)                                   |
| Driver / CUDA | 575.57.08 / 12.8                                                             |
| PyTorch       | 2.9.1+cu128                                                                  |
| TileLang      | 0.1.9                                                                        |
| Profiler      | CUPTI (primary); CUDA event+median fallback when CUPTI singleton unavailable |

## Index

| Category    | Checklist                        | Evidence                                           |
| ----------- | -------------------------------- | -------------------------------------------------- |
| Elementwise | [elementwise.md](elementwise.md) | [elementwise-evidence.md](elementwise-evidence.md) |
