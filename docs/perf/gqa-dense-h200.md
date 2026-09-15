# Dense GQA decode on H200

This change addresses the 17 decode workloads reported in
[issue #2131](https://github.com/tile-ai/TileOPs/issues/2131). The issue's worst
external-baseline/TileOPs device-time ratio was 0.7535. On the test system below,
all 17 workloads reach 1.00; the lowest three-round median ratio is 1.028.
The nightly image must reproduce this result before closing the issue.

## Comparator audit

- Q is contiguous BSHD with one query token; K and V are contiguous BSHD.
- FA3 consumes the same tensors and the full cache length. Its noncausal default
  and TileOPs' bottom-right causal mask cover the same keys for one query token.
  No cache append, layout conversion, or head repetition is timed for FA3.
- FlashInfer's batched adapter views that storage as consecutive 256-token pages;
  its batch-one adapter reads the contiguous cache directly. It does not get a
  separately packed copy. Planning is outside timing for both revisions.
- Before timing, every available comparator and both TileOPs revisions passed
  independent FP32 grouped matmul/softmax checks, with TF32 disabled and the
  benchmark's normal dtype tolerances. The reference groups Q heads instead of
  materializing repeated KV heads. The softcap case uses the FP32 torch reference
  because the existing FA3/FlashInfer adapters exclude softcap.

## Change

For Hopper decode with batch greater than one, head dimension 128, at most eight
Q heads per KV head, and no fused RoPE, use 64-token KV tiles. Choose the smallest
power-of-two split count targeting 256 partial CTAs across batch and KV heads,
capped at 16. The existing runtime sequence-length clamp still applies. Batch 32
with eight KV heads already supplies 256 CTAs, so it needs no split or combine.

The batch-one long-context implementation keeps its 64-token tiles through
128K KV tokens and uses 128-token tiles above that boundary on Hopper. Wider
tiles amortize loop overhead at 256K but lose at shorter lengths. Its Op cache
includes this two-valued configuration tier so reused Ops select the right tile
size without keying every exact sequence length. Explicit configs and autotuning
continue to take precedence over defaults.

## Measurements

Measured on 2026-09-15, based on commit
`b5b225d4332d3c62d5888946b147b27bc31c4937`:

- H200, SM clock 1500 MHz, memory clock 3201 MHz, power limit 700 W.
- Driver 595.71.05; PyTorch 2.10.0+cu129.
- TileLang 0.1.11+cu129.git65dbc983; FA3 3.0.0b1;
  FlashInfer 0.6.11.post2; cupti-python 13.2.0.
- Repository CUPTI timing with an L2 flush before each sample, 25 ms warmup and
  100 ms repeat budgets, up to 200 samples per measurement. Device busy time
  excludes launch gaps. Three rounds alternate implementation order on one GPU;
  the table reports the median of the three per-round medians.
- Original code is also loaded under a separate custom-op namespace for paired
  measurements, following a separate pristine-checkout baseline run.

Times are microseconds. `Ratio` is the best external time divided by the new time.

| Workload                      | Dtype | Before |  After | Best external      |  Ratio |
| ----------------------------- | ----- | -----: | -----: | ------------------ | -----: |
| Llama 8B, B32, 4K             | FP16  | 198.02 | 134.58 | FA3 151.15         |  1.123 |
| Llama 8B, B32, 4K             | BF16  | 198.26 | 134.35 | FA3 150.83         |  1.123 |
| Llama 8B, B8, 32K             | FP16  | 266.59 | 256.55 | FA3 267.07         |  1.041 |
| Llama 8B, B8, 32K             | BF16  | 266.77 | 256.77 | FA3 266.50         |  1.038 |
| Llama 70B, B16, 4K            | FP16  |  98.88 |  75.78 | FA3 84.48          |  1.115 |
| Llama 70B, B16, 4K            | BF16  |  98.27 |  75.84 | FA3 84.03          |  1.108 |
| Llama 70B, B4, 32K            | FP16  | 142.37 | 136.62 | FA3 148.72         |  1.089 |
| Llama 70B, B4, 32K            | BF16  | 141.86 | 135.81 | FA3 147.68         |  1.087 |
| Llama 8B, B32, 4K, softcap 50 | FP16  | 212.67 | 137.23 | torch-ref 13327.90 | 97.119 |
| Qwen3 30B A3B, B1, 1K         | FP16  |   5.57 |   5.57 | FlashInfer 9.57    |  1.718 |
| Qwen3 30B A3B, B1, 4K         | FP16  |   8.35 |   8.35 | FlashInfer 11.62   |  1.391 |
| Qwen3 30B A3B, B1, 8K         | FP16  |  11.10 |  11.10 | FlashInfer 14.10   |  1.269 |
| Qwen3 30B A3B, B1, 16K        | FP16  |  16.48 |  16.45 | FlashInfer 21.82   |  1.327 |
| Qwen3 30B A3B, B1, 32K        | FP16  |  26.59 |  26.61 | FlashInfer 34.69   |  1.304 |
| Qwen3 30B A3B, B1, 64K        | FP16  |  43.90 |  43.84 | FlashInfer 53.12   |  1.212 |
| Qwen3 30B A3B, B1, 128K       | FP16  |  76.74 |  76.51 | FlashInfer 83.34   |  1.089 |
| Qwen3 30B A3B, B1, 256K       | FP16  | 145.06 | 137.68 | FlashInfer 141.50  |  1.028 |

The 256K margin is only 2.8%, and nightly uses different PyTorch, TileLang and
baseline builds. These measurements support the change; they do not replace a
nightly result. The FP8/fused-RoPE prefill workloads added after the issue are
outside this 17-row performance comparison.

## Other experiments

- Swept generic KV tiles 64/128/256, splits 1/2/4/8/16/32 and pipeline stages 1/2.
  Reducing splits accounts for the large batched improvement. Some 256-token
  configurations exceed H200 shared-memory capacity.
- Tested the existing explicit Hopper producer/consumer kernel with KV tiles
  64/128/256 and splits 1/2/4/8/16. It did not beat the best generic configuration
  on the worst workload.
- Reordered the split grid to put KV heads before batch. Gains were about 1% or
  absent across the tested geometries, so the grid is unchanged.
- At batch one and 256K, 16 splits were too few (about 246 us); 64 splits with
  64-token tiles gave about 141 us. Keeping 32 splits and widening the tile gave
  about 137 us. At 1K–128K the original 64-token, two-stage choice was preferable.

## Reproduce

Install the checkout in an isolated environment with the versions above, and run
on an idle H200 with matching clocks:

```bash
CUDA_VISIBLE_DEVICES=2 python -m pytest benchmarks/ops/attention/bench_gqa.py \
  -k test_gqa_dense_decode_bench -v -s --junitxml=decode-results.xml
CUDA_VISIBLE_DEVICES=2 python -m pytest tests/ops/attention/test_gqa.py -k 'not bwd' -q
```

The regressions include batched FP16/BF16 tails, crossing the long-context tile
boundary in both directions on one Op, and bounded reuse of the two cache tiers.

## Validation

- Forward tests: 39 passed (four backward cases deselected).
- Native decode benchmark entry: 17 passed; all 17 external-baseline ratios exceeded 1.00, with a minimum of 1.036 in that separate run.
- Ruff lint/format and the shipped-reference, Op docstring, and TileLang idiom checks passed.
