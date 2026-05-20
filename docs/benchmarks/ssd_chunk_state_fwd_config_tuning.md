# SSDChunkStateFwdOp Benchmark — chunk_state config tuning

## Environment

| | |
|---|---|
| GPU | NVIDIA H200 (GPU 1) |
| SM clock | **1830 MHz (locked)** |
| CUDA | 12.9 |
| PyTorch | 2.12.0.dev20260401+cu129 |
| Baseline | commit `81ebb55` — default config `block_n=64, block_p=64, block_l=64, threads=128` |
| This PR | commit `1323910` — `block_n=128, block_p=64, block_l=128, threads=256` (non-seqidx); `block_n=128, block_p=64, block_l=128, threads=128` (seqidx) |

Timing method: L2-flush CUDA events, 50 warmup + 500 timed iterations per shape.
Inputs randomly seeded. Both kernels pre-compiled before timing.

---

## Benchmark

### ssd_chunk_state_fwd

| shape | dtype | has_seq_idx | before (µs) | after (µs) | speedup |
|---|---:|---:|---:|---:|---:|
| 370m-b1: B=1,C=8,L=256,H=32,P=64,N=128,G=1 | fp16 | false | 139.06 | 138.36 | **1.01×** |
| 370m-b4: B=4,C=8,L=256,H=32,P=64,N=128,G=1 | fp16 | false | 276.30 | 252.11 | **1.10×** |
| 1.3b-b1: B=1,C=8,L=256,H=64,P=64,N=128,G=1 | fp16 | false | 183.43 | 182.10 | **1.01×** |
| 1.3b-b4: B=4,C=8,L=256,H=64,P=64,N=128,G=1 | fp16 | false | 435.27 | 389.18 | **1.12×** |
| 2.7b-b1: B=1,C=8,L=256,H=80,P=64,N=128,G=1 | bf16 | false | 206.40 | 206.50 | **1.00×** |
| 1.3b-b1-L8k: B=1,C=32,L=256,H=64,P=64,N=128,G=1 | fp16 | false | 434.45 | 388.80 | **1.12×** |
| 1.3b-b1-seqidx: B=1,C=8,L=256,H=64,P=64,N=128,G=1 | fp16 | true | 205.72 | 184.50 | **1.12×** |
