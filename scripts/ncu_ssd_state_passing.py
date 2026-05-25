"""
NCU profiling target for SSDStatePassingFwdKernel.

Run with:
  ncu --set full -o ncu_ssd_state_passing \
      conda run -n flashmlaenv env TMPDIR=/home/yuxian.du/tmp \
      python scripts/ncu_ssd_state_passing.py

Or for a specific config:
  ncu --set full -o ncu_ssd_state_passing \
      conda run -n flashmlaenv env TMPDIR=/home/yuxian.du/tmp \
      python scripts/ncu_ssd_state_passing.py \
      --batch 1 --chunks 16 --heads 32 --dstate 128

Pass --no-tune to use a fixed config instead of autotuning.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tileops.kernels.mamba.ssd_state_passing import SSDStatePassingFwdKernel

parser = argparse.ArgumentParser()
parser.add_argument("--batch",   type=int, default=1)
parser.add_argument("--chunks",  type=int, default=16)
parser.add_argument("--heads",   type=int, default=32)
parser.add_argument("--dstate",  type=int, default=128)
parser.add_argument("--dtype",   type=str, default="float16")
parser.add_argument("--block_d", type=int, default=None)
parser.add_argument("--threads", type=int, default=None)
parser.add_argument("--no-tune", action="store_true")
args = parser.parse_args()

dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
B, C, H, D = args.batch, args.chunks, args.heads, args.dstate

use_tune = not args.no_tune and args.block_d is None and args.threads is None
explicit_config = None
if args.block_d is not None or args.threads is not None:
    explicit_config = {
        "block_d": args.block_d or 64,
        "threads": args.threads or 128,
    }

kernel = SSDStatePassingFwdKernel(
    B, C, H, D,
    has_initial_states=True,
    dtype=dtype,
    config=explicit_config,
    tune=use_tune,
)

states = torch.randn(B, C, H, D, dtype=dtype, device="cuda")
da_chunk_cumsum    = torch.randn(B, H, C, dtype=torch.float32, device="cuda")
init  = torch.randn(B, H, D, dtype=torch.float32, device="cuda")

# warmup (excluded from NCU capture via cudaProfilerStart/Stop)
for _ in range(3):
    kernel.forward(states, da_chunk_cumsum, init)
torch.cuda.synchronize()

# NCU captures this region
torch.cuda.cudart().cudaProfilerStart()
kernel.forward(states, da_chunk_cumsum, init)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
