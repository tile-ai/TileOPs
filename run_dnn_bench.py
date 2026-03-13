"""Quick DNN-sized benchmark for PR 463 description.

Shapes: (tokens, hidden_dim) with DNN-realistic hidden dimensions.
Reports bandwidth in TB/s and latency in μs using CUDA event timing.
"""

import sys

sys.path.insert(0, "/home/cy/TileOps-dev/TileOPs/.claude/worktrees/issue-438")

import torch
import torch.nn.functional as F
from tilelang.profiler import do_bench

from tileops.ops.elementwise import (
    BitwiseAndOp,
    BitwiseOrOp,
    BitwiseXorOp,
    DivOp,
    EqOp,
    FloorDivideOp,
    GeluAndMulOp,
    GeluTanhAndMulOp,
    GeOp,
    GtOp,
    LeOp,
    LerpOp,
    LogicalAndOp,
    LogicalOrOp,
    LtOp,
    MaximumOp,
    MinimumOp,
    MulOp,
    NeOp,
    PowOp,
    RemainderOp,
    SubOp,
)

# DNN-realistic shapes: (tokens, hidden_dim)
M = 1024
HIDDEN_DIMS = [4096, 10240, 20480]
WARMUP = 200
REP = 500


def bench(fn, *inputs):
    """Latency in μs using CUDA event backend."""
    def f():
        return fn(*inputs)
    with torch.no_grad():
        lat_ms = do_bench(f, warmup=WARMUP, rep=REP, backend='event')
    return lat_ms * 1000.0  # → μs


def bw_tbs(n_bytes, lat_us):
    """Bandwidth in TB/s."""
    return n_bytes / (lat_us * 1e-6) / 1e12


def fmt(v, decimals=2):
    return f"{v:.{decimals}f}"


def randn_pair(shape, dtype):
    return torch.randn(*shape, device="cuda", dtype=dtype), torch.randn(*shape, device="cuda", dtype=dtype)


def pos_pair(shape, dtype):
    return (torch.rand(*shape, device="cuda", dtype=dtype) + 0.1,
            torch.rand(*shape, device="cuda", dtype=dtype) + 0.1)


def int_pair(shape):
    return (torch.randint(-1000, 1000, shape, device="cuda", dtype=torch.int32),
            torch.randint(-1000, 1000, shape, device="cuda", dtype=torch.int32))


# ---------------------------------------------------------------------------
print("\n### Binary Arithmetic (fp16, bandwidth TB/s)")
print("| Op | Shape (M×N) | TileOPs | PyTorch | Speedup |")
print("|---|---|---|---|---|")

arith_ops = [
    ("sub",          SubOp,         torch.sub,                           randn_pair),
    ("mul",          MulOp,         torch.mul,                           randn_pair),
    ("div",          DivOp,         torch.div,                           pos_pair),
    ("remainder",    RemainderOp,   torch.remainder,                     pos_pair),
    ("pow",          PowOp,         torch.pow,                           pos_pair),
    ("floor_divide", FloorDivideOp, torch.floor_divide,                  pos_pair),
    ("lerp",         LerpOp,        lambda a, b: torch.lerp(a, b, 0.5), randn_pair),
    ("maximum",      MaximumOp,     torch.maximum,                       randn_pair),
    ("minimum",      MinimumOp,     torch.minimum,                       randn_pair),
]

dtype = torch.float16
elem = dtype.itemsize
for op_name, op_cls, pt_fn, gen in arith_ops:
    for N in HIDDEN_DIMS:
        shape = (M, N)
        n_total = M * N
        a, b = gen(shape, dtype)
        op = op_cls(a_shape=shape, b_shape=shape, dtype=dtype)
        t = bench(op, a, b)
        p = bench(pt_fn, a, b)
        n_bytes = n_total * (2 * elem + elem)  # 2 reads + 1 write
        bw_t = bw_tbs(n_bytes, t)
        bw_p = bw_tbs(n_bytes, p)
        speedup = bw_t / max(bw_p, 0.001)
        print(f"| {op_name} | {M}×{N} | {fmt(bw_t)} | {fmt(bw_p)} | {fmt(speedup)}x |")

# ---------------------------------------------------------------------------
print("\n### Comparison (fp16, bandwidth TB/s)")
print("| Op | Shape (M×N) | TileOPs | PyTorch | Speedup |")
print("|---|---|---|---|---|")

cmp_ops = [
    ("eq", EqOp, torch.eq),
    ("ne", NeOp, torch.ne),
    ("gt", GtOp, torch.gt),
    ("lt", LtOp, torch.lt),
    ("ge", GeOp, torch.ge),
    ("le", LeOp, torch.le),
]

for op_name, op_cls, pt_fn in cmp_ops:
    for N in HIDDEN_DIMS:
        shape = (M, N)
        n_total = M * N
        a, b = randn_pair(shape, dtype)
        op = op_cls(a_shape=shape, b_shape=shape, dtype=dtype)
        t = bench(op, a, b)
        p = bench(pt_fn, a, b)
        n_bytes = n_total * (2 * elem + 1)  # 2 fp16 reads + 1 bool write
        bw_t = bw_tbs(n_bytes, t)
        bw_p = bw_tbs(n_bytes, p)
        speedup = bw_t / max(bw_p, 0.001)
        print(f"| {op_name} | {M}×{N} | {fmt(bw_t)} | {fmt(bw_p)} | {fmt(speedup)}x |")

# ---------------------------------------------------------------------------
print("\n### Logical (fp16 input, bandwidth TB/s)")
print("| Op | Shape (M×N) | TileOPs | PyTorch | Speedup |")
print("|---|---|---|---|---|")

log_ops = [
    ("logical_and", LogicalAndOp, torch.logical_and),
    ("logical_or",  LogicalOrOp,  torch.logical_or),
]

for op_name, op_cls, pt_fn in log_ops:
    for N in HIDDEN_DIMS:
        shape = (M, N)
        n_total = M * N
        a = torch.randn(*shape, device="cuda", dtype=dtype)
        b = torch.randn(*shape, device="cuda", dtype=dtype)
        op = op_cls(a_shape=shape, b_shape=shape, dtype=dtype)
        t = bench(op, a, b)
        a_bool = (a > 0)
        b_bool = (b > 0)
        p = bench(pt_fn, a_bool, b_bool)
        n_bytes = n_total * (2 * elem + 1)  # 2 fp16 reads + 1 bool write
        bw_t = bw_tbs(n_bytes, t)
        bw_p = bw_tbs(n_bytes, p)
        speedup = bw_t / max(bw_p, 0.001)
        print(f"| {op_name} | {M}×{N} | {fmt(bw_t)} | {fmt(bw_p)} | {fmt(speedup)}x |")

# ---------------------------------------------------------------------------
print("\n### Bitwise (int32, bandwidth TB/s)")
print("| Op | Shape (M×N) | TileOPs | PyTorch | Speedup |")
print("|---|---|---|---|---|")

bit_ops = [
    ("bitwise_and", BitwiseAndOp, torch.bitwise_and),
    ("bitwise_or",  BitwiseOrOp,  torch.bitwise_or),
    ("bitwise_xor", BitwiseXorOp, torch.bitwise_xor),
]

int_elem = torch.int32.itemsize
for op_name, op_cls, pt_fn in bit_ops:
    for N in HIDDEN_DIMS:
        shape = (M, N)
        n_total = M * N
        a, b = int_pair(shape)
        op = op_cls(a_shape=shape, b_shape=shape, dtype=torch.int32)
        t = bench(op, a, b)
        p = bench(pt_fn, a, b)
        n_bytes = n_total * 3 * int_elem  # 2 reads + 1 write
        bw_t = bw_tbs(n_bytes, t)
        bw_p = bw_tbs(n_bytes, p)
        speedup = bw_t / max(bw_p, 0.001)
        print(f"| {op_name} | {M}×{N} | {fmt(bw_t)} | {fmt(bw_p)} | {fmt(speedup)}x |")

# ---------------------------------------------------------------------------
print("\n### Fused Gated (fp16, bandwidth TB/s)")
print("| Op | Shape (M×N) | TileOPs | PyTorch | Speedup |")
print("|---|---|---|---|---|")

def gelu_and_mul_pt(x):
    half = x.shape[-1] // 2
    return F.gelu(x[..., :half]) * x[..., half:]

def gelu_tanh_and_mul_pt(x):
    half = x.shape[-1] // 2
    return F.gelu(x[..., :half], approximate="tanh") * x[..., half:]

fused_ops = [
    ("gelu_and_mul",      GeluAndMulOp,     gelu_and_mul_pt),
    ("gelu_tanh_and_mul", GeluTanhAndMulOp, gelu_tanh_and_mul_pt),
]

for op_name, op_cls, pt_fn in fused_ops:
    for N in HIDDEN_DIMS:
        n_total = M * N
        x = torch.randn(M, 2 * N, device="cuda", dtype=dtype)
        op = op_cls(M=M, N=N, dtype=dtype)
        t = bench(op, x)
        p = bench(pt_fn, x)
        n_bytes = n_total * 3 * elem  # read (M,2N) + write (M,N)
        bw_t = bw_tbs(n_bytes, t)
        bw_p = bw_tbs(n_bytes, p)
        speedup = bw_t / max(bw_p, 0.001)
        print(f"| {op_name} | {M}×{N} | {fmt(bw_t)} | {fmt(bw_p)} | {fmt(speedup)}x |")

# ---------------------------------------------------------------------------
print("\n### Broadcast: Bias-Add Pattern (fp16, bandwidth TB/s)")
print("| Op | a_shape | b_shape | TileOPs | PyTorch | Speedup |")
print("|---|---|---|---|---|---|")

bcast_ops = [
    ("sub", SubOp, torch.sub, randn_pair),
    ("mul", MulOp, torch.mul, randn_pair),
    ("div", DivOp, torch.div, pos_pair),
]

for op_name, op_cls, pt_fn, gen in bcast_ops:
    for N in HIDDEN_DIMS:
        a_shape = (M, N)
        b_shape = (1, N)
        n_total = M * N
        a = gen(a_shape, dtype)[0]
        b = gen(b_shape, dtype)[0]
        op = op_cls(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
        t = bench(op, a, b)
        p = bench(pt_fn, a, b)
        # Memory: read a (M*N) + read b (1*N, broadcast) + write out (M*N)
        n_bytes = (n_total + N + n_total) * elem
        bw_t = bw_tbs(n_bytes, t)
        bw_p = bw_tbs(n_bytes, p)
        speedup = bw_t / max(bw_p, 0.001)
        print(f"| {op_name} | {M}×{N} | 1×{N} | {fmt(bw_t)} | {fmt(bw_p)} | {fmt(speedup)}x |")

print()
