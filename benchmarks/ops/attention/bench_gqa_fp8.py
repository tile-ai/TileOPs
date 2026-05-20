from dataclasses import dataclass

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, bench_kernel
from benchmarks.ops.attention.gqa_fp8_utils import (
    quantize_kv_fa3_descale,
    quantize_q_fa3_gqa_descale,
)
from tileops.manifest import load_workloads
from tileops.ops import GroupedQueryAttentionPrefillFP8TensorCoreFwdOp

_OP_NAME = "GroupedQueryAttentionPrefillFP8TensorCoreFwdOp"


@dataclass(frozen=True)
class GQAFp8TensorCoreBenchCase:
    batch: int
    seq_len: int
    heads: int
    heads_kv: int
    dim: int
    out_dtype: torch.dtype
    label: str


def _manifest_cases() -> list[GQAFp8TensorCoreBenchCase]:
    cases: list[GQAFp8TensorCoreBenchCase] = []
    for workload in load_workloads(_OP_NAME):
        batch, seq_len, heads, dim = workload["q_shape"]
        _, _, heads_kv, _ = workload["kv_shape"]
        for dtype_name in workload["dtypes"]:
            out_dtype = getattr(torch, dtype_name)
            cases.append(
                GQAFp8TensorCoreBenchCase(
                    batch=batch,
                    seq_len=seq_len,
                    heads=heads,
                    heads_kv=heads_kv,
                    dim=dim,
                    out_dtype=out_dtype,
                    label=f"{workload['label']}-{dtype_name}",
                ))
    return cases


def _make_inputs(case: GQAFp8TensorCoreBenchCase) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    q = torch.randn(
        case.batch, case.seq_len, case.heads, case.dim, device="cuda",
        dtype=torch.float16) * 0.25
    k = torch.randn(
        case.batch, case.seq_len, case.heads_kv, case.dim, device="cuda",
        dtype=torch.float16) * 0.25
    v = torch.randn(
        case.batch, case.seq_len, case.heads_kv, case.dim, device="cuda",
        dtype=torch.float16) * 0.25
    q_fp8, q_descale = quantize_q_fa3_gqa_descale(q, case.heads_kv)
    k_fp8, k_descale = quantize_kv_fa3_descale(k)
    v_fp8, v_descale = quantize_kv_fa3_descale(v)
    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale


@pytest.mark.parametrize("case", [pytest.param(c, id=c.label) for c in _manifest_cases()])
def test_gqa_prefill_fp8_tensor_core_bench(case: GQAFp8TensorCoreBenchCase) -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch fp8 is unavailable")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires Hopper FP8 WGMMA")

    op = GroupedQueryAttentionPrefillFP8TensorCoreFwdOp(
        batch=case.batch,
        heads=case.heads,
        heads_kv=case.heads_kv,
        seq_len=case.seq_len,
        dim=case.dim,
        is_causal=False,
        dtype=case.out_dtype,
    )
    inputs = _make_inputs(case)
    op(*inputs)
    torch.cuda.synchronize()
    latency_ms = bench_kernel(op, args=inputs, n_warmup=1, n_repeat=3, n_trials=1)
    flops, bytes_moved = op.eval_roofline()
    result = {
        "latency_ms": latency_ms,
        "tflops": flops / latency_ms * 1e-9 if latency_ms > 0 else 0.0,
        "gbps": bytes_moved / latency_ms * 1e-6 if latency_ms > 0 else 0.0,
        "flops": flops,
        "bytes": bytes_moved,
    }
    BenchmarkReport.record(op, {"case": case.label}, result, tag="tileops")
