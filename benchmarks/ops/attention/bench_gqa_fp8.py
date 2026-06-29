from dataclasses import dataclass

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, bench_kernel
from tileops.manifest import load_workloads
from tileops.ops import GroupedQueryAttentionPrefillFwdOp
from tileops.testing.gqa_fp8_utils import (
    quantize_kv_fa3_descale,
    quantize_q_fa3_gqa_descale,
)

_OP_NAME = "GroupedQueryAttentionPrefillFwdOp"


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
        if workload.get("backend") != "fp8":
            continue
        batch = workload["batch"]
        seq_len = workload["max_seqlen_q"]
        heads = workload["heads"]
        heads_kv = workload["heads_kv"]
        dim = workload["dim"]
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
                )
            )
    return cases


def _make_inputs(case: GQAFp8TensorCoreBenchCase) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    q = (
        torch.randn(
            case.batch, case.seq_len, case.heads, case.dim, device="cuda", dtype=torch.float16
        )
        * 0.25
    )
    k = (
        torch.randn(
            case.batch, case.seq_len, case.heads_kv, case.dim, device="cuda", dtype=torch.float16
        )
        * 0.25
    )
    v = (
        torch.randn(
            case.batch, case.seq_len, case.heads_kv, case.dim, device="cuda", dtype=torch.float16
        )
        * 0.25
    )
    q_fp8, q_descale = quantize_q_fa3_gqa_descale(q, case.heads_kv)
    k_fp8, k_descale = quantize_kv_fa3_descale(k)
    v_fp8, v_descale = quantize_kv_fa3_descale(v)
    cu = torch.tensor([0, case.seq_len], device="cuda", dtype=torch.int32)
    return (
        q_fp8.reshape(case.batch * case.seq_len, case.heads, case.dim).contiguous(),
        k_fp8.reshape(case.batch * case.seq_len, case.heads_kv, case.dim).contiguous(),
        v_fp8.reshape(case.batch * case.seq_len, case.heads_kv, case.dim).contiguous(),
        cu,
        cu,
        q_descale,
        k_descale,
        v_descale,
    )


def _fa3_gqa_fp8_fwd(case: GQAFp8TensorCoreBenchCase):
    try:
        from flash_attn_interface import flash_attn_func  # noqa: PLC0415
    except Exception:
        return None

    def _run(q, k, v, cu_q, cu_kv, q_descale, k_descale, v_descale):
        del cu_q, cu_kv
        return flash_attn_func(
            q.reshape(case.batch, case.seq_len, case.heads, case.dim),
            k.reshape(case.batch, case.seq_len, case.heads_kv, case.dim),
            v.reshape(case.batch, case.seq_len, case.heads_kv, case.dim),
            causal=False,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    return _run


@pytest.mark.parametrize("case", [pytest.param(c, id=c.label) for c in _manifest_cases()])
def test_gqa_prefill_fp8_tensor_core_bench(case: GQAFp8TensorCoreBenchCase) -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch fp8 is unavailable")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires Hopper FP8 WGMMA")

    op = GroupedQueryAttentionPrefillFwdOp(
        batch=case.batch,
        heads=case.heads,
        heads_kv=case.heads_kv,
        dim=case.dim,
        max_seqlen_q=case.seq_len,
        max_seqlen_kv=case.seq_len,
        is_causal=False,
        dtype=case.out_dtype,
        backend="fp8",
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

    fa3_fn = _fa3_gqa_fp8_fwd(case)
    if fa3_fn is not None:
        fa3_latency_ms = bench_kernel(fa3_fn, args=inputs, n_warmup=1, n_repeat=3, n_trials=1)
        fa3_result = {
            "latency_ms": fa3_latency_ms,
            "tflops": flops / fa3_latency_ms * 1e-9 if fa3_latency_ms > 0 else 0.0,
            "gbps": bytes_moved / fa3_latency_ms * 1e-6 if fa3_latency_ms > 0 else 0.0,
            "flops": flops,
            "bytes": bytes_moved,
        }
        BenchmarkReport.record(op, {"case": case.label}, fa3_result, tag="fa3")
