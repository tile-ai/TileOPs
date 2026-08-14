from dataclasses import dataclass

import pytest
import torch

from benchmarks.benchmark_base import ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops import GroupedQueryAttentionPrefillDenseFwdOp
from workloads.gqa_fp8_utils import (
    quantize_kv_fa3_descale,
    quantize_q_fa3_gqa_descale,
)

_OP_NAME = "GroupedQueryAttentionPrefillDenseFwdOp"


@dataclass(frozen=True)
class GQAFp8TensorCoreBenchCase:
    batch: int
    seq_len_q: int
    seq_len_kv: int
    heads: int
    heads_kv: int
    dim: int
    is_causal: bool
    sm_scale: float | None
    softcap: float | None
    window_size_left: int
    window_size_right: int
    out_dtype: torch.dtype
    label: str


def _manifest_cases() -> list[GQAFp8TensorCoreBenchCase]:
    cases: list[GQAFp8TensorCoreBenchCase] = []
    for workload in load_workloads(_OP_NAME):
        if workload.get("input_dtype") != "float8_e4m3fn":
            continue
        batch, seq_len_q, heads, dim = workload["q_shape"]
        _, seq_len_kv, heads_kv, _ = workload["kv_shape"]
        for dtype_name in workload["dtypes"]:
            out_dtype = getattr(torch, dtype_name)
            cases.append(
                GQAFp8TensorCoreBenchCase(
                    batch=batch,
                    seq_len_q=seq_len_q,
                    seq_len_kv=seq_len_kv,
                    heads=heads,
                    heads_kv=heads_kv,
                    dim=dim,
                    is_causal=workload.get("is_causal", True),
                    sm_scale=workload.get("sm_scale"),
                    softcap=workload.get("softcap"),
                    window_size_left=workload.get("window_size_left", -1),
                    window_size_right=workload.get("window_size_right", -1),
                    out_dtype=out_dtype,
                    label=f"{workload['label']}-{dtype_name}",
                )
            )
    return cases


def _make_inputs(case: GQAFp8TensorCoreBenchCase) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    q = (
        torch.randn(
            case.batch,
            case.seq_len_q,
            case.heads,
            case.dim,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.25
    )
    k = (
        torch.randn(
            case.batch,
            case.seq_len_kv,
            case.heads_kv,
            case.dim,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.25
    )
    v = (
        torch.randn(
            case.batch,
            case.seq_len_kv,
            case.heads_kv,
            case.dim,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.25
    )
    q_fp8, q_descale = quantize_q_fa3_gqa_descale(q, case.heads_kv)
    k_fp8, k_descale = quantize_kv_fa3_descale(k)
    v_fp8, v_descale = quantize_kv_fa3_descale(v)
    return (
        q_fp8.contiguous(),
        k_fp8.contiguous(),
        v_fp8.contiguous(),
        q_descale,
        k_descale,
        v_descale,
    )


def _fa3_gqa_fp8_fwd(case: GQAFp8TensorCoreBenchCase):
    try:
        from flash_attn_interface import flash_attn_func
    except Exception:
        return None

    def _run(q, k, v, q_descale, k_descale, v_descale):
        return flash_attn_func(
            q,
            k,
            v,
            softmax_scale=case.sm_scale,
            causal=case.is_causal,
            window_size=(case.window_size_left, case.window_size_right),
            softcap=case.softcap or 0.0,
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

    op = GroupedQueryAttentionPrefillDenseFwdOp(
        batch=case.batch,
        heads=case.heads,
        heads_kv=case.heads_kv,
        seq_len=case.seq_len_q,
        seq_len_kv=case.seq_len_kv,
        dim=case.dim,
        is_causal=case.is_causal,
        sm_scale=case.sm_scale,
        softcap=case.softcap,
        window_size_left=case.window_size_left,
        window_size_right=case.window_size_right,
        dtype=case.out_dtype,
    )
    inputs = _make_inputs(case)
    op(*inputs)
    torch.cuda.synchronize()

    bm = ManifestBenchmark(_OP_NAME, op, case)
    functors = {"tileops": op}
    fa3_fn = _fa3_gqa_fp8_fwd(case)
    if fa3_fn is not None:
        functors["fa3"] = fa3_fn

    bm.compare(functors, *inputs, record_as=op, params={"case": case.label})
