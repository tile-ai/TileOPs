from dataclasses import dataclass

import pytest
import torch

from benchmarks.baselines import assert_matches_reference
from benchmarks.benchmark_base import ManifestBenchmark, workload_params
from tileops.manifest import load_workloads
from tileops.ops import GroupedQueryAttentionPrefillFwdOp
from workloads.gqa_fp8_utils import (
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
    validate_uniform_cu_seqlens: bool
    out_dtype: torch.dtype


def _fp8_case_args(workload: dict, dtype: torch.dtype) -> tuple:
    """One case object; the fp8 tensor-core path is the only one this bench runs."""
    return (
        GQAFp8TensorCoreBenchCase(
            batch=workload["batch"],
            seq_len=workload["max_seqlen_q"],
            heads=workload["heads"],
            heads_kv=workload["heads_kv"],
            dim=workload["dim"],
            validate_uniform_cu_seqlens=workload.get("validate_uniform_cu_seqlens", True),
            out_dtype=dtype,
        ),
    )


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


def _torch_sdpa_dequant_fwd(case: GQAFp8TensorCoreBenchCase):
    """Dequantize with the descales, then attend in the output dtype.

    Needs nothing optional, so the row reports a comparison whether or not flash-attn 3
    is installed. ``scaled_dot_product_attention`` avoids the ``seq_len x seq_len`` score
    matrix a per-head loop would materialize -- 12 GiB at the largest row.
    """
    group_size = case.heads // case.heads_kv

    def _run(q, k, v, cu_q, cu_kv, q_descale, k_descale, v_descale):
        del cu_q, cu_kv
        batch, seq_len, dim = case.batch, case.seq_len, case.dim
        heads, heads_kv = case.heads, case.heads_kv
        q_deq = q.float().reshape(batch, seq_len, heads_kv, group_size, dim)
        q_deq = (q_deq * q_descale[:, None, :, None, None]).reshape(batch, seq_len, heads, dim)
        k_deq = k.float().reshape(batch, seq_len, heads_kv, dim) * k_descale[:, None, :, None]
        v_deq = v.float().reshape(batch, seq_len, heads_kv, dim) * v_descale[:, None, :, None]
        out = torch.nn.functional.scaled_dot_product_attention(
            q_deq.transpose(1, 2).to(case.out_dtype),
            k_deq.transpose(1, 2).to(case.out_dtype),
            v_deq.transpose(1, 2).to(case.out_dtype),
            is_causal=False,
            enable_gqa=True,
        )
        return out.transpose(1, 2).reshape(batch * seq_len, heads, dim)

    return _run


def _fa3_gqa_fp8_fwd(case: GQAFp8TensorCoreBenchCase):
    try:
        from flash_attn_interface import flash_attn_func
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


@pytest.mark.parametrize(
    "case",
    workload_params(
        [w for w in load_workloads(_OP_NAME) if w.get("backend") == "fp8"],
        _fp8_case_args,
    ),
)
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
        validate_uniform_cu_seqlens=case.validate_uniform_cu_seqlens,
    )
    inputs = _make_inputs(case)
    op(*inputs)
    torch.cuda.synchronize()

    bm = ManifestBenchmark(op, case)
    sdpa_fn = _torch_sdpa_dequant_fwd(case)
    # The tolerance tests/ops/attention/test_gqa_fp8.py holds its own dequantized
    # reference to: fp8 quantization is the whole difference between the two.
    assert_matches_reference(op, sdpa_fn, *inputs, atol=5e-2, rtol=5e-2)

    functors = {"tileops": op, "torch-sdpa-dequant": sdpa_fn}
    fa3_fn = _fa3_gqa_fp8_fwd(case)
    if fa3_fn is not None:
        functors["fa3"] = fa3_fn

    bm.compare(
        functors,
        *inputs,
        params={
            "batch": case.batch,
            "seq_len": case.seq_len,
            "heads": case.heads,
            "heads_kv": case.heads_kv,
            "dim": case.dim,
            "dtype": case.out_dtype,
        },
    )
