from dataclasses import dataclass

import pytest
import torch

from benchmarks.baselines import assert_matches_reference
from benchmarks.benchmark_base import ManifestBenchmark, then_dtype, workload_params
from tileops.manifest import load_workloads
from tileops.ops import GroupedQueryAttentionDenseFwdOp
from workloads.gqa_fp8_utils import (
    quantize_kv_fa3_descale,
    quantize_q_fa3_gqa_descale,
)

_OP_NAME = "GroupedQueryAttentionDenseFwdOp"


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


def _fp8_case_args(workload: dict, dtype: torch.dtype) -> tuple:
    """One case object; the fp8 tensor-core path is the only one this bench runs."""
    batch, seq_len_q, heads, dim = workload["q_shape"]
    _, seq_len_kv, heads_kv, _ = workload["kv_shape"]
    return (
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
            out_dtype=dtype,
        ),
    )


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


def _torch_sdpa_dequant_fwd(case: GQAFp8TensorCoreBenchCase):
    """Dequantize with the descales, then attend in the output dtype.

    Needs nothing optional, so the row reports a comparison whether or not flash-attn 3
    is installed. ``scaled_dot_product_attention`` avoids the ``seq_len x seq_len`` score
    matrix a per-head loop would materialize -- 12 GiB at the largest row.
    """
    group_size = case.heads // case.heads_kv

    if case.softcap or case.window_size_left >= 0 or case.window_size_right >= 0:
        return None

    def _run(q, k, v, q_descale, k_descale, v_descale):
        batch, dim = case.batch, case.dim
        heads, heads_kv = case.heads, case.heads_kv
        q_deq = q.float().reshape(batch, case.seq_len_q, heads_kv, group_size, dim)
        q_deq = (q_deq * q_descale[:, None, :, None, None]).reshape(
            batch, case.seq_len_q, heads, dim
        )
        k_deq = k.float() * k_descale[:, None, :, None]
        v_deq = v.float() * v_descale[:, None, :, None]
        out = torch.nn.functional.scaled_dot_product_attention(
            q_deq.transpose(1, 2).to(case.out_dtype),
            k_deq.transpose(1, 2).to(case.out_dtype),
            v_deq.transpose(1, 2).to(case.out_dtype),
            is_causal=case.is_causal,
            enable_gqa=True,
            scale=case.sm_scale,
        )
        return out.transpose(1, 2).contiguous()

    return _run


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


@pytest.mark.parametrize(
    "case",
    workload_params(
        [w for w in load_workloads(_OP_NAME) if w.get("input_dtype") == "float8_e4m3fn"],
        then_dtype(_fp8_case_args),
    ),
)
def test_gqa_prefill_fp8_tensor_core_bench(case: GQAFp8TensorCoreBenchCase) -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch fp8 is unavailable")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires Hopper FP8 WGMMA")

    op = GroupedQueryAttentionDenseFwdOp(
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
    sdpa_fn = _torch_sdpa_dequant_fwd(case)
    fa3_fn = _fa3_gqa_fp8_fwd(case)
    reference_fn = fa3_fn or sdpa_fn
    if reference_fn is None:
        pytest.skip("this FP8 feature combination requires FlashAttention-3")
    # The tolerance tests/ops/attention/test_gqa_fp8.py holds its own dequantized
    # reference to: fp8 quantization is the whole difference between the two.
    assert_matches_reference(op, reference_fn, *inputs, atol=5e-2, rtol=5e-2)

    functors = {"tileops": op}
    if sdpa_fn is not None:
        functors["torch-sdpa-dequant"] = sdpa_fn
    if fa3_fn is not None:
        functors["fa3"] = fa3_fn

    bm.compare(
        functors,
        *inputs,
        record_as=op,
        params={
            "batch": case.batch,
            "seq_len_q": case.seq_len_q,
            "seq_len_kv": case.seq_len_kv,
            "heads": case.heads,
            "heads_kv": case.heads_kv,
            "dim": case.dim,
            "dtype": case.out_dtype,
        },
    )
