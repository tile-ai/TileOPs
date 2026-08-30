import pytest
import torch

from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    reference_tolerance,
)
from benchmarks.benchmark_base import (
    ManifestBenchmark,
    then_dtype,
    workload_params,
)
from benchmarks.ops.attention.workload_args import dsa_decode_args
from tileops.manifest import load_workloads
from tileops.ops import DeepSeekSparseAttentionDecodeWithKVCacheFwdOp
from workloads.attention.deepseek import DsaDecodeWorkload

_OP_NAME = "DeepSeekSparseAttentionDecodeWithKVCacheFwdOp"


_DSA_DECODE_BENCH_PARAMS = workload_params(
    load_workloads(_OP_NAME),
    then_dtype(
        dsa_decode_args,
        tune=False,
    ),
)


def _torch_sdpa_dsa(test: DsaDecodeWorkload):
    """SDPA over the selection ``ref_program`` masks, or None for a row it cannot serve.

    Same computation, without the reference's float32 upcast and materialized score
    tensor. A single kv head lets the mask and the cache broadcast over the query heads.
    """
    if test.heads_kv != 1:
        return None

    def fn(q, kv, indices):
        b, sq, h, dim_q = q.shape
        sk = kv.shape[1]
        dim = test.dim
        mask = test.selection_mask(indices).expand(b, h, sq, sk)
        k = kv.permute(0, 2, 1, 3).expand(b, h, sk, dim_q)
        v = kv[..., :dim].permute(0, 2, 1, 3).expand(b, h, sk, dim)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k,
            v,
            attn_mask=mask,
            scale=test.sm_scale if test.sm_scale is not None else dim_q**-0.5,
        )
        return out.permute(0, 2, 1, 3).reshape(b, sq, h, dim).to(torch.float16)

    return fn


def _torch_gather_dsa(test: DsaDecodeWorkload):
    """Dense attention over only the gathered selection, or None when it buys nothing.

    Gathering beats masking only where the selection is smaller than the cache.
    """
    if test.heads_kv != 1 or test.topk >= test.seq_len_kv:
        return None

    def fn(q, kv, indices):
        b, sq, h, dim_q = q.shape
        sk = kv.shape[1]
        dim, topk = test.dim, indices.shape[-1]
        idx = indices.transpose(1, 2).clamp(max=sk - 1).long()
        valid = torch.gather(test.selection_mask(indices), 3, idx)
        gathered = torch.gather(
            kv.squeeze(2), 1, idx.reshape(b, sq * topk, 1).expand(-1, -1, dim_q)
        ).view(b, sq, topk, dim_q)
        scale = test.sm_scale if test.sm_scale is not None else dim_q**-0.5
        scores = torch.einsum("bhqd,bqkd->bhqk", q.permute(0, 2, 1, 3), gathered)
        probs = scores.float().mul(scale).masked_fill(~valid, float("-inf")).softmax(-1)
        out = torch.einsum("bhqk,bqkd->bhqd", probs.to(kv.dtype), gathered[..., :dim])
        return out.permute(0, 2, 1, 3).reshape(b, sq, h, dim).to(torch.float16)

    return fn


@pytest.mark.parametrize(
    "batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, q_start_index_s, sm_scale, dtype, tune",
    _DSA_DECODE_BENCH_PARAMS,
)
def test_dsa_decode_bench(
    batch: int,
    heads: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    dim_tail: int,
    topk: int,
    stride_kv: int,
    heads_kv: int,
    q_start_index_s: int,
    sm_scale: float,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = DsaDecodeWorkload(
        batch,
        heads,
        seq_len_q,
        seq_len_kv,
        dim,
        dim_tail,
        topk,
        stride_kv,
        heads_kv,
        q_start_index_s,
        sm_scale=sm_scale,
        dtype=dtype,
    )
    inputs = test.gen_inputs()

    op = DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(
        batch,
        heads,
        seq_len_q,
        seq_len_kv,
        dim,
        dim_tail,
        topk,
        stride_kv,
        heads_kv,
        q_start_index_s,
        sm_scale=sm_scale,
        tune=tune,
    )
    bm = ManifestBenchmark(op, test)

    baselines = {}
    sdpa_fn = _torch_sdpa_dsa(test)
    if sdpa_fn is not None:
        baselines["torch-sdpa"] = sdpa_fn
    gather_fn = _torch_gather_dsa(test)
    if gather_fn is not None:
        baselines["torch-gather"] = gather_fn
    for fn in baselines.values():
        assert_matches_reference(fn, test.ref_program, *inputs, **reference_tolerance(dtype))

    bm.compare(
        {
            "tileops": op,
            "torch-ref": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
            **baselines,
        },
        *inputs,
        params=locals(),
    )
