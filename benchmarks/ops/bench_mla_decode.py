"""MLA decode against FlashInfer and vLLM's Hopper-compatible FlashMLA."""

import pytest
import torch

from benchmarks.baselines import FLASHINFER_TAG, flashinfer_op, vllm_op
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import MLADecodeWithKVCacheFwdOp
from workloads.attention.mla import MLADecodeCall


@pytest.mark.parametrize("call", manifest_calls(MLADecodeWithKVCacheFwdOp))
def test_mla_decode_bench(call) -> None:
    workload = MLADecodeCall(call)
    inputs = workload.gen_inputs()
    q, q_pe, k, _ = inputs
    batch, heads, dim = q.shape
    length, dim_pe = (k.shape[1], q_pe.shape[-1])
    page = 64
    pages = length // page
    indices = torch.arange(batch * pages, dtype=torch.int32, device=q.device)
    lengths = torch.full((batch,), length, dtype=torch.int32, device=q.device)
    wrapper_cls = flashinfer_op("mla.BatchMLAPagedAttentionWrapper")
    wrapper = wrapper_cls(torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device))
    wrapper.plan(
        torch.arange(batch + 1, dtype=torch.int32, device=q.device),
        torch.arange(batch + 1, dtype=torch.int32, device=q.device) * pages,
        indices,
        lengths,
        heads,
        dim,
        dim_pe,
        page,
        False,
        (dim + dim_pe) ** (-0.5),
        q.dtype,
        k.dtype,
    )

    def flashinfer_fn(q, q_pe, k, k_pe):
        return wrapper.run(q, q_pe, k.reshape(-1, page, dim), k_pe.reshape(-1, page, dim_pe))

    flashmla = vllm_op("flash_mla_with_kvcache", "v1.attention.ops.flashmla")
    metadata, _ = vllm_op("get_mla_metadata", "v1.attention.ops.flashmla")()
    table = indices.reshape(batch, pages)

    def flashmla_fn(q, q_pe, k, k_pe):
        query = torch.cat((q, q_pe), -1).unsqueeze(1)
        cache = torch.cat((k, k_pe), -1).reshape(-1, page, 1, dim + dim_pe)
        out, _ = flashmla(query, cache, table, lengths, dim, metadata)
        return out.squeeze(1)

    op = MLADecodeWithKVCacheFwdOp(**workload.arguments(), tune=True)
    functors = {"tileops": op, FLASHINFER_TAG: flashinfer_fn, "flashmla": flashmla_fn}
    ManifestBenchmark(op, workload).compare(functors, *inputs, count_copies=True)
