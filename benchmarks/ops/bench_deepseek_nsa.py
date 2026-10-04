"""NSA varlen benchmarks against FLA forward kernels, using the same precompressed K/V.

Top-k uses ``lse=None`` to include LSE computation on both sides. FLA's other two
passes also write LSE; its cached sequence metadata helpers run warm during timing.
"""

import pytest

from benchmarks.baselines import FLA_TAG, assert_output_spec, fla_op
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.attention import NSACompressedVarlenFwdOp, NSATopKVarlenFwdOp, NSAVarlenFwdOp
from workloads.attention.nsa import NsaCmpFwdCall, NsaFwdCall, NsaTopkCall


def _setup(op_cls, workload_cls, call):
    workload = workload_cls(call)
    op = op_cls(**workload.arguments())
    return workload, workload.gen_inputs(), ManifestBenchmark(op, workload), op


def _fla_nsa_fwd(workload: NsaFwdCall):
    """Selected-block attention. None where fla cannot serve the row."""
    # parallel_nsa_fwd masks each selected block against the query's own position, so it has no
    # non-causal form.
    if not workload.is_causal:
        return None
    fwd = fla_op("ops.nsa.parallel.parallel_nsa_fwd")

    def fn(q, k, v, block_indices, block_counts, offsets, token_indices):
        o, _ = fwd(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            block_indices=block_indices.unsqueeze(0),
            block_counts=block_counts.unsqueeze(0),
            block_size=workload.block_size,
            scale=workload.scale,
            cu_seqlens_q=offsets,
            cu_seqlens_k=offsets,
            token_indices_q=token_indices,
        )
        return o.squeeze(0)

    return fn


def _fla_nsa_cmp_fwd(workload: NsaCmpFwdCall):
    """Compression attention over the same compressed k/v. Adapts LSE to the workload's output dtype."""
    fwd = fla_op("ops.nsa.compression.parallel_nsa_compression_fwd")

    def fn(q, k_cmp, v_cmp, offsets, _chunk_offsets, token_indices):
        o, lse = fwd(
            q=q.unsqueeze(0),
            k=k_cmp.unsqueeze(0),
            v=v_cmp.unsqueeze(0),
            TK=workload.c_seq_len,
            block_size=workload.bs,
            scale=workload.scale,
            cu_seqlens_q=offsets,
            cu_seqlens_k=offsets,
            token_indices_q=token_indices,
        )
        return o.squeeze(0), lse.squeeze(0).to(q.dtype)

    return fn


def _fla_nsa_topk(workload: NsaTopkCall):
    """Unmodified FLA selection, including LSE computation from the same Q/K."""
    topk = fla_op("ops.nsa.parallel.parallel_nsa_topk")

    def fn(q, k_cmp, offsets, _chunk_offsets, _token_indices):
        return topk(
            q=q.unsqueeze(0),
            k=k_cmp.unsqueeze(0),
            TK=workload.c_seq_len,
            lse=None,
            block_counts=workload.selected_block_num,
            block_size=workload.bs,
            scale=workload.scale,
            cu_seqlens=offsets,
        ).squeeze(0)

    return fn


@pytest.mark.parametrize("call", manifest_calls(NSACompressedVarlenFwdOp))
def test_nsa_cmp_fwd_varlen_bench(call) -> None:
    workload, inputs, bm, op = _setup(NSACompressedVarlenFwdOp, NsaCmpFwdCall, call)
    fla_fn = _fla_nsa_cmp_fwd(workload)
    bm.compare({"tileops": op, FLA_TAG: fla_fn}, *inputs)


@pytest.mark.parametrize("call", manifest_calls(NSATopKVarlenFwdOp))
def test_nsa_topk_varlen_bench(call) -> None:
    workload, inputs, bm, op = _setup(NSATopKVarlenFwdOp, NsaTopkCall, call)
    fla_fn = _fla_nsa_topk(workload)

    bm.compare({"tileops": op, FLA_TAG: fla_fn}, *inputs)


@pytest.mark.parametrize("call", manifest_calls(NSAVarlenFwdOp))
def test_nsa_fwd_varlen_bench(call) -> None:
    workload, inputs, bm, op = _setup(NSAVarlenFwdOp, NsaFwdCall, call)
    fla_fn = _fla_nsa_fwd(workload)
    if fla_fn is None:
        bm.compare({"tileops": op, "torch-ref": workload.ref_program}, *inputs)
        return
    assert_output_spec(fla_fn(*inputs), call.specs["o_slc"], FLA_TAG)
    bm.compare({"tileops": op, FLA_TAG: fla_fn}, *inputs)
