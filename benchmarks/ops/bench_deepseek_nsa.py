"""Benchmarks for the three Native Sparse Attention (NSA) varlen passes.

The compression and selected-block passes are timed against the triton kernels
``flash-linear-attention`` ships for them, each checked against the torch reference before it is
timed. The top-k pass keeps the torch reference: fla selects by different rules, so no ratio
between them would compare two implementations of one function.

Two asymmetries fla's API gives no way to remove, both charging fla for work these ops do not do,
so each reading is a lower bound on how far fla is ahead: both forward entry points always write
a float32 ``[B, TQ, HQ]`` lse, and ``parallel_nsa_compression_fwd`` takes no ``chunk_offsets``,
so fla rederives what the workload holds. That derivation is ``@tensor_cache``d, so timed
iterations see hits as they would in a serving loop.

fla wants ``[B, T, H, D]`` with ``B == 1`` and ``cu_seqlens``, so the adapters unsqueeze and
squeeze. They call the forward entry points, not the exported wrappers: ``parallel_nsa``
mean-pools k and v on every call, work none of these ops does.
"""

import pytest

from benchmarks.baselines import (
    FLA_TAG,
    assert_matches_reference,
    assert_output_spec,
    fla_op,
    reference_tolerance,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.attention import NSACmpVarlenFwdOp, NSATopkVarlenFwdOp, NSAVarlenFwdOp
from workloads.deepseek_attention import NsaCmpFwdCall, NsaFwdCall, NsaTopkCall


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
    """Compression attention over the same compressed k/v. Returns fla's float32 lse unconverted."""
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
        return o.squeeze(0), lse.squeeze(0)

    return fn


@pytest.mark.parametrize("call", manifest_calls(NSACmpVarlenFwdOp))
def test_nsa_cmp_fwd_varlen_bench(call) -> None:
    workload, inputs, bm, op = _setup(NSACmpVarlenFwdOp, NsaCmpFwdCall, call)
    fla_fn = _fla_nsa_cmp_fwd(workload)

    # fla writes a float32 lse, the manifest declares the input dtype. The conversion is this
    # benchmark's, not fla's, so it stays out of the timed callable.
    def checked(*args):
        o, lse = fla_fn(*args)
        return o, lse.to(workload.dtype)

    o, lse = checked(*inputs)
    assert_output_spec(o, call.specs["o"], FLA_TAG)
    assert_output_spec(lse, call.specs["lse"], FLA_TAG)
    # tests/ops/test_deepseek_nsa.py's tolerance for this op: the reference accumulates a whole
    # sequence in float32, both kernels accumulate one block at a time.
    assert_matches_reference(checked, workload.ref_program, *inputs, rtol=1e-5, atol=4e-3)

    bm.compare({"tileops": op, FLA_TAG: fla_fn}, *inputs)


@pytest.mark.parametrize("call", manifest_calls(NSATopkVarlenFwdOp))
def test_nsa_topk_varlen_bench(call) -> None:
    workload, inputs, bm, op = _setup(NSATopkVarlenFwdOp, NsaTopkCall, call)
    # No fla comparator: its selection forces blocks 0, IC-1 and IC to importance 1.0 while this
    # op forces only IC, and it ranks raw scores where this op treats a gap under 1e-5 as a tie
    # and prefers the larger block id. The two select by different rules, so a ratio between them
    # would not be a ratio between implementations of one function.
    bm.compare({"tileops": op, "torch-ref": workload.ref_program}, *inputs)


@pytest.mark.parametrize("call", manifest_calls(NSAVarlenFwdOp))
def test_nsa_fwd_varlen_bench(call) -> None:
    workload, inputs, bm, op = _setup(NSAVarlenFwdOp, NsaFwdCall, call)
    fla_fn = _fla_nsa_fwd(workload)
    if fla_fn is None:
        bm.compare({"tileops": op, "torch-ref": workload.ref_program}, *inputs)
        return
    assert_output_spec(fla_fn(*inputs), call.specs["o_slc"], FLA_TAG)
    assert_matches_reference(
        fla_fn, workload.ref_program, *inputs, **reference_tolerance(workload.dtype)
    )
    bm.compare({"tileops": op, FLA_TAG: fla_fn}, *inputs)
