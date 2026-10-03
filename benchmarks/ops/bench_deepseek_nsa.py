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
import torch

from benchmarks.baselines import (
    FLA_TAG,
    assert_matches_reference,
    assert_output_spec,
    fla_op,
    reference_tolerance,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from benchmarks.verification import Custom, zeroed_input
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


@pytest.mark.parametrize("call", manifest_calls(NSACompressedVarlenFwdOp))
def test_nsa_cmp_fwd_varlen_bench(call) -> None:
    workload, inputs, bm, op = _setup(NSACompressedVarlenFwdOp, NsaCmpFwdCall, call)
    fla_fn = _fla_nsa_cmp_fwd(workload)

    def validate(got, expected):
        # FLA writes its LSE in FP32; both kernels use blockwise accumulation.
        for output, target in zip(got, expected, strict=True):
            torch.testing.assert_close(output.float(), target.float(), rtol=1e-5, atol=4e-3)

    checked = Custom(validate, "both outputs checked at the NSA unit-test bound; LSE may be FP32")
    bm.compare(
        {"tileops": op, FLA_TAG: fla_fn},
        *inputs,
        evidence={"tileops": checked, FLA_TAG: checked},
    )


@pytest.mark.parametrize("call", manifest_calls(NSATopKVarlenFwdOp))
def test_nsa_topk_varlen_bench(call) -> None:
    workload, inputs, bm, op = _setup(NSATopKVarlenFwdOp, NsaTopkCall, call)

    # No fla comparator: its selection forces blocks 0, IC-1 and IC to importance 1.0 while this
    # op forces only IC, and it ranks raw scores where this op treats a gap under 1e-5 as a tie
    # and prefers the larger block id. The two select by different rules, so a ratio between them
    # would not be a ratio between implementations of one function.
    def validate(got, expected):
        assert got.shape == expected.shape and got.dtype == expected.dtype
        assert torch.equal(got < 0, expected < 0), "unfilled top-k slots differ"
        # Match the NSA unit-test contract for ranks at floating-point score ties.
        assert (got != expected).float().mean() <= 1e-3, "top-k mismatch exceeds 0.1%"

    checked = Custom(
        validate,
        "top-k index mismatch <= 0.1%; padding matches exactly",
        controls=(zeroed_input(0, "query-zeroed"),),
    )
    bm.compare(
        {"tileops": op, "torch-ref": workload.ref_program}, *inputs, evidence={"tileops": checked}
    )


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
