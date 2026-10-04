"""NSA varlen benchmarks against FLA forward kernels, using the same precompressed K/V.

Top-k uses ``lse=None`` to include LSE computation on both sides. FLA's other two
passes also write LSE; its cached sequence metadata helpers run warm during timing.
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
from tileops.attention import NSACompressedVarlenFwdOp, NSATopKVarlenFwdOp, NSAVarlenFwdOp
from workloads.attention.nsa import NsaCmpFwdCall, NsaFwdCall, NsaTopkCall
from workloads.numerics import Custom, zeroed_input


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
    tolerance = (
        reference_tolerance(torch.bfloat16)
        if inputs[0].dtype == torch.bfloat16
        else {"rtol": 1e-5, "atol": 4e-3}
    )

    def validate(got, expected):
        assert got[0].dtype == expected[0].dtype
        # FLA writes its LSE in FP32; TileOPs returns the input dtype.
        assert got[1].dtype in (expected[1].dtype, torch.float32)
        for output, target in zip(got, expected, strict=True):
            torch.testing.assert_close(output.float(), target.float(), **tolerance)

    checked = Custom(
        validate,
        "both outputs checked at the existing FP16 bound or standard BF16 bound; LSE may be FP32",
        controls=(zeroed_input(0, "query-zeroed"),),
    )
    bm.compare(
        {"tileops": op, FLA_TAG: fla_fn},
        *inputs,
        evidence={"tileops": checked, FLA_TAG: checked},
    )


@pytest.mark.parametrize("call", manifest_calls(NSATopKVarlenFwdOp))
def test_nsa_topk_varlen_bench(call) -> None:
    workload, inputs, bm, op = _setup(NSATopKVarlenFwdOp, NsaTopkCall, call)

    fla_fn = _fla_nsa_topk(workload)

    def validate(got, expected):
        assert_output_spec(got, call.specs["block_indices"], "NSA top-k")
        assert torch.equal(got == -1, expected == -1), "unfilled top-k slots differ"
        current = inputs[-1][:, 1, None, None] // workload.bs
        assert ((got >= -1) & (got <= current)).all(), "non-causal or invalid block id"
        ordered = got.sort(-1).values
        assert not ((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] >= 0)).any(), (
            "duplicate selected block"
        )
        torch.testing.assert_close(
            workload.selection_scores(got, *inputs),
            workload.selection_scores(expected, *inputs),
            rtol=1e-5,
            atol=1e-6,
        )

    checked = Custom(
        validate,
        "every selected score checked at rtol=1e-5/atol=1e-6; valid unique indices and exact padding",
        controls=(zeroed_input(0, "query-zeroed"),),
    )
    bm.compare(
        {"tileops": op, FLA_TAG: fla_fn},
        *inputs,
        evidence={"tileops": checked, FLA_TAG: checked},
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
