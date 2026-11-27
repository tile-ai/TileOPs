"""NSA varlen benchmarks against FLA forward kernels, using the same precompressed K/V.

Top-k uses ``lse=None`` to include LSE computation on both sides. FLA's other two
passes also write LSE; its cached sequence metadata helpers run warm during timing.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import FLA_TAG, fla_op, optional_baseline
from tileops.attention import NSACompressedVarlenFwdOp, NSATopKVarlenFwdOp, NSAVarlenFwdOp
from workloads.attention.nsa import NSACompressedFwdCall, NSAFwdCall, NSATopKCall


def _fla_nsa_fwd(workload: NSAFwdCall):
    """Selected-block attention. None where fla cannot serve the row."""
    # parallel_nsa_fwd masks each selected block against the query's own position, so it has no
    # non-causal form; its tiles take a power-of-two query group of at least 16 and blocks of
    # at least 16 keys.
    group = workload.groups
    if not workload.is_causal or workload.block_size < 16 or group < 16 or group & (group - 1):
        return None
    fwd = fla_op("ops.nsa.parallel.parallel_nsa_fwd")

    def fn(q, k, v, block_indices, block_counts, cu_seqlens, token_indices, cu_seqlens_k=None):
        o, _ = fwd(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            block_indices=block_indices.unsqueeze(0),
            block_counts=block_counts.unsqueeze(0),
            block_size=workload.block_size,
            scale=workload.scale,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens if cu_seqlens_k is None else cu_seqlens_k,
            token_indices_q=token_indices,
        )
        return o.squeeze(0)

    return fn


def _fa3_paged_nsa_fwd(workload: NSAFwdCall, inputs: tuple):
    """FA3 decode over a page table of the selected blocks, for a causal row fla's tiles
    refuse. None where FA3 cannot express the row.

    A page is one selected block: one batch element per token and KV head, whose pages are
    the token's live slots and whose ``cache_seqlens`` is its visible key count. That is the
    selection only when every request's keys start on a block boundary and each token's live
    slots lead its list with the one block its position cuts, if any, last. The page table
    is built before timing, as FlashInfer's ``plan``.
    """
    fwd = optional_baseline("flash_attn_interface", "fa3", "flash_attn_with_kvcache")
    if fwd is None:
        return None
    q, k, v, block_indices, block_counts, cu_seqlens, token_indices, *rest = inputs
    cu_seqlens_k = rest[0] if rest and rest[0] is not None else cu_seqlens
    bs, groups = workload.block_size, workload.groups
    if bool((cu_seqlens_k % bs != 0).any()):
        return None
    request = token_indices[:, 0].long()
    n_q = (cu_seqlens[1:] - cu_seqlens[:-1]).long()[request]
    n_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).long()[request]
    limit = token_indices[:, 1].long() + n_k - n_q + 1
    bos_k = cu_seqlens_k[:-1].long()[request]
    slots = torch.arange(block_indices.shape[-1], device=q.device)
    calls = []
    for h in range(k.shape[1]):
        start = block_indices[:, h].long() * bs
        live = (slots < block_counts[:, h, None]) & (start >= 0) & (start < limit[:, None])
        keys = torch.where(live, (limit[:, None] - start).clamp(max=bs), 0)
        n_live = live.sum(-1, keepdim=True)
        last = slots == n_live - 1
        if not bool((live == (slots < n_live)).all()) or bool((live & ~last & (keys < bs)).any()):
            return None
        calls.append(
            (
                q[:, h * groups : (h + 1) * groups].unsqueeze(1).contiguous(),
                k.view(-1, bs, k.shape[1], k.shape[2])[:, :, h : h + 1],
                v.view(-1, bs, v.shape[1], v.shape[2])[:, :, h : h + 1],
                keys.sum(-1).to(torch.int32),
                torch.where(live, (bos_k[:, None] + start) // bs, 0).to(torch.int32),
            )
        )

    def fn(*_inputs):
        return torch.cat(
            [
                fwd(qh, kc, vc, cache_seqlens=lens, page_table=pages, softmax_scale=workload.scale)[
                    :, 0
                ]
                for qh, kc, vc, lens, pages in calls
            ],
            dim=1,
        )

    return fn


def _fla_nsa_compressed_fwd(workload: NSACompressedFwdCall):
    """Compression attention over the same compressed k/v. FLA's LSE is token-major with 0
    for a token that has closed no chunk; the adapter returns the op's head-major LSE with
    -inf there."""
    fwd = fla_op("ops.nsa.compression.parallel_nsa_compression_fwd")

    def fn(q, k_cmp, v_cmp, cu_seqlens, _chunk_offsets, token_indices):
        o, lse = fwd(
            q=q.unsqueeze(0),
            k=k_cmp.unsqueeze(0),
            v=v_cmp.unsqueeze(0),
            TK=workload.c_seq_len,
            block_size=workload.bs,
            scale=workload.scale,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            token_indices_q=token_indices,
        )
        empty = (token_indices[:, 1] + 1 < workload.bs)[:, None]
        return o.squeeze(0), lse.squeeze(0).masked_fill(empty, float("-inf")).T

    return fn


def _fla_nsa_topk(workload: NSATopKCall):
    """Unmodified FLA selection, including LSE computation from the same Q/K."""
    topk = fla_op("ops.nsa.parallel.parallel_nsa_topk")

    def fn(q, k_cmp, cu_seqlens, _chunk_offsets, _token_indices):
        return topk(
            q=q.unsqueeze(0),
            k=k_cmp.unsqueeze(0),
            TK=workload.c_seq_len,
            lse=None,
            block_counts=workload.selected_block_num,
            block_size=workload.bs,
            scale=workload.scale,
            cu_seqlens=cu_seqlens,
        ).squeeze(0)

    return fn


@pytest.mark.parametrize("case", bench.cases(NSACompressedVarlenFwdOp))
def test_nsa_compressed_fwd_varlen_bench(case) -> None:
    op = NSACompressedVarlenFwdOp(**case.arguments)
    fla_fn = _fla_nsa_compressed_fwd(case.workload)
    bench.Runner(op, case).compare({FLA_TAG: fla_fn})


@pytest.mark.parametrize("case", bench.cases(NSATopKVarlenFwdOp))
def test_nsa_topk_varlen_bench(case) -> None:
    op = NSATopKVarlenFwdOp(**case.arguments)
    fla_fn = _fla_nsa_topk(case.workload)

    bench.Runner(op, case).compare({FLA_TAG: fla_fn})


@pytest.mark.parametrize("case", bench.cases(NSAVarlenFwdOp))
def test_nsa_fwd_varlen_bench(case) -> None:
    op = NSAVarlenFwdOp(**case.arguments)
    fla_fn = _fla_nsa_fwd(case.workload)
    if fla_fn is not None:
        bench.Runner(op, case).compare({FLA_TAG: fla_fn})
        return
    implementations = {"torch-ref": case.reference}
    # A causal row fla refuses is one its tiles do not take.
    if case.workload.is_causal:
        fa3_fn = _fa3_paged_nsa_fwd(case.workload, case.inputs)
        if fa3_fn is not None:
            implementations["fa3"] = fa3_fn
    bench.Runner(op, case).compare(implementations)
