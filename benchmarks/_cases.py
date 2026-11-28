"""How a benchmark calls each manifest op on its case.

``ENTRIES`` holds the ops whose case needs more than the default. The workload of an op's call
comes from ``workloads.calls``.
"""

import dataclasses
import functools
from collections.abc import Callable
from typing import Any

import torch

from benchmarks.api import Implementation
from benchmarks.baselines import private_inputs
from workloads.attention.gated_pool_compress import gated_pool_compress_result
from workloads.attention.mla_cache_write import mla_cache_write_result
from workloads.attention.paged_cache_write import paged_cache_write_result
from workloads.linear_attention.deltanet import DeltaNetChunkwiseCall
from workloads.linear_attention.gla import GLAChunkwiseCall
from workloads.mamba import conv_state_result, ssd_decode_result
from workloads.norm import batch_norm_forward_result, fused_qk_norm_rope_result
from workloads.sequence_modeling.ngram_hash import ngram_hash_result

__all__ = ["Entry", "default_binder", "entry", "static_weight"]


def default_binder(op: Any, case: Any) -> Implementation:
    """The op called on the case's inputs, which it must leave unchanged; it receives a
    ``kw_only`` input by name, as its ``forward`` takes it."""
    signature = case._call.signature

    def run(*inputs: Any) -> Any:
        args, keywords = signature.forward_arguments(inputs)
        return op(*args, **keywords)

    return Implementation(run=run)


@dataclasses.dataclass(frozen=True)
class Entry:
    """How a benchmark calls one op on its case.

    ``inputs`` is for an op whose inputs are not the workload's ``gen_inputs()``. ``binder``
    turns the op into the implementation recorded as ``tileops``.
    """

    count_copies: bool = False
    binder: Callable[[Any, Any], Implementation] = default_binder
    inputs: Callable[[Any], tuple] | None = None


def entry(name: str) -> Entry:
    """How a benchmark calls manifest op *name*: its registry entry, else the default."""
    return ENTRIES.get(name, Entry())


def _paged_cache_gather_binder(op, case):
    """Time the write-only gather on its private destination."""
    return private_inputs(default_binder(op, case).run, case.inputs, len(case.inputs) - 1)


def _paged_cache_write_binder(op: Any, case: Any) -> Implementation:
    """The op on private pages it scatters k and v into, returning both caches."""
    return private_inputs(lambda *args: paged_cache_write_result(op, *args), case.inputs, 2, 3)


def _mla_cache_write_binder(op: Any, case: Any) -> Implementation:
    """The op on a private cache it writes kv_c and k_pe into, returning the cache."""
    return private_inputs(lambda *args: mla_cache_write_result(op, *args), case.inputs, 2)


def _gated_pool_compress_binder(op: Any, case: Any) -> Implementation:
    """The op on private rings it writes, returning its entries, valid flags and both rings."""
    return private_inputs(lambda *args: gated_pool_compress_result(op, *args), case.inputs, 3, 4)


def static_weight(fn: Callable, case) -> Implementation:
    """*fn* called on ``x`` alone, the weight and bias bound across calls; *fn* is the op or a
    baseline taking ``(x, weight, bias)``."""
    x, weight, bias = case.inputs

    def run(x_i):
        return fn(x_i, weight, bias)

    return Implementation(run=run, args=(x,))


def _prepacked_weight(op, case) -> Implementation:
    """The op called on its own repack of the case's row-major weight, made once, untimed."""
    activation, packed_weight, *scales = case.inputs
    return Implementation(run=op, args=(activation, op.repack(packed_weight), *scales))


def _deltanet_bwd_inputs(workload: DeltaNetChunkwiseCall) -> tuple:
    """The backward's inputs with the saved tensors a forward run produces for them."""
    from tileops.linear_attention import DeltaNetChunkFwdOp

    do, q, k, v, beta, *_saved = workload.gen_inputs()
    fwd_op = DeltaNetChunkFwdOp(workload.arguments()["chunk_size"])
    _o, h, Aw, Au, w, u = fwd_op(q, k, v, beta)
    return do, q, k, v, beta, h, Aw, Au, w, u


def _gla_bwd_inputs(workload: GLAChunkwiseCall) -> tuple:
    """The backward's inputs with the chunk states a forward run produces and a zero final-state
    gradient; the forward starts from a random state when the backward has one."""
    from tileops.linear_attention import GLAChunkFwdOp

    do, _dfinal_state, q, k, v, g, _h = workload.gen_inputs()
    arguments = workload.arguments()
    fwd_op = GLAChunkFwdOp(arguments["chunk_size"], arguments["scale"])
    initial_state = (
        torch.randn_like(_dfinal_state) * 0.1 if arguments["has_initial_state"] else None
    )
    _o, _final_state, h = fwd_op(q, k, v, g, initial_state)
    dfinal_state = torch.zeros_like(_dfinal_state)
    return do, dfinal_state, q, k, v, g, h


def _state_pool_binder(op: Any, case: Any) -> Implementation:
    """The op; given ``state_indices``, on a private copy of the state pool it updates in place."""
    implementation = default_binder(op, case)
    if case.inputs[10] is None:
        return implementation
    return private_inputs(implementation.run, case.inputs, 5)


def _ssd_decode_binder(op: Any, case: Any) -> Implementation:
    """The op on a private copy of the recurrent state it updates, returning output and state."""
    return private_inputs(lambda *args: ssd_decode_result(op, *args), case.inputs, 5)


def _conv_state_binder(op: Any, case: Any) -> Implementation:
    """The op on a private copy of the convolution state it overwrites, returning output and
    state."""
    run = default_binder(functools.partial(conv_state_result, op), case).run
    return private_inputs(run, case.inputs, 2)


def _private_output(op, case) -> Implementation:
    """The op writing a private ``out`` buffer, which it overwrites whole, so no reset."""
    output = torch.empty_like(case.inputs[0])

    def run(*args):
        return op(*args, out=output)

    return Implementation(run=run, args=case.inputs)


def _topk_inputs(workload) -> tuple:
    """The router logits, the correction bias only where the call passes one or given experts,
    and the given experts only where the call passes them."""
    gating_output, correction_bias, expert_ids = workload.gen_inputs()
    if expert_ids is not None:
        return gating_output, correction_bias, expert_ids
    return (gating_output,) if correction_bias is None else (gating_output, correction_bias)


def _ngram_hash_binder(op, case) -> Implementation:
    """The op on a private copy of the context pool it updates, returning ids and the pool."""
    return private_inputs(lambda *args: ngram_hash_result(op, *args), case.inputs, 1)


def _batch_norm_fwd(op, case) -> Implementation:
    """The op on private running statistics, exposing them beside its output."""
    return private_inputs(lambda *a: batch_norm_forward_result(op, *a), case.inputs, 1, 2)


def _instance_norm_fwd(op, case) -> Implementation:
    """The op on private running statistics, which it updates when it uses input statistics."""
    return private_inputs(op, case.inputs, 1, 2)


def _fused_qk_norm_rope(op, case) -> Implementation:
    """The op on a private qkv it normalizes and rotates in place, exposing that qkv."""
    return private_inputs(lambda *a: fused_qk_norm_rope_result(op, *a), case.inputs, 0)


def _quantize_binder(op: Any, case: Any) -> Implementation:
    """The op on private copies of what it writes: ``x`` where it fake-quantizes ``x`` in
    place, otherwise any ``out_q`` the call passes after ``x``."""
    positions = (0,) if case.arguments["inplace"] else range(1, len(case.inputs))
    return private_inputs(default_binder(op, case).run, case.inputs, *positions)


def _rope_positions(op, case) -> Implementation:
    """The op on private copies of the query and key a call in place rewrites."""
    inputs = list(case._call.signature.inputs)
    written = sorted(inputs.index(name) for name in case._call.written)
    return private_inputs(default_binder(op, case).run, case.inputs, *written)


def _rope_cos_sin(op, case) -> Implementation:
    """The op on a private ``x`` where it rotates ``x`` in place."""
    if not case.arguments["inplace"]:
        return default_binder(op, case)
    return private_inputs(op, case.inputs, 0)


ENTRIES = {
    # attention
    "PagedKVCacheGatherFwdOp": Entry(binder=_paged_cache_gather_binder, count_copies=True),
    "MergeAttentionStatesFwdOp": Entry(count_copies=True),
    "GQADenseFwdOp": Entry(count_copies=True),
    "PagedKVCacheWriteFwdOp": Entry(binder=_paged_cache_write_binder),
    "MLADenseDecodeFwdOp": Entry(count_copies=True),
    "MLAKVCacheWriteFwdOp": Entry(binder=_mla_cache_write_binder),
    "FP8LightningIndexerFwdOp": Entry(count_copies=True),
    "GatedPoolCompressFwdOp": Entry(binder=_gated_pool_compress_binder),
    # convolution
    "Conv1dFwdOp": Entry(binder=static_weight),
    # elementwise: the reference clones then fills in place; the clone is a device copy and counts.
    "MaskedFillFwdOp": Entry(count_copies=True),
    # gemm
    "GEMMFP8FwdOp": Entry(count_copies=True),
    "GEMMW4A8FwdOp": Entry(binder=_prepacked_weight),
    # linear attention
    "DeltaNetChunkBwdOp": Entry(inputs=_deltanet_bwd_inputs),
    "GDNFwdOp": Entry(binder=_state_pool_binder),
    "GLAChunkBwdOp": Entry(inputs=_gla_bwd_inputs),
    "KDAFwdOp": Entry(binder=_state_pool_binder),
    # mamba
    "CausalConv1dFwdOp": Entry(binder=_conv_state_binder),
    "CausalConv1dUpdateFwdOp": Entry(binder=_conv_state_binder),
    "SSDRecurrentFwdOp": Entry(binder=_ssd_decode_binder),
    # moe
    "FusedTopKFwdOp": Entry(inputs=_topk_inputs),
    "FusedMoEExpertsFwdOp": Entry(count_copies=True, binder=_private_output),
    "IndexedExpertMLPFwdOp": Entry(count_copies=True, binder=_private_output),
    "FusedMoEFwdOp": Entry(count_copies=True),
    "FusedMoESharedExpertFwdOp": Entry(count_copies=True),
    "SharedExpertMLPFwdOp": Entry(count_copies=True),
    # ngram hash
    "NGramHashFwdOp": Entry(binder=_ngram_hash_binder),
    # norm
    "BatchNormFwdOp": Entry(binder=_batch_norm_fwd),
    "InstanceNormFwdOp": Entry(binder=_instance_norm_fwd),
    "FusedQKNormRoPEFwdOp": Entry(binder=_fused_qk_norm_rope),
    # pool
    "MeanPoolingFwdOp": Entry(count_copies=True),
    # quantization
    "FP4QuantPerGroupFwdOp": Entry(binder=_quantize_binder),
    "FP8QuantPerTokenGroupFwdOp": Entry(binder=_quantize_binder),
    # rope
    "RoPECosSinFwdOp": Entry(binder=_rope_cos_sin),
    "RoPEFwdOp": Entry(binder=_rope_positions),
}
