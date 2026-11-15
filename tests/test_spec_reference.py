"""Reference conformance of spec-only entries that have no implementation yet.

Each entry's reference is the torch expression its issue names, or the library reference
where one exists; an entry whose workload module holds its reference calls that one.
Every workload row is instantiated; data tensors live on ``meta`` and metadata tensors on
the CPU with the row's generated values, so the reference runs as written at the row's
full size. What is checked depends on the row only through its discriminant point, its dtypes
and its parameters, so each point, dtype case and set of parameter values runs its row with
the fewest input elements: a recurrent reference steps once per token even on ``meta``. Checked against the signature: the
reference's outputs have the inferred names, shapes and dtypes, and it writes exactly the
inputs the call's effects mark written. One call the signature rejects is rejected by the
reference too.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from tests.roofline_binder import signature_class
from tileops._ops._signature_codegen import leaves
from tileops.manifest import load_adts, load_manifest
from tileops.manifest._plan import entry_plan
from tileops.manifest._workload import TensorSpec, instantiate
from workloads import moe, sampling
from workloads.attention.gated_pool_compress import gated_pool_compress
from workloads.attention.merge_states import MergeAttentionStatesWorkload
from workloads.quantization import int8_dequant
from workloads.quantization import quantize as quantization
from workloads.sequence_modeling import hc_head, ngram_hash

pytestmark = pytest.mark.smoke

_INF = float("inf")


# ---------------------------------------------------------------- quantization


def _dequant(reference):
    def run(p, t):
        return {"x": reference(t["q"], t["scale"], p["out_dtype"])}

    return run


def _outputs(names, reference, *inputs, **params):
    """A spec reference that runs a workload reference and names its outputs."""

    def run(p, t):
        values = reference(*(t[n] for n in inputs), **{k: p[v] for k, v in params.items()})
        return dict(zip(names, values, strict=True))

    return run


_int8_per_tensor = _outputs(("q", "scale"), quantization.int8_quant_per_tensor, "x")
_int8_per_channel = _outputs(("q", "scale"), quantization.int8_quant_per_channel, "w")
_int8_per_block = _outputs(("q", "scale"), quantization.int8_quant_per_block, "x")
_fp8_per_block = _outputs(("q", "scale"), quantization.fp8_quant_per_block, "w")
_int4_per_group = _outputs(
    ("packed_weight", "params"),
    quantization.int4_quant_per_group,
    "w",
    group_size="group_size",
)
_smooth_quant = _outputs(("q", "scale"), quantization.smooth_quant, "x", "smooth")


def _fp4_per_group(p, t):
    result = quantization.fp4_quant_per_group(
        t["x"], p["block_size"], p["inplace"], p["scale_dtype"]
    )
    q, scale, output = result
    if p["inplace"]:
        assert q is None and scale is None and output is t["x"]
        return {"output": output}
    assert output is None
    return {"q": q, "scale": scale}


# ---------------------------------------------------------------- sampling


def _top_k_mask(p, t):
    return {"masked_logits": sampling.top_k_mask(t["logits"], t["top_k"])}


def _min_p_mask(p, t):
    return {"masked_logits": sampling.min_p_mask(t["logits"], t["min_p"])}


def _top_p_mask(p, t):
    return {"masked_logits": sampling.top_p_mask(t["logits"], t["top_p"])}


def _top_k_top_p_mask(p, t):
    masked = sampling.top_p_mask(sampling.top_k_mask(t["logits"], t["top_k"]), t["top_p"])
    return {"masked_logits": masked}


def _sampling_from_probs(p, t):
    return {"samples": sampling.sampling_from_probs(t["probs"], t["seed"], t["offset"])}


def _chain_speculative_sampling(p, t):
    names = ("draft_probs", "draft_token_ids", "target_probs", "seed", "offset")
    tokens, num = sampling.chain_speculative_sampling(*(t[n] for n in names))
    return {"output_token_ids": tokens, "num_accepted": num}


# ---------------------------------------------------------------- attention and caches


def _attend(q, k, v, scale, visible):
    """Float32 softmax attention of ``q [S, H, Dk]`` over ``k [N, Dk]``/``v [N, Dv]`` shared by
    every head, or per head when ``k`` is ``[N, H, Dk]``; ``visible [S, N]`` on the CPU."""
    kh = k if k.dim() == 3 else k[:, None].expand(-1, q.shape[1], -1)
    vh = v if v.dim() == 3 else v[:, None].expand(-1, q.shape[1], -1)
    scores = torch.einsum("shd,nhd->hsn", q.float(), kh.float()) * scale
    scores = scores.masked_fill(~visible.to(q.device)[None], -_INF)
    return torch.einsum("hsn,nhd->shd", scores.softmax(-1), vh.float()), scores.logsumexp(-1).T


def _causal(queries, keys, is_causal):
    """Bottom-right aligned visibility of ``queries`` rows over ``keys``."""
    rows = torch.arange(queries)[:, None] + keys - queries
    return (
        torch.arange(keys)[None] <= rows
        if is_causal
        else torch.ones(queries, keys, dtype=torch.bool)
    )


def _paged_rows(cache, table, start, end):
    """Rows ``[start, end)`` of one request, read through its block table."""
    page_size = cache.shape[1]
    pages = cache[table[: -(-end // page_size)].long()]
    return pages.flatten(0, 1)[start:end]


def _mla_paged(p, t):
    q, cache, table, lens = t["q"], t["kv_cache"], t["block_table"], t["cache_seqlens"]
    batch, s_q, _h, dk = q.shape
    rank = p["kv_lora_rank"]
    scale = p["sm_scale"] if p["sm_scale"] is not None else dk**-0.5
    outs, lses = [], []
    for b in range(batch):
        kv = _paged_rows(cache, table[b], 0, int(lens[b])).float()
        if t.get("kv_scale") is not None:
            kv = kv * t["kv_scale"]
        o, lse = _attend(q[b], kv, kv[:, :rank], scale, _causal(s_q, kv.shape[0], p["is_causal"]))
        outs.append(o)
        lses.append(lse)
    return {"o": torch.stack(outs).to(q.dtype), "lse": torch.stack(lses).transpose(1, 2)}


def _mla_varlen(p, t):
    q, k_nope, k_pe, v, cu = t["q"], t["k_nope"], t["k_pe"], t["v"], t["cu_seqlens"].tolist()
    heads = q.shape[1]
    scale = p["sm_scale"] if p["sm_scale"] is not None else q.shape[-1] ** -0.5
    outs, lses = [], []
    for a, e in zip(cu, cu[1:], strict=False):
        k = torch.cat([k_nope[a:e], k_pe[a:e, None].expand(-1, heads, -1)], -1)
        o, lse = _attend(q[a:e], k, v[a:e], scale, _causal(e - a, e - a, p["is_causal"]))
        outs.append(o)
        lses.append(lse)
    return {"o": torch.cat(outs).to(q.dtype), "lse": torch.cat(lses).transpose(0, 1)}


def _stored(rows, cache, scale):
    """Rows as ``cache`` stores them, one cache row each: divided by the scale when it has one."""
    rows = rows.reshape(rows.shape[0], *cache.shape[2:])
    return (rows.float() / scale if scale is not None else rows).to(cache.dtype)


def _paged_kv_cache_write(p, t):
    slots = t["slot_mapping"]
    tokens = (slots >= 0).nonzero().squeeze(1)
    for name, pages, scale in (("k", "k_cache", "k_scale"), ("v", "v_cache", "v_scale")):
        cache = t[pages]
        cache.view(-1, *cache.shape[2:])[slots[tokens]] = _stored(
            t[name][tokens], cache, t.get(scale)
        )
    return {}


def _nsa_varlen(p, t):
    # fla.ops.nsa.parallel.parallel_nsa_fwd: request i's queries are the last of its keys
    q, k, v, picks, counts = (t[n] for n in ("q", "k", "v", "block_indices", "block_counts"))
    if k.shape != v.shape or q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]:
        raise ValueError("q, k and v must agree")
    cu = t["cu_seqlens"].tolist()
    keys = cu if t["cu_seqlens_k"] is None else t["cu_seqlens_k"].tolist()
    size, group = p["block_size"], q.shape[1] // k.shape[1]
    scale = p["sm_scale"] if p["sm_scale"] is not None else q.shape[-1] ** -0.5
    outs = []
    for a, e, a_k, e_k in zip(cu, cu[1:], keys, keys[1:], strict=False):
        n, m = e - a, e_k - a_k
        blocks = -(-m // size)
        # Each slot past block_counts, or naming no block of the keys, selects nothing; a
        # block selected twice is attended twice.
        slot = picks[a:e].long()
        live = torch.arange(slot.shape[-1]) < counts[a:e, :, None]
        slot = torch.where(live & (slot >= 0) & (slot < blocks), slot, blocks)
        times = torch.zeros(n, k.shape[1], blocks + 1).scatter_add(2, slot, torch.ones(slot.shape))
        times = times[..., torch.arange(m) // size]
        if p["is_causal"]:
            times *= torch.arange(m)[None, None] <= torch.arange(m - n, m)[:, None, None]
        times = times.repeat_interleave(group, 1).permute(1, 0, 2).to(q.device)
        kh, vh = (x[a_k:e_k].float().repeat_interleave(group, 1) for x in (k, v))
        scores = torch.einsum("shd,mhd->hsm", q[a:e].float(), kh) * scale + times.log()
        weights = scores.softmax(-1).nan_to_num(0.0)
        outs.append(torch.einsum("hsm,mhd->shd", weights, vh))
    return {"o": torch.cat(outs).to(q.dtype)}


def _rope(x, cos_sin, positions, layout):
    """Rotate ``x [N, ..., R']`` on its first ``R`` columns by ``cos_sin [P, R]`` at ``positions``."""
    half = cos_sin.shape[1] // 2
    cos, sin = cos_sin[positions].float().chunk(2, -1)
    shape = (x.shape[0],) + (1,) * (x.dim() - 2) + (half,)
    cos, sin = cos.view(shape), sin.view(shape)
    rot, rest = x[..., : 2 * half].float(), x[..., 2 * half :]
    if layout == "neox":
        a, b = rot[..., :half], rot[..., half:]
        rotated = torch.cat([a * cos - b * sin, b * cos + a * sin], -1)
    else:
        a, b = rot[..., 0::2], rot[..., 1::2]
        rotated = torch.stack([a * cos - b * sin, b * cos + a * sin], -1).flatten(-2)
    return torch.cat([rotated.to(x.dtype), rest], -1)


def _mla_kv_cache_write(p, t):
    slots, cache = t["slot_mapping"], t["kv_cache"]
    tokens = (slots >= 0).nonzero().squeeze(1)
    k_pe = t["k_pe"][tokens]
    if p["fuse_rope"]:
        k_pe = _rope(k_pe, t["cos_sin_cache"], t["positions"][tokens], p["rope_layout"])
    rows = torch.cat([t["kv_c"][tokens], k_pe], -1)
    cache.view(-1, cache.shape[-1])[slots[tokens]] = _stored(rows, cache, t.get("kv_scale"))
    return {}


def _fused_qk_norm_rope(p, t):
    qkv, eps = t["qkv"], p["eps"]
    heads, kv_heads = p["num_heads"], p["num_kv_heads"]
    dim = t["q_weight"].shape[0]
    tokens = qkv.shape[0]
    for first, count, weight in ((0, heads, t["q_weight"]), (heads, kv_heads, t["k_weight"])):
        view = qkv[:, first * dim : (first + count) * dim].view(tokens, count, dim)
        normed = F.rms_norm(view.float(), (dim,), weight.float(), eps).to(qkv.dtype)
        view.copy_(_rope(normed, t["cos_sin_cache"], t["positions"], p["rope_layout"]))
    return {}


def _merge_attention_states(p, t):
    workload = MergeAttentionStatesWorkload(*t["v_a"].shape, t["v_a"].dtype, lse_base=p["lse_base"])
    v, s = workload.ref_program(t["v_a"], t["s_a"], t["v_b"], t["s_b"])
    return {"v": v, "s": s}


def _dsa_paged(p, t):
    q, cache, table, lens, indices = (
        t["q"],
        t["kv_cache"],
        t["block_table"],
        t["cache_seqlens"],
        t["indices"],
    )
    batch, s_q, _h, dk = q.shape
    page_size = cache.shape[1]
    scale = p["sm_scale"] if p["sm_scale"] is not None else dk**-0.5
    flat = cache.view(-1, cache.shape[-1])
    fp8 = cache.dtype == torch.uint8
    if (cache.shape[-1] != 656 or dk != 576) if fp8 else cache.shape[-1] != dk:
        raise ValueError("q and the cache row disagree")
    sink = t["attn_sink"]
    outs, lses = [], []
    for b in range(batch):
        for s in range(s_q):
            pos = indices[b, s].long()
            pos = pos[(pos >= 0) & (pos < lens[b])]
            rows = flat[table[b][pos // page_size].long() * page_size + pos % page_size]
            if fp8:
                latent = rows[:, :512].view(torch.float8_e4m3fn).float()
                scales = rows[:, 512:528].view(torch.float32).repeat_interleave(128, 1)
                value = latent * scales
                key = torch.cat([value, rows[:, 528:].view(torch.bfloat16).float()], -1)
            else:
                key = rows.float()
                value = key[:, : dk - p["dim_tail"]]
            visible = torch.ones(1, key.shape[0], dtype=torch.bool)
            o, lse = _attend(q[b, s : s + 1], key, value, scale, visible)
            if sink is not None:
                # The sink joins the denominator: o * e^lse / (e^lse + e^sink); lse unchanged.
                o = o * torch.sigmoid(lse - sink.float()).nan_to_num(0.0)[..., None]
            outs.append(o[0])
            lses.append(lse[0])
    shape = (batch, s_q)
    return {
        "o": torch.stack(outs).view(*shape, *outs[0].shape).to(q.dtype),
        "lse": torch.stack(lses).view(*shape, -1).transpose(1, 2),
    }


def _paged_kv_cache_gather(p, t):
    out, cache, table = t["out"], t["kv_cache"], t["block_table"]
    cu = t["cu_seqlens"].tolist()
    starts = t["seq_starts"].tolist() if t.get("seq_starts") is not None else [0] * (len(cu) - 1)
    for b, (a, e) in enumerate(zip(cu, cu[1:], strict=False)):
        rows = _paged_rows(cache, table[b], starts[b], starts[b] + e - a)
        if t.get("kv_scale") is not None:
            rows = rows.float() * t["kv_scale"]
        out[a:e] = rows.to(out.dtype)
    return {"output": out}


def _gated_pool_compress(p, t):
    names = ("kv", "score", "ape", "kv_state", "score_state", "cu_seqlens", "start_pos")
    entries, valid = gated_pool_compress(
        *(t[n] for n in names), t["state_slots"], p["ratio"], p["overlap"]
    )
    return {"entries": entries, "valid": valid}


# ---------------------------------------------------------------- moe


def _moe_grouped_gemm_mxfp4(p, t):
    names = ("a", "a_scale", "b", "b_scale", "layout_metadata")
    return {"output": moe.ref_moe_grouped_gemm_mxfp4(*(t[n] for n in names), p["layout"])}


# ---------------------------------------------------------------- sequence modeling


def _hyper_connection_pre(p, t):
    # vllm.model_executor.kernels.mhc.torch.mhc_pre_torch
    residual, fn, scale, base = t["residual"], t["fn"], t["hc_scale"], t["hc_base"]
    n, c = residual.shape[-2:]
    x = residual.flatten(-2).float()
    mixes = (x @ fn.T) * torch.rsqrt(x.square().mean(-1, keepdim=True) + p["rms_eps"])
    pre = torch.sigmoid(mixes[..., :n] * scale[0] + base[:n]) + p["hc_pre_eps"]
    post = torch.sigmoid(mixes[..., n : 2 * n] * scale[1] + base[n : 2 * n])
    comb = mixes[..., 2 * n :].unflatten(-1, (n, n)) * scale[2] + base[2 * n :].view(n, n)
    eps = p["hc_sinkhorn_eps"]
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(p["sinkhorn_repeat"] - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(-2).to(torch.bfloat16)
    post_mix = (post * p["hc_post_mult_value"]).unsqueeze(-1)
    return {"post_mix": post_mix, "comb_mix": comb, "layer_input": layer_input}


def _hyper_connection_post(p, t):
    # vllm.model_executor.kernels.mhc.torch.mhc_post_torch
    x, residual = t["x"], t["residual"]
    mixed = torch.einsum("...ij,...ih->...jh", t["comb_res_mix"].float(), residual.float())
    out = t["post_layer_mix"].float() * x.unsqueeze(-2).float() + mixed
    return {"output": out.to(residual.dtype)}


def _hyper_connection_head(p, t):
    names = ("hidden_states", "hc_fn", "hc_scale", "hc_base")
    return {"output": hc_head(*(t[n] for n in names), p["rms_norm_eps"], p["hc_eps"])}


def _ngram_hash(p, t):
    names = ("input_ids", "context", "multipliers", "primes", "offsets")
    optional = ("cu_seqlens", "context_indices", "num_accepted_tokens")
    ids = ngram_hash(*(t[n] for n in names), p["eos_token_id"], *(t[n] for n in optional))
    return {"ngram_ids": ids}


# ---------------------------------------------------------------- indexing


def _embedding(p, t):
    params = ("padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse")
    return {"output": F.embedding(t["input"], t["weight"], **{k: p[k] for k in params})}


def _masked_scatter(p, t):
    return {"output": torch.masked_scatter(t["input"], t["mask"], t["source"])}


# ---------------------------------------------------------------- norm


def _rms_norm_gated(p, t):
    # mamba_ssm.ops.triton.layernorm_gated.rms_norm_ref; the sigmoid gate follows
    # fla.modules.norm.fused_norm_gate.rms_norm_gated
    x, z, weight, bias = t["x"], t["z"], t["weight"], t["bias"]
    n = x.shape[-1]
    group = n if p["group_size"] is None else p["group_size"]
    if (
        weight.shape != (n,)
        or (bias is not None and bias.shape != (n,))
        or (z is not None and z.shape != x.shape)
        or group <= 0
        or n % group
    ):
        raise ValueError("weight, bias, z and group_size must match x")
    h = x.float()
    act = F.silu if p["activation"] == "silu" else torch.sigmoid
    gate = act(z.float()) if z is not None else None
    if gate is not None and not p["norm_before_gate"]:
        h = h * gate
    g = h.unflatten(-1, (n // group, group))
    h = (g * (1 / torch.sqrt(g.square().mean(-1, keepdim=True) + p["eps"]))).flatten(-2)
    h = h * weight.float()
    if bias is not None:
        h = h + bias.float()
    if gate is not None and p["norm_before_gate"]:
        h = h * gate
    return {"output": h.to(x.dtype)}


# ---------------------------------------------------------------- rope


def _rope_cos_sin(p, t):
    # flash_attn.layers.rotary.apply_rotary_emb: one offset, or one per sequence
    x, cos, sin, cu = t["x"], t["cos"], t["sin"], t["cu_seqlens"]
    if cos.shape != sin.shape or 2 * cos.shape[-1] > x.shape[-1]:
        raise ValueError("cos and sin must share a shape of at most half the head")
    r, off = 2 * cos.shape[-1], t["seqlen_offsets"]
    lengths = [x.shape[1]] * x.shape[0] if cu is None else (cu[1:] - cu[:-1]).tolist()
    offsets = off.tolist() if isinstance(off, torch.Tensor) else [off] * len(lengths)
    rows = [torch.arange(n) + o for n, o in zip(lengths, offsets, strict=True)]
    if cu is None:
        rows = torch.stack(rows) if rows else torch.zeros(0, x.shape[1], dtype=torch.long)
        c, s = cos[rows][:, :, None].float(), sin[rows][:, :, None].float()
    else:
        rows = torch.cat(rows)
        c, s = cos[rows][:, None].float(), sin[rows][:, None].float()
    h = x[..., :r].float()
    if p["interleaved"]:
        x1, x2 = h[..., 0::2], h[..., 1::2]
        rotated = torch.stack((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).flatten(-2)
    else:
        x1, x2 = h.chunk(2, dim=-1)
        rotated = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
    out = x if p["inplace"] else x.clone()
    out[..., :r] = rotated.to(x.dtype)
    return {"output": out}


# ---------------------------------------------------------------- entries


def _narrow(name, axis=-1):
    """A rejected call: tensor *name* one element shorter along *axis*."""

    def edit(tensors):
        x = tensors[name]
        tensors[name] = x.narrow(axis, 0, x.shape[axis] - 1)

    return edit


def _unsqueeze(name):
    def edit(tensors):
        tensors[name] = tensors[name][None]

    return edit


def _as_float(name):
    def edit(tensors):
        tensors[name] = tensors[name].float()

    return edit


def _empty_rows(name):
    def edit(tensors):
        tensors[name] = tensors[name][:0]

    return edit


# ---------------------------------------------------------------- indexing


def _concat(p, t):
    return {"output": torch.cat(t["tensors"], p["dim"])}


def _narrow_element(name, k, axis):
    """A rejected call: element *k* of tensor sequence *name* one element shorter along *axis*."""

    def edit(tensors):
        items = list(tensors[name])
        items[k] = items[k].narrow(axis, 0, items[k].shape[axis] - 1)
        tensors[name] = items

    return edit


REFERENCES = {
    # op: (reference, the edit of the first row's tensors that the signature rejects)
    "INT8QuantPerTensorFwdOp": (_int8_per_tensor, _empty_rows("x")),
    "INT8QuantPerChannelFwdOp": (_int8_per_channel, _unsqueeze("w")),
    "INT8QuantPerTokenGroupFwdOp": (_int8_per_block, _unsqueeze("x")),
    "INT4QuantPerGroupFwdOp": (_int4_per_group, _narrow("w")),
    "SmoothQuantFwdOp": (_smooth_quant, _narrow("smooth")),
    "INT8DequantPerTensorFwdOp": (_dequant(int8_dequant.int8_dequant_per_tensor), _unsqueeze("q")),
    "INT8DequantPerChannelFwdOp": (
        _dequant(int8_dequant.int8_dequant_per_channel),
        _narrow("scale"),
    ),
    "INT8DequantPerBlockFwdOp": (
        _dequant(int8_dequant.int8_dequant_per_block),
        _narrow("scale", 0),
    ),
    "FP8QuantPerBlockFwdOp": (_fp8_per_block, _unsqueeze("w")),
    "TopKMaskFwdOp": (_top_k_mask, _narrow("top_k")),
    "MinPMaskFwdOp": (_min_p_mask, _narrow("min_p", 0)),
    "TopPMaskFwdOp": (_top_p_mask, _narrow("top_p", 0)),
    "TopKTopPMaskFwdOp": (_top_k_top_p_mask, _narrow("top_p", 0)),
    "SamplingFromProbsFwdOp": (_sampling_from_probs, _unsqueeze("probs")),
    "ChainSpeculativeSamplingFwdOp": (_chain_speculative_sampling, _narrow("draft_token_ids")),
    "MLAPagedFwdOp": (_mla_paged, _narrow("kv_cache")),
    "MLAVarlenFwdOp": (_mla_varlen, _narrow("k_nope")),
    "PagedKVCacheWriteFwdOp": (_paged_kv_cache_write, _narrow("v")),
    "FusedQKNormRoPEFwdOp": (_fused_qk_norm_rope, _narrow("k_weight")),
    "MLAKVCacheWriteFwdOp": (_mla_kv_cache_write, _narrow("kv_cache")),
    "MergeAttentionStatesFwdOp": (_merge_attention_states, _narrow("v_b")),
    "DSAPagedFwdOp": (_dsa_paged, _narrow("kv_cache")),
    "PagedKVCacheGatherFwdOp": (_paged_kv_cache_gather, _narrow("out")),
    "EmbeddingFwdOp": (_embedding, _as_float("input")),
    "MaskedScatterFwdOp": (_masked_scatter, _narrow("mask", 0)),
    "RMSNormGatedFwdOp": (_rms_norm_gated, _narrow("weight")),
    "RoPECosSinFwdOp": (_rope_cos_sin, _narrow("sin")),
    "ConcatFwdOp": (_concat, _narrow_element("tensors", 1, 0)),
    "GatedPoolCompressFwdOp": (_gated_pool_compress, _narrow("ape")),
    "NSAVarlenFwdOp": (_nsa_varlen, _narrow("v")),
    "FP4QuantPerGroupFwdOp": (_fp4_per_group, _narrow("x")),
    "MoEGroupedGEMMMXFP4FwdOp": (_moe_grouped_gemm_mxfp4, _narrow("b_scale")),
    "HyperConnectionPreFwdOp": (_hyper_connection_pre, _narrow("hc_base")),
    "HyperConnectionPostFwdOp": (_hyper_connection_post, _narrow("comb_res_mix")),
    "HyperConnectionHeadFwdOp": (_hyper_connection_head, _narrow("hc_base")),
    "NGramHashFwdOp": (_ngram_hash, _narrow("offsets")),
}


class _Writes(TorchDispatchMode):
    """The storages the dispatched ops write, by their schema's mutable arguments."""

    def __init__(self):
        super().__init__()
        self.storages = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for i, arg in enumerate(func._schema.arguments):
            value = args[i] if i < len(args) else kwargs.get(arg.name)
            if (
                arg.alias_info is not None
                and arg.alias_info.is_write
                and isinstance(value, torch.Tensor)
            ):
                self.storages.add(value.untyped_storage()._cdata)
        return func(*args, **kwargs)


def _calls(name):
    """Per row and dtype case: the instantiated call, its meta tensors and the reference's
    tensors (metadata on the CPU with the row's values)."""
    entry = load_manifest()[name]
    plan = entry_plan(name, entry, load_adts())
    for row in entry["workloads"]:
        for case in row.get("dtype_cases") or [{}]:
            call = instantiate(plan, row, case)
            meta = call.materialize("meta")
            host = dict(meta)
            for n, spec in call.specs.items():
                if isinstance(spec, TensorSpec) and spec.values is not None and n in meta:
                    host[n] = torch.tensor(spec.values, dtype=getattr(torch, spec.dtype)).reshape(
                        spec.shape
                    )
            yield row["label"], case, plan, call, meta, host


def _numel(values):
    """The elements of the tensors among *values*, a tensor sequence's counted over its elements."""
    return sum(t.numel() for t in leaves(tuple(values)) if isinstance(t, torch.Tensor))


def _smallest_calls(name, cls):
    """`_calls` narrowed to the call with the fewest input elements per discriminant point,
    dtype case and parameter values, each with its constructed op."""
    smallest = {}
    for label, case, plan, call, meta, host in _calls(name):
        sig = plan.sig
        arguments = call.arguments()
        op = cls(**arguments)
        inputs = {n: meta[n] for n in sig.inputs}
        params = sorted((p, repr(arguments.get(p))) for p in sig.params)
        key = (cls._signature.key(cls._signature.point(op, inputs)), sorted(case.items()), params)
        size = _numel(inputs.values())
        if repr(key) not in smallest or size < smallest[repr(key)][0]:
            smallest[repr(key)] = (size, (label, case, plan, call, meta, host, op))
    return [c for _, c in smallest.values()]


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_reference_agrees_with_the_signature(name):
    entry = load_manifest()[name]
    reference, _reject = REFERENCES[name]
    cls = signature_class(name, entry)
    for label, case, plan, call, meta, host, op in _smallest_calls(name, cls):
        where = f"{name} {label} {case}"
        checked = cls._signature.check(op, {n: meta[n] for n in plan.sig.inputs})
        inputs = {n: host[n] for n in plan.sig.inputs}
        with _Writes() as writes:
            outputs = reference(call.arguments(), inputs)
        declared = {n: call.tensors[n] for n in plan.sig.outputs if n in call.tensors}
        got = {n: (tuple(v.shape), str(v.dtype).removeprefix("torch.")) for n, v in outputs.items()}
        assert got == {n: (tuple(s), d) for n, (s, d) in declared.items()}, where
        written = {
            n
            for n, v in inputs.items()
            if any(
                t.untyped_storage()._cdata in writes.storages
                for t in leaves((v,))
                if isinstance(t, torch.Tensor)
            )
        }
        assert written == set(checked.written), where


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_a_call_the_signature_rejects_the_reference_rejects(name):
    reference, reject = REFERENCES[name]
    cls = signature_class(name, load_manifest()[name])
    label, case, plan, call, meta, host, op = min(
        _smallest_calls(name, cls),
        key=lambda c: _numel(c[4][n] for n in c[2].sig.inputs),
    )
    bad_meta = {n: meta[n] for n in plan.sig.inputs}
    bad_host = {n: host[n] for n in plan.sig.inputs}
    reject(bad_meta)
    reject(bad_host)
    with pytest.raises((ValueError, TypeError)):
        cls._signature.check(op, bad_meta)
    with pytest.raises((RuntimeError, ValueError, IndexError, TypeError)):
        reference(call.arguments(), bad_host)


def test_a_write_only_buffer_is_written_whole():
    """`PagedKVCacheGatherFwdOp`'s `out` is write-only: on a small call with real values the
    reference leaves no row of it unwritten."""
    name = "PagedKVCacheGatherFwdOp"
    plan = entry_plan(name, load_manifest()[name], load_adts())
    row = {"T_q": 7, "NP": 6, "PS": 4, "W": 3, "E": [2, 3], "seq_lens": [3, 4], "starts": [2, 5]}
    call = instantiate(plan, {**row, "some": ["seq_starts"], "label": "s"}, {"KV": "bfloat16"})
    tensors = call.materialize("cpu")
    tensors["out"].fill_(math.nan)
    _paged_kv_cache_gather(call.arguments(), {n: tensors[n] for n in plan.sig.inputs})
    assert not tensors["out"].isnan().any()


def test_the_nsa_reference_attends_a_block_selected_twice_twice():
    """parallel_nsa_fwd attends each selected slot, so a repeated block weighs double."""
    params = {"block_size": 1, "sm_scale": None, "is_causal": False}
    tensors = {
        "q": torch.zeros(1, 1, 1),
        "k": torch.zeros(2, 1, 1),
        "v": torch.tensor([0.0, 3.0]).view(2, 1, 1),
        "block_indices": torch.tensor([[[0, 0, 1]]], dtype=torch.int32),
        "block_counts": torch.tensor([[3]], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, 1], dtype=torch.int32),
        "token_indices": torch.tensor([[0, 0]], dtype=torch.int32),
        "cu_seqlens_k": torch.tensor([0, 2], dtype=torch.int32),
    }
    assert _nsa_varlen(params, tensors)["o"].item() == pytest.approx(1.0)
