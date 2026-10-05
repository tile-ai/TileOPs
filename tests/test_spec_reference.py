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
from tileops.manifest import load_adts, load_manifest
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
from workloads import sampling
from workloads.quantization import int8_dequant
from workloads.quantization import quantize as quantization

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
    ("packed_weight", "weight_scale", "weight_zero"),
    quantization.int4_quant_per_group,
    "w",
    group_size="group_size",
)
_smooth_quant = _outputs(("q", "scale"), quantization.smooth_quant, "x", "smooth")


# ---------------------------------------------------------------- sampling


def _top_k_mask(p, t):
    return {"masked_logits": sampling.top_k_mask(t["logits"], t["k"])}


def _min_p_mask(p, t):
    return {"masked_logits": sampling.min_p_mask(t["logits"], t["min_p"])}


def _top_p_mask(p, t):
    return {"masked_logits": sampling.top_p_mask(t["logits"], t["p"])}


def _top_k_top_p_mask(p, t):
    masked = sampling.top_p_mask(sampling.top_k_mask(t["logits"], t["k"]), t["p"])
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
    return {"o": torch.stack(outs).to(q.dtype), "lse": torch.stack(lses)}


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
    return {"o": torch.cat(outs).to(q.dtype), "lse": torch.cat(lses)}


def _stored(rows, cache, scale):
    """Rows as ``cache`` stores them, one cache row each: divided by the scale when it has one."""
    rows = rows.reshape(rows.shape[0], *cache.shape[2:])
    return (rows.float() / scale if scale is not None else rows).to(cache.dtype)


def _paged_kv_cache_write(p, t):
    slots = t["slot_mapping"]
    tokens = (slots >= 0).nonzero().squeeze(1)
    for name, pages, scale in (("k", "k_pages", "k_scale"), ("v", "v_pages", "v_scale")):
        cache = t[pages]
        cache.view(-1, *cache.shape[2:])[slots[tokens]] = _stored(
            t[name][tokens], cache, t.get(scale)
        )
    return {}


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
    cache.view(-1, cache.shape[-1])[slots[tokens]] = _stored(rows, cache, t.get("scale"))
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
    s_a, s_b = t["s_a"], t["s_b"]
    top = torch.maximum(s_a, s_b)
    empty = top == -_INF
    shift = torch.where(empty, torch.zeros_like(top), top)
    w_a, w_b = (s_a - shift).exp(), (s_b - shift).exp()
    total = torch.where(empty, torch.ones_like(top), w_a + w_b)
    v = t["v_a"].float() * (w_a / total)[..., None] + t["v_b"].float() * (w_b / total)[..., None]
    return {"v": v.to(t["v_a"].dtype), "s": torch.where(empty, top, top + total.log())}


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
    outs, lses = [], []
    for b in range(batch):
        for s in range(s_q):
            pos = indices[b, s].long()
            pos = pos[(pos >= 0) & (pos < lens[b])]
            rows = flat[table[b][pos // page_size].long() * page_size + pos % page_size]
            latent = rows[:, :512].view(torch.float8_e4m3fn).float()
            scales = rows[:, 512:528].view(torch.float32).repeat_interleave(128, 1)
            value = latent * scales
            key = torch.cat([value, rows[:, 528:].view(torch.bfloat16).float()], -1)
            visible = torch.ones(1, key.shape[0], dtype=torch.bool)
            o, lse = _attend(q[b, s : s + 1], key, value, scale, visible)
            outs.append(o[0])
            lses.append(lse[0])
    shape = (batch, s_q)
    return {
        "o": torch.stack(outs).view(*shape, *outs[0].shape).to(q.dtype),
        "lse": torch.stack(lses).view(*shape, -1),
    }


def _paged_kv_cache_gather(p, t):
    dst, cache, table = t["dst"], t["cache"], t["block_table"]
    cu = t["cu_seq_lens"].tolist()
    starts = t["seq_starts"].tolist() if t.get("seq_starts") is not None else [0] * (len(cu) - 1)
    for b, (a, e) in enumerate(zip(cu, cu[1:], strict=False)):
        rows = _paged_rows(cache, table[b], starts[b], starts[b] + e - a)
        if t.get("scale") is not None:
            rows = rows.float() * t["scale"]
        dst[a:e] = rows.to(dst.dtype)
    return {}


# ---------------------------------------------------------------- linear attention


def _kda(p, t):
    naive = pytest.importorskip(
        "fla.ops.kda.naive",
        reason="KimiDeltaAttentionFwdOp unverified: its reference, FLA (package `fla`), is not installed",
    ).naive_recurrent_kda
    q, k, v, g, beta = t["q"], t["k"], t["v"], t["g"], t["beta"]
    if p["use_qk_l2norm_in_kernel"]:
        q, k = (
            F.normalize(q.float(), dim=-1).to(q.dtype),
            F.normalize(k.float(), dim=-1).to(k.dtype),
        )
    if p["use_gate_in_kernel"]:
        heads = v.shape[2]
        a = t["A_log"].exp()[:, None]
        x = g.float() + (t["dt_bias"].view(heads, -1) if t.get("dt_bias") is not None else 0)
        lower = p["lower_bound"]
        g = lower * torch.sigmoid(a * x) if lower is not None else -a * F.softplus(x)
    if p["use_beta_sigmoid_in_kernel"]:
        beta = beta.float().sigmoid() * (2 if p["allow_neg_eigval"] else 1)
    cu = t["cu_seqlens"].tolist() if t.get("cu_seqlens") is not None else None
    spans = list(zip(cu, cu[1:], strict=False)) if cu else [(0, q.shape[1])]
    init = t.get("initial_state")
    if init is not None and p["state_v_first"]:
        init = init.transpose(-1, -2)
    outs, states = [], []
    for i, (a, e) in enumerate(spans):
        sel = slice(a, e)
        h0 = init[i : i + 1] if cu and init is not None else init
        o, s = naive(q[:, sel], k[:, sel], v[:, sel], g[:, sel], beta[:, sel], p["scale"], h0, True)
        outs.append(o)
        states.append(s)
    state = torch.cat(states) if cu else states[0]
    if p["state_v_first"]:
        state = state.transpose(-1, -2)
    return {"o": torch.cat(outs, 1).to(v.dtype), "final_state": state.float()}


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


def _empty_rows(name):
    def edit(tensors):
        tensors[name] = tensors[name][:0]

    return edit


REFERENCES = {
    # op: (reference, the edit of the first row's tensors that the signature rejects)
    "INT8QuantPerTensorFwdOp": (_int8_per_tensor, _empty_rows("x")),
    "INT8QuantPerChannelFwdOp": (_int8_per_channel, _unsqueeze("w")),
    "INT8QuantPerBlockFwdOp": (_int8_per_block, _unsqueeze("x")),
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
    "TopKMaskFwdOp": (_top_k_mask, _narrow("k")),
    "MinPMaskFwdOp": (_min_p_mask, _narrow("min_p", 0)),
    "TopPMaskFwdOp": (_top_p_mask, _narrow("p", 0)),
    "TopKTopPMaskFwdOp": (_top_k_top_p_mask, _narrow("p", 0)),
    "SamplingFromProbsFwdOp": (_sampling_from_probs, _unsqueeze("probs")),
    "ChainSpeculativeSamplingFwdOp": (_chain_speculative_sampling, _narrow("draft_token_ids")),
    "MultiHeadLatentAttentionPagedFwdOp": (_mla_paged, _narrow("kv_cache")),
    "MultiHeadLatentAttentionVarlenFwdOp": (_mla_varlen, _narrow("k_nope")),
    "PagedKVCacheWriteFwdOp": (_paged_kv_cache_write, _narrow("v")),
    "FusedQKNormRopeFwdOp": (_fused_qk_norm_rope, _narrow("k_weight")),
    "MultiHeadLatentAttentionKVCacheWriteFwdOp": (_mla_kv_cache_write, _narrow("kv_cache")),
    "MergeAttentionStatesFwdOp": (_merge_attention_states, _narrow("v_b")),
    "DeepSeekSparseAttentionPagedFwdOp": (_dsa_paged, _narrow("kv_cache")),
    "PagedKVCacheGatherFwdOp": (_paged_kv_cache_gather, _narrow("dst")),
    "KimiDeltaAttentionFwdOp": (_kda, _narrow("g")),
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
                if spec is not None and spec.values is not None and n in meta:
                    host[n] = torch.tensor(spec.values, dtype=getattr(torch, spec.dtype)).reshape(
                        spec.shape
                    )
            yield row["label"], case, plan, call, meta, host


def _smallest_calls(name, cls):
    """`_calls` narrowed to the call with the fewest input elements per discriminant point,
    dtype case and parameter values, each with its constructed op."""
    smallest = {}
    for label, case, plan, call, meta, host in _calls(name):
        sig = plan.sig
        arguments = call.arguments(meta)
        op = cls(**arguments)
        inputs = {n: meta[n] for n in sig.inputs}
        params = sorted((p, repr(arguments.get(p))) for p in sig.params)
        key = (cls._signature.key(cls._signature.point(op, inputs)), sorted(case.items()), params)
        size = sum(t.numel() for t in inputs.values() if t is not None)
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
            outputs = reference(call.arguments(meta), inputs)
        declared = {n: call.tensors[n] for n in plan.sig.outputs}
        got = {n: (tuple(v.shape), str(v.dtype).removeprefix("torch.")) for n, v in outputs.items()}
        assert got == {n: (tuple(s), d) for n, (s, d) in declared.items()}, where
        written = {
            n
            for n, v in inputs.items()
            if v is not None and v.untyped_storage()._cdata in writes.storages
        }
        assert written == set(checked.written), where


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_a_call_the_signature_rejects_the_reference_rejects(name):
    reference, reject = REFERENCES[name]
    cls = signature_class(name, load_manifest()[name])
    label, case, plan, call, meta, host, op = min(
        _smallest_calls(name, cls),
        key=lambda c: sum(t.numel() for n in c[2].sig.inputs if (t := c[4][n]) is not None),
    )
    bad_meta = {n: meta[n] for n in plan.sig.inputs}
    bad_host = {n: host[n] for n in plan.sig.inputs}
    reject(bad_meta)
    reject(bad_host)
    with pytest.raises((ValueError, TypeError)):
        cls._signature.check(op, bad_meta)
    with pytest.raises((RuntimeError, ValueError, IndexError, TypeError)):
        reference(call.arguments(meta), bad_host)


def test_a_write_only_buffer_is_written_whole():
    """`PagedKVCacheGatherFwdOp`'s `dst` is write-only: on a small call with real values the
    reference leaves no row of it unwritten."""
    name = "PagedKVCacheGatherFwdOp"
    plan = entry_plan(name, load_manifest()[name], load_adts())
    row = {"T_q": 7, "NP": 6, "PS": 4, "W": 3, "E": [2, 3], "seq_lens": [3, 4], "starts": [2, 5]}
    call = instantiate(plan, {**row, "some": ["seq_starts"], "label": "s"}, {"KV": "bfloat16"})
    tensors = call.materialize("cpu")
    tensors["dst"].fill_(math.nan)
    _paged_kv_cache_gather(call.arguments(tensors), {n: tensors[n] for n in plan.sig.inputs})
    assert not tensors["dst"].isnan().any()
