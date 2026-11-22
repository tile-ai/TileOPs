"""DeepSeek-V4's official ``inference/`` code, with torch in place of its TileLang kernels.

``official_model`` imports the checkpoint repository's ``inference/model.py`` with a torch module
standing in for its ``kernel.py``, so the reference shares no kernel with TileOPs.
``DEEPSEEK_V4_FLASH_CONFIG`` is DeepSeek-V4-Flash's published ``config.json``.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import math
import os
import sys
import types
from pathlib import Path

import torch

from workloads.quantization.fp8_quant import fp8_quant_per_group
from workloads.quantization.quantize import fp4_quant_per_group, unpack_e2m1
from workloads.sequence_modeling.hyper_connection import hc_split_mixes
from workloads.transform import hadamard_transform

__all__ = [
    "DEEPSEEK_V4_FLASH_CONFIG",
    "check_index_selection",
    "check_selection_agreement",
    "official_defaults",
    "official_model",
    "random_checkpoint",
]


def act_quant(
    x: torch.Tensor,
    block_size: int,
    scale_fmt: str | None,
    scale_dtype: torch.dtype,
    inplace: bool = False,
):
    """``kernel.act_quant`` as the official model calls it, with a ``scale_fmt``:
    :func:`fp8_quant_per_group` with ``amax`` floored at 1e-4 and scales rounded up to a power of
    two."""
    assert scale_fmt is not None, "the official model quantizes activations under a scale_fmt"
    return fp8_quant_per_group(x, block_size, 1e-4, True, scale_dtype, inplace)


def fp4_act_quant(x: torch.Tensor, block_size: int, inplace: bool):
    """``kernel.fp4_act_quant`` as the official model calls it, ``inplace``: ``x`` overwritten
    by its e2m1 codes per ``block_size`` elements under a power-of-two scale, dequantized."""
    assert inplace, "the official model quantizes FP4 activations in place only"
    fp4_quant_per_group(x, block_size, inplace=True)
    return x


def _dequant_rows(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Float32 values of ``codes`` (``[..., K]``) under per-row-block ``scale`` (``[..., K/b]``)."""
    block = codes.shape[-1] // scale.shape[-1]
    return (codes.float().unflatten(-1, (-1, block)) * scale.float().unsqueeze(-1)).flatten(-2)


def _dequant_blocks(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Float32 values of FP8 ``w`` (``[N, K]``) under 128x128 block ``scale``."""
    return _dequant_rows(w, scale.float().repeat_interleave(128, 0)[: w.shape[0]])


def _gemm(a: torch.Tensor, a_s: torch.Tensor, b32: torch.Tensor) -> torch.Tensor:
    """``a @ b32^T`` in float32, ``a`` FP8 under per-row-block scales ``a_s``, returned in the
    default dtype."""
    k = a.shape[-1]
    a32 = _dequant_rows(a.reshape(-1, k), a_s.reshape(-1, a_s.shape[-1]))
    return (a32 @ b32.T).to(torch.get_default_dtype()).view(*a.shape[:-1], b32.shape[0])


def fp8_gemm(a, a_s, b, b_s, scale_dtype):
    """``kernel.fp8_gemm``: ``a @ b^T`` in float32 over dequantized FP8 operands, ``b`` under
    128x128 block scales, returned in the default dtype; the scales are read as float32
    whatever their ``scale_dtype``."""
    return _gemm(a, a_s, _dequant_blocks(b, b_s))


def fp4_gemm(a, a_s, b, b_s, scale_dtype):
    """``kernel.fp4_gemm``: ``a @ b^T`` in float32, ``b`` e2m1 codes two per byte (even element
    in the low nibble) under per-32 scales, returned in the default dtype; the scales are read as
    float32 whatever their ``scale_dtype``."""
    return _gemm(a, a_s, _dequant_rows(unpack_e2m1(b.view(torch.uint8)), b_s))


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, p_dtype=None):
    """``kernel.sparse_attn``: each query attends in float32 to the ``kv`` rows its indices
    name (-1 skips), ``exp(attn_sink[h])`` joining head ``h``'s softmax denominator.

    ``p_dtype`` rounds the unnormalized probabilities before they weight the values, as the
    TileLang kernel rounds them to bfloat16. Queries run 256 at a time, which bounds the float32
    ``kv`` rows gathered for one block of queries."""
    chunk = 256
    b, m, h, d = q.shape
    out = torch.empty_like(q)
    sink = attn_sink.float().view(1, 1, h, 1)
    for start in range(0, m, chunk):
        idx = topk_idxs[:, start : start + chunk].long()
        valid = idx >= 0
        rows = torch.gather(
            kv.float().unsqueeze(1).expand(-1, idx.shape[1], -1, -1),
            2,
            idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, d),
        )
        qc = q[:, start : start + chunk].float()
        scores = torch.einsum("bmhd,bmkd->bmhk", qc, rows) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        peak = torch.maximum(scores.amax(-1, keepdim=True), sink)
        p = torch.exp(scores - peak)
        denom = p.sum(-1, keepdim=True) + torch.exp(sink - peak)
        if p_dtype is not None:
            p = p.to(p_dtype).float()
        o = torch.einsum("bmhk,bmkd->bmhd", p, rows) / denom
        out[:, start : start + chunk] = o.to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    """``kernel.hc_split_sinkhorn``: :func:`hc_split_mixes` with one ``eps``, ``post`` doubled."""
    pre, post, comb = hc_split_mixes(mixes, hc_scale, hc_base, hc_mult, eps, eps, sinkhorn_iters)
    return pre, 2 * post, comb


def _load_official(inference_dir: str | os.PathLike) -> types.ModuleType:
    """Import the official ``inference/model.py`` in ``inference_dir`` against a ``kernel``
    module of this file's torch functions. ``rotate_activation`` is replaced too, so
    ``fast_hadamard_transform`` is not needed."""
    path = Path(inference_dir) / "model.py"
    kernel = types.ModuleType("kernel")
    for fn in (act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn):
        setattr(kernel, fn.__name__, fn)
    saved = sys.modules.get("kernel")
    sys.modules["kernel"] = kernel
    try:
        spec = importlib.util.spec_from_file_location("deepseek_v4_official_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            del sys.modules["kernel"]
        else:
            sys.modules["kernel"] = saved
    module.rotate_activation = hadamard_transform
    return module


DEEPSEEK_V4_FLASH_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "expert_dtype": "fp4",
    "hc_eps": 1e-06,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "head_dim": 512,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "index_head_dim": 128,
    "index_n_heads": 64,
    "index_topk": 512,
    "initializer_range": 0.02,
    "max_position_embeddings": 1048576,
    "model_type": "deepseek_v4",
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 64,
    "num_experts_per_tok": 6,
    "num_hidden_layers": 43,
    "num_hash_layers": 3,
    "num_key_value_heads": 1,
    "num_nextn_predict_layers": 1,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "q_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    },
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn",
    },
    "rope_theta": 10000,
    "routed_scaling_factor": 1.5,
    "scoring_func": "sqrtsoftplus",
    "sliding_window": 128,
    "swiglu_limit": 10.0,
    "tie_word_embeddings": False,
    "topk_method": "noaux_tc",
    "torch_dtype": "bfloat16",
    "transformers_version": "4.57.1",
    "use_cache": True,
    "vocab_size": 129280,
    "compress_rope_theta": 160000,
    "compress_ratios": [0, 0] + [4, 128] * 20 + [4, 0],
}


def _official_args(module: types.ModuleType, config: dict, max_seq_len: int):
    """The official ``ModelArgs`` of a Hugging Face ``config`` dict for one request, without MTP
    layers."""
    rope = config["rope_scaling"]
    return module.ModelArgs(
        max_batch_size=1,
        max_seq_len=max_seq_len,
        dtype="fp8",
        scale_fmt="ue8m0",
        expert_dtype="fp4",
        scale_dtype="fp8",
        vocab_size=config["vocab_size"],
        dim=config["hidden_size"],
        moe_inter_dim=config["moe_intermediate_size"],
        n_layers=config["num_hidden_layers"],
        n_hash_layers=config["num_hash_layers"],
        n_mtp_layers=0,
        n_heads=config["num_attention_heads"],
        n_routed_experts=config["n_routed_experts"],
        n_shared_experts=config["n_shared_experts"],
        n_activated_experts=config["num_experts_per_tok"],
        score_func=config["scoring_func"],
        route_scale=config["routed_scaling_factor"],
        swiglu_limit=config["swiglu_limit"],
        q_lora_rank=config["q_lora_rank"],
        head_dim=config["head_dim"],
        rope_head_dim=config["qk_rope_head_dim"],
        norm_eps=config["rms_norm_eps"],
        o_groups=config["o_groups"],
        o_lora_rank=config["o_lora_rank"],
        window_size=config["sliding_window"],
        compress_ratios=tuple(config["compress_ratios"]),
        compress_rope_theta=config["compress_rope_theta"],
        original_seq_len=rope["original_max_position_embeddings"],
        rope_theta=config["rope_theta"],
        rope_factor=rope["factor"],
        beta_fast=rope["beta_fast"],
        beta_slow=rope["beta_slow"],
        index_n_heads=config["index_n_heads"],
        index_head_dim=config["index_head_dim"],
        index_topk=config["index_topk"],
        hc_mult=config["hc_mult"],
        hc_sinkhorn_iters=config["hc_sinkhorn_iters"],
        hc_eps=config["hc_eps"],
    )


def _official_state_dict(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Checkpoint tensors in the dtypes the official modules hold, as ``convert.py`` writes
    them, for their ``load_state_dict``: ``wo_a`` dequantized to bfloat16 by its block scale,
    routed-expert bytes as ``float4_e2m1fn_x2``; MTP tensors and ``wo_a`` scales are dropped."""
    out = {}
    for name, tensor in tensors.items():
        if name.startswith("mtp.") or name.endswith("wo_a.scale"):
            continue
        if name.endswith("wo_a.weight"):
            tensor = _dequant_blocks(tensor, tensors[name.replace("weight", "scale")]).bfloat16()
        elif ".experts." in name and tensor.dtype == torch.int8:
            tensor = tensor.view(torch.float4_e2m1fn_x2)
        out[name] = tensor
    return out


@contextlib.contextmanager
def official_defaults(device: torch.device | str):
    """Make bfloat16 and ``device`` torch's defaults, which the official model is built and run
    under, and restore the previous ones after."""
    dtype, previous = torch.get_default_dtype(), torch.get_default_device()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_dtype(dtype)
        torch.set_default_device(previous)


def official_model(
    inference_dir: str | os.PathLike,
    tensors: dict[str, torch.Tensor],
    config: dict,
    max_seq_len: int,
    *,
    rounded: bool = False,
):
    """The official ``Transformer`` of ``config`` from ``inference_dir``, for one request and
    holding the checkpoint ``tensors``; ``rounded`` rounds its attention probabilities to
    bfloat16, as its TileLang kernel does. Build and run it under :func:`official_defaults`."""
    module = _load_official(inference_dir)
    if rounded:
        module.sparse_attn = functools.partial(sparse_attn, p_dtype=torch.bfloat16)
    model = module.Transformer(_official_args(module, config, max_seq_len))
    missing, unexpected = model.load_state_dict(_official_state_dict(tensors), strict=False)
    buffers = ("kv_cache", "freqs_cis", "kv_state", "score_state")
    assert [m for m in missing if not any(b in m for b in buffers)] == []
    assert unexpected == []
    return model


def random_checkpoint(config: dict, seed: int) -> dict[str, torch.Tensor]:
    """Random weights for ``config`` under the official tensor names, dtypes and layouts.

    Dense linears are standard normal FP8 codes under one power-of-two scale near ``k ** -0.5``
    per 128x128 block, routed experts uniformly random e2m1 bytes under per-32 power-of-two
    scales near ``k ** -0.5 / 2.9``, 2.9 being the RMS of a uniformly random e2m1 code, each
    jittered by up to an octave: magnitudes near the published checkpoint's. MTP tensors are
    not generated.
    """
    g = torch.Generator().manual_seed(seed)

    def normal(*shape, std=1.0, mean=0.0):
        return torch.randn(*shape, generator=g) * std + mean

    dim, n_heads, head_dim = (
        config["hidden_size"],
        config["num_attention_heads"],
        config["head_dim"],
    )
    q_rank, o_groups, o_rank = config["q_lora_rank"], config["o_groups"], config["o_lora_rank"]
    inter, n_exp = config["moe_intermediate_size"], config["n_routed_experts"]
    hc, top_k = config["hc_mult"], config["num_experts_per_tok"]
    idx_heads, idx_dim = config["index_n_heads"], config["index_head_dim"]
    mix = (2 + hc) * hc
    t: dict[str, torch.Tensor] = {}

    def pow2_scale(shape, scale):
        exponent = torch.full(shape, round(math.log2(scale)), dtype=torch.float32)
        exponent += torch.randint(-1, 2, shape, generator=g)
        return torch.exp2(exponent).to(torch.float8_e8m0fnu)

    def fp8_linear(name, n, k):
        t[f"{name}.weight"] = normal(n, k).to(torch.float8_e4m3fn)
        t[f"{name}.scale"] = pow2_scale((-(-n // 128), k // 128), k**-0.5)

    def norm(name, d):
        t[f"{name}.weight"] = normal(d, std=0.1, mean=1.0).bfloat16()

    def compressor(name, ratio, d):
        width = (2 if ratio == 4 else 1) * d
        t[f"{name}.wkv.weight"] = normal(width, dim, std=dim**-0.5).bfloat16()
        t[f"{name}.wgate.weight"] = normal(width, dim, std=dim**-0.5).bfloat16()
        t[f"{name}.ape"] = normal(ratio, width, std=0.3)
        norm(f"{name}.norm", d)

    def fp4_expert(name, n, k):
        t[f"{name}.weight"] = torch.randint(-128, 128, (n, k // 2), generator=g, dtype=torch.int8)
        t[f"{name}.scale"] = pow2_scale((n, k // 32), k**-0.5 / 2.9)

    t["embed.weight"] = normal(config["vocab_size"], dim).bfloat16()
    for i, ratio in enumerate(config["compress_ratios"][: config["num_hidden_layers"]]):
        p = f"layers.{i}."
        norm(p + "attn_norm", dim)
        norm(p + "ffn_norm", dim)
        for part in ("attn", "ffn"):
            t[f"{p}hc_{part}_fn"] = normal(mix, hc * dim, std=0.05)
            t[f"{p}hc_{part}_base"] = normal(mix)
            t[f"{p}hc_{part}_scale"] = normal(3, std=0.05, mean=0.1)
        fp8_linear(p + "attn.wq_a", q_rank, dim)
        norm(p + "attn.q_norm", q_rank)
        fp8_linear(p + "attn.wq_b", n_heads * head_dim, q_rank)
        fp8_linear(p + "attn.wkv", head_dim, dim)
        norm(p + "attn.kv_norm", head_dim)
        fp8_linear(p + "attn.wo_a", o_groups * o_rank, n_heads * head_dim // o_groups)
        fp8_linear(p + "attn.wo_b", dim, o_groups * o_rank)
        t[p + "attn.attn_sink"] = normal(n_heads, std=0.5)
        if ratio:
            compressor(p + "attn.compressor", ratio, head_dim)
        if ratio == 4:
            fp8_linear(p + "attn.indexer.wq_b", idx_heads * idx_dim, q_rank)
            t[p + "attn.indexer.weights_proj.weight"] = normal(idx_heads, dim, std=0.3).bfloat16()
            compressor(p + "attn.indexer.compressor", ratio, idx_dim)
        t[p + "ffn.gate.weight"] = normal(n_exp, dim, std=0.03).bfloat16()
        if i < config["num_hash_layers"]:
            ids = torch.rand(config["vocab_size"], n_exp, generator=g).argsort(-1)[:, :top_k]
            t[p + "ffn.gate.tid2eid"] = ids.to(torch.int64).contiguous()
        else:
            t[p + "ffn.gate.bias"] = normal(n_exp, std=0.1)
        for w in ("w1", "w3"):
            fp8_linear(f"{p}ffn.shared_experts.{w}", inter, dim)
        fp8_linear(p + "ffn.shared_experts.w2", dim, inter)
        for e in range(n_exp):
            fp4_expert(f"{p}ffn.experts.{e}.w1", inter, dim)
            fp4_expert(f"{p}ffn.experts.{e}.w3", inter, dim)
            fp4_expert(f"{p}ffn.experts.{e}.w2", dim, inter)
    norm("norm", dim)
    t["head.weight"] = normal(config["vocab_size"], dim, std=dim**-0.5).bfloat16()
    t["hc_head_fn"] = normal(hc, hc * dim, std=0.05)
    t["hc_head_base"] = normal(hc)
    t["hc_head_scale"] = normal(1, std=0.05, mean=0.1)
    return t


def check_index_selection(
    indices: torch.Tensor, start_pos: int, offset: int, ratio: int, top_k: int
) -> None:
    """Assert each query of a call starting at ``start_pos`` selects exactly
    ``min(top_k, visible)`` distinct compressed entries, all closed before or at it: the entry
    of window ``j`` closes at position ``(j + 1) * ratio - 1``. ``indices`` (``[1, S, k]``) holds
    entry numbers plus ``offset``, -1 padding."""
    seqlen = indices.shape[1]
    pos = torch.arange(start_pos, start_pos + seqlen, device=indices.device).view(1, -1, 1)
    visible = (pos + 1) // ratio
    entries = torch.where(indices >= 0, indices - offset, -1)
    chosen = entries >= 0
    assert bool(((entries < visible) | ~chosen).all()), "an index selects an entry not yet closed"
    assert bool((chosen.sum(-1, keepdim=True) == visible.clamp(max=top_k)).all()), (
        "a query selects other than min(top_k, closed entries)"
    )
    ordered = entries.sort(-1).values
    assert bool(((ordered[..., 1:] != ordered[..., :-1]) | (ordered[..., 1:] < 0)).all()), (
        "a query selects an entry twice"
    )


def check_selection_agreement(selections: list, official: list, top_k: int) -> None:
    """Assert the indexer's selections agree with the official indexer's on the same inputs.

    ``selections`` holds each call's ``(indices, start_pos, offset, ratio)`` and ``official`` the
    official indexer's output for the same call. Over the queries that see more than ``top_k``
    closed entries, and so choose, on average at most ``top_k / 256`` of the entries kept may be
    missing from the official selection: FP4 rounding ties move a few entries across the cut,
    a scoring fault moves many.
    """
    differing = choosing = 0
    for (indices, start_pos, _, ratio), theirs in zip(selections, official, strict=True):
        pos = torch.arange(start_pos, start_pos + indices.shape[1], device=indices.device)
        chooses = (pos + 1) // ratio > top_k
        if not chooses.any():
            continue
        width = int(max(indices.max(), theirs.max())) + 2
        kept = torch.zeros(*theirs.shape[:2], width, dtype=torch.bool, device=indices.device)
        kept.scatter_(-1, theirs.long() + 1, True)
        ours_only = (indices >= 0) & ~kept.gather(-1, indices.long() + 1)
        differing += int(ours_only.sum(-1)[:, chooses].sum())
        choosing += int(chooses.sum()) * indices.shape[0]
    assert differing <= choosing * top_k / 256, (
        f"{differing} selected entries over {choosing} choosing queries differ from the official "
        f"indexer's, bound {top_k / 256:.3g} per query"
    )
