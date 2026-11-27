"""Batch-1 prefill and decode tokens/s of the models under ``models/``, on one GPU, for the
nightly report.

The weights are random: like a model test, the run reads no checkpoint. Qwen3.8-27B (54 GB)
runs whole. DeepSeek-V4-Flash (156 GB) does not fit one H200, so it runs one layer at a time
(:func:`layer_by_layer`). A whole-model call is timed from an idle device to an idle device.
The prefills cover ``PREFILL_LENGTHS``, a short, a medium and a long prompt. Decode follows a
prefill; its steps take one token each with no host read between them, replay CUDA graphs, and
are timed as one run of ``DECODE_STEPS`` steps, which spreads the run's start from an idle
device over them. DeepSeek-V4-Flash's per-layer graphs leave out what a whole-model replay adds
each step: the host work of moving the step and copying the token in and the logits out, and
the derivation of the step's key indices, which its prefill seconds leave out as well.

Usage: ``python scripts/model_throughput.py --output model_throughput.json``
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from models.causal_lm import indexed_device
from models.deepseek_v4_flash import DeepseekV4FlashConfig, DeepseekV4FlashForCausalLM
from models.deepseek_v4_flash import loader as deepseek_loader
from models.deepseek_v4_flash.layers import Block, Step
from models.deepseek_v4_flash.ops import ModelOps
from models.qwen3_8 import Qwen3_8Config, Qwen3_8ForCausalLM
from models.qwen3_8 import loader as qwen_loader
from workloads.models import deepseek_v4_flash as deepseek_reference
from workloads.models import qwen3_8 as qwen_reference

PREFILL_LENGTHS = (512, 2048, 4096)
DECODE_STEPS = 32


def _timed(fn: Callable[[], object]) -> tuple[object, float]:
    """``fn``'s result and its wall seconds, from an idle device to an idle device."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def _median_of_three(reading: Callable[[], float]) -> float:
    """The median of three ``reading()`` values."""
    return statistics.median(reading() for _ in range(3))


def _seconds(fn: Callable[[], object]) -> float:
    """The wall seconds of ``fn`` after one untimed call, which compiles and tunes."""
    fn()
    return _median_of_three(lambda: _timed(fn)[1])


def _layer_call_seconds(call: Callable[[], object]) -> float:
    """The seconds ``call`` adds to a run of calls: two calls less one, which leaves out the
    idle device a timed call starts from."""
    call()
    return _median_of_three(lambda: _timed(lambda: (call(), call()))[1] - _timed(call)[1])


def _check_finite(logits: Sequence[torch.Tensor]) -> None:
    """Raise unless every one of ``logits`` is finite: a run on overflowed random activations
    is not the run real weights make."""
    if not all(torch.isfinite(x).all() for x in logits):
        raise ValueError("the random-weight run produced non-finite logits")


def _sample_layer(sample: dict[str, torch.Tensor], j: int, device: torch.device) -> dict:
    """Layer ``j``'s tensors of ``sample`` on ``device``, named relative to ``layers.<j>.``."""
    key = f"layers.{j}."
    return {n.split(key, 1)[1]: t.to(device) for n, t in sample.items() if key in n}


def _sample_outside(sample: dict[str, torch.Tensor], device: torch.device) -> dict:
    """The tensors of ``sample`` outside its layers, on ``device``."""
    return {n: t.to(device) for n, t in sample.items() if "layers." not in n}


def _first_of_each_kind(kinds: Sequence[object], sampled: int) -> dict[object, int]:
    """The first of the ``sampled`` leading layers of each kind in ``kinds``: a random checkpoint
    of those layers gives every layer of a kind the same weights. Every kind must have one."""
    first: dict[object, int] = {}
    for j in range(sampled):
        first.setdefault(kinds[j], j)
    missing = set(kinds) - set(first)
    if missing:
        raise ValueError(f"the first {sampled} layers hold no layer of kinds {sorted(missing)}")
    return first


def _decode_after_prefill(
    layer: Block,
    ops: ModelOps,
    prefill: tuple[torch.Tensor, torch.Tensor, Step],
    inputs: Sequence[torch.Tensor],
    tokens: Sequence[torch.Tensor],
    steps: Sequence[Step],
) -> tuple[torch.Tensor, list[torch.Tensor], float]:
    """Run ``layer``'s decode steps on ``inputs`` once eagerly, so they compile, on the state its
    last prefill of the whole prompt left, capture them in one CUDA graph and replay it once,
    which uploads it to the device; prefill the layer on ``prefill`` (streams, ids, step), which
    starts the request afresh, and time a second replay. Return the prefill's output, the steps'
    outputs and the second replay's seconds."""

    def run() -> list[torch.Tensor]:
        return [layer(ops, x, tok, s) for x, tok, s in zip(inputs, tokens, steps, strict=True)]

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    graph.replay()
    out = layer(ops, *prefill)
    _, seconds = _timed(graph.replay)
    return out, outputs, seconds


def layer_by_layer(
    model: DeepseekV4FlashForCausalLM,
    weights_of: Callable[[int], dict[str, torch.Tensor]],
    prompt: torch.Tensor,
    tokens: Sequence[torch.Tensor],
    lengths: Sequence[int],
) -> tuple[dict[int, float], float, list[torch.Tensor]]:
    """Run ``model`` one layer at a time on its current device, dropping each layer after use.

    Returns the prefill seconds of the first ``n`` tokens of ``prompt`` (``[1, S]``) for each
    ``n`` in ``lengths``, the seconds of one decode step after the whole prompt, and the logits
    of the whole prompt and of each decode step taking ``tokens`` (each ``[1, 1]``), which equal
    ``model.forward``'s. Layer ``i`` is filled from ``weights_of(i)``. The last of ``lengths``
    must be ``S``.

    A layer's prefill time is that of two calls less one, which drops the idle-device start a
    layer inside a whole forward never meets; a layer's decode steps replay one CUDA graph.
    """
    cfg, ops, device = model.config, model.ops, model.device
    total = prompt.shape[1]
    if lengths[-1] != total:
        raise ValueError(f"the last length must be the prompt's {total} tokens, got {lengths[-1]}")
    if total + len(tokens) > model.max_seq_len:
        raise ValueError(
            f"{total} + {len(tokens)} positions exceed max_seq_len {model.max_seq_len}"
        )
    prefill_steps = {n: Step.build(cfg, 0, n, device) for n in lengths}
    decode_steps = [Step.build(cfg, total + t, 1, device) for t in range(len(tokens))]

    def outside_prefill(n: int) -> torch.Tensor:
        h = model.streams(prompt[:, :n])
        Step.build(cfg, 0, n, device)
        return model.logits(h)

    def outside_decode() -> None:
        for token in tokens:
            ops.argmax(model.logits(model.streams(token)))

    prefill = {n: _seconds(lambda n=n: outside_prefill(n)) for n in lengths}
    outside_decode()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outside_decode()
    decode = _seconds(graph.replay) / len(tokens)
    h = model.streams(prompt)
    hd = [model.streams(token) for token in tokens]
    for i, layer in enumerate(model.layers):
        deepseek_loader.load_layer(layer, weights_of(i), cfg.n_routed_experts)
        for n in lengths:
            prefill[n] += _layer_call_seconds(
                functools.partial(
                    layer, ops, h[:, :n].contiguous(), prompt[:, :n], prefill_steps[n]
                )
            )
        h, hd, seconds = _decode_after_prefill(
            layer, ops, (h, prompt, prefill_steps[total]), hd, tokens, decode_steps
        )
        decode += seconds / len(tokens)
        model.layers[i] = None
    return prefill, decode, [model.logits(h)] + [model.logits(x) for x in hd]


def _deepseek_v4_flash(device: torch.device) -> dict:
    """DeepSeek-V4-Flash, one layer at a time, decoding after the longest prefill. Its first
    five layers hold the four kinds: sliding-window and ratio-4 hash-routed, ratio 128 and
    ratio 4."""
    published = deepseek_reference.DEEPSEEK_V4_FLASH_CONFIG
    cfg = DeepseekV4FlashConfig.from_hf(published)
    total = max(PREFILL_LENGTHS)
    model = DeepseekV4FlashForCausalLM(cfg, device=device, max_seq_len=total + DECODE_STEPS)
    kinds = [(r, i < cfg.n_hash_layers) for i, r in enumerate(cfg.compress_ratios[: cfg.n_layers])]
    sampled = 5
    first = _first_of_each_kind(kinds, sampled)
    sample = deepseek_reference.random_checkpoint(
        {**published, "num_hidden_layers": sampled}, seed=0
    )
    deepseek_loader.load_outside(model, _sample_outside(sample, device))
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(cfg.vocab_size, (1, total), generator=g).to(device)
    tokens = [
        torch.randint(cfg.vocab_size, (1, 1), generator=g).to(device) for _ in range(DECODE_STEPS)
    ]
    prefill, decode, logits = layer_by_layer(
        model,
        lambda i: _sample_layer(sample, first[kinds[i]], device),
        prompt,
        tokens,
        PREFILL_LENGTHS,
    )
    _check_finite(logits)
    return {
        "run": "layer by layer",
        "prefill_tokens_per_s": {str(n): n / s for n, s in prefill.items()},
        "decode_tokens_per_s": {str(total): 1 / decode},
    }


def _greedy_steps(model: Qwen3_8ForCausalLM, logits: torch.Tensor, pos: int) -> torch.Tensor:
    """``DECODE_STEPS`` decode steps from ``pos``, each taking the argmax of the logits before
    it; return the last step's logits."""
    for p in range(pos, pos + DECODE_STEPS):
        logits = model.forward(model.ops.argmax(logits).view(1, 1), p)
    return logits


def _qwen3_8(device: torch.device) -> dict:
    """Qwen3.8-27B whole, decoding after the shortest and the longest prefill: its
    full-attention layers read the whole KV cache. Its first four layers hold both kinds."""
    published = qwen_reference.QWEN3_8_27B_CONFIG
    text = published["text_config"]
    cfg = Qwen3_8Config.from_hf(published)
    total = max(PREFILL_LENGTHS)
    model = Qwen3_8ForCausalLM(cfg, device=device, max_seq_len=total + DECODE_STEPS)
    sampled = 4
    first = _first_of_each_kind(cfg.layer_types, sampled)
    sample = qwen_reference.random_checkpoint(
        {
            **published,
            "text_config": {
                **text,
                "num_hidden_layers": sampled,
                "layer_types": text["layer_types"][:sampled],
            },
        },
        seed=0,
    )
    for i, layer in enumerate(model.layers):
        qwen_loader.load_layer(layer, _sample_layer(sample, first[cfg.layer_types[i]], device))
    qwen_loader.load_outside(model, _sample_outside(sample, device))
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(cfg.vocab_size, (1, total), generator=g).to(device)
    prefill = {
        n: n / _seconds(lambda n=n: model.forward(prompt[:, :n], 0)) for n in PREFILL_LENGTHS
    }
    decode = {}
    for ctx in (min(PREFILL_LENGTHS), total):
        _greedy_steps(model, model.forward(prompt[:, :ctx], 0), ctx)
        start = model.forward(prompt[:, :ctx], 0)
        last, seconds = _timed(functools.partial(_greedy_steps, model, start, ctx))
        _check_finite([start, last])
        decode[str(ctx)] = DECODE_STEPS / seconds
    return {
        "run": "whole model",
        "prefill_tokens_per_s": {str(n): v for n, v in prefill.items()},
        "decode_tokens_per_s": decode,
    }


_MODELS = {"deepseek_v4_flash": _deepseek_v4_flash, "qwen3_8": _qwen3_8}


def main() -> int:
    """Time each model and write ``{model: result}`` to ``--output`` before the runs and after
    each one; a model not yet timed reads ``{"error": "the run did not finish"}``, so a run killed
    midway still lists it. Exit 1 if a model failed, its result then ``{"error": ...}``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, help="Path of the JSON to write")
    output = Path(parser.parse_args().output)
    device = indexed_device("cuda")
    results = {name: {"error": "the run did not finish"} for name in _MODELS}
    output.write_text(json.dumps(results, indent=2) + "\n")
    for name, run in _MODELS.items():
        try:
            with torch.inference_mode():
                results[name] = run(device)
        except Exception as e:
            traceback.print_exc()
            results[name] = {"error": f"{type(e).__name__}: {e}"}
        output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({name: results[name]}), flush=True)
    return int(any("error" in r for r in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
