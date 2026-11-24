"""DeepseekV4FlashForCausalLM against DeepSeek's official ``inference/model.py`` run on torch kernels.

A three-layer model (sliding window with hash routing, ratio-4 compressed attention with the
indexer, ratio-128 compressed attention) at DeepSeek-V4-Flash's widths, on random weights in the
checkpoint's formats, is written as a Hugging Face checkpoint and loaded by both. Each layer's
output and the logits are compared over the prefill and over the decode steps by
``check_within_rounding``, the rounded reference being the official model with its attention
probabilities rounded to bfloat16, as its TileLang kernel rounds them. The long prompt is longer
than four times the indexer's top-k, so the indexer selects from more entries than it keeps, and
leaves partial windows that the decode steps close at both compression ratios; the short one closes
no window, so the decode steps close the first.
"""

import dataclasses
import importlib.util
import time
from pathlib import Path

import pytest
import torch

from models.deepseek_v4_flash import DeepseekV4FlashConfig, DeepseekV4FlashForCausalLM
from models.deepseek_v4_flash.layers import Step
from models.deepseek_v4_flash.loader import load_outside
from models.deepseek_v4_flash.ops import RoPETable
from workloads.device import run_device
from workloads.models.checkpoint import write_checkpoint
from workloads.models.deepseek_v4_flash import (
    DEEPSEEK_V4_FLASH_CONFIG,
    check_index_selection,
    check_selection_agreement,
    official_defaults,
    official_model,
    random_checkpoint,
)
from workloads.models.rounding import check_argmax_within_rounding, check_within_rounding

pytestmark = pytest.mark.nightly

_SMALL = {
    **DEEPSEEK_V4_FLASH_CONFIG,
    "num_hidden_layers": 3,
    "num_hash_layers": 1,
    "compress_ratios": [0, 4, 128, 0],
    "n_routed_experts": 8,
    "vocab_size": 1024,
}
_TINY = {
    **_SMALL,
    "num_hidden_layers": 1,
    "compress_ratios": [0, 0],
    "vocab_size": 64,
    "hidden_size": 256,
    "moe_intermediate_size": 256,
    "num_attention_heads": 2,
    "head_dim": 128,
    "q_lora_rank": 128,
    "o_groups": 1,
    "o_lora_rank": 128,
}


def test_config_reads_the_published_flash_config():
    cfg = DeepseekV4FlashConfig.from_hf(DEEPSEEK_V4_FLASH_CONFIG)
    assert (cfg.n_layers, cfg.n_hash_layers, cfg.n_routed_experts) == (43, 3, 256)
    assert cfg.compress_ratios[:4] == (0, 0, 4, 128)
    assert (cfg.rope_factor, cfg.original_seq_len, cfg.route_scale) == (16, 65536, 1.5)


def test_rope_table_takes_a_zero_width_yarn_range():
    """A YaRN setting whose interpolated dimensions span no width, here an original length of six
    positions, still gives finite cosines and sines."""
    rope = {**DEEPSEEK_V4_FLASH_CONFIG["rope_scaling"], "original_max_position_embeddings": 6}
    cfg = DeepseekV4FlashConfig.from_hf({**DEEPSEEK_V4_FLASH_CONFIG, "rope_scaling": rope})
    table = RoPETable(cfg, cfg.compress_rope_theta, True, 16, torch.device(run_device()))
    assert torch.isfinite(table.cos).all() and torch.isfinite(table.sin).all()


def _quantized(**changes) -> dict:
    return {
        **DEEPSEEK_V4_FLASH_CONFIG,
        "quantization_config": {**DEEPSEEK_V4_FLASH_CONFIG["quantization_config"], **changes},
    }


@pytest.mark.parametrize(
    ("config", "match"),
    [
        pytest.param({"model_type": "llama"}, "model_type", id="model-type"),
        pytest.param({"compress_ratios": [0] * 43}, "compress_ratios", id="ratio-list-length"),
        pytest.param({"n_shared_experts": 2}, "shared expert", id="shared-experts"),
        pytest.param({"attention_bias": True}, "attention_bias", id="attention-bias"),
        pytest.param({"num_key_value_heads": 8}, "num_key_value_heads", id="kv-heads"),
        pytest.param({"hidden_act": "gelu"}, "hidden_act", id="activation"),
        pytest.param(
            {"rope_scaling": {**DEEPSEEK_V4_FLASH_CONFIG["rope_scaling"], "type": "linear"}},
            "rope_scaling",
            id="rope-scaling",
        ),
        pytest.param({"tie_word_embeddings": True}, "tie_word_embeddings", id="tied"),
        pytest.param({"torch_dtype": "float16"}, "torch_dtype", id="dtype"),
        pytest.param({"expert_dtype": "fp8"}, "expert_dtype", id="expert-dtype"),
        pytest.param(_quantized(quant_method="int8"), "quantization_config", id="quant-method"),
        pytest.param(_quantized(fmt="e5m2"), "quantization_config", id="quant-fmt"),
        pytest.param(_quantized(scale_fmt=None), "quantization_config", id="quant-scale-fmt"),
        pytest.param(
            _quantized(weight_block_size=[64, 64]), "quantization_config", id="quant-block"
        ),
    ],
)
def test_config_refuses_what_the_model_does_not_compute(config, match):
    with pytest.raises(ValueError, match=match):
        DeepseekV4FlashConfig.from_hf({**DEEPSEEK_V4_FLASH_CONFIG, **config})


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """The small model's checkpoint directory and its tensors."""
    path = tmp_path_factory.mktemp("dsv4-small")
    tensors = random_checkpoint(_SMALL, seed=0)
    write_checkpoint(path, _SMALL, tensors)
    return path, tensors


@pytest.mark.parametrize(
    ("change", "extra", "match"),
    [
        pytest.param({}, "head.bias", "unused checkpoint tensors", id="model"),
        pytest.param({}, "layers.1.ffn.gate.weight", "unused checkpoint tensors", id="layer-1"),
        pytest.param({}, "layers.0.attn.bias", "layer 0: unused tensors", id="in-layer"),
        pytest.param({"moe_intermediate_size": 64}, None, "128-row blocks", id="partial-block"),
    ],
)
def test_from_pretrained_refuses_a_checkpoint_it_cannot_read_whole(tmp_path, change, extra, match):
    """A tensor the model has no place for, and a stacked FP8 weight whose first part ends
    inside a scale block, are refused rather than dropped or misaligned."""
    config = {**_TINY, **change}
    tensors = random_checkpoint(config, seed=0)
    if extra is not None:
        tensors[extra] = torch.zeros(1)
    write_checkpoint(tmp_path, config, tensors)
    with pytest.raises(ValueError, match=match):
        DeepseekV4FlashForCausalLM.from_pretrained(tmp_path, device=run_device())


def test_from_pretrained_refuses_a_checkpoint_missing_a_layer(tmp_path):
    tensors = random_checkpoint(_TINY, seed=0)
    kept = {n: t for n, t in tensors.items() if not n.startswith("layers.0.")}
    write_checkpoint(tmp_path, _TINY, kept)
    with pytest.raises(ValueError, match="holds no tensors of layers \\[0\\]"):
        DeepseekV4FlashForCausalLM.from_pretrained(tmp_path, device=run_device())


def test_from_pretrained_refuses_a_layer_missing_some_tensors(tmp_path):
    tensors = random_checkpoint(_TINY, seed=0)
    del tensors["layers.0.attn.kv_norm.weight"]
    write_checkpoint(tmp_path, _TINY, tensors)
    with pytest.raises(ValueError, match="layer 0: missing tensor 'attn.kv_norm.weight'"):
        DeepseekV4FlashForCausalLM.from_pretrained(tmp_path, device=run_device())


def test_from_pretrained_skips_the_mtp_tensors(tmp_path):
    tensors = random_checkpoint(_TINY, seed=0)
    tensors["mtp.0.norm.weight"] = torch.ones(_TINY["hidden_size"], dtype=torch.bfloat16)
    write_checkpoint(tmp_path, _TINY, tensors)
    model = DeepseekV4FlashForCausalLM.from_pretrained(tmp_path, device=run_device())
    assert model.head.shape == (_TINY["vocab_size"], _TINY["hidden_size"])


def _check_decode_state(block, official_block, positions: int) -> None:
    """Assert a TileOPs layer's decoding state matches the official block's after both took the
    same inputs over ``positions`` tokens; entries are compared once a window has closed.

    The window ring and the compressed entries hold FP8 fake-quantized rows, the indexer's
    entries FP4 ones. Where the two sides' inputs differ by bfloat16 rounding, a value on a
    rounding boundary takes the neighbouring code; 1e-2 for FP8 and 2e-2 for the coarser FP4
    codes, in relative L2, admit such flips on a few values, while a wrong position, rotation or
    slot moves the state by order one.
    """
    fp8_tol, fp4_tol = 1e-2, 2e-2
    attn, official = block.attn, official_block.attn
    ring = slice(0, attn.window)
    pairs = [("window ring", attn.kv_cache[:, ring], official.kv_cache[:, ring], fp8_tol)]
    if attn.ratio and positions >= attn.ratio:
        entries = slice(attn.window, attn.window + positions // attn.ratio)
        pairs.append(("entries", attn.kv_cache[:, entries], official.kv_cache[:, entries], fp8_tol))
    if attn.indexer is not None and positions >= attn.indexer.ratio:
        n = positions // attn.indexer.ratio
        actual, expected = attn.indexer.cache[:, :n], official.indexer.kv_cache[:, :n]
        pairs.append(("indexer entries", actual, expected, fp4_tol))
    for name, actual, expected, tol in pairs:
        rel = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
        assert rel <= tol, f"{name}: {rel:.3e} relative L2 from the official state, bound {tol:.0e}"


class _Recorded:
    """An indexer that records each call's selection, start position, offset and ratio."""

    def __init__(self, indexer, record):
        self._indexer = indexer
        self._record = record

    def __getattr__(self, name):
        return getattr(self._indexer, name)

    def __call__(self, ops, rope, x, qr, step):
        indices = self._indexer(ops, rope, x, qr, step)
        self._record.append((indices, step.start_pos, step.offset, self._indexer.ratio))
        return indices


@pytest.mark.sm90
@pytest.mark.parametrize(
    ("prompt_len", "decode_steps"),
    [pytest.param(2170, 8, id="long-prompt"), pytest.param(3, 6, id="short-prompt")],
)
def test_layers_and_logits_match_the_official_model(checkpoint, prompt_len, decode_steps):
    """Each layer, fed the official model's input to it, against that layer of the official model,
    over the prefill and over the decode steps, and its decoding state at the end; the entries
    the indexer selects, against the official indexer's; and the logits of the whole model.
    ``max_seq_len`` 2304 holds the long request's 2178 tokens in whole ratio-128 windows."""
    path, tensors = checkpoint
    max_seq_len = 2304
    mine, mine_tf = (
        DeepseekV4FlashForCausalLM.from_pretrained(
            path, device=run_device(), max_seq_len=max_seq_len
        )
        for _ in range(2)
    )
    with official_defaults(run_device()):
        on_device = {k: v.to(run_device()) for k, v in tensors.items()}
        ref = official_model(on_device, _SMALL, max_seq_len)
        rounded, rounded_tf = (
            official_model(on_device, _SMALL, max_seq_len, rounded=True) for _ in range(2)
        )
        trace, official_selections = [], []
        for i, layer in enumerate(ref.layers):
            layer.register_forward_hook(
                lambda _m, args, out, i=i: trace.append((i, args[0].clone(), out.clone()))
            )
        for layer in rounded_tf.layers:
            if getattr(layer.attn, "indexer", None) is not None:
                layer.attn.indexer.register_forward_hook(
                    lambda _m, _args, out: official_selections.append(out)
                )
        g = torch.Generator().manual_seed(1)
        shape = (1, prompt_len + decode_steps)
        ids = torch.randint(0, _SMALL["vocab_size"], shape, generator=g, device="cpu")
        ids = ids.to(run_device())
        phases = {"prefill": [0], "decode": list(range(prompt_len, prompt_len + decode_steps))}
        selections = []
        for layer in mine_tf.layers:
            if layer.attn.indexer is not None:
                layer.attn.indexer = _Recorded(layer.attn.indexer, selections)
        with torch.inference_mode():
            for phase, positions in phases.items():
                logits = ([], [], [])
                layers = {i: ([], [], []) for i in range(len(ref.layers))}
                for pos in positions:
                    tokens = ids[:, :prompt_len] if pos == 0 else ids[:, pos : pos + 1]
                    trace.clear()
                    for out, model in zip(logits, (mine, ref, rounded), strict=True):
                        out.append(model(tokens, pos))
                    step = Step.build(mine_tf.config, pos, tokens.shape[1], mine_tf.device)
                    for i, h_in, h_out in trace:
                        actual, expected, rnd = layers[i]
                        layer = mine_tf.layers[i]
                        actual.append(layer(mine_tf.ops, h_in.contiguous(), tokens, step))
                        expected.append(h_out)
                        rnd.append(rounded_tf.layers[i](h_in.clone(), pos, tokens))
                for i, outs in layers.items():
                    check_within_rounding(*outs, f"{phase} layer {i} outputs")
                check_within_rounding(*logits, f"{phase} logits")
            for layer, official in zip(mine_tf.layers, ref.layers, strict=True):
                _check_decode_state(layer, official, prompt_len + decode_steps)
    top_k = _SMALL["index_topk"]
    assert len(selections) == 1 + decode_steps
    for indices, start_pos, offset, ratio in selections:
        check_index_selection(indices, start_pos, offset, ratio, top_k)
    check_selection_agreement(selections, official_selections, top_k)


@pytest.mark.sm90
def test_forward_takes_one_request_position_by_position(checkpoint):
    """``forward`` refuses a batch, an empty input, a decode step out of order or of more than
    one token, and a sequence past ``max_seq_len``; a refused call leaves the request where it
    was."""
    path, _ = checkpoint
    max_seq_len = 18
    model = DeepseekV4FlashForCausalLM.from_pretrained(
        path, device=run_device(), max_seq_len=max_seq_len
    )
    ids = torch.zeros(1, max_seq_len + 1, dtype=torch.long, device=run_device())
    refused = [
        (ids[:, :4].expand(2, -1), 0, "batch of 2"),
        (ids[:, :0], 0, "at least one token"),
        (ids[:, :1], 4, "prefill"),
    ]
    for step, pos, match in refused:
        with pytest.raises(ValueError, match=match):
            model(step, pos)
    model(ids[:, :16], 0)
    refused = [
        (ids, 0, "max_seq_len"),
        (ids[:, :1], 17, "position 16"),
        (ids[:, :2], 16, "one token"),
    ]
    for step, pos, match in refused:
        with pytest.raises(ValueError, match=match):
            model(step, pos)
    model(ids[:, :1], 16)
    model(ids[:, :1], 17)
    model(ids[:, :max_seq_len], 0)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(ids[:, :1], max_seq_len)


@pytest.mark.sm90
def test_generate_takes_the_official_argmax_until_eos(checkpoint):
    """``generate`` from a three-token prompt: each token's official logit, given the tokens
    before it, is the official maximum to within twice the reference's own bfloat16 rounding of
    the logits; with an EOS id, given or the config's, the run stops after that token."""
    path, tensors = checkpoint
    max_seq_len = 64
    g = torch.Generator().manual_seed(2)
    prompt = torch.randint(0, _SMALL["vocab_size"], (3,), generator=g).tolist()
    mine = DeepseekV4FlashForCausalLM.from_pretrained(
        path, device=run_device(), max_seq_len=max_seq_len
    )
    forward, calls = mine.forward, []
    mine.forward = lambda *args: calls.append(args) or forward(*args)
    out = mine.generate(prompt, 12, eos_token_id=-1)
    assert (len(out), len(calls)) == (12, 12), "a prefill and one decode step per fed-back token"
    mine.forward = forward
    eos = out[5]
    assert mine.generate(prompt, 12, eos_token_id=eos) == out[: out.index(eos) + 1]
    mine.config = dataclasses.replace(mine.config, eos_token_id=eos)
    assert mine.generate(prompt, 12) == out[: out.index(eos) + 1], "config.eos_token_id stops it"
    with official_defaults(run_device()), torch.inference_mode():
        on_device = {k: v.to(run_device()) for k, v in tensors.items()}
        ref, rounded = (
            official_model(on_device, _SMALL, max_seq_len, rounded=r) for r in (False, True)
        )
        ids = torch.tensor([prompt + out[:-1]])
        steps = [(ids[:, : len(prompt)], 0)]
        steps += [(ids[:, pos : pos + 1], pos) for pos in range(len(prompt), ids.shape[1])]
        for token, (step, pos) in zip(out, steps, strict=True):
            check_argmax_within_rounding(
                token, ref(step, pos)[0].float(), rounded(step, pos)[0].float(), f"position {pos}"
            )


@pytest.mark.sm90
def test_generate_stops_when_the_sequence_fills_max_seq_len(checkpoint):
    """From a three-token prompt with ``max_seq_len`` 8 and no EOS, ``generate`` returns five
    tokens to fill its eight positions."""
    path, _ = checkpoint
    model = DeepseekV4FlashForCausalLM.from_pretrained(path, device=run_device(), max_seq_len=8)
    assert len(model.generate([1, 2, 3], 12, eos_token_id=-1)) == 5


def _throughput_script():
    """``scripts/model_throughput.py`` as a module."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "model_throughput.py"
    spec = importlib.util.spec_from_file_location("model_throughput", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_throughput_run_samples_the_first_layer_of_each_kind():
    """The throughput run gives every layer the weights of the first sampled layer of its kind,
    and refuses a sample that leaves a kind out."""
    throughput = _throughput_script()
    kinds = ["window", "window", "ratio-4", "ratio-128", "ratio-4"]
    assert throughput._first_of_each_kind(kinds, 4) == {"window": 0, "ratio-4": 2, "ratio-128": 3}
    with pytest.raises(ValueError, match="ratio-128"):
        throughput._first_of_each_kind(kinds, 3)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_the_throughput_run_refuses_non_finite_logits(bad):
    """A run whose logits overflowed reports an error, not its tokens/s."""
    throughput = _throughput_script()
    throughput._check_finite([torch.zeros(2), torch.ones(2)])
    with pytest.raises(ValueError, match="non-finite"):
        throughput._check_finite([torch.zeros(2), torch.tensor([0.0, bad])])


def test_a_layer_call_counts_the_seconds_one_more_call_adds(monkeypatch):
    """Two calls less one reads one call and leaves out the start from an idle device: a call
    that takes 20 ms, and 10 ms more when a synchronize has left the device idle, reads 20 ms."""
    throughput = _throughput_script()
    idle = [True]
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: idle.__setitem__(0, True))

    def call():
        time.sleep(0.03 if idle[0] else 0.02)
        idle[0] = False

    assert throughput._layer_call_seconds(call) == pytest.approx(0.02, abs=0.004)


@pytest.mark.sm90
def test_a_layer_by_layer_run_gives_the_forward_logits(checkpoint):
    """The nightly's layer-by-layer timing (``scripts/model_throughput.py``) runs what
    ``forward`` runs: with each layer filled, run over the prefills and decode steps, and dropped
    before the next, its logits of the prompt and of every decode step equal ``forward``'s."""
    path, tensors = checkpoint
    throughput = _throughput_script()
    lengths, steps = (130, 260), 6
    max_seq_len = lengths[-1] + steps
    whole = DeepseekV4FlashForCausalLM.from_pretrained(
        path, device=run_device(), max_seq_len=max_seq_len
    )
    alone = DeepseekV4FlashForCausalLM(
        DeepseekV4FlashConfig.from_hf(_SMALL), device=run_device(), max_seq_len=max_seq_len
    )
    device = alone.device
    load_outside(alone, throughput._sample_outside(tensors, device))
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(_SMALL["vocab_size"], (1, lengths[-1]), generator=g).to(device)
    tokens = [
        torch.randint(_SMALL["vocab_size"], (1, 1), generator=g).to(device) for _ in range(steps)
    ]
    with torch.accelerator.device_index(device.index), torch.inference_mode():
        prefill, decode, logits = throughput.layer_by_layer(
            alone, lambda i: throughput._sample_layer(tensors, i, device), prompt, tokens, lengths
        )
    expected = [whole(prompt, 0)]
    expected += [whole(token, lengths[-1] + t) for t, token in enumerate(tokens)]
    for got, want in zip(logits, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert alone.layers == [None] * len(alone.layers)
    assert min(prefill.values()) > 0 and decode > 0
