"""DeepseekV4ForCausalLM against DeepSeek's official ``inference/model.py`` run on torch kernels.

A three-layer model (sliding window with hash routing, ratio-4 compressed attention with the
indexer, ratio-128 compressed attention) at DeepSeek-V4-Flash's widths, on random weights in the
checkpoint's formats, is written as a Hugging Face checkpoint and loaded by both. Each layer's
output and the logits are compared over the prefill and over the decode steps by
``check_within_rounding``, the rounded reference being the official model with its attention
probabilities rounded to bfloat16, as its TileLang kernel rounds them. The long prompt is longer than four times the
indexer's top-k, so the indexer selects from more entries than it keeps, and leaves partial
windows that the decode steps close at both compression ratios; the short one closes no
window, so the decode steps close the first.
"""

import contextlib
import functools
import json
import os

import pytest
import torch

from models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from models.deepseek_v4.layers import Step
from workloads.device import run_device
from workloads.models.deepseek_v4 import (
    DEEPSEEK_V4_FLASH_CONFIG,
    INFERENCE_DIR_ENV,
    check_decode_state,
    check_index_selection,
    check_selection_agreement,
    load_official,
    official_args,
    official_state_dict,
    random_checkpoint,
    sparse_attn,
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


def test_config_reads_the_published_flash_config():
    cfg = DeepseekV4Config.from_hf(DEEPSEEK_V4_FLASH_CONFIG)
    assert (cfg.n_layers, cfg.n_hash_layers, cfg.n_routed_experts) == (43, 3, 256)
    assert cfg.compress_ratios[:4] == (0, 0, 4, 128)
    assert (cfg.rope_factor, cfg.original_seq_len, cfg.route_scale) == (16, 65536, 1.5)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        pytest.param({"compress_ratios": [0] * 43}, "compress_ratios", id="ratio-list-length"),
        pytest.param({"model_type": "llama"}, "model_type", id="model-type"),
        pytest.param({"n_shared_experts": 2}, "shared expert", id="shared-experts"),
    ],
)
def test_config_rejects_an_invalid_config(change, match):
    with pytest.raises(ValueError, match=match):
        DeepseekV4Config.from_hf({**DEEPSEEK_V4_FLASH_CONFIG, **change})


def _write_checkpoint(path, config, tensors):
    """Write ``tensors`` and ``config`` as a Hugging Face checkpoint whose tensors alternate
    between two shards, so every layer spans both."""
    from safetensors.torch import save_file

    names = sorted(tensors)
    weight_map = {n: f"model-0000{1 + i % 2}-of-00002.safetensors" for i, n in enumerate(names)}
    for shard in set(weight_map.values()):
        save_file({n: tensors[n] for n in names if weight_map[n] == shard}, str(path / shard))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (path / "config.json").write_text(json.dumps(config))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """The small model's checkpoint directory and its tensors."""
    path = tmp_path_factory.mktemp("dsv4-small")
    tensors = random_checkpoint(_SMALL, seed=0)
    _write_checkpoint(path, _SMALL, tensors)
    return path, tensors


@pytest.mark.parametrize(
    ("change", "extra", "match"),
    [
        pytest.param({}, "head.bias", "unused checkpoint tensors", id="unused-tensor"),
        pytest.param({}, "layers.0.attn.bias", "layer 0: unused tensors", id="unused-layer-tensor"),
        pytest.param({"moe_intermediate_size": 64}, None, "128-row blocks", id="partial-block"),
    ],
)
def test_from_pretrained_refuses_a_checkpoint_it_cannot_read_whole(tmp_path, change, extra, match):
    """A tensor the model has no place for, and a stacked FP8 weight whose first part ends
    inside a scale block, are refused rather than dropped or misaligned."""
    config = {**_SMALL, "num_hidden_layers": 1, "compress_ratios": [0, 0], **change}
    tensors = random_checkpoint(config, seed=0)
    if extra is not None:
        tensors[extra] = torch.zeros(1)
    _write_checkpoint(tmp_path, config, tensors)
    with pytest.raises(ValueError, match=match):
        DeepseekV4ForCausalLM.from_pretrained(tmp_path, devices=[run_device()])


class _Recorded:
    """An indexer that records each call's selection, start position and offset."""

    def __init__(self, indexer, record):
        self._indexer = indexer
        self._record = record

    def __getattr__(self, name):
        return getattr(self._indexer, name)

    def __call__(self, ops, rope, x, qr, step):
        indices = self._indexer(ops, rope, x, qr, step)
        self._record.append((indices, step.start_pos, step.offset))
        return indices


@contextlib.contextmanager
def _official_defaults():
    """The default dtype and device the official model is built and run under, restored after."""
    dtype, device = torch.get_default_dtype(), torch.get_default_device()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(run_device())
    try:
        yield
    finally:
        torch.set_default_dtype(dtype)
        torch.set_default_device(device)


def _official_model(tensors, max_seq_len, *, rounded=False):
    """The official ``Transformer`` on the run device, holding ``tensors``; ``rounded`` rounds
    its attention probabilities to bfloat16, as its TileLang kernel does; built and run under
    ``_official_defaults``."""
    inference_dir = os.environ.get(INFERENCE_DIR_ENV)
    assert inference_dir, f"{INFERENCE_DIR_ENV} names no directory; the dev image sets it"
    module = load_official(inference_dir)
    if rounded:
        module.sparse_attn = functools.partial(sparse_attn, p_dtype=torch.bfloat16)
    model = module.Transformer(official_args(module, _SMALL, max_seq_len, 1))
    missing, unexpected = model.load_state_dict(official_state_dict(tensors), strict=False)
    buffers = ("kv_cache", "freqs_cis", "kv_state", "score_state")
    assert [m for m in missing if not any(b in m for b in buffers)] == []
    assert unexpected == []
    return model


@pytest.mark.sm90
@pytest.mark.parametrize(
    ("prompt_len", "decode_steps"),
    [pytest.param(2170, 8, id="long-prompt"), pytest.param(3, 6, id="short-prompt")],
)
def test_layers_and_logits_match_the_official_model(checkpoint, prompt_len, decode_steps):
    """Each layer, fed the official model's input to it, against that layer of the official model,
    over the prefill and over the decode steps, and its decoding state at the end; the entries
    the indexer selects, against the official indexer's; and the logits of the whole model."""
    path, tensors = checkpoint
    max_seq_len = 2304
    mine, mine_tf = (
        DeepseekV4ForCausalLM.from_pretrained(path, devices=[run_device()], max_seq_len=max_seq_len)
        for _ in range(2)
    )
    with _official_defaults():
        on_device = {k: v.to(run_device()) for k, v in tensors.items()}
        ref = _official_model(on_device, max_seq_len)
        rounded, rounded_tf = (
            _official_model(on_device, max_seq_len, rounded=True) for _ in range(2)
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
        mine_tf.reset()
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
                    step = Step.build(mine_tf.config, pos, tokens.shape[1], mine_tf.first_device)
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
                check_decode_state(layer, official, prompt_len + decode_steps)
    top_k = _SMALL["index_topk"]
    assert len(selections) == len(phases["prefill"]) + len(phases["decode"])
    for indices, start_pos, offset in selections:
        check_index_selection(indices, start_pos, offset, 4, top_k)
    check_selection_agreement(selections, official_selections, 4, top_k)


@pytest.mark.sm90
def test_forward_takes_one_request_position_by_position(checkpoint):
    """``forward`` refuses a batch, an empty input, a decode step out of order or of more than
    one token, and a sequence past ``max_seq_len``; a refused call leaves the request where it
    was."""
    path, _ = checkpoint
    max_seq_len = 18
    model = DeepseekV4ForCausalLM.from_pretrained(
        path, devices=[run_device()], max_seq_len=max_seq_len
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
def test_layers_split_over_two_devices_give_the_one_device_logits(checkpoint, two_devices):
    """The model spread over two devices moves the streams and the token ids between its layer
    runs, and returns the logits it returns on one device, over a prefill that closes windows at
    both compression ratios, the ratio-128 one on the second device, and a decode step; it
    leaves the current device as it found it."""
    if any(torch.cuda.get_device_capability(d)[0] != 9 for d in two_devices):
        pytest.skip("needs two devices of compute capability 9.x")
    path, _ = checkpoint
    one, two = (
        DeepseekV4ForCausalLM.from_pretrained(path, devices=devices, max_seq_len=256)
        for devices in (two_devices[:1], two_devices)
    )
    assert two.layer_devices == [two_devices[0], two_devices[0], two_devices[1]]
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, _SMALL["vocab_size"], (1, 131), generator=g)
    current = torch.cuda.current_device()
    for step, pos in ((ids[:, :130], 0), (ids[:, 130:], 130)):
        expected = one(step, pos)
        torch.testing.assert_close(two(step, pos).to(expected.device), expected, rtol=0, atol=0)
    assert torch.cuda.current_device() == current


@pytest.mark.sm90
def test_generate_takes_the_official_argmax_until_eos_or_max_seq_len(checkpoint):
    """``generate`` from a three-token prompt: each token's official logit, given the tokens
    before it, is the official maximum to within twice the reference's own bfloat16 rounding of
    the logits; the run stops once the sequence fills ``max_seq_len``, and with an EOS id it
    stops after that token."""
    path, tensors = checkpoint
    max_seq_len = 16
    g = torch.Generator().manual_seed(2)
    prompt = torch.randint(0, _SMALL["vocab_size"], (3,), generator=g).tolist()
    mine = DeepseekV4ForCausalLM.from_pretrained(
        path, devices=[run_device()], max_seq_len=max_seq_len
    )
    out = mine.generate(prompt, max_seq_len, eos_token_id=-1)
    assert len(out) == max_seq_len - len(prompt) + 1
    eos = out[2]
    assert mine.generate(prompt, max_seq_len, eos_token_id=eos) == out[: out.index(eos) + 1]
    with _official_defaults(), torch.inference_mode():
        on_device = {k: v.to(run_device()) for k, v in tensors.items()}
        ref, rounded = (_official_model(on_device, max_seq_len, rounded=r) for r in (False, True))
        ids = torch.tensor([prompt + out[:-1]])
        steps = [(ids[:, : len(prompt)], 0)]
        steps += [(ids[:, pos : pos + 1], pos) for pos in range(len(prompt), max_seq_len)]
        for token, (step, pos) in zip(out, steps, strict=True):
            check_argmax_within_rounding(
                token, ref(step, pos)[0].float(), rounded(step, pos)[0].float(), f"position {pos}"
            )
