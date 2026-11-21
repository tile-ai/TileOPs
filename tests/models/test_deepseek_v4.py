"""DeepseekV4ForCausalLM against DeepSeek's official ``inference/model.py`` run on torch kernels.

A three-layer model (sliding window with hash routing, ratio-4 compressed attention with the
indexer, ratio-128 compressed attention) at DeepSeek-V4-Flash's widths, on random weights in the
checkpoint's formats, is written as a Hugging Face checkpoint and loaded by both. Each layer's
output and the logits are compared over the prefill and over the decode steps, within the
reference's own bfloat16 rounding noise. The long prompt is longer than four times the
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

from tileops.models import DeepseekV4Config, DeepseekV4ForCausalLM
from workloads.device import run_device
from workloads.models.deepseek_v4 import (
    DEEPSEEK_V4_FLASH_CONFIG,
    INFERENCE_DIR_ENV,
    check_decode_state,
    check_index_selection,
    check_within_rounding,
    load_official,
    official_args,
    official_state_dict,
    random_checkpoint,
    sparse_attn,
)

_SMALL = {
    **DEEPSEEK_V4_FLASH_CONFIG,
    "num_hidden_layers": 3,
    "num_hash_layers": 1,
    "compress_ratios": [0, 4, 128, 0],
    "n_routed_experts": 8,
    "vocab_size": 1024,
}


@pytest.mark.smoke
def test_config_reads_the_published_flash_config():
    cfg = DeepseekV4Config.from_hf(DEEPSEEK_V4_FLASH_CONFIG)
    assert (cfg.n_layers, cfg.n_hash_layers, cfg.n_routed_experts) == (43, 3, 256)
    assert cfg.compress_ratios[:4] == (0, 0, 4, 128)
    assert (cfg.rope_factor, cfg.original_seq_len, cfg.route_scale) == (16, 65536, 1.5)


@pytest.mark.smoke
def test_config_rejects_a_ratio_list_of_the_wrong_length():
    with pytest.raises(ValueError, match="compress_ratios"):
        DeepseekV4Config.from_hf({**DEEPSEEK_V4_FLASH_CONFIG, "compress_ratios": [0] * 43})


@pytest.mark.smoke
def test_forward_refuses_a_batch_of_two():
    model = DeepseekV4ForCausalLM(DeepseekV4Config.from_hf(_SMALL), devices=[run_device()])
    with pytest.raises(ValueError, match="batch of 2"):
        model(torch.zeros(2, 4, dtype=torch.long))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp("dsv4-small")
    tensors = random_checkpoint(_SMALL, seed=0)
    save_file(tensors, str(path / "model-00001-of-00001.safetensors"))
    weight_map = dict.fromkeys(tensors, "model-00001-of-00001.safetensors")
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (path / "config.json").write_text(json.dumps(_SMALL))
    return path, tensors


class _Recorded:
    """An indexer that records each call's selection, start position and offset."""

    def __init__(self, indexer, record):
        self._indexer = indexer
        self._record = record

    def __getattr__(self, name):
        return getattr(self._indexer, name)

    def __call__(self, ops, rope, x, qr, batch, start_pos, offset):
        indices = self._indexer(ops, rope, x, qr, batch, start_pos, offset)
        self._record.append((indices, start_pos, offset))
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


@pytest.mark.full
@pytest.mark.sm90
@pytest.mark.parametrize(
    ("prompt_len", "decode_steps"),
    [pytest.param(2170, 8, id="long-prompt"), pytest.param(3, 6, id="short-prompt")],
)
def test_layers_and_logits_match_the_official_model(checkpoint, prompt_len, decode_steps):
    """Each layer, fed the official model's input to it, against that layer of the official model,
    over the prefill and over the decode steps, and its decoding state at the end; the entries
    the indexer selects; and the logits of the whole model."""
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
        trace = []
        for i, layer in enumerate(ref.layers):
            layer.register_forward_hook(
                lambda _m, args, out, i=i: trace.append((i, args[0].clone(), out.clone()))
            )
        g = torch.Generator().manual_seed(1)
        shape = (1, prompt_len + decode_steps)
        ids = torch.randint(0, _SMALL["vocab_size"], shape, generator=g, device="cpu")
        ids = ids.to(run_device())
        phases = {"prefill": [0], "decode": list(range(prompt_len, prompt_len + decode_steps))}
        mine_tf.reset(1)
        selections = []
        for layer in mine_tf.layers:
            if layer.attn.indexer is not None:
                layer.attn.indexer = _Recorded(layer.attn.indexer, selections)
        with torch.inference_mode():
            for phase, positions in phases.items():
                logits = ([], [], [])
                layers = {i: ([], [], []) for i in range(len(ref.layers))}
                for pos in positions:
                    step = ids[:, :prompt_len] if pos == 0 else ids[:, pos : pos + 1]
                    trace.clear()
                    for out, model in zip(logits, (mine, ref, rounded), strict=True):
                        out.append(model(step, pos))
                    for i, h_in, h_out in trace:
                        actual, expected, rnd = layers[i]
                        actual.append(mine_tf.layers[i](mine_tf.ops, h_in.contiguous(), step, pos))
                        expected.append(h_out)
                        rnd.append(rounded_tf.layers[i](h_in.clone(), pos, step))
                for i, outs in layers.items():
                    check_within_rounding(*outs, f"{phase} layer {i} outputs")
                check_within_rounding(*logits, f"{phase} logits")
            for layer, official in zip(mine_tf.layers, ref.layers, strict=True):
                check_decode_state(layer, official, prompt_len + decode_steps)
    top_k = _SMALL["index_topk"]
    assert len(selections) == len(phases["prefill"]) + len(phases["decode"])
    for indices, start_pos, offset in selections:
        check_index_selection(indices, start_pos, offset, 4, top_k)


@pytest.mark.full
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
            expected, noisy = ref(step, pos)[0].float(), rounded(step, pos)[0].float()
            slack = 2 * (noisy - expected).abs().max()
            assert expected[token] >= expected.max() - slack, (
                f"position {pos}: token {token} is {expected.max() - expected[token]:.3f} below the "
                f"official maximum, the reference's rounding moves logits by {slack / 2:.3f}"
            )
