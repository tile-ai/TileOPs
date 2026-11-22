"""Qwen3_8ForCausalLM against Hugging Face transformers' ``qwen3_5`` text model run on torch only.

A four-layer model (three Gated DeltaNet layers, then a full-attention layer) at Qwen3.8-27B's
widths with a small vocabulary, on random weights in the checkpoint's format, is written as a
sharded Hugging Face checkpoint and loaded by both. HF runs in float32, the truth, and in
bfloat16. Over every call of a request (a prefill, the chunks of a chunked prefill, decode
steps) each layer's output and the last token's logits are compared, and at the end the
decoding state. The bound is ``check_within_rounding``: in L2 over all calls, this model is at
most twice as far from HF float32 as HF bfloat16 is.
"""

import dataclasses
import json

import pytest
import torch

from models.qwen3_8 import Qwen3_8Config, Qwen3_8ForCausalLM
from models.qwen3_8.layers import GatedDeltaNet
from workloads.device import run_device
from workloads.models.qwen3_8 import (
    QWEN3_8_27B_CONFIG,
    ReferenceRequest,
    check_argmax_within_rounding,
    check_within_rounding,
    random_checkpoint,
    reference_model,
    torch_paths,
)

pytestmark = pytest.mark.nightly

_TEXT = QWEN3_8_27B_CONFIG["text_config"]


def _config(**text_changes) -> dict:
    return {**QWEN3_8_27B_CONFIG, "text_config": {**_TEXT, **text_changes}}


_SMALL = _config(num_hidden_layers=4, layer_types=_TEXT["layer_types"][:4], vocab_size=1024)
# Two layers at narrow widths, for the loader's refusals; no kernel runs on them.
_TINY = _config(
    num_hidden_layers=2,
    layer_types=_TEXT["layer_types"][2:4],
    vocab_size=64,
    hidden_size=64,
    intermediate_size=64,
    num_attention_heads=2,
    num_key_value_heads=1,
    linear_num_key_heads=1,
    linear_num_value_heads=2,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
)
_MAX_SEQ_LEN = 2304


def test_config_reads_the_published_config():
    cfg = Qwen3_8Config.from_hf(QWEN3_8_27B_CONFIG)
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size) == (64, 5120, 248320)
    assert cfg.layer_types.count("full_attention") == 16
    assert cfg.layer_types[3] == "full_attention"
    assert (cfg.rotary_dim, cfg.mrope_section, cfg.rope_theta) == (64, (11, 11, 10), 1e7)
    assert Qwen3_8Config.from_hf(_TEXT) == cfg


@pytest.mark.parametrize(
    ("config", "match"),
    [
        pytest.param({**QWEN3_8_27B_CONFIG, "model_type": "qwen3"}, "model_type", id="model-type"),
        pytest.param(_config(model_type="qwen3_next"), "model_type", id="text-model-type"),
        pytest.param(_config(attention_bias=True), "attention_bias", id="attention-bias"),
        pytest.param(_config(attn_output_gate=False), "attn_output_gate", id="ungated"),
        pytest.param(_config(tie_word_embeddings=True), "tie_word_embeddings", id="tied"),
        pytest.param(_config(dtype="float16"), "dtype", id="dtype"),
        pytest.param(
            _config(rope_parameters={**_TEXT["rope_parameters"], "mrope_interleaved": False}),
            "mrope_interleaved",
            id="consecutive-mrope",
        ),
        pytest.param(
            _config(rope_parameters={**_TEXT["rope_parameters"], "rope_type": "yarn"}),
            "rope_type",
            id="rope-type",
        ),
        pytest.param(_config(num_hidden_layers=63), "layer_types", id="layer-count"),
        pytest.param(
            _config(layer_types=["sliding_attention"] * 64), "layer_types", id="layer-type"
        ),
        pytest.param(_config(linear_value_head_dim=64), "square", id="non-square-state"),
        pytest.param(_config(hidden_act="gelu"), "hidden_act", id="activation"),
        pytest.param(_config(output_gate_type="sigmoid"), "output_gate_type", id="gdn-gate"),
        pytest.param(_config(mamba_ssm_dtype="bfloat16"), "mamba_ssm_dtype", id="state-dtype"),
        pytest.param(_config(num_key_value_heads=5), "num_key_value_heads", id="gqa-heads"),
        pytest.param(_config(linear_num_key_heads=20), "linear_num_key_heads", id="gdn-heads"),
        pytest.param(
            _config(rope_parameters={**_TEXT["rope_parameters"], "mrope_section": [11, 11, 11]}),
            "does not cover",
            id="mrope-section",
        ),
    ],
)
def test_config_refuses_what_the_model_does_not_compute(config, match):
    with pytest.raises(ValueError, match=match):
        Qwen3_8Config.from_hf(config)


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
    """The small model's checkpoint directory."""
    path = tmp_path_factory.mktemp("qwen3_8-small")
    _write_checkpoint(path, _SMALL, random_checkpoint(_SMALL, seed=0))
    return path


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        pytest.param("model.language_model.rotary_emb.inv_freq", "unused checkpoint", id="model"),
        pytest.param("model.language_model.layers.2.mlp.bias", "unused checkpoint", id="layer-2"),
        pytest.param("model.language_model.layers.1.mlp.bias", "layer 1: unused", id="in-layer"),
    ],
)
def test_from_pretrained_refuses_a_tensor_it_does_not_read(tmp_path, extra, match):
    """A tensor the text decoder has no place for is refused, not dropped; the vision and MTP
    tensors every checkpoint carries are skipped."""
    tensors = random_checkpoint(_TINY, seed=0)
    tensors[extra] = torch.zeros(1, dtype=torch.bfloat16)
    _write_checkpoint(tmp_path, _TINY, tensors)
    with pytest.raises(ValueError, match=match):
        Qwen3_8ForCausalLM.from_pretrained(tmp_path, devices=[run_device()], max_seq_len=64)


def test_from_pretrained_refuses_a_checkpoint_missing_a_layer(tmp_path):
    tensors = random_checkpoint(_TINY, seed=0)
    layer = "model.language_model.layers.1."
    _write_checkpoint(
        tmp_path, _TINY, {n: t for n, t in tensors.items() if not n.startswith(layer)}
    )
    with pytest.raises(ValueError, match="holds no tensors of layers \\[1\\]"):
        Qwen3_8ForCausalLM.from_pretrained(tmp_path, devices=[run_device()], max_seq_len=64)


def test_from_pretrained_skips_the_vision_and_mtp_tensors(tmp_path):
    tensors = random_checkpoint(_TINY, seed=0)
    assert any(n.startswith("model.visual.") for n in tensors)
    assert any(n.startswith("mtp.") for n in tensors)
    _write_checkpoint(tmp_path, _TINY, tensors)
    model = Qwen3_8ForCausalLM.from_pretrained(tmp_path, devices=[run_device()], max_seq_len=64)
    assert model.lm_head.shape == (64, 64)


class _Recorded:
    """A decoder layer that records each call's hidden state, ``out + residual``."""

    def __init__(self, layer, record):
        self._layer = layer
        self._record = record

    def __getattr__(self, name):
        return getattr(self._layer, name)

    def __call__(self, ops, x, residual, step):
        out, residual = self._layer(ops, x, residual, step)
        self._record.append(out.float() + residual.float())
        return out, residual


# Per request: the lengths of the calls that prefill it, then the decode steps. GDNFwdOp splits
# each call into 64-token chunks from its first token: 130 is 64 + 64 + 2, 67 is 64 + 3, and 64 is
# one whole chunk. The 64-row KV-cache pages count from position 0, so the 67- and 64-token calls
# start mid-page, at 130 and 197. The 2100-token prefill runs 32 whole chunks and a 52-token tail,
# so the recurrent state's rounding builds up over a long walk.
_REQUESTS = {
    "short": ([5], 6),
    "chunked-prefill": ([130, 67, 64], 4),
    "long": ([2100], 3),
}


@pytest.mark.sm90
@pytest.mark.parametrize(("chunks", "decode_steps"), list(_REQUESTS.values()), ids=list(_REQUESTS))
def test_layers_logits_and_state_match_hf(checkpoint, chunks, decode_steps):
    """Every layer's output and the logits over every call, then the decoding state."""
    mine = Qwen3_8ForCausalLM.from_pretrained(
        checkpoint, devices=[run_device()], max_seq_len=_MAX_SEQ_LEN
    )
    records = [[] for _ in mine.layers]
    mine.layers = [_Recorded(layer, r) for layer, r in zip(mine.layers, records, strict=True)]
    lengths = chunks + [1] * decode_steps
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, _SMALL["text_config"]["vocab_size"], (1, sum(lengths)), generator=g)
    ids = ids.to(run_device())
    with torch_paths() as ran:
        exact, rounded = (
            ReferenceRequest(reference_model(checkpoint, dtype, run_device()))
            for dtype in (torch.float32, torch.bfloat16)
        )
        logits = ([], [], [])
        layers = [([], [], []) for _ in mine.layers]
        start = 0
        for n in lengths:
            step = ids[:, start : start + n]
            for out, model in zip(logits, (mine, exact, rounded), strict=True):
                out.append(model(step, start))
            for i, (actual, expected, rnd) in enumerate(layers):
                actual.append(records[i].pop())
                expected.append(exact.layers[i][0])
                rnd.append(rounded.layers[i][0])
            start += n
    assert ran["torch_chunk_gated_delta_rule"] == 2 * 3 * len(chunks)
    assert ran["torch_recurrent_gated_delta_rule"] == 2 * 3 * decode_steps
    assert ran["causal_conv1d_fn"] == 2 * 3 * len(chunks)
    assert ran["causal_conv1d_update"] == 2 * 3 * decode_steps
    assert ran["eager_attention_forward"] == 2 * len(lengths)
    for i, outs in enumerate(layers):
        check_within_rounding(*outs, f"layer {i} outputs")
    check_within_rounding(*logits, "logits")
    for i, layer in enumerate(mine.layers):
        mixer = layer.mixer
        states = [ref.cache.layers[i] for ref in (exact, rounded)]
        if isinstance(mixer, GatedDeltaNet):
            width = mixer.conv_state.shape[-1]
            check_within_rounding(
                [mixer.state], *([s.recurrent_states[0]] for s in states), f"layer {i} GDN state"
            )
            check_within_rounding(
                [mixer.conv_state],
                *([s.conv_states[0][..., -width:]] for s in states),
                f"layer {i} conv state",
            )
        else:
            for name, cache in (("keys", mixer.k_cache), ("values", mixer.v_cache)):
                ours = cache.flatten(0, 1)[:start].permute(1, 0, 2)
                check_within_rounding(
                    [ours], *([getattr(s, name)[0]] for s in states), f"layer {i} cached {name}"
                )


@pytest.mark.sm90
def test_forward_takes_one_request_in_order(checkpoint):
    """``forward`` refuses a batch, an empty call, a call at ``start_pos > 0`` before a
    prefill or out of order, and a sequence past ``max_seq_len``; a refused call leaves the
    request where it was."""
    max_seq_len = 18
    model = Qwen3_8ForCausalLM.from_pretrained(
        checkpoint, devices=[run_device()], max_seq_len=max_seq_len
    )
    ids = torch.zeros(1, max_seq_len + 1, dtype=torch.long, device=run_device())
    refused = [
        (ids[:, :4].expand(2, -1), 0, "batch of 2"),
        (ids[:, :0], 0, "at least one token"),
        (ids[:, :1], 4, "prefill at start_pos 0"),
    ]
    for step, pos, match in refused:
        with pytest.raises(ValueError, match=match):
            model(step, pos)
    model(ids[:, :10], 0)
    refused = [
        (ids, 0, "max_seq_len"),
        (ids[:, :1], 11, "continues at position 10"),
        (ids[:, :9], 10, "max_seq_len"),
    ]
    for step, pos, match in refused:
        with pytest.raises(ValueError, match=match):
            model(step, pos)
    model(ids[:, :3], 10)
    model(ids[:, :1], 13)


@pytest.mark.sm90
def test_generate_takes_the_hf_argmax_until_eos(checkpoint):
    """``generate`` from a three-token prompt: each token's HF float32 logit, given the tokens
    before it, is the HF float32 maximum to within twice HF bfloat16's largest logit error;
    with an EOS id the run stops after that token."""
    g = torch.Generator().manual_seed(2)
    prompt = torch.randint(0, _SMALL["text_config"]["vocab_size"], (3,), generator=g).tolist()
    mine = Qwen3_8ForCausalLM.from_pretrained(checkpoint, devices=[run_device()], max_seq_len=64)
    forward, calls = mine.forward, []
    mine.forward = lambda *args: calls.append(args) or forward(*args)
    out = mine.generate(prompt, 12, eos_token_id=-1)
    assert (len(out), len(calls)) == (12, 12), "a prefill and one decode step per fed-back token"
    mine.forward = forward
    eos = out[5]
    assert mine.generate(prompt, 12, eos_token_id=[-1, eos]) == out[: out.index(eos) + 1]
    mine.config = dataclasses.replace(mine.config, eos_token_id=eos)
    assert mine.generate(prompt, 12) == out[: out.index(eos) + 1], "config.eos_token_id stops it"
    with torch_paths():
        exact, rounded = (
            ReferenceRequest(reference_model(checkpoint, dtype, run_device()))
            for dtype in (torch.float32, torch.bfloat16)
        )
        ids = torch.tensor([prompt + out[:-1]], device=run_device())
        steps = [(ids[:, : len(prompt)], 0)]
        steps += [(ids[:, pos : pos + 1], pos) for pos in range(len(prompt), ids.shape[1])]
        for token, (step, pos) in zip(out, steps, strict=True):
            check_argmax_within_rounding(
                token, exact(step, pos)[0], rounded(step, pos)[0], f"position {pos}"
            )


@pytest.mark.sm90
def test_generate_stops_when_the_sequence_fills_max_seq_len(checkpoint):
    """From a three-token prompt with ``max_seq_len`` 8 and no EOS, ``generate`` returns six
    tokens: the last one completes the sequence without a forward call past position 7."""
    model = Qwen3_8ForCausalLM.from_pretrained(checkpoint, devices=[run_device()], max_seq_len=8)
    assert len(model.generate([1, 2, 3], 12, eos_token_id=-1)) == 6


def test_layers_split_over_two_devices_give_the_one_device_logits(checkpoint, two_devices):
    """The model spread over two devices moves the hidden state, the residual and each device's
    position metadata between its layer runs, and returns the logits it returns on one device,
    over a prefill and a decode step."""
    if any(torch.cuda.get_device_capability(d)[0] != 9 for d in two_devices):
        pytest.skip("needs two devices of compute capability 9.x")
    one = Qwen3_8ForCausalLM.from_pretrained(checkpoint, devices=two_devices[:1], max_seq_len=64)
    two = Qwen3_8ForCausalLM.from_pretrained(checkpoint, devices=two_devices, max_seq_len=64)
    assert set(two.layer_devices) == set(two_devices)
    ids = torch.arange(1, 8).view(1, -1)
    for step, pos in ((ids[:, :6], 0), (ids[:, 6:], 6)):
        expected = one(step, pos)
        torch.testing.assert_close(two(step, pos).to(expected.device), expected, rtol=0, atol=0)
