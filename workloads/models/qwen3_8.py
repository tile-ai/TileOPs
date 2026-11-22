"""Hugging Face transformers' ``qwen3_5`` text model, the reference for ``models/qwen3_8``.

The model runs on its pure-torch paths only: ``torch_paths`` swaps the GDN, causal-conv1d and
attention functions of ``modeling_qwen3_5``, which import fla and causal_conv1d kernels when
those packages are installed, for the torch implementations they decorate, and counts the
calls each takes. Needs ``transformers`` and ``safetensors``.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import torch

__all__ = [
    "QWEN3_8_27B_CONFIG",
    "ReferenceRequest",
    "check_within_rounding",
    "random_checkpoint",
    "reference_model",
    "torch_paths",
]

_LAYER_TYPES = (["linear_attention"] * 3 + ["full_attention"]) * 16

# Qwen3.8-27B's published ``config.json``, its vision part left out.
QWEN3_8_27B_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "tie_word_embeddings": False,
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "bos_token_id": 248044,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 17408,
        "layer_types": _LAYER_TYPES,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 262144,
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "num_key_value_heads": 4,
        "output_gate_type": "swish",
        "pad_token_id": None,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 248320,
    },
}

_PREFIX = "model.language_model."
# The vision tower and the MTP layer, which the text model does not read.
_SKIPPED = ("model.visual.", "mtp.")
_TORCH_PATHS = (
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "eager_attention_forward",
)


def random_checkpoint(config: dict, seed: int) -> dict[str, torch.Tensor]:
    """Random bfloat16 tensors under the names and shapes of a ``qwen3_5`` checkpoint.

    Projections are scaled by ``fan_in ** -0.5`` so activations stay of unit size;
    ``1 + w`` norm weights are ``N(0, 0.1)``, the gated norm's ``w`` is ``1 + N(0, 0.1)``,
    ``A_log`` spans decay rates from about 0.02 to 7. One ``model.visual.`` and one ``mtp.``
    tensor stand in for the parts the text decoder skips.
    """
    c = config["text_config"]
    g = torch.Generator().manual_seed(seed)
    hidden, heads, kv_heads, head_dim = (
        c["hidden_size"],
        c["num_attention_heads"],
        c["num_key_value_heads"],
        c["head_dim"],
    )
    hk, hv = c["linear_num_key_heads"], c["linear_num_value_heads"]
    key_dim, value_dim = hk * c["linear_key_head_dim"], hv * c["linear_value_head_dim"]
    conv_dim, width = 2 * key_dim + value_dim, c["linear_conv_kernel_dim"]

    def randn(*shape: int, std: float = 1.0, mean: float = 0.0) -> torch.Tensor:
        return (torch.randn(shape, generator=g) * std + mean).to(torch.bfloat16)

    def linear(out_features: int, in_features: int) -> torch.Tensor:
        return randn(out_features, in_features, std=in_features**-0.5)

    t = {
        f"{_PREFIX}embed_tokens.weight": randn(c["vocab_size"], hidden),
        f"{_PREFIX}norm.weight": randn(hidden, std=0.1),
        "lm_head.weight": linear(c["vocab_size"], hidden),
        "model.visual.patch_embed.proj.weight": randn(1),
        "mtp.fc.weight": randn(1),
    }
    for i, kind in enumerate(c["layer_types"]):
        p = f"{_PREFIX}layers.{i}."
        t[p + "input_layernorm.weight"] = randn(hidden, std=0.1)
        t[p + "post_attention_layernorm.weight"] = randn(hidden, std=0.1)
        t[p + "mlp.gate_proj.weight"] = linear(c["intermediate_size"], hidden)
        t[p + "mlp.up_proj.weight"] = linear(c["intermediate_size"], hidden)
        t[p + "mlp.down_proj.weight"] = linear(hidden, c["intermediate_size"])
        if kind == "linear_attention":
            a = p + "linear_attn."
            t[a + "in_proj_qkv.weight"] = linear(conv_dim, hidden)
            t[a + "in_proj_z.weight"] = linear(value_dim, hidden)
            t[a + "in_proj_b.weight"] = linear(hv, hidden)
            t[a + "in_proj_a.weight"] = linear(hv, hidden)
            t[a + "conv1d.weight"] = randn(conv_dim, 1, width, std=width**-0.5)
            t[a + "A_log"] = (torch.rand(hv, generator=g) * 6 - 4).to(torch.bfloat16)
            t[a + "dt_bias"] = randn(hv, std=0.5)
            t[a + "norm.weight"] = randn(c["linear_value_head_dim"], std=0.1, mean=1.0)
            t[a + "out_proj.weight"] = linear(hidden, value_dim)
        else:
            a = p + "self_attn."
            t[a + "q_proj.weight"] = linear(2 * heads * head_dim, hidden)
            t[a + "k_proj.weight"] = linear(kv_heads * head_dim, hidden)
            t[a + "v_proj.weight"] = linear(kv_heads * head_dim, hidden)
            t[a + "q_norm.weight"] = randn(head_dim, std=0.1)
            t[a + "k_norm.weight"] = randn(head_dim, std=0.1)
            t[a + "o_proj.weight"] = linear(hidden, heads * head_dim)
    return t


@contextlib.contextmanager
def torch_paths() -> Iterator[Counter]:
    """Run ``modeling_qwen3_5``'s GDN, conv1d and attention functions as pure torch within the
    block; yield a counter of the calls each takes, by function name."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    calls: Counter = Counter()
    saved = {name: getattr(m, name) for name in _TORCH_PATHS}

    def recorded(name: str) -> object:
        fn = inspect.unwrap(saved[name])
        assert fn.__module__ == m.__name__, f"{name} resolves to {fn.__module__}.{fn.__name__}"

        @functools.wraps(fn)
        def run(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)

        return run

    try:
        for name in _TORCH_PATHS:
            setattr(m, name, recorded(name))
        yield calls
    finally:
        for name, fn in saved.items():
            setattr(m, name, fn)


def reference_model(path: str | Path, dtype: torch.dtype, device: torch.device | str):
    """HF's ``Qwen3_5ForCausalLM`` with eager attention, holding the text decoder of the
    ``qwen3_5`` checkpoint at ``path`` in ``dtype`` on ``device``."""
    from safetensors import safe_open
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    path = Path(path)
    config = Qwen3_5TextConfig(**json.loads((path / "config.json").read_text())["text_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(path / shard), framework="pt", device=str(device)) as f:
            for name in f.keys():
                if not name.startswith(_SKIPPED):
                    state[name.replace(_PREFIX, "model.")] = f.get_tensor(name).to(dtype)
    model.load_state_dict(state, strict=True, assign=True)
    # The rotary frequencies are a buffer no checkpoint holds; build them on the device.
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config).to(device)
    return model.eval()


class ReferenceRequest:
    """One request through a ``reference_model``, call by call, as ``Qwen3_8ForCausalLM``
    takes it; records each decoder layer's output of the last call in ``layers``."""

    def __init__(self, model) -> None:
        self.model = model
        self.cache = None
        self.layers: list[torch.Tensor] = []
        for layer in model.model.layers:
            layer.register_forward_hook(lambda _m, _args, out: self.layers.append(out))

    @torch.inference_mode()
    def __call__(self, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Run ``input_ids`` ``[1, S]`` from ``start_pos``; return the float32 logits of its
        last token, ``[1, vocab_size]``."""
        if start_pos == 0:
            self.cache = None
        device = self.model.lm_head.weight.device
        positions = torch.arange(start_pos, start_pos + input_ids.shape[1], device=device)
        self.layers = []
        out = self.model(
            input_ids=input_ids.to(device),
            position_ids=positions.view(1, -1),
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        self.cache = out.past_key_values
        return out.logits[:, -1].float()


def check_within_rounding(
    actual: list[torch.Tensor], expected: list[torch.Tensor], rounded: list[torch.Tensor], what: str
) -> None:
    """Assert ``actual`` is at most twice as far from ``expected`` as ``rounded`` is.

    Each list holds one tensor per step; distances are L2 over all of them. ``expected`` comes
    from the reference in float32 and ``rounded`` from the same reference in bfloat16: their
    distance is the reference's own bfloat16 rounding noise.
    """

    def distance(xs: list[torch.Tensor]) -> float:
        return (
            sum(
                (x.float().cpu() - e.float().cpu()).square().sum().item()
                for x, e in zip(xs, expected, strict=True)
            )
            ** 0.5
        )

    error, noise = distance(actual), distance(rounded)
    assert error <= 2 * noise, (
        f"{what} differ from the float32 reference by {error:.4g} in L2, its bfloat16 run by "
        f"{noise:.4g}"
    )
