"""The text decoder of Hugging Face ``qwen3_5`` checkpoints (Qwen3.8-27B), composed from TileOPs ops."""

from models.qwen3_5.config import Qwen3_5Config
from models.qwen3_5.model import Qwen3_5ForCausalLM

__all__ = ["Qwen3_5Config", "Qwen3_5ForCausalLM"]
