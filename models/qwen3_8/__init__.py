"""The Qwen3.8-27B text decoder composed from TileOPs ops.

Qwen3.8-27B ships as Hugging Face ``model_type`` ``qwen3_5``. Qwen3.5-27B and Qwen3.6-27B have the
same architecture and tensor names, so their checkpoints load as well.
"""

from models.qwen3_8.config import Qwen3_8Config
from models.qwen3_8.model import Qwen3_8ForCausalLM

__all__ = ["Qwen3_8Config", "Qwen3_8ForCausalLM"]
