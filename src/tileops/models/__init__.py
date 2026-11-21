"""Language models composed end to end from TileOPs ops.

Each model is a subpackage named for its Hugging Face ``model_type`` and loads that
model's published checkpoint:

    from tileops.models import DeepseekV4ForCausalLM

    model = DeepseekV4ForCausalLM.from_pretrained(path, devices=[0, 1])
"""

from tileops.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM

__all__ = ["DeepseekV4Config", "DeepseekV4ForCausalLM"]
