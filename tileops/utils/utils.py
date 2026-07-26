import functools

import torch

str2dtype = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
    "int32": torch.int32
}


def is_hopper():
    return torch.cuda.get_device_capability() == (9, 0)


@functools.lru_cache(maxsize=1)
def is_h200():
    if not torch.cuda.is_available():
        return False
    return "H200" in torch.cuda.get_device_name().upper()


def get_sm_version():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor
