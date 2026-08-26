"""Performance evaluation — roofline analysis and GPU hardware profiles."""

from .profile import (
    find_profile,
    get_profile_path,
    load_profile,
    resolve_roof,
    tensor_core_roof,
)

__all__ = [
    "find_profile",
    "get_profile_path",
    "load_profile",
    "resolve_roof",
    "tensor_core_roof",
]
