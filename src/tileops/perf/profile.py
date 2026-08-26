"""GPU profile loader.

Reads YAML profiles from src/tileops/perf/profiles/ and returns them as dicts.
This is the M6 -> M5 data contract interface (see docs/design/architecture.md).

YAML files store only ``theoretical`` and ``calibration`` values.
``effective = theoretical * calibration`` is computed at load time.
"""

from pathlib import Path

import yaml

_PROFILES_DIR = Path(__file__).parent / "profiles"

# Keys whose values are numeric but arrive as strings from PyYAML
# (scientific notation like 4800e9 is not YAML-native float syntax).
_NUMERIC_KEYS = frozenset({"theoretical", "calibration", "calibration_burst", "effective"})


def get_profile_path(gpu_name: str) -> Path:
    """Return the path to a GPU profile YAML.

    Args:
        gpu_name: Profile name without extension (e.g. "h200").

    Returns:
        Path to the YAML file.

    Raises:
        FileNotFoundError: If no profile exists for the given name.
    """
    path = _PROFILES_DIR / f"{gpu_name}.yaml"
    if not path.exists():
        available = [p.stem for p in _PROFILES_DIR.glob("*.yaml")]
        raise FileNotFoundError(f"No GPU profile '{gpu_name}'. Available: {available}")
    return path


def _coerce_numeric_strings(obj, key=None):
    """Recursively convert known numeric string values to floats.

    Only converts values whose dict key is in ``_NUMERIC_KEYS``, avoiding
    unintended coercion of string fields like ``compute_capability``.
    """
    if isinstance(obj, dict):
        return {k: _coerce_numeric_strings(v, key=k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numeric_strings(v) for v in obj]
    if isinstance(obj, str) and key in _NUMERIC_KEYS:
        try:
            return float(obj)
        except ValueError:
            return obj
    return obj


def _inject_effective(profile):
    """Compute effective = theoretical * calibration for hbm and the compute sections.

    A section with only ``theoretical`` is left alone: profiles are created from
    datasheet numbers first and calibrated by benchmarks/hardware/ afterwards.
    """
    sections = [profile.get("hbm")]
    for group in ("tensor_core", "cuda_core"):
        sections.extend(profile.get(group, {}).values())
    for section in sections:
        if isinstance(section, dict) and "effective" not in section and "calibration" in section:
            section["effective"] = section["theoretical"] * section["calibration"]


# Tensor-core dtype keys, by the dtype the contraction consumes. fp32 maps to
# tf32 because that is the unit an fp32 contraction runs on when tensor cores
# serve it. Encode side of the roof-key format; ``resolve_roof`` is the decode.
_TENSOR_CORE_DTYPE_KEYS = {
    "float16": "fp16",
    "bfloat16": "bf16",
    "float32": "tf32",
    "float8_e4m3fn": "fp8",
    "float8_e5m2": "fp8",
}


def tensor_core_roof(dtype) -> str:
    """Tensor-core roof key for a contraction computing at *dtype*.

    Args:
        dtype: The dtype the matmul consumes — a ``torch.dtype`` or its
            string name. Ops pass ``self.dtype`` directly.

    Returns:
        A GPU-profile key such as ``"tensor_core.bf16"``.

    Raises:
        ValueError: If *dtype* has no tensor-core section in the profile
            schema (including ``None`` — the op has not bound a dtype yet).
    """
    name = str(dtype).removeprefix("torch.") if dtype is not None else None
    key = _TENSOR_CORE_DTYPE_KEYS.get(name) if name is not None else None
    if key is None:
        raise ValueError(
            f"no tensor-core roof for dtype {dtype!r}; known dtypes: "
            f"{sorted(_TENSOR_CORE_DTYPE_KEYS)}"
        )
    return f"tensor_core.{key}"


def find_profile(device_name: str) -> dict | None:
    """Load the profile whose ``gpu`` field names *device_name*, or ``None``.

    Args:
        device_name: The device name as CUDA reports it
            (``torch.cuda.get_device_name()``), e.g. ``"NVIDIA H200"``.

    Returns:
        The loaded profile dict, or ``None`` when no profile claims the
        device — the caller leaves speed-of-light readings blank rather
        than guessing a ceiling.
    """
    for path in _PROFILES_DIR.glob("*.yaml"):
        profile = load_profile(path.stem)
        if profile.get("gpu") == device_name:
            return profile
    return None


def resolve_roof(profile: dict, key: str) -> dict | None:
    """Resolve a roof key like ``"tensor_core.bf16"`` to its profile section.

    Args:
        profile: A dict from :func:`load_profile`.
        key: ``"<unit>.<dtype>"`` as declared by ``Op.compute_roof()``.

    Returns:
        The section dict (with ``theoretical`` / ``effective``), or ``None``
        when the profile has no calibrated entry for the key.
    """
    unit, _, dt = key.partition(".")
    section = (profile.get(unit) or {}).get(dt)
    if isinstance(section, dict) and "effective" in section and "theoretical" in section:
        return section
    return None


def load_profile(gpu_name: str) -> dict:
    """Load a GPU profile as a dict.

    Args:
        gpu_name: Profile name without extension (e.g. "h200").

    Returns:
        Dict with keys: gpu, compute_capability, hbm, tensor_core.
        Each hbm/tensor_core section includes a computed ``effective`` field.
    """
    path = get_profile_path(gpu_name)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data = _coerce_numeric_strings(data)
    _inject_effective(data)
    return data
