"""GPU profile loader.

Reads YAML profiles from tileops/perf/profiles/ and returns them as dicts.
This is the M6 -> M5 data contract interface (see docs/architecture.md).

YAML files store only ``theoretical`` and ``calibration`` values.
``effective = theoretical * calibration`` is computed at load time.
"""

from pathlib import Path

import yaml

_PROFILES_DIR = Path(__file__).parent / "profiles"

# Keys whose values are numeric but arrive as strings from PyYAML
# (scientific notation like 4800e9 is not YAML-native float syntax).
_NUMERIC_KEYS = frozenset({"theoretical", "calibration", "effective"})


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
        raise FileNotFoundError(
            f"No GPU profile '{gpu_name}'. Available: {available}"
        )
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
    """Compute effective = theoretical * calibration for hbm and tensor_core."""
    hbm = profile.get("hbm")
    if hbm and "effective" not in hbm:
        hbm["effective"] = hbm["theoretical"] * hbm["calibration"]

    for section in profile.get("tensor_core", {}).values():
        if isinstance(section, dict) and "effective" not in section:
            section["effective"] = section["theoretical"] * section["calibration"]


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
