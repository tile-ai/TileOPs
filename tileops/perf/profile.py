"""GPU profile loader.

Reads YAML profiles from tileops/perf/profiles/ and returns them as dicts.
This is the M6 -> M5 data contract interface (see docs/architecture.md).
"""

from pathlib import Path

import yaml

_PROFILES_DIR = Path(__file__).parent / "profiles"


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


def _coerce_numeric_strings(obj: object) -> object:
    """Recursively convert numeric strings (e.g. '4800e9') to floats.

    PyYAML does not parse scientific-notation values like ``4800e9`` as
    floats; they arrive as strings.  This helper walks the loaded structure
    and converts any string that parses as a float.
    """
    if isinstance(obj, dict):
        return {k: _coerce_numeric_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numeric_strings(v) for v in obj]
    if isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return obj
    return obj


def load_profile(gpu_name: str) -> dict:
    """Load a GPU profile as a dict.

    Args:
        gpu_name: Profile name without extension (e.g. "h200").

    Returns:
        Dict with keys: gpu, compute_capability, hbm, tensor_core.
    """
    path = get_profile_path(gpu_name)
    with open(path) as f:
        data = yaml.safe_load(f)
    return _coerce_numeric_strings(data)
