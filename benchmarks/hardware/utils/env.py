"""Environment info and GPU profile integration."""

import subprocess

import torch

from tileops.perf import load_profile

# Map torch GPU names to profile file names.
# Add entries as new profiles are created in tileops/perf/profiles/.
_GPU_PROFILE_MAP = {
    "NVIDIA H200": "h200",
    "NVIDIA H100 80GB HBM3": "h100",
}


def _detect_profile_name():
    """Detect GPU and return matching profile name, or None."""
    if not torch.cuda.is_available():
        return None
    gpu_name = torch.cuda.get_device_properties(0).name
    if gpu_name in _GPU_PROFILE_MAP:
        return _GPU_PROFILE_MAP[gpu_name]
    for key, profile in _GPU_PROFILE_MAP.items():
        if key in gpu_name or gpu_name in key:
            return profile
    return None


def get_theoretical_peaks():
    """Get theoretical peak specs for the current GPU from profile YAML.

    Returns:
        Dict with keys like hbm_bw_gbs, fp16_tensor_tflops, etc.
        None if GPU is not recognized.
    """
    profile_name = _detect_profile_name()
    if profile_name is None:
        return None
    profile = load_profile(profile_name)
    hbm_bw_gbs = profile["hbm"]["theoretical"] / 1e9
    tc = profile["tensor_core"]
    return {
        "hbm_bw_gbs": hbm_bw_gbs,
        "fp16_tensor_tflops": tc["fp16"]["theoretical"] / 1e12,
        "bf16_tensor_tflops": tc["bf16"]["theoretical"] / 1e12,
        "fp8_tensor_tflops": tc["fp8"]["theoretical"] / 1e12,
        "tf32_tensor_tflops": tc["tf32"]["theoretical"] / 1e12,
    }


def _get_tilelang_version():
    try:
        import tilelang
        return tilelang.__version__
    except Exception:
        return "N/A"


def get_env_info():
    """Collect full environment info dict."""
    props = torch.cuda.get_device_properties(0)
    try:
        nvsmi = subprocess.run(
            ["nvidia-smi", "--id=0",
             "--query-gpu=driver_version,clocks.max.sm,clocks.max.mem,mig.mode.current",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        parts = [p.strip() for p in nvsmi.stdout.strip().split(",")]
        driver, sm_clock, mem_clock, mig = parts[0], parts[1], parts[2], parts[3]
    except Exception:
        driver = sm_clock = mem_clock = mig = "N/A"

    return {
        "gpu_name": props.name,
        "sm": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "total_memory_gb": f"{props.total_memory / (1024**3):.1f}",
        "l2_cache_mb": f"{props.L2_cache_size / (1024**2):.1f}",
        "mig_mode": mig,
        "driver": driver,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "tilelang": _get_tilelang_version(),
        "clock_sm_mhz": sm_clock,
        "clock_mem_mhz": mem_clock,
    }


def print_env_header():
    """Print environment info as header."""
    info = get_env_info()
    print("=" * 80)
    print(f"GPU: {info['gpu_name']}  |  SM: {info['sm']}  |  SMs: {info['sm_count']}")
    print(f"Memory: {info['total_memory_gb']} GB  |  L2: {info['l2_cache_mb']} MB  |  "
          f"MIG: {info['mig_mode']}")
    print(f"Driver: {info['driver']}  |  CUDA: {info['cuda']}  |  "
          f"Torch: {info['torch']}  |  TileLang: {info['tilelang']}")
    print(f"Clock: SM {info['clock_sm_mhz']}  |  Mem {info['clock_mem_mhz']}")
    print("=" * 80)
    return info
