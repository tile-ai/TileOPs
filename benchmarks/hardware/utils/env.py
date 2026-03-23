"""Environment info and theoretical peak specs."""

import subprocess

import torch

# ---- Theoretical peak specs (per GPU model) ----
# Source: NVIDIA datasheets
# Note: dense (no sparsity) peaks for GEMM benchmarking.
# Sparse (2:4) peaks are 2x these values.
THEORETICAL_PEAKS = {
    "NVIDIA H200": {
        "hbm_bw_gbs": 4800.0,           # 4.8 TB/s HBM3e
        "fp16_tensor_tflops": 989.5,     # dense; sparse = 1979
        "bf16_tensor_tflops": 989.5,     # dense; sparse = 1979
        "fp8_tensor_tflops": 1979.0,     # dense; sparse = 3958
        "tf32_tensor_tflops": 494.7,     # dense; sparse = 989.5
        "fp32_tflops": 67.0,
    },
    "NVIDIA H100 80GB HBM3": {
        "hbm_bw_gbs": 3350.0,
        "fp16_tensor_tflops": 989.5,
        "bf16_tensor_tflops": 989.5,
        "fp8_tensor_tflops": 1979.0,
        "tf32_tensor_tflops": 494.7,
        "fp32_tflops": 67.0,
    },
}


def get_theoretical_peaks():
    """Get theoretical peak specs for the current GPU."""
    name = torch.cuda.get_device_properties(0).name
    if name in THEORETICAL_PEAKS:
        return THEORETICAL_PEAKS[name]
    for key in THEORETICAL_PEAKS:
        if key in name or name in key:
            return THEORETICAL_PEAKS[key]
    return None


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
