"""Tests for benchmarks/hardware/ directory structure and module imports."""

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HARDWARE_DIR = REPO_ROOT / "benchmarks" / "hardware"

pytestmark = pytest.mark.smoke


class TestDirectoryStructure:
    """AC-1: benchmarks/hardware/ directory exists with memory/, compute/, system/."""

    def test_hardware_dir_exists(self):
        assert HARDWARE_DIR.is_dir()

    def test_memory_subdir_exists(self):
        assert (HARDWARE_DIR / "memory").is_dir()

    def test_compute_subdir_exists(self):
        assert (HARDWARE_DIR / "compute").is_dir()

    def test_system_subdir_exists(self):
        assert (HARDWARE_DIR / "system").is_dir()

    def test_utils_subdir_exists(self):
        assert (HARDWARE_DIR / "utils").is_dir()

    def test_results_subdir_exists(self):
        assert (HARDWARE_DIR / "results").is_dir()


class TestMemoryBenchmarks:
    """AC-1: Memory benchmarks are present."""

    def test_hbm_bandwidth_exists(self):
        assert (HARDWARE_DIR / "memory" / "hbm_bandwidth.py").is_file()

    def test_l2_bandwidth_exists(self):
        assert (HARDWARE_DIR / "memory" / "l2_bandwidth.py").is_file()

    def test_shared_bandwidth_exists(self):
        assert (HARDWARE_DIR / "memory" / "shared_bandwidth.py").is_file()

    def test_latency_exists(self):
        assert (HARDWARE_DIR / "memory" / "latency.py").is_file()

    def test_hbm_saturation_cu_exists(self):
        assert (HARDWARE_DIR / "memory" / "hbm_saturation.cu").is_file()

    def test_pointer_chase_cu_exists(self):
        assert (HARDWARE_DIR / "memory" / "pointer_chase.cu").is_file()


class TestComputeBenchmarks:
    """AC-1 + AC-3: Compute benchmarks including BF16 GEMM."""

    def test_gemm_throughput_exists(self):
        assert (HARDWARE_DIR / "compute" / "gemm_throughput.py").is_file()

    def test_gemm_has_bf16(self):
        """AC-3: BF16 GEMM benchmark is included alongside FP16."""
        content = (HARDWARE_DIR / "compute" / "gemm_throughput.py").read_text()
        assert "bf16" in content or "bfloat16" in content


class TestSystemBenchmarks:
    """AC-1: System (CUDA) benchmarks are present."""

    def test_sync_overhead_exists(self):
        assert (HARDWARE_DIR / "system" / "sync_overhead.cu").is_file()

    def test_atomic_overhead_exists(self):
        assert (HARDWARE_DIR / "system" / "atomic_overhead.cu").is_file()

    def test_bank_conflict_exists(self):
        assert (HARDWARE_DIR / "system" / "bank_conflict.cu").is_file()

    def test_bank_conflict_v2_exists(self):
        assert (HARDWARE_DIR / "system" / "bank_conflict_v2.cu").is_file()

    def test_async_copy_exists(self):
        assert (HARDWARE_DIR / "system" / "async_copy.cu").is_file()

    def test_warp_spec_exists(self):
        assert (HARDWARE_DIR / "system" / "warp_spec.cu").is_file()

    def test_occupancy_latency_exists(self):
        assert (HARDWARE_DIR / "system" / "occupancy_latency.cu").is_file()

    def test_register_spill_exists(self):
        assert (HARDWARE_DIR / "system" / "register_spill.cu").is_file()

    def test_stream_sync_exists(self):
        assert (HARDWARE_DIR / "system" / "stream_sync.py").is_file()


class TestUtils:
    """AC-1: Shared utilities are migrated."""

    def test_bench_utils_exists(self):
        assert (HARDWARE_DIR / "utils" / "bench.py").is_file()

    def test_env_utils_exists(self):
        assert (HARDWARE_DIR / "utils" / "env.py").is_file()

    def test_output_utils_exists(self):
        assert (HARDWARE_DIR / "utils" / "output.py").is_file()

    def test_init_exists(self):
        assert (HARDWARE_DIR / "utils" / "__init__.py").is_file()
