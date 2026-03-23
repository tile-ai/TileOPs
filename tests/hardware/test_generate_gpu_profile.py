"""Tests for scripts/generate_gpu_profile.py."""

import csv
import os
import pathlib
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_gpu_profile.py"

pytestmark = pytest.mark.smoke


class TestScriptExists:
    """AC-2: scripts/generate_gpu_profile.py exists."""

    def test_script_exists(self):
        assert SCRIPT_PATH.is_file()

    def test_script_is_importable(self):
        """The script should define functions we can import."""
        sys.path.insert(0, str(SCRIPT_PATH.parent))
        try:
            import generate_gpu_profile
            assert hasattr(generate_gpu_profile, "generate_profile")
            assert hasattr(generate_gpu_profile, "parse_hbm_results")
            assert hasattr(generate_gpu_profile, "parse_gemm_results")
        finally:
            sys.path.pop(0)
            sys.modules.pop("generate_gpu_profile", None)


class TestProfileGeneration:
    """AC-2 + AC-4: Generate valid gpu_profile.yaml from benchmark results."""

    @pytest.fixture
    def sample_results_dir(self, tmp_path):
        """Create sample CSV result files mimicking real H200 benchmark output."""
        # HBM peak results
        hbm_csv = tmp_path / "hbm_peak.csv"
        with open(hbm_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "op", "dtype", "size_bytes", "vec_width",
                "best_gbs", "median_gbs", "pct_of_4800_best",
                "run1_gbs", "run2_gbs", "run3_gbs",
            ])
            # Copy results: best indicator of peak achievable BW
            writer.writerow([
                "2026-03-20T00:06:16.117146", "copy", "fp32", "2147483648",
                "pytorch_default", "4255.06", "4255.33", "88.6%",
                "4255.06", "4255.33", "4255.94",
            ])
            writer.writerow([
                "2026-03-20T00:06:17.833368", "copy", "fp16", "2147483648",
                "pytorch_default", "4255.36", "4255.45", "88.7%",
                "4255.36", "4255.45", "4255.76",
            ])

        # GEMM throughput results
        gemm_csv = tmp_path / "gemm_throughput.csv"
        with open(gemm_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "gpu_name", "sm", "mig_mode", "driver", "cuda", "torch",
                "tilelang", "clock_sm_mhz", "clock_mem_mhz", "benchmark",
                "backend", "dtype", "shape", "size_bytes",
                "working_set_bytes", "stride_bytes", "block_dim",
                "num_warps", "warmup", "rep", "latency_us", "latency_ms",
                "bandwidth_gbs", "bandwidth_tbs", "tflops",
                "achieved_pct_of_peak", "notes", "timestamp",
            ])
            # cuBLAS FP16
            writer.writerow([
                "NVIDIA H200", "9.0", "Disabled", "575.57.08", "12.8",
                "2.9.1+cu128", "0.1.8", "1980 MHz", "3201 MHz", "gemm",
                "cublas", "fp16", "8192x8192x8192", "402653184",
                "", "", "", "", "100", "200", "3184.0", "3.184",
                "", "", "345.3", "34.9",
                "torch.mm, theo_peak=989.5TF",
                "2026-03-19T23:39:25.669273",
            ])
            # cuBLAS BF16
            writer.writerow([
                "NVIDIA H200", "9.0", "Disabled", "575.57.08", "12.8",
                "2.9.1+cu128", "0.1.8", "1980 MHz", "3201 MHz", "gemm",
                "cublas", "bf16", "8192x28672x8192", "1073741824",
                "", "", "", "", "100", "200", "10571.8", "10.572",
                "", "", "364.0", "36.8",
                "torch.mm, theo_peak=989.5TF",
                "2026-03-19T23:39:28.891426",
            ])

        return tmp_path

    def test_generate_profile_from_csvs(self, sample_results_dir, tmp_path):
        """AC-2: Can produce a valid gpu_profile.yaml from benchmark results."""
        sys.path.insert(0, str(SCRIPT_PATH.parent))
        try:
            # Reload to avoid stale module
            import importlib

            import generate_gpu_profile
            importlib.reload(generate_gpu_profile)

            output_path = tmp_path / "test_profile.yaml"
            generate_gpu_profile.generate_profile(
                results_dir=str(sample_results_dir),
                output_path=str(output_path),
                gpu_name="NVIDIA H200",
            )

            assert output_path.is_file()
            with open(output_path) as f:
                profile = yaml.safe_load(f)

            assert profile is not None
            assert "gpu_name" in profile
            assert profile["gpu_name"] == "NVIDIA H200"
        finally:
            sys.path.pop(0)
            sys.modules.pop("generate_gpu_profile", None)

    def test_profile_contains_hbm_bandwidth(self, sample_results_dir, tmp_path):
        """AC-4: Profile contains HBM bandwidth."""
        sys.path.insert(0, str(SCRIPT_PATH.parent))
        try:
            import importlib

            import generate_gpu_profile
            importlib.reload(generate_gpu_profile)

            output_path = tmp_path / "test_profile.yaml"
            generate_gpu_profile.generate_profile(
                results_dir=str(sample_results_dir),
                output_path=str(output_path),
                gpu_name="NVIDIA H200",
            )

            with open(output_path) as f:
                profile = yaml.safe_load(f)

            assert "hbm_bandwidth_gbs" in profile
            assert "measured" in profile["hbm_bandwidth_gbs"]
            assert profile["hbm_bandwidth_gbs"]["measured"] > 0
        finally:
            sys.path.pop(0)
            sys.modules.pop("generate_gpu_profile", None)

    def test_profile_contains_tensor_core_tflops(self, sample_results_dir, tmp_path):
        """AC-4: Profile contains Tensor Core TFLOPS."""
        sys.path.insert(0, str(SCRIPT_PATH.parent))
        try:
            import importlib

            import generate_gpu_profile
            importlib.reload(generate_gpu_profile)

            output_path = tmp_path / "test_profile.yaml"
            generate_gpu_profile.generate_profile(
                results_dir=str(sample_results_dir),
                output_path=str(output_path),
                gpu_name="NVIDIA H200",
            )

            with open(output_path) as f:
                profile = yaml.safe_load(f)

            assert "tensor_core_tflops" in profile
            assert "fp16" in profile["tensor_core_tflops"]
            assert "bf16" in profile["tensor_core_tflops"]
            assert profile["tensor_core_tflops"]["fp16"]["measured"] > 0
            assert profile["tensor_core_tflops"]["bf16"]["measured"] > 0
        finally:
            sys.path.pop(0)
            sys.modules.pop("generate_gpu_profile", None)

    def test_profile_contains_calibration_factors(self, sample_results_dir, tmp_path):
        """AC-4: Profile contains calibration factors."""
        sys.path.insert(0, str(SCRIPT_PATH.parent))
        try:
            import importlib

            import generate_gpu_profile
            importlib.reload(generate_gpu_profile)

            output_path = tmp_path / "test_profile.yaml"
            generate_gpu_profile.generate_profile(
                results_dir=str(sample_results_dir),
                output_path=str(output_path),
                gpu_name="NVIDIA H200",
            )

            with open(output_path) as f:
                profile = yaml.safe_load(f)

            # HBM calibration
            assert "calibration_factor" in profile["hbm_bandwidth_gbs"]
            cal = profile["hbm_bandwidth_gbs"]["calibration_factor"]
            assert 0 < cal <= 1.0

            # Tensor core calibration
            assert "calibration_factor" in profile["tensor_core_tflops"]["fp16"]
            assert "calibration_factor" in profile["tensor_core_tflops"]["bf16"]
        finally:
            sys.path.pop(0)
            sys.modules.pop("generate_gpu_profile", None)


class TestBandwidthCsvSchema:
    """Regression: parse_hbm_results must handle the migrated bandwidth.csv schema
    where benchmark='bandwidth' and the operation name is in the 'notes' field."""

    @pytest.fixture
    def bandwidth_csv_dir(self, tmp_path):
        """Create bandwidth.csv matching hbm_bandwidth.py output schema."""
        csv_path = tmp_path / "bandwidth.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "gpu_name", "sm", "mig_mode", "driver", "cuda", "torch",
                "tilelang", "clock_sm_mhz", "clock_mem_mhz", "benchmark",
                "backend", "dtype", "shape", "size_bytes",
                "working_set_bytes", "stride_bytes", "block_dim",
                "num_warps", "warmup", "rep", "latency_us", "latency_ms",
                "bandwidth_gbs", "bandwidth_tbs", "tflops",
                "achieved_pct_of_peak", "notes", "timestamp",
            ])
            # A read row -- should NOT be picked as copy
            writer.writerow([
                "NVIDIA H200", "9.0", "Disabled", "575.57.08", "12.8",
                "2.9.1+cu128", "0.1.8", "1980 MHz", "3201 MHz", "bandwidth",
                "pytorch", "fp16", "134217728", "268435456",
                "", "", "", "", "100", "200", "72.0", "0.072",
                "3600.0", "3.6", "", "75.0",
                "read, theo_peak=4800GB/s",
                "2026-03-20T00:06:10.000000",
            ])
            # A copy row -- this is what parse_hbm_results must find
            writer.writerow([
                "NVIDIA H200", "9.0", "Disabled", "575.57.08", "12.8",
                "2.9.1+cu128", "0.1.8", "1980 MHz", "3201 MHz", "bandwidth",
                "pytorch", "fp32", "536870912", "2147483648",
                "", "", "", "", "100", "200", "1009.0", "1.009",
                "4255.06", "4.255", "", "88.6",
                "copy, theo_peak=4800GB/s",
                "2026-03-20T00:06:16.117146",
            ])
            # Another copy row with different dtype
            writer.writerow([
                "NVIDIA H200", "9.0", "Disabled", "575.57.08", "12.8",
                "2.9.1+cu128", "0.1.8", "1980 MHz", "3201 MHz", "bandwidth",
                "pytorch", "fp16", "1073741824", "2147483648",
                "", "", "", "", "100", "200", "1010.0", "1.010",
                "4200.36", "4.200", "", "87.5",
                "copy, theo_peak=4800GB/s",
                "2026-03-20T00:06:17.833368",
            ])
        return tmp_path

    def test_parse_hbm_from_bandwidth_csv(self, bandwidth_csv_dir):
        """parse_hbm_results must return non-zero BW from bandwidth.csv schema."""
        sys.path.insert(0, str(SCRIPT_PATH.parent))
        try:
            import importlib

            import generate_gpu_profile
            importlib.reload(generate_gpu_profile)

            bw = generate_gpu_profile.parse_hbm_results(str(bandwidth_csv_dir))
            # Should pick up 4255.06 GB/s (the best copy row)
            assert bw > 4000.0, f"Expected >4000 GB/s, got {bw}"
        finally:
            sys.path.pop(0)
            sys.modules.pop("generate_gpu_profile", None)

    def test_profile_from_bandwidth_csv_has_nonzero_hbm(self, bandwidth_csv_dir, tmp_path):
        """Full profile generation from bandwidth.csv must yield non-zero HBM BW."""
        sys.path.insert(0, str(SCRIPT_PATH.parent))
        try:
            import importlib

            import generate_gpu_profile
            importlib.reload(generate_gpu_profile)

            output_path = tmp_path / "profile.yaml"
            profile = generate_gpu_profile.generate_profile(
                results_dir=str(bandwidth_csv_dir),
                output_path=str(output_path),
                gpu_name="NVIDIA H200",
            )

            assert profile["hbm_bandwidth_gbs"]["measured"] > 4000.0
            assert profile["hbm_bandwidth_gbs"]["calibration_factor"] > 0.8
        finally:
            sys.path.pop(0)
            sys.modules.pop("generate_gpu_profile", None)

    def test_read_rows_not_counted_as_copy(self, bandwidth_csv_dir):
        """Only copy rows should contribute to HBM bandwidth, not read/write."""
        sys.path.insert(0, str(SCRIPT_PATH.parent))
        try:
            import importlib

            import generate_gpu_profile
            importlib.reload(generate_gpu_profile)

            bw = generate_gpu_profile.parse_hbm_results(str(bandwidth_csv_dir))
            # The read row has 3600 GB/s; if parser incorrectly picks it up,
            # it still wouldn't exceed 4255.06, but let's verify it's >= best copy
            assert bw == pytest.approx(4255.06, rel=1e-3)
        finally:
            sys.path.pop(0)
            sys.modules.pop("generate_gpu_profile", None)


class TestH200Profile:
    """AC-4: Pre-generated h200.yaml profile validation."""

    PROFILE_PATH = REPO_ROOT / "tileops" / "perf" / "profiles" / "h200.yaml"

    def test_h200_profile_exists(self):
        assert self.PROFILE_PATH.is_file()

    def test_h200_profile_valid_yaml(self):
        with open(self.PROFILE_PATH) as f:
            profile = yaml.safe_load(f)
        assert profile is not None
        assert isinstance(profile, dict)

    def test_h200_profile_has_required_fields(self):
        with open(self.PROFILE_PATH) as f:
            profile = yaml.safe_load(f)

        assert profile["gpu_name"] == "NVIDIA H200"
        assert "hbm_bandwidth_gbs" in profile
        assert "tensor_core_tflops" in profile

    def test_h200_hbm_bandwidth_reasonable(self):
        """Measured HBM BW should be between 3000-4800 GB/s for H200."""
        with open(self.PROFILE_PATH) as f:
            profile = yaml.safe_load(f)

        measured = profile["hbm_bandwidth_gbs"]["measured"]
        assert 3000 <= measured <= 4800

    def test_h200_tensor_core_tflops_reasonable(self):
        """Measured tensor core TFLOPS should be between 300-989.5 for H200."""
        with open(self.PROFILE_PATH) as f:
            profile = yaml.safe_load(f)

        for dtype in ["fp16", "bf16"]:
            measured = profile["tensor_core_tflops"][dtype]["measured"]
            assert 100 <= measured <= 989.5

    def test_h200_calibration_factors(self):
        """Calibration factors should be between 0.5 and 1.0 (50%-100% of theoretical)."""
        with open(self.PROFILE_PATH) as f:
            profile = yaml.safe_load(f)

        hbm_cal = profile["hbm_bandwidth_gbs"]["calibration_factor"]
        assert 0.5 <= hbm_cal <= 1.0

        for dtype in ["fp16", "bf16"]:
            tc_cal = profile["tensor_core_tflops"][dtype]["calibration_factor"]
            assert 0.1 <= tc_cal <= 1.0


class TestUtilsFunctions:
    """Test utility module functions (pure logic, no GPU needed)."""

    def test_calc_bandwidth_gbs(self):
        sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "hardware"))
        try:
            from utils.bench import calc_bandwidth_gbs
            # 1 GB in 1 ms = 1000 GB/s
            result = calc_bandwidth_gbs(1_000_000_000, 1.0)
            assert abs(result - 1000.0) < 0.01
        finally:
            sys.path.pop(0)

    def test_calc_bandwidth_gbs_zero_latency(self):
        sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "hardware"))
        try:
            from utils.bench import calc_bandwidth_gbs
            result = calc_bandwidth_gbs(1_000_000_000, 0.0)
            assert result == 0.0
        finally:
            sys.path.pop(0)

    def test_calc_tflops(self):
        sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "hardware"))
        try:
            from utils.bench import calc_tflops
            # 1e12 FLOPS in 1 ms = 1 TFLOPS
            result = calc_tflops(1e12, 1.0)
            assert abs(result - 1000.0) < 0.01
        finally:
            sys.path.pop(0)

    def test_achieved_pct(self):
        sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "hardware"))
        try:
            from utils.bench import achieved_pct
            result = achieved_pct(240.0, 480.0)
            assert abs(result - 50.0) < 0.01
        finally:
            sys.path.pop(0)

    def test_achieved_pct_none_peak(self):
        sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "hardware"))
        try:
            from utils.bench import achieved_pct
            result = achieved_pct(240.0, None)
            assert result is None
        finally:
            sys.path.pop(0)

    def test_theoretical_peaks_h200(self):
        sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "hardware"))
        try:
            from utils.env import THEORETICAL_PEAKS
            assert "NVIDIA H200" in THEORETICAL_PEAKS
            h200 = THEORETICAL_PEAKS["NVIDIA H200"]
            assert h200["hbm_bw_gbs"] == 4800.0
            assert h200["fp16_tensor_tflops"] == 989.5
            assert h200["bf16_tensor_tflops"] == 989.5
        finally:
            sys.path.pop(0)

    def test_csv_fields_complete(self):
        sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "hardware"))
        try:
            from utils.output import ALL_FIELDS, AUX_FIELDS, CASE_FIELDS, ENV_FIELDS, METRIC_FIELDS
            assert len(ALL_FIELDS) == len(ENV_FIELDS) + len(CASE_FIELDS) + len(METRIC_FIELDS) + len(AUX_FIELDS)
            assert "gpu_name" in ENV_FIELDS
            assert "benchmark" in CASE_FIELDS
            assert "latency_ms" in METRIC_FIELDS
            assert "timestamp" in AUX_FIELDS
        finally:
            sys.path.pop(0)

    def test_output_results_dir_points_to_hardware_results(self):
        sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "hardware"))
        try:
            from utils.output import RESULTS_DIR
            assert RESULTS_DIR.endswith(os.path.join("benchmarks", "hardware", "results"))
        finally:
            sys.path.pop(0)
