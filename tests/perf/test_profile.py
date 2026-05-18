import pytest

from tileops.perf.profile import get_profile_path, load_profile


class TestGetProfilePath:
    @pytest.mark.smoke
    def test_h200_exists(self):
        path = get_profile_path("h200")
        assert path.exists()
        assert path.suffix == ".yaml"

    @pytest.mark.smoke
    def test_unknown_gpu_raises(self):
        with pytest.raises(FileNotFoundError):
            get_profile_path("nonexistent_gpu")


class TestLoadProfile:
    @pytest.mark.smoke
    def test_h200_top_level_keys(self):
        profile = load_profile("h200")
        assert profile["gpu"] == "NVIDIA H200"
        assert "hbm" in profile
        assert "tensor_core" in profile
