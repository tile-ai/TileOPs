import pytest

from scripts import nightly_report

pytestmark = pytest.mark.smoke


def test_full_benchmark_results_uses_strongest_baseline_for_color(monkeypatch):
    monkeypatch.setattr(nightly_report, "_get_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(nightly_report, "_get_gpu_name", lambda: "Test GPU")

    report = nightly_report.generate_report(
        test_ops=None,
        bench_ops={
            "RMSNormFwdOp": {
                "module": None,
                "configs": [
                    {
                        "name": "cfg",
                        "tileops_latency_ms": 1.0,
                        "baseline_tag": "flashinfer",
                        "baseline_ratio": 0.5,
                        "baselines": {"torch_inductor": {"ratio": 2.0}},
                    },
                ],
            },
        },
        bench_failures=[],
        regressions=[],
        improvements=[],
        baseline_alerts=[],
    )

    assert f"| {nightly_report._RED} | RMSNormFwdOp | cfg " in report
