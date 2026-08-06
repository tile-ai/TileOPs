"""Tests for benchmark provenance from JUnit properties into history."""

import xml.etree.ElementTree as ET

from benchmarks.conftest import _append_bench_provenance
from scripts import nightly_report


class _FakeItem:
    def __init__(self) -> None:
        self.user_properties: list[tuple[str, str]] = []


def _write_junit(path, properties: list[tuple[str, str]]) -> None:
    suites = ET.Element("testsuites")
    suite = ET.SubElement(suites, "testsuite")
    case = ET.SubElement(
        suite,
        "testcase",
        {"classname": "benchmarks.ops.bench_fake", "name": "test_fake[case]"},
    )
    props = ET.SubElement(case, "properties")
    for name, value in properties:
        ET.SubElement(props, "property", {"name": name, "value": value})
    ET.ElementTree(suites).write(path, encoding="utf-8", xml_declaration=True)


def test_benchmark_provenance_flows_from_junit_into_history(tmp_path, monkeypatch):
    item = _FakeItem()
    item.user_properties.extend(
        [
            ("op", "FakeOp"),
            ("tileops_latency_ms", "0.1000"),
            ("baseline_tag", "torch"),
            ("baseline_latency_ms", "0.2000"),
            ("torch_latency_ms", "0.2000"),
        ]
    )
    provenance = {
        "timing": "native-cupti",
        "cupti_sampled_calls": 50,
        "cupti_expected_kernel_count": 2,
        "cupti_begin_tolerance_us": 2.0,
        "cupti_end_tolerance_us": 8.0,
        "cupti_repeat_guard_us": 16.0,
        "input_policy": "shifting-pool",
        "input_policy_seed": 0,
    }
    _append_bench_provenance(item, "tileops", provenance)
    _append_bench_provenance(item, "baseline", provenance)
    _append_bench_provenance(item, "torch", provenance)

    junit = tmp_path / "bench.xml"
    _write_junit(junit, item.user_properties)
    parsed = nightly_report.parse_bench_xml(str(junit))
    aggregated = nightly_report.aggregate_bench_results(parsed)

    config = aggregated["FakeOp"]["configs"][0]
    assert config["tileops_provenance"] == provenance
    assert config["baseline_provenance"] == provenance
    assert config["baselines"]["torch"]["provenance"] == provenance

    rows = nightly_report.summarize_bench_provenance(aggregated)
    assert [(row["role"], row["count"]) for row in rows] == [
        ("tileops", 1),
        ("torch", 1),
    ]

    monkeypatch.setattr(nightly_report, "_get_git_commit", lambda: "abc1234")
    monkeypatch.setattr(nightly_report, "_get_gpu_name", lambda: "Test GPU")
    history = nightly_report.build_history_entry(aggregated)
    history_config = history["ops"]["FakeOp"]["test_fake[case]"]
    assert history_config["tileops"]["benchmark"] == provenance
    assert history_config["torch"]["benchmark"] == provenance
