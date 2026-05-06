import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, _bench_results

# Skip NSA benchmarks until op failures are resolved (see #696).
collect_ignore_glob = [
    "ops/attention/bench_deepseek_nsa*.py",
]

TILELANG_019_BENCH_SKIP_REASON = (
    "Temporarily skipped while tracking TileLang 0.1.9 benchmark migration "
    "failures (#1067)."
)

TILELANG_019_BENCH_PATHS = {
    # Linear attention / recurrent ops.
    "benchmarks/ops/bench_deltanet.py",
    "benchmarks/ops/bench_deltanet_recurrence.py",
    "benchmarks/ops/bench_engram.py",
    "benchmarks/ops/bench_gated_deltanet.py",
    "benchmarks/ops/bench_gated_deltanet_recurrence.py",
    "benchmarks/ops/bench_gla_chunkwise.py",
    "benchmarks/ops/bench_gla_recurrence.py",
    # Attention.
    "benchmarks/ops/attention/bench_deepseek_dsa_decode.py",
    "benchmarks/ops/attention/bench_deepseek_mla_decode.py",
    "benchmarks/ops/attention/bench_gqa_decode.py",
    "benchmarks/ops/attention/bench_gqa_decode_paged.py",
    "benchmarks/ops/attention/bench_gqa_sliding_window.py",
    "benchmarks/ops/attention/bench_gqa_sliding_window_varlen.py",
    "benchmarks/ops/attention/bench_mha.py",
    "benchmarks/ops/attention/bench_mha_decode.py",
    "benchmarks/ops/attention/bench_mha_decode_paged.py",
    # Other operators.
    "benchmarks/ops/bench_fft.py",
    "benchmarks/ops/bench_mhc_post.py",
    "benchmarks/ops/bench_mhc_pre.py",
    "benchmarks/ops/bench_rope.py",
    "benchmarks/ops/bench_topk_selector.py",
    # Normalization.
    "benchmarks/ops/bench_fused_add_layer_norm.py",
    "benchmarks/ops/bench_instance_norm.py",
    # MoE.
    "benchmarks/ops/bench_moe_permute.py",
    "benchmarks/ops/bench_moe_permute_align.py",
    # GEMM.
    "benchmarks/ops/bench_grouped_gemm.py",
}

TILELANG_019_BENCH_PREFIXES = (
    "benchmarks/ops/attention/bench_gqa.py::test_gqa_bwd_bench",
    "benchmarks/ops/bench_convolution.py::test_conv2d_bench",
    "benchmarks/ops/bench_convolution.py::test_conv3d_bench",
)

TILELANG_019_BENCH_NODEIDS = {
    "benchmarks/ops/bench_gemm.py::test_gemm_bench[wide-fp16]",
    "benchmarks/ops/bench_group_norm.py::test_group_norm_bench[tail-spatial-g16-float16]",
}


def _normalized_benchmark_nodeid(item: pytest.Item) -> str:
    nodeid = item.nodeid
    if nodeid.startswith("benchmarks/"):
        return nodeid
    if nodeid.startswith("ops/"):
        return f"benchmarks/{nodeid}"
    return nodeid


def _is_fp8_e4m3_benchmark(item: pytest.Item) -> bool:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return False
    return callspec.params.get("dtype") == torch.float8_e4m3fn


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(1235)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1235)


def pytest_sessionstart(session):
    BenchmarkReport.clear()


def pytest_sessionfinish(session, exitstatus):
    BenchmarkReport.dump("profile_run.log")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    tilelang_019_skip = pytest.mark.skip(reason=TILELANG_019_BENCH_SKIP_REASON)
    fp8_e4m3_skip = pytest.mark.skip(
        reason=(
            "Temporarily skipped while tracking TileLang 0.1.9 fp8 e4m3 "
            "benchmark migration failures (#1067)."
        )
    )

    for item in items:
        nodeid = _normalized_benchmark_nodeid(item)
        path = nodeid.split("::", 1)[0]

        if path in TILELANG_019_BENCH_PATHS:
            item.add_marker(tilelang_019_skip)
            continue

        if nodeid.startswith(TILELANG_019_BENCH_PREFIXES):
            item.add_marker(tilelang_019_skip)
            continue

        if nodeid in TILELANG_019_BENCH_NODEIDS:
            item.add_marker(tilelang_019_skip)
            continue

        if (
            path == "benchmarks/ops/bench_elementwise_fp8.py"
            and _is_fp8_e4m3_benchmark(item)
        ):
            item.add_marker(fp8_e4m3_skip)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """After bench test execution, attach perf data to the item as properties."""
    _bench_results.entries = []
    yield
    entries = getattr(_bench_results, "entries", [])
    if not entries:
        return

    # Separate tileops entry (tag starts with "tileops") from baselines.
    tileops_entry = None
    baseline_entries = []
    for e in entries:
        if e["tag"].startswith("tileops"):
            if tileops_entry is None:
                tileops_entry = e
        else:
            baseline_entries.append(e)

    if tileops_entry:
        item.user_properties.append(("op", tileops_entry["op"]))
        if "op_module" in tileops_entry:
            item.user_properties.append(("op_module", tileops_entry["op_module"]))
        tag = tileops_entry["tag"]
        if tag != "tileops" and tag.startswith("tileops_"):
            item.user_properties.append(("tileops_variant", tag[len("tileops_"):]))
        item.user_properties.append(("tileops_latency_ms",
                                     f"{tileops_entry.get('latency_ms', 0):.4f}"))
        tflops = tileops_entry.get("tflops")
        if tflops is not None:
            item.user_properties.append(("tileops_tflops", f"{tflops:.2f}"))
        bw = tileops_entry.get("bandwidth_tbs")
        if bw is not None:
            item.user_properties.append(("tileops_bandwidth_tbs", f"{bw:.2f}"))

    # Write all baselines into JUnit XML properties.
    # The first baseline uses the legacy unprefixed names (baseline_tag, etc.)
    # for backward compatibility.  Additional baselines use "{tag}_latency_ms",
    # "{tag}_tflops", "{tag}_ratio" so the report can display multiple columns.
    for idx, be in enumerate(baseline_entries):
        tag = be["tag"]
        bl_latency = be.get("latency_ms", 0)
        bl_tflops = be.get("tflops")

        if idx == 0:
            # Legacy unprefixed keys — consumed by existing nightly_report.py
            item.user_properties.append(("baseline_tag", tag))
            item.user_properties.append(("baseline_latency_ms", f"{bl_latency:.4f}"))
            if bl_tflops is not None:
                item.user_properties.append(("baseline_tflops", f"{bl_tflops:.2f}"))
            if tileops_entry:
                tl = tileops_entry.get("latency_ms", 0)
                if tl > 0 and bl_latency > 0:
                    item.user_properties.append(("baseline_ratio",
                                                 f"{bl_latency / tl:.4f}"))

        # Tag-prefixed keys — always written for every baseline
        item.user_properties.append((f"{tag}_latency_ms", f"{bl_latency:.4f}"))
        if bl_tflops is not None:
            item.user_properties.append((f"{tag}_tflops", f"{bl_tflops:.2f}"))
        if tileops_entry:
            tl = tileops_entry.get("latency_ms", 0)
            if tl > 0 and bl_latency > 0:
                item.user_properties.append((f"{tag}_ratio", f"{bl_latency / tl:.4f}"))

    _bench_results.entries = []
