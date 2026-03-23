"""Benchmark helpers and metric calculators."""

from tilelang.profiler import do_bench

# Backend selection: 'event' works reliably for all ops including in-place
DEFAULT_BACKEND = "event"

# ---- Measured peak reference (set once by bandwidth.py) ----
_MEASURED_PEAK_BW_GBS = None


def bench(fn, warmup=100, rep=200, backend=None):
    """Benchmark a function, return latency in ms."""
    backend = backend or DEFAULT_BACKEND
    return do_bench(fn, warmup=warmup, rep=rep, backend=backend)


def calc_bandwidth_gbs(bytes_total, latency_ms):
    """Calculate bandwidth in GB/s. Returns 0 if latency is 0."""
    if latency_ms <= 0:
        return 0.0
    return bytes_total / latency_ms * 1e-6


def calc_tflops(flops_total, latency_ms):
    """Calculate throughput in TFLOPS."""
    if latency_ms <= 0:
        return 0.0
    return flops_total / latency_ms * 1e-9


def set_measured_peak_bw(bw_gbs):
    global _MEASURED_PEAK_BW_GBS
    _MEASURED_PEAK_BW_GBS = bw_gbs


def get_measured_peak_bw():
    return _MEASURED_PEAK_BW_GBS


def achieved_pct(value, peak):
    """Calculate achieved percentage of peak."""
    if peak is None or peak <= 0:
        return None
    return value / peak * 100.0
