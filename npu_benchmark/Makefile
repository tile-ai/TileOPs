.PHONY: test bench bench-smoke help

test:
	python -m pytest -q tests

bench:
	python -m pytest benchmarks/

bench-smoke:
	python -m pytest benchmarks/ -m smoke

help:
	@echo "Available targets:"
	@echo "  test        Run correctness tests"
	@echo "  bench       Run all benchmarks"
	@echo "  bench-smoke Run smoke-tier benchmarks only"
	@echo ""
	@echo "Environment variables:"
	@echo "  NPU_BENCHMARK_FORCE_CUDA=1  Use CUDA backend (for local dev)"
	@echo "  NPU_BENCH_FLUSH_CACHE=0     Disable L2 cache flush"
