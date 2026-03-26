.PHONY: install install-bench lint test test-smoke test-full test-nightly bench clean help

install:
	pip install -e '.[dev]' -v
	pre-commit install

install-bench:
	PIP_NO_BUILD_ISOLATION=1 pip install -e '.[dev,bench]' -v
	pip install git+https://github.com/fla-org/native-sparse-attention.git@bd67af59b90afa34b25f61d2922e612d10dba3bd
	pre-commit install

lint:
	pre-commit run --all-files

test:
	python -m pytest -q tests

test-smoke:
	python -m pytest -q tests -m smoke

test-full:
	python -m pytest -q tests -m "smoke or full"

test-nightly:
	python -m pytest -q tests -m "smoke or full or nightly"

bench:
	python -m pytest benchmarks/

clean:
	rm -rf build/ dist/ tileops.egg-info

help:
	@echo "Available targets:"
	@echo "  install       Install dependencies and pre-commit hooks"
	@echo "  install-bench Install with benchmark baseline libraries"
	@echo "  lint       Run linters on all files"
	@echo "  test       Run the test suite"
	@echo "  test-smoke Run smoke-tier tests"
	@echo "  test-full  Run smoke + full-tier tests"
	@echo "  test-nightly Run smoke + full + nightly-tier tests"
	@echo "  bench      Run benchmarks"
	@echo "  clean      Remove build artifacts"
	@echo "  help       Show this help message"
