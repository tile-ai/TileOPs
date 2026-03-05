.PHONY: install lint test bench clean help

install:
	pip install -e '.[dev]' -v
	pre-commit install

lint:
	pre-commit run --all-files

test:
	python -m pytest -q tests

bench:
	python -m pytest benchmarks/

clean:
	rm -rf build/ dist/ tileops.egg-info

help:
	@echo "Available targets:"
	@echo "  install    Install dependencies and pre-commit hooks"
	@echo "  lint       Run linters on all files"
	@echo "  test       Run the test suite"
	@echo "  bench      Run benchmarks"
	@echo "  clean      Remove build artifacts"
	@echo "  help       Show this help message"
