.PHONY: install lint test bench clean

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
	rm -rf build/ dist/ *.egg-info
