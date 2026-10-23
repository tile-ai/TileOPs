# Development

How to install, test, lint, and benchmark TileOPs from a source checkout.

Commands here are the ones CI runs. Where CI passes `-c constraints.txt`, use it locally too — that file pins the versions CI validates, and omitting it resolves a different dependency set than the one your PR is tested against.

## Install from source

A CUDA-capable GPU is required to run the test suite. See [Prerequisites](../README.md#installation) for supported Python, PyTorch, CUDA, and TileLang versions.

```bash
git clone https://github.com/tile-ai/TileOPs
cd TileOPs
pip install -e '.[dev]' -c constraints.txt
pre-commit install
```

If CUDA and TileLang are already installed system-wide and the build fails while re-resolving them, skip build isolation:

```bash
PIP_NO_BUILD_ISOLATION=1 pip install -e '.[dev]' -c constraints.txt
```

`[dev]` adds ruff, codespell, pytest, pytest-xdist, and pyyaml. `[bench]` adds the baseline libraries the benchmarks compare against — see [Benchmarks](#benchmarks).

## Dev Docker image

The prebuilt dev image ships the whole stack — CUDA 13.2, PyTorch 2.13 (cu132), the TileLang release TileOPs declares, and the benchmark baselines — so nothing needs resolving locally:

```bash
docker run --rm -it --gpus all \
  -v "$(pwd)":/workspace -w /workspace \
  ghcr.io/tile-ai/tileops-dev:cuda13.2-torch2.13-tilelang0.1.12

# inside the container
pip install -e . --no-deps
python -m pytest -q tests -m smoke
```

`--no-deps` is deliberate: the image already carries the pinned stack, and letting pip resolve dependencies would replace it.

Tags follow `cuda<major.minor>-torch<major.minor>-tilelang<version>`, where `<version>` is the release [Prerequisites](../README.md#installation) names; pull that one rather than a floating tag. `tileops-dev` is the stage the CI image `tileops-ci-runner` builds on: the same software stack, without the runner agent and its user, so a green run inside it is the same environment CI reports on.

To build the image, roll out a new one, or bump TileLang, see [`.github/runner/README.md`](../.github/runner/README.md).

## Tests

Tests are tiered by marker. Pick the tier by how much you need to cover, not by how long you can wait:

| Command                                                   | Covers                                  |
| --------------------------------------------------------- | --------------------------------------- |
| `python -m pytest -q tests -m smoke`                      | Fast critical path. What every PR runs. |
| `python -m pytest -q tests -m "smoke or full"`            | Standard correctness coverage.          |
| `python -m pytest -q tests -m "smoke or full or nightly"` | Exhaustive and long-running cases.      |
| `python -m pytest -q tests`                               | Everything, including unmarked tests.   |

Narrow to one file or case the usual way — `python -m pytest -q tests/ops/test_gemm.py -k tuned`.

Two suites do not need a GPU and are worth running before pushing:

```bash
python -m pytest -q tests/test_validate_manifest.py   # manifest spec validator
python -m pytest -q benchmarks/tests                  # benchmark base-class contract
```

CI runs on SM90, so it skips every test marked `sm80`, `sm86` or `sm89`: those guard paths a device of that architecture selects. On such a device, run its group across tiers, for example on SM89:

```bash
python -m pytest -q tests -m sm89
```

The `packaging` marker is separate from the tiers: it is a minimal wheel-install sanity check, one case per op family, run against an installed wheel rather than a source checkout.

### CUDA Graph validation

The release validation target is SM90/H200 with the [verified stack](../README.md#installation).
Other architectures are outside this release gate.

[`tests/cuda_graph_contract.py`](../tests/cuda_graph_contract.py) lists the ops with qualified
calls. Each `cuda_graph(op=...)` test defines its supported shapes, dtypes and options; the
qualification does not extend to untested specializations or other ops. Print the mapping
without a GPU, then run all its evidence on H200:

```bash
python scripts/ci/check_cuda_graph_coverage.py
python -m pytest -q tests/ops -m cuda_graph
```

- The collection gate rejects missing evidence, unknown ops and collection errors.
- Tests carry `sm90`, so a skip on H200 fails the run; their smoke/full tiers run in PR and nightly CI.
- Warm up with the intended options and any requested tuning before capture.
- Replay changes tensor contents at captured addresses. Changes to shapes, strides, dtypes, host metadata or dispatch options require a new graph.
- Compare replay outputs and state with the workload reference.

## Lint

`pre-commit install` (above) runs the hooks on every commit. To check the whole tree as CI does:

```bash
pre-commit run --all-files
```

## Docstrings

Docstrings are the API reference on
[the docs site](https://tile-ai.github.io/TileOPs.github.io/), and mkdocstrings
renders them **as Markdown** — reStructuredText reaches the page as literal text.

Three docstrings per op, each answering one question:

| Docstring  | Answers             | Sections                                               |
| ---------- | ------------------- | ------------------------------------------------------ |
| the class  | what the op is      | prose: the formula, the shapes, when a kernel rebuilds |
| `__init__` | how to construct it | `Args:`                                                |
| `forward`  | how to call it      | `Args:`, `Returns:`, `Raises:`, `Example:` last        |

Both members need one: the page gives every member an entry, so a missing
docstring publishes a heading with nothing under it. Parameters go on `__init__`,
not in the class's `Args:`, and the example goes last in `forward` — a class
docstring renders above both signatures.

How to write each element:

| Element                | Write                                                              | Not                                                              |
| ---------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------- |
| Code example           | a fenced block inside `Example:`: ```` ```python linenums="1" ```` | `>>>` prompts — `>` is a Markdown blockquote                     |
| Tensor shape           | `$[B \\times M \\times K]$`                                        | `` `[B, M, K]` ``                                                |
| Formula                | `$d_i = a_i \\mathbin{@} b_i$`, or `$$…$$` on its own line         | `.. math::`                                                      |
| Four or more variants  | a table                                                            | an indented bullet list, which Markdown folds into one paragraph |
| Callout                | `!!! note "Title"`, body indented four spaces                      | `.. note::`                                                      |
| Cross-reference        | `` `torch.nn.functional.rms_norm` ``                               | `:func:` and the other roles                                     |
| Identifier, path, flag | inline code                                                        | math                                                             |

Two that bite:

- **Double every backslash.** A docstring is a regular string literal, so
  `\\times` is a tab followed by `imes`. Write `\\\\times`, or make the docstring raw.
- **A shape is a product of dimensions.** `` `(flops, bytes)` `` is a return pair
  and stays code.

[`bmm.py`](../src/tileops/ops/gemm/bmm.py) carries all of it and is the one to copy
from. `scripts/lint/op_docstrings_lint.py` fails a missing docstring,
reStructuredText, or an `Args:` left on a class; it runs as the
`op-docstrings-lint` pre-commit hook and in the `pre-commit` CI job.

## Benchmarks

Benchmarks compare against external baselines, which the `bench` extra pulls in:

```bash
PIP_NO_BUILD_ISOLATION=1 pip install -e '.[dev,bench]' -c constraints.txt
```

`PIP_NO_BUILD_ISOLATION=1` is required here: several baselines build against the installed PyTorch, and an isolated build environment would fetch a different one.

Prefer the dev Docker image, which carries FlashAttention-2/3, flash-linear-attention, vLLM and flashinfer prebuilt against its own CUDA and PyTorch — the FA3 build in particular takes a long time from source. It does not install `sgl-kernel`, so a benchmark that needs that baseline has to install it in the container.

```bash
python -m pytest benchmarks/            # all benchmarks
python -m pytest benchmarks/ops/bench_gqa.py -q
```

The nightly coverage gate requires every implemented op's manifest cases to appear once in
the passing benchmark rows with correctness evidence. Missing, empty, skipped or wholly
unverified results fail. Declared `Partial` checks count for their checked outputs; coverage
does not require a performance ratio against a baseline.

```bash
python scripts/check_bench_coverage.py --bench-xml bench_results.xml
```

A partial local benchmark run is useful for development but does not satisfy this gate.

Benchmark reporting rules and tolerances are in [design/testing.md](design/testing.md).

## Packaging check

To reproduce the wheel checks CI runs before a release:

```bash
pip install build twine
python -m build
twine check dist/*
```

On H200, verify a built wheel outside the checkout using the pinned dev image dependencies:

```bash
python scripts/ci/check_packaging_coverage.py
scripts/ci/verify_installed_wheel.sh --wheel dist/<wheel>.whl --deps image
```

- Coverage fails on collection errors or a family with no packaging case.
- Wheel verification uses a fresh venv, checks installed imports/resources, and requires every packaging case to run.
- Each invocation creates empty TileLang, Triton, Inductor and CUDA driver caches, including when `--work-dir` is reused. Their paths, resolved versions and device are recorded in `env.json`.
- `--deps pypi` resolves the wheel's declared dependencies in an isolated venv and runs `pip check`.

## Working on a change

Design docs and `src/tileops/manifest/` are the authoritative spec — code conforms to the spec, not the other way around. Start from [design/architecture.md](design/architecture.md) for the module map, and [design/ops-design.md](design/ops-design.md) to add an op.
