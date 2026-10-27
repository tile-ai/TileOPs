# Development

How to install, test, lint, and benchmark TileOPs from a source checkout.

Commands here are the ones CI runs. Where CI passes `-c constraints.txt`, use it locally too — that file pins the versions CI validates, and omitting it resolves a different dependency set than the one your PR is tested against.

## Install from source

A CUDA-capable GPU is required to run the test suite. See [Prerequisites](../README.md#installation) for supported Python, PyTorch, CUDA, and TileLang versions.

```bash
git clone https://github.com/tile-ai/TileOPs
cd TileOPs
pip install -e . --group dev -c constraints.txt
pre-commit install
```

If CUDA and TileLang are already installed system-wide and the build fails while re-resolving them, skip build isolation:

```bash
PIP_NO_BUILD_ISOLATION=1 pip install -e . --group dev -c constraints.txt
```

The `dev` dependency group (pip 25.1 or later) adds the linters, pre-commit, pytest and its plugins, and the references tests import (einops, flash-linear-attention). The `bench` dependency group adds the baseline libraries the benchmarks compare against — see [Benchmarks](#benchmarks).

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

Tags follow `cuda<major.minor>-torch<major.minor>-tilelang<version>`, where `<version>` is the release [Prerequisites](../README.md#installation) names; pull that one rather than a floating tag. CI runs on the same software stack.

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

The `packaging` marker is separate from the tiers: its cases run against an installed wheel rather than a source checkout.

### CUDA Graph validation

[`tests/cuda_graph_contract.py`](../tests/cuda_graph_contract.py) lists the ops with qualified
calls. Each `cuda_graph(op=...)` test defines its supported shapes, dtypes and options; the
qualification does not extend to untested specializations or other ops. Run them on H200:

```bash
python -m pytest -q tests/ops -m cuda_graph
```

- Warm up with the intended options and any requested tuning before capture.
- Replay changes tensor contents at captured addresses. Changes to shapes, strides, dtypes, host metadata or dispatch options require a new graph.

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

[`bmm.py`](../src/tileops/_ops/gemm/bmm.py) carries all of it and is the one to copy
from. The `op-docstrings-lint` pre-commit hook fails a missing docstring,
reStructuredText, or an `Args:` left on a class.

## Benchmarks

Benchmarks compare against external baselines, which the `bench` dependency group pulls in (pip 25.1 or later):

```bash
PIP_NO_BUILD_ISOLATION=1 pip install -e . --group dev --group bench -c constraints.txt
```

FlashAttention-3 (`flash_attn_interface`) is not on PyPI and is not in the group; build it from the FlashAttention repository's `hopper/` directory, or use the dev image.

`PIP_NO_BUILD_ISOLATION=1` is required here: several baselines build against the installed PyTorch, and an isolated build environment would fetch a different one.

Prefer the dev Docker image, which carries FlashAttention-2/3, flash-linear-attention, vLLM and flashinfer prebuilt against its own CUDA and PyTorch — the FA3 build in particular takes a long time from source. It does not install `sgl-kernel`, so a benchmark that needs that baseline has to install it in the container.

```bash
python -m pytest benchmarks/            # all benchmarks
python -m pytest benchmarks/ops/bench_gqa.py -q
```

Benchmark reporting rules and tolerances are in [design/testing.md](design/testing.md).

## Packaging check

To build the wheel and check its metadata:

```bash
pip install build twine
python -m build
twine check dist/*
```
