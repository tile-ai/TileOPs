"""A persisted tuning result is reused only while the inputs it was timed on are unchanged.

TileLang leaves the input supplier out of the key of a tuning result it stores, so a result
tuned on one supplier's inputs survives a change to the supplier. ``tileops._kernels.autotune``
adds the supplier's fingerprint to that key.
"""

import importlib
import linecache
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest
import tilelang
import tilelang.language as T
import torch
from tilelang.autotuner.tuner import AutoTuner

from tileops._kernels.autotune import supply_fingerprint
from tileops._kernels.kernel_base import Kernel
from workloads.device import run_device

_REPO = Path(__file__).resolve().parents[2]

# The modules a supplier is built from: a constant, a helper it calls, and the supplier.
_MODULES = {
    "consts": "ROWS = 4\n",
    "helper": """
import torch

FILL = 8

def lengths(rows):
    return torch.full((rows,), FILL, dtype=torch.int32)
""",
    "supplier": """
import torch

import helper
from consts import ROWS

FILL = [8]
OFFSETS = torch.zeros(2, dtype=torch.int32)
# Iterated, and NAMES filled, in the per-process string-hash order.
TAGS = {("lengths", "int32"), ("offsets", "int32"), ("scales", "float32")}
NAMES = dict.fromkeys(name for name, _ in TAGS)

class Shape:
    def __init__(self, rows):
        self.rows = rows

    def supply(self, params):
        return [helper.lengths(self.rows)]

class Slotted:
    __slots__ = ("rows",)

    def __init__(self, rows):
        self.rows = rows

    def supply(self, params):
        return [helper.lengths(self.rows)]

class State:
    def __init__(self):
        self.rows = 4

class Held:
    state = State()

    def __call__(self, params):
        return [helper.lengths(self.state.rows)]

def make(rows):
    def supply_prog(params):
        return [helper.lengths(rows), sorted(TAGS), sorted(NAMES)]
    return supply_prog

def supply_prog(params):
    return [helper.lengths(ROWS) * FILL[0] + OFFSETS[0]]
""",
}


def _set_fill(module) -> None:
    module.FILL[0] = 9


def _set_offset(module) -> None:
    module.OFFSETS[0] = 2


def _set_held_rows(module) -> None:
    module.Held.state.rows = 5


# change -> (supplier before, supplier after, module edits, runtime change in between). Each
# changes one fact the supplied inputs depend on. A supplier is "function", "held",
# "closure:<rows>", "method:<rows>" or "slotted:<rows>".
_CHANGES = {
    "supplier source": (
        "function",
        "function",
        {"supplier": _MODULES["supplier"].replace("lengths(ROWS)", "lengths(ROWS + 1)")},
        None,
    ),
    "called helper source": (
        "function",
        "function",
        {"helper": _MODULES["helper"].replace("FILL = 8", "FILL = 16")},
        None,
    ),
    "imported constant": ("function", "function", {"consts": "ROWS = 5\n"}, None),
    "mutated global": ("function", "function", {}, _set_fill),
    "integer tensor values": ("function", "function", {}, _set_offset),
    "closed-over value": ("closure:4", "closure:5", {}, None),
    "bound object attribute": ("method:4", "method:5", {}, None),
    "slotted object attribute": ("slotted:4", "slotted:5", {}, None),
    "class-held object attribute": ("held", "held", {}, _set_held_rows),
}


def _write_modules(root: Path, edits: dict[str, str]) -> None:
    for name, source in {**_MODULES, **edits}.items():
        (root / f"{name}.py").write_text(textwrap.dedent(source))


def _reload():
    """The ``supplier`` module, with every module it is built from reloaded from disk."""
    linecache.clearcache()
    importlib.invalidate_caches()
    for name in _MODULES:
        module = importlib.reload(importlib.import_module(name))
    return module


def _supplier(module, spec: str):
    kind, _, rows = spec.partition(":")
    if kind == "closure":
        return module.make(int(rows))
    if kind == "method":
        return module.Shape(int(rows)).supply
    if kind == "slotted":
        return module.Slotted(int(rows)).supply
    if kind == "held":
        return module.Held()
    return module.supply_prog


@pytest.fixture
def supplier_modules(tmp_path, monkeypatch):
    """The ``_MODULES`` in *tmp_path*, importable and reloadable."""
    _write_modules(tmp_path, {})
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    yield tmp_path
    for name in _MODULES:
        sys.modules.pop(name, None)


@pytest.mark.smoke
@pytest.mark.parametrize("change", list(_CHANGES))
def test_fingerprint_changes_with_what_the_supplier_feeds(supplier_modules, change):
    """Each fact the supplied inputs depend on enters the fingerprint."""
    before_spec, after_spec, edits, mutate = _CHANGES[change]
    before = supply_fingerprint(_supplier(_reload(), before_spec))
    assert supply_fingerprint(_supplier(_reload(), before_spec)) == before

    _write_modules(supplier_modules, edits)
    module = _reload()
    if mutate is not None:
        mutate(module)
    assert supply_fingerprint(_supplier(module, after_spec)) != before


@pytest.mark.smoke
def test_fingerprint_is_the_same_in_another_process(supplier_modules):
    """An unchanged supplier keeps its fingerprint across processes, string hash seed included."""
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); import supplier;"
        "from tileops._kernels.autotune import supply_fingerprint;"
        "print(supply_fingerprint(supplier.make(4)), supply_fingerprint(supplier.Shape(4).supply))"
    )
    module = _reload()
    expected = [
        supply_fingerprint(_supplier(module, "closure:4")),
        supply_fingerprint(_supplier(module, "method:4")),
    ]
    # Two seeds that iterate ``TAGS`` and fill ``NAMES`` in different orders.
    for seed in ("1", "2"):
        run = subprocess.run(
            [sys.executable, "-c", script, str(supplier_modules)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONPATH": str(_REPO / "src"), "PYTHONHASHSEED": seed},
        )
        assert run.stdout.split() == expected, f"hash seed {seed}"


@pytest.mark.smoke
def test_fingerprint_tracks_imported_helper_runtime_state(supplier_modules):
    module = _reload()
    before = supply_fingerprint(module.supply_prog)
    module.helper.FILL = 16
    assert supply_fingerprint(module.supply_prog) != before


@pytest.mark.smoke
def test_fingerprint_tracks_replaced_function_code(supplier_modules):
    module = _reload()
    before = supply_fingerprint(module.supply_prog)
    module.supply_prog.__code__ = (lambda params: []).__code__
    assert supply_fingerprint(module.supply_prog) != before


@pytest.mark.smoke
def test_fingerprint_tracks_non_kernel_config(supplier_modules):
    module = _reload()
    supplier = module.Shape(4)
    supplier.config = {"rows": 4}
    before = supply_fingerprint(supplier.supply)
    supplier.config["rows"] = 8
    assert supply_fingerprint(supplier.supply) != before


_EXTENT = 4096


def _doubling_builder():
    extent = _EXTENT

    @tilelang.jit(out_idx=[1])
    def program(block: int = 256):
        @T.prim_func
        def main(x: T.Tensor((extent,), "float32"), y: T.Tensor((extent,), "float32")):
            with T.Kernel(T.ceildiv(extent, block), threads=128) as bx:
                for i in T.Parallel(block):
                    y[bx * block + i] = x[bx * block + i] * 2.0

        return main

    return program


def _filled_supplier(fill: float):
    def supply_prog(params):
        # Read by the test that runs this process.
        print("supplied", flush=True)
        return [torch.full((_EXTENT,), fill, dtype=torch.float32, device=run_device())]

    return supply_prog


class _DoublingKernel(Kernel):
    autotune_configs = [{"block": 256}, {"block": 512}]

    def __init__(self, supply_prog) -> None:
        super().__init__()
        self.kernel = _doubling_builder()
        self._supply_prog = supply_prog
        self.init_config()

    @property
    def default_config(self) -> dict:
        return {"block": 256}

    @property
    def autotune_supply_prog(self):
        return self._supply_prog

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel(**self.config)(x)


def _tune_once(cache_dir: str, fill: float) -> None:
    """Tune a fresh kernel on inputs filled with *fill*, its results stored in *cache_dir*.

    Run in a process of its own, as each nightly run is.
    """
    AutoTuner.cache_dir = property(lambda self: Path(cache_dir))
    tilelang.env.is_cache_enabled = lambda: True
    tilelang.env.is_autotune_cache_disabled = lambda: False
    _DoublingKernel(_filled_supplier(fill)).autotune(warmup=1, rep=1)


def _supplied_in_new_process(cache_dir: Path, fill: float, seed: str) -> int:
    """How often the supplier fed a tuning run in a new process with hash seed *seed*."""
    script = (
        "import importlib.util, sys;"
        "spec = importlib.util.spec_from_file_location('supply_key_case', sys.argv[1]);"
        "case = importlib.util.module_from_spec(spec); sys.modules['supply_key_case'] = case;"
        "spec.loader.exec_module(case); case._tune_once(sys.argv[2], float(sys.argv[3]))"
    )
    run = subprocess.run(
        [sys.executable, "-c", script, __file__, str(cache_dir), str(fill)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(_REPO / "src"), str(_REPO)]),
            "PYTHONHASHSEED": seed,
        },
    )
    assert run.returncode == 0, run.stderr[-2000:]
    return run.stdout.split().count("supplied")


@pytest.mark.smoke
@pytest.mark.cuda_only
def test_persisted_result_is_tuned_again_after_the_supplier_changes():
    """Across processes, the stored result is read for an unchanged supplier and tuned
    again for a changed one."""
    # TileLang moves a result from its temporary directory into the cache by rename, so the
    # cache stands on the same file system.
    os.makedirs(tilelang.env.TILELANG_TMP_DIR, exist_ok=True)
    cache_dir = Path(tempfile.mkdtemp(dir=tilelang.env.TILELANG_TMP_DIR))
    try:
        assert _supplied_in_new_process(cache_dir, 1.0, "1"), "the first run found a result"
        assert _supplied_in_new_process(cache_dir, 1.0, "2") == 0, (
            "an unchanged supplier tuned again instead of reading its result"
        )
        assert _supplied_in_new_process(cache_dir, 2.0, "3"), (
            "a changed supplier reused the result tuned on the old inputs"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
