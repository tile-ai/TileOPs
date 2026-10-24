"""A kernel's published generation: one configuration and the programs compiled from it
(docs/design/ops-design-reference.md § Kernel base class attributes).

The kernel here runs on the CPU and compiles a Python function, so publication and the program
table are checked without a device.
"""

import threading
import time

import pytest
import torch

from tileops.kernels.kernel_base import Kernel

pytestmark = pytest.mark.smoke


class _Scaling(Kernel):
    """Scales its input by the configured factor, one program per row count."""

    devices = frozenset({"cpu"})
    autotune_configs = [{"scale": 3}]

    def __init__(self) -> None:
        super().__init__()
        self.init_config()
        self.kernel = object()
        self.compiled: list[tuple[int, int]] = []

    @property
    def default_config(self) -> dict:
        return {"scale": 2}

    def _compile(self, cfg: dict, rows: int):
        self.compiled.append((cfg["scale"], rows))
        scale = cfg["scale"]
        return lambda x: x * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.program(self._compile, x.shape[0])(x)

    def tune_jit_kernel(self, jit_kernel, configs, **kwargs):
        return type("Tuned", (), {"config": dict(configs[0])})()


def test_a_launch_runs_a_program_of_the_published_generation():
    """A program is compiled once per generation and launch facts; a new generation's
    launches compile again from its configuration and never run an earlier one's."""
    kernel = _Scaling()
    one, two = torch.ones(1), torch.ones(2)
    assert kernel(one).tolist() == [2]
    kernel(one)
    kernel(two)
    kernel(one)
    assert kernel.compiled == [(2, 1), (2, 2)]

    kernel.config = {"scale": 5}
    assert kernel(one).tolist() == [5]
    assert kernel.compiled[-1] == (5, 1)
    with pytest.raises(TypeError, match="does not change in place"):
        kernel.config["scale"] = 7

    published = {"scale": 5, "nested": {"scale": 5}, "scales": [5]}
    kernel.config = published
    published["nested"]["scale"] = 6
    published["scales"][0] = 6
    assert kernel.config["nested"] == {"scale": 5} and kernel.config["scales"] == (5,)
    with pytest.raises(TypeError, match="does not change in place"):
        kernel.config["nested"]["scale"] = 7
    with pytest.raises(TypeError):
        kernel.config["scales"][0] = 7


def test_a_launch_runs_one_generation_while_another_is_published():
    """A configuration published during a launch serves the next launch; the running one
    keeps the configuration and programs it started with."""
    kernel = _Scaling()
    kernel(torch.ones(1))
    seen = []

    def compile_and_publish(cfg, rows):
        kernel.config = {"scale": 9}
        seen.append(kernel.config["scale"])
        return lambda x: x * cfg["scale"]

    kernel._compile = compile_and_publish
    assert kernel(torch.ones(2)).tolist() == [2, 2]
    assert seen == [2]
    assert kernel.config == {"scale": 9}


def test_a_tried_configuration_is_seen_only_by_the_thread_trying_it():
    """A sweep's candidate runs this thread's launches and no other's, and is not
    published."""
    kernel = _Scaling()
    one = torch.ones(1)
    kernel(one)
    elsewhere = []
    with kernel._trying({"scale": 4}):
        assert kernel(one).tolist() == [4]
        thread = threading.Thread(target=lambda: elsewhere.append(kernel(one).tolist()))
        thread.start()
        thread.join()
    assert elsewhere == [[2]]
    assert kernel.config == {"scale": 2} and kernel(one).tolist() == [2]


def test_tuning_publishes_only_a_configuration_it_found():
    """A tuning that raises publishes nothing, and neither does a build that raises; the
    published generation keeps serving, and a later build retries."""
    kernel = _Scaling()
    one = torch.ones(1)
    kernel(one)
    published = kernel._generation

    def failing(*args, **kwargs):
        raise RuntimeError("no candidate compiled")

    kernel.tune_jit_kernel = failing
    with pytest.raises(RuntimeError, match="no candidate"):
        kernel.autotune()
    assert kernel._generation is published and kernel(one).tolist() == [2]

    def broken(cfg, rows):
        raise RuntimeError("the build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        kernel.program(broken, 7)
    assert kernel._generation is published
    assert kernel.program(lambda cfg, rows: rows * cfg["scale"], 7) == 14

    del kernel.tune_jit_kernel
    kernel.autotune()
    assert kernel.config == {"scale": 3} and kernel(one).tolist() == [3]


def test_a_configuration_published_with_a_failing_build_is_not_published():
    """Publishing a configuration with its program compiled publishes nothing when the
    compile raises."""
    kernel = _Scaling()
    kernel(torch.ones(1))
    published = kernel._generation

    def broken(cfg):
        raise RuntimeError("the build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        kernel._publish({"scale": 6}, broken)
    assert kernel._generation is published
    kernel._publish({"scale": 6}, lambda cfg: cfg["scale"])
    assert kernel.config == {"scale": 6}


def test_launches_entering_a_kernel_together_each_run_one_generation():
    """Threads making their first launches together, while another thread publishes, each
    run every program of one launch from one generation."""

    class Twice(_Scaling):
        def _compile(self, cfg: dict, rows: int):
            time.sleep(0.001)
            scale = cfg["scale"]
            return lambda x: x * scale

        def forward(self, x: torch.Tensor) -> tuple:
            first = self.program(self._compile, x.shape[0])(x)
            return first, self.program(self._compile, -x.shape[0])(x)

    for _ in range(5):
        kernel = Twice()
        start = threading.Barrier(9)
        mixed, failed, launched = [], [], []

        def launch(kernel=kernel, start=start, mixed=mixed, failed=failed, launched=launched):
            start.wait()
            try:
                for rows in range(1, 9):
                    first, second = kernel(torch.ones(rows))
                    if first.tolist() != second.tolist():
                        mixed.append((first.tolist(), second.tolist()))
                    launched.append(rows)
            except Exception as exc:
                failed.append(exc)

        def publish(kernel=kernel, start=start):
            start.wait()
            for scale in range(3, 40):
                kernel.config = {"scale": scale}
                time.sleep(0.0005)

        threads = [threading.Thread(target=launch) for _ in range(8)]
        threads.append(threading.Thread(target=publish))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert (mixed, failed, len(launched)) == ([], [], 8 * 8)


def test_a_row_tiled_sweep_whose_rebuild_raises_publishes_nothing(monkeypatch):
    """The tile_n sweep rebuilds the winner's kernel before publishing its configuration,
    so a rebuild that raises leaves the published generation and tile_n."""
    import tileops.kernels.autotune as autotune
    from tileops.kernels.reduction._primitives import RowTiledAutotuneMixin

    class Rows(RowTiledAutotuneMixin, Kernel):
        devices = frozenset({"cpu"})
        autotune_configs = [
            {"block_m": 1, "threads": 128, "tile_n": 64},
            {"block_m": 1, "threads": 128, "tile_n": 32},
        ]

        def __init__(self) -> None:
            super().__init__()
            self._tile_n = 64
            self.built: list[int] = []
            self.kernel = self._build_row_kernel(64)
            self.init_config()

        @property
        def default_config(self) -> dict:
            return {"block_m": 1, "threads": 128, "tile_n": self._tile_n}

        def _build_row_kernel(self, tile_n: int):
            if tile_n in self.built and tile_n != 64:
                raise RuntimeError("the rebuild failed")
            self.built.append(tile_n)
            return object()

        def _call_autotuned_kernel(self, tuned, kernel, config):
            return tuned

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    latencies = iter([2.0, 1.0])

    def tuner(**kwargs):
        # The sweep times tile 64, then tile 32 faster, and picks tile 32.
        tuned = type("Tuned", (), {"latency": next(latencies), "config": kwargs["configs"][0]})
        return lambda kernel: tuned

    monkeypatch.setattr(autotune, "autotune", tuner)
    kernel = Rows()
    published = kernel._generation
    with pytest.raises(RuntimeError, match="rebuild failed"):
        kernel.autotune()
    assert kernel._generation is published and kernel._tile_n == 64


def test_launches_missing_a_program_together_build_it_once():
    """Launches that miss one program of one generation together compile it once."""
    kernel = _Scaling()
    built = []

    def compile_slowly(cfg, rows):
        built.append(rows)
        time.sleep(0.01)
        scale = cfg["scale"]
        return lambda x: x * scale

    kernel._compile = compile_slowly
    start = threading.Barrier(4)
    results = []

    def launch():
        start.wait()
        results.append(kernel(torch.ones(3)).tolist())

    threads = [threading.Thread(target=launch) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert (built, results) == ([3], [[2, 2, 2]] * 4)


def test_a_launch_leaves_the_tuning_fingerprint_of_its_kernel_alone():
    """What a launch records for its own thread is not what tuning keys a supplier by."""
    from tileops.kernels.autotune import supply_fingerprint

    kernel = _Scaling()
    kernel._compile = lambda cfg, rows: lambda x: x * cfg["scale"]
    before = supply_fingerprint(kernel.forward)
    kernel(torch.ones(1))
    assert supply_fingerprint(kernel.forward) == before


@pytest.mark.cuda_only
def test_a_decode_sweep_where_no_candidate_compiles_publishes_nothing(monkeypatch):
    """A tuning sweep that times no candidate leaves the published generation."""
    import tileops.kernels.linear_attention.gla.recurrent as recurrent

    kernel = recurrent.GLADecodeFP32Kernel(1, 2, 64, 64)
    published = kernel._generation

    def refuse(*args, **kwargs):
        raise RuntimeError("the candidate did not compile")

    monkeypatch.setattr(recurrent, "_gla_decode_fp32_tl", refuse)
    kernel.autotune(warmup=1, rep=1)
    assert kernel._generation is published


def test_a_compiled_program_returns_its_outputs_in_the_order_it_names_them():
    """A program naming its outputs out of parameter order returns them in its own order,
    as TileLang does."""
    from types import SimpleNamespace

    from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter

    from tileops.kernels.kernel_base import CompiledProgram

    def param():
        dtype = SimpleNamespace(bits=32)
        return SimpleNamespace(shape=[1], dtype=dtype, torch_dtype=lambda: torch.float32)

    def executable(x, first, second):
        first.fill_(10.0)
        second.fill_(20.0)

    adapter = object.__new__(TVMFFIKernelAdapter)
    vars(adapter).update(
        result_idx=[2, 1], params=[param(), param(), param()], executable=executable
    )
    program = CompiledProgram(SimpleNamespace(adapter=adapter))
    assert [t.item() for t in program(torch.zeros(1))] == [20.0, 10.0]


def test_a_program_evicted_by_newer_launch_facts_is_compiled_again():
    """A generation holds a bounded number of programs: past it the oldest is dropped and
    compiled again on its next launch."""
    kernel = _Scaling()
    for rows in range(1, 100):
        kernel(torch.ones(rows))
    compiled = len(kernel.compiled)
    kernel(torch.ones(99))
    assert len(kernel.compiled) == compiled
    assert kernel(torch.ones(1)).tolist() == [2] and len(kernel.compiled) == compiled + 1
