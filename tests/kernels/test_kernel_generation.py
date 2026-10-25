"""A kernel's published generation: one configuration and the programs compiled from it
(docs/design/ops-design-reference.md § Kernel base class attributes).

The kernel here runs on the CPU and compiles a Python function, so publication and the program
table are checked without a device.
"""

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from tileops._kernels.kernel_base import Kernel

pytestmark = pytest.mark.smoke


class _Scaling(Kernel):
    """Scales its input by the configured factor, one program per row count."""

    devices = frozenset({"cpu"})
    autotune_configs = [{"scale": 3}]
    default_config = property(lambda self: {"scale": 2})

    def __init__(self) -> None:
        super().__init__()
        self.init_config()
        self.kernel, self.compiled = object(), []

    def _compile(self, cfg: dict, rows: int):
        self.compiled.append((cfg["scale"], rows))
        return lambda x, scale=cfg["scale"]: x * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.program(self._compile, x.shape[0])(x)

    def tune_jit_kernel(self, jit_kernel, configs, **kwargs):
        return SimpleNamespace(config=dict(configs[0]))


def _together(*targets) -> None:
    """Run each of *targets* on its own thread, all released at once, and wait for them."""
    start = threading.Barrier(len(targets))
    threads = [threading.Thread(target=lambda t=t: (start.wait(), t())) for t in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _raising(*args, **kwargs):
    raise RuntimeError("the build failed")


def test_a_launch_runs_a_program_of_the_published_generation():
    """A program is compiled once per generation and launch facts; a new generation's
    launches compile again from its configuration, which no one changes in place."""
    kernel, one = _Scaling(), torch.ones(1)
    for x in (one, one, torch.ones(2), one):
        kernel(x)
    assert kernel.compiled == [(2, 1), (2, 2)]
    kernel.config = {"scale": 5}
    assert kernel(one).tolist() == [5] and kernel.compiled[-1] == (5, 1)
    published = {"scale": 5, "nested": {"scale": 5}, "scales": [5]}
    kernel.config = published
    published["nested"]["scale"] = published["scales"][0] = 6
    assert kernel.config["nested"] == {"scale": 5} and kernel.config["scales"] == (5,)
    config = kernel.config
    for container, key in ((config, "scale"), (config["nested"], "scale"), (config["scales"], 0)):
        with pytest.raises(TypeError):
            container[key] = 7


def test_a_launch_runs_the_one_generation_pinned_for_it():
    """A launch keeps the generation it started with while another is published, a tried
    configuration runs only the trying thread's launches, and a pin leaves the tuning
    fingerprint of the kernel alone."""
    from tileops._kernels.autotune import supply_fingerprint

    kernel, one, seen = _Scaling(), torch.ones(1), []
    kernel._compile = lambda cfg, rows: lambda x: x * cfg["scale"]
    fingerprint = supply_fingerprint(kernel.forward)
    kernel(one)
    assert supply_fingerprint(kernel.forward) == fingerprint

    def compile_and_publish(cfg, rows):
        kernel.config = {"scale": 9}
        seen.append(kernel.config["scale"])
        return lambda x: x * cfg["scale"]

    kernel._compile = compile_and_publish
    assert kernel(torch.ones(2)).tolist() == [2, 2] and seen == [2]
    del kernel._compile
    elsewhere = []
    with kernel._trying({"scale": 4}):
        assert kernel(one).tolist() == [4]
        _together(lambda: elsewhere.append(kernel(one).tolist()))
    assert elsewhere == [[9]] and kernel.config == {"scale": 9}


def test_a_failed_tuning_or_build_publishes_nothing(monkeypatch):
    """A tuning, a build, a publication's compile, a build of the first launch under a tuned
    config or a row-tiled sweep's rebuild that raises leaves the published generation
    serving, and a later attempt that succeeds publishes."""
    import tileops._kernels.autotune as autotune
    from tileops._kernels.reduction._primitives import RowTiledAutotuneMixin

    kernel, one = _Scaling(), torch.ones(1)
    kernel(one)
    published = kernel._generation
    kernel.tune_jit_kernel = _raising
    for attempt in (kernel.autotune, lambda: kernel.program(_raising, 7)):
        with pytest.raises(RuntimeError, match="build failed"):
            attempt()
    with pytest.raises(RuntimeError, match="build failed"):
        kernel._publish({"scale": 6}, _raising)
    assert kernel._generation is published and kernel(one).tolist() == [2]
    assert kernel.program(lambda cfg, rows: rows * cfg["scale"], 7) == 14
    kernel._publish({"scale": 6}, lambda cfg: cfg["scale"])
    del kernel.tune_jit_kernel
    # A launch whose first program builds and whose second raises.
    previous = kernel._generation
    kernel.forward = lambda x: kernel.program(lambda cfg, n: x.mul, 0)(2) + kernel.program(_raising)
    kernel.autotune()
    with pytest.raises(RuntimeError, match="build failed"):
        kernel(one)
    assert kernel._generation is previous, "a tuned config whose launch failed to build stays"
    del kernel.forward
    kernel.autotune()
    assert kernel.config == {"scale": 3} and kernel(one).tolist() == [3]

    class Rows(RowTiledAutotuneMixin, Kernel):
        devices = frozenset({"cpu"})
        autotune_configs = [{"block_m": 1, "threads": 128, "tile_n": n} for n in (64, 32)]
        default_config = property(lambda self: {**self.autotune_configs[0], "tile_n": 64})

        forward = lambda self, x: x  # noqa: E731

        def __init__(self) -> None:
            super().__init__()
            self._tile_n, self.built = 64, []
            self.kernel = self._build_row_kernel(64)
            self.init_config()

        def _build_row_kernel(self, tile_n: int):
            # The sweep builds each candidate once; the winner's rebuild raises.
            if tile_n in self.built and tile_n != 64:
                raise RuntimeError("the rebuild failed")
            self.built.append(tile_n)
            return object()

        def _call_autotuned_kernel(self, tuned, kernel, config):
            return tuned

    # The sweep times tile 64, then tile 32 faster, and picks tile 32.
    latencies = iter([2.0, 1.0])

    def tuner(configs, **kwargs):
        return lambda kernel: SimpleNamespace(latency=next(latencies), config=configs[0])

    monkeypatch.setattr(autotune, "autotune", tuner)
    rows = Rows()
    published = rows._generation
    with pytest.raises(RuntimeError, match="rebuild failed"):
        rows.autotune()
    assert rows._generation is published and rows._tile_n == 64


@pytest.mark.cuda_only
def test_a_decode_sweep_where_no_candidate_compiles_publishes_nothing(monkeypatch):
    """A real tuning sweep that times no candidate leaves the published generation."""
    import tileops._kernels.linear_attention.gla.recurrent as recurrent

    kernel = recurrent.GLADecodeFP32Kernel(1, 2, 64, 64)
    published = kernel._generation
    monkeypatch.setattr(recurrent, "_gla_decode_fp32_tl", _raising)
    kernel.autotune(warmup=1, rep=1)
    assert kernel._generation is published


class _Slow(_Scaling):
    """Compiles slowly, and launches two programs per call."""

    def _compile(self, cfg: dict, rows: int):
        time.sleep(self.delay)
        return super()._compile(cfg, rows)

    def forward(self, x: torch.Tensor) -> tuple:
        return tuple(self.program(self._compile, n)(x) for n in (x.shape[0], -x.shape[0]))


def test_concurrent_launches_build_each_program_once_from_one_generation():
    """Launches that miss a program together compile it once, and launches entering a kernel
    while another thread publishes each run every program of one launch from one generation."""
    kernel, results = _Slow(), []
    kernel.delay = 0.01
    _together(*[lambda: results.append(kernel(torch.ones(3))[0].tolist())] * 4)
    assert (kernel.compiled, results) == ([(2, 3), (2, -3)], [[2, 2, 2]] * 4)

    for _ in range(5):
        kernel, mixed = _Slow(), []
        kernel.delay = 0.001

        def launch(kernel=kernel, mixed=mixed):
            for rows in range(1, 9):
                first, second = kernel(torch.ones(rows))
                mixed.append(first.tolist() != second.tolist())

        def publish(kernel=kernel):
            for scale in range(3, 40):
                kernel.config = {"scale": scale}
                time.sleep(0.0005)

        _together(*[launch] * 8, publish)
        assert mixed == [False] * 64


def test_a_compiled_program_returns_its_outputs_in_the_order_it_names_them():
    """A program naming its outputs out of parameter order returns them in its own order,
    as TileLang does."""
    from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter

    from tileops._kernels.kernel_base import CompiledProgram

    param = SimpleNamespace(
        shape=[1], dtype=SimpleNamespace(bits=32), torch_dtype=lambda: torch.float32
    )
    adapter = object.__new__(TVMFFIKernelAdapter)
    executable = lambda x, first, second: (first.fill_(10.0), second.fill_(20.0))  # noqa: E731
    vars(adapter).update(result_idx=[2, 1], params=[param] * 3, executable=executable)
    program = CompiledProgram(SimpleNamespace(adapter=adapter))
    assert [t.item() for t in program(torch.zeros(1))] == [20.0, 10.0]


def test_a_compiled_program_sizes_a_dynamic_output_from_the_input_binding_it():
    """An output extent a dynamic symbol names is the input extent that binds the symbol, read
    on each launch."""
    from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter

    from tileops._kernels.kernel_base import CompiledProgram

    def param(*shape):
        dtype = SimpleNamespace(bits=32)
        return SimpleNamespace(shape=list(shape), dtype=dtype, torch_dtype=lambda: torch.float32)

    adapter = object.__new__(TVMFFIKernelAdapter)
    executable = lambda x, y: y.copy_(x * 2)  # noqa: E731
    vars(adapter).update(
        result_idx=[1],
        params=[param("rows", 2), param("rows", 2)],
        executable=executable,
        dynamic_symbolic_map={"rows": (0, 0, 0, 1)},
    )
    program = CompiledProgram(SimpleNamespace(adapter=adapter))
    assert program.adapter is adapter, "launched through TileLang, not its executable"
    for rows in (3, 5):
        assert program(torch.ones(rows, 2)).tolist() == [[2.0, 2.0]] * rows


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


class _BuiltAtLaunch(_Scaling):
    """Builds its program, and exposes it to tuning, at its first launch."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel = None

    def _compile(self, cfg: dict, rows: int):
        self.kernel = object()
        return super()._compile(cfg, rows)


@pytest.mark.parametrize("kernel_type", [_Scaling, _BuiltAtLaunch])
def test_a_request_whose_tuning_raised_can_be_made_again(kernel_type):
    """A tuning that raises, now or deferred to the launch, lets the next request tune again."""
    kernel, one = kernel_type(), torch.ones(1)
    kernel.tune_jit_kernel = _raising
    with pytest.raises(RuntimeError, match="build failed"):
        kernel.request_tune()
        kernel(one)
    del kernel.tune_jit_kernel
    kernel.request_tune()
    assert kernel.config == {"scale": 3} and kernel(one).tolist() == [3]


def test_a_build_that_raises_leaves_the_kernel_handle_as_it_was():
    """A build that raises leaves ``self.kernel`` as it was, absent included."""
    kernel = _Scaling()
    held = kernel.kernel

    def points_then_raises(cfg, rows):
        kernel.kernel = "the failed build's program"
        raise RuntimeError("the build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        kernel.program(points_then_raises, 1)
    assert kernel.kernel is held
    del kernel.kernel
    with pytest.raises(RuntimeError, match="build failed"):
        kernel.program(points_then_raises, 2)
    assert "kernel" not in kernel.__dict__
