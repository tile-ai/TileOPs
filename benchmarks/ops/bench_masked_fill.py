"""Benchmark for MaskedFillFwdOp (Tensor-value masked_fill).

Baseline:
  - PyTorch reference: ``input.masked_fill(mask, value)`` with ``value`` a
    0-dim Tensor on the same device.

Workload shapes come from the manifest entry's ``workloads`` (via
``load_workloads``); the benchmark reports TileOPs latency alongside the
manifest-derived roofline (``op.eval_roofline()``).

Usage:
    conda run -n tileops python -m pytest benchmarks/ops/bench_masked_fill.py -vvs
    conda run -n tileops python benchmarks/ops/bench_masked_fill.py
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.manifest import load_manifest, load_workloads
from tileops.ops.elementwise import MaskedFillFwdOp
from workloads.workload_base import WorkloadBase

_OP_NAME = "MaskedFillFwdOp"

# Skip the whole module while the manifest entry is still spec-only:
# the L4 `eval_roofline()` body is codegen-installed only on `implemented`
# ops, so `op.eval_roofline()` would hit the L1 base stub.
_OP_STATUS = load_manifest().get(_OP_NAME, {}).get("status", "spec-only")
pytestmark = pytest.mark.skipif(
    _OP_STATUS != "implemented",
    reason=f"{_OP_NAME} manifest status is {_OP_STATUS!r}; "
    f"eval_roofline body not installed until status flips to 'implemented'",
)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class MaskedFillTest(WorkloadBase):
    """Manifest-shaped inputs for the Tensor-value masked_fill kernel.

    ``input`` and ``mask`` are sized per the manifest workload; ``value``
    is a 0-dim Tensor as required by ``MaskedFillFwdOp``.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        mask_shape: tuple[int, ...],
        dtype: torch.dtype,
    ):
        self.input_shape = tuple(input_shape)
        self.mask_shape = tuple(mask_shape)
        self.dtype = dtype

    def gen_inputs(self):
        torch.manual_seed(42)
        dev = "cuda"
        x = torch.randn(self.input_shape, dtype=self.dtype, device=dev)
        # Roughly half-true mask exercises both branches of the predicated select.
        mask = torch.randint(0, 2, self.mask_shape, device=dev).bool()
        value = torch.zeros((), dtype=self.dtype, device=dev)
        return x, mask, value

    def ref_program(self, *args):
        return None


class MaskedFillBenchmark(BenchmarkBase[MaskedFillTest]):

    _roofline_cache: Optional[tuple[float, float]] = None

    def __init__(self, test: MaskedFillTest, op: MaskedFillFwdOp):
        super().__init__(test)
        self._op = op

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = self._op.eval_roofline()
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


def _manifest_params():
    """Convert manifest workloads to pytest params."""
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        input_shape = tuple(w["input_shape"])
        mask_shape = tuple(w["mask_shape"])
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                input_shape, mask_shape, dtype_str,
                id=f"{label}-{dtype_str}",
            ))
    return params


@pytest.mark.parametrize(
    "input_shape, mask_shape, dtype_str",
    _manifest_params(),
)
def test_masked_fill_bench(
    input_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
    dtype_str: str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    test = MaskedFillTest(input_shape, mask_shape, dtype)
    x, mask, value = test.gen_inputs()

    op = MaskedFillFwdOp(
        input=input_shape, mask=mask_shape, value=(),
        dtype=dtype,
    )
    bm = MaskedFillBenchmark(test, op)

    # Warmup: trigger JIT compilation before timed profiling.
    op(x, mask, value)
    torch.cuda.synchronize()

    result = bm.profile(op, x, mask, value)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def _torch_fn(x, mask, value):
        return x.masked_fill(mask, value)

    _torch_fn(x, mask, value)  # warmup
    torch.cuda.synchronize()

    result_torch = bm.profile(_torch_fn, x, mask, value)
    BenchmarkReport.record(op, locals(), result_torch, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
