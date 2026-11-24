import pytest
import torch

from tests.compile_contract import register_compile_contract
from tests.workload_test_base import TestBase
from tileops._kernels.indexing import ConcatKernel
from tileops._kernels.indexing.concat import _GROUP, _THREADS
from tileops.indexing import ConcatFwdOp
from workloads.device import run_device
from workloads.indexing import ConcatWorkload, TransposedConcatWorkload
from workloads.workload_base import manifest_call


class ConcatTest(ConcatWorkload, TestBase):
    pass


class TransposedConcatTest(TransposedConcatWorkload, TestBase):
    pass


_BF16, _FP16, _FP32 = "bfloat16", "float16", "float32"
_SMOKE = pytest.mark.smoke
_FULL = pytest.mark.full
# One case per dtype and per copy path: whole 16-byte reads, narrower vectors, one or several
# vectors per thread, rows that end inside a column tile, one input, several launches, and
# empty elements.
_CASES = (
    # Two inputs that each fill whole column tiles, along a negative dim: the MTP decode-4 row.
    pytest.param([[4, 5120], [4, 5120]], -1, [_BF16] * 2, marks=_SMOKE, id="bf16-tiles"),
    # Each input's row ends inside a column tile.
    pytest.param([[3, 40], [3, 24]], 1, [_FP16] * 2, marks=_SMOKE, id="fp16-tails"),
    pytest.param([[2, 3, 8], [2, 5, 8]], 1, [_FP32] * 2, marks=_SMOKE, id="fp32-middle-dim"),
    # Rows whose one-vector CTAs are all resident and outnumber the SMs: 16-byte vectors,
    # doubled to four per thread, where the launch fits the SMs.
    pytest.param([[32, 5120], [32, 5120]], -1, [_BF16] * 2, marks=_FULL, id="bf16-resident-rows"),
    # A resident launch of 1-element 2-byte vectors: four per thread, each an element store.
    pytest.param([[8, 4097], [8, 3]], 1, [_BF16] * 2, marks=_FULL, id="bf16-2-byte-resident"),
    # Rows whose one-vector CTAs are all resident and outnumber the SMs at two vectors, where
    # the next doubling removes no CTA: two vectors per thread.
    pytest.param(
        [[80, 1032], [80, 1032]], -1, [_BF16] * 2, marks=_FULL, id="bf16-resident-no-removal"
    ),
    # A launch just beyond one resident wave: the float32 input keeps one 16-byte vector and
    # the bfloat16 input doubles to two, which brings the launch within the wave; there both
    # double to the cap of eight, with more CTAs than SMs left.
    pytest.param(
        [[136, 4096], [136, 4096]], -1, [_FP32, _BF16], marks=_FULL, id="wave-then-cap-eight"
    ),
    # The MTP draft row: its CTAs are all resident at once, so each thread copies eight
    # vectors, the last ones past the input's row end.
    pytest.param(
        [[64, 5120], [64, 5120]], -1, [_BF16] * 2, marks=_FULL, id="bf16-eight-per-thread"
    ),
    # Widths that make a vector 8 bytes, on more one-vector CTAs than SMs: two per thread, each
    # an 8-byte store, the second one past the row end in the last tile.
    pytest.param([[32, 1036], [32, 1044]], 1, [_BF16] * 2, marks=_FULL, id="bf16-8-byte-vectors"),
    # Widths that make a vector 4 bytes: four per thread, each a 4-byte store of two elements.
    pytest.param([[32, 1034], [32, 1030]], 1, [_BF16] * 2, marks=_FULL, id="bf16-4-byte-vectors"),
    # bfloat16 and float16 promote to a float32 output: 8-byte reads, two per thread, converted.
    pytest.param(
        [[20, 1200], [20, 1032], [20, 8]], -1, [_BF16, _FP16, _BF16], marks=_FULL, id="promote"
    ),
    # float16 and float32 promote to float32: the float16 input takes twice the float32
    # input's vectors a thread, so each input's tiles take its own count.
    pytest.param(
        [[256, 4096], [256, 4096]],
        -1,
        [_FP16, _FP32],
        marks=_FULL,
        id="mixed-reads-per-thread",
    ),
    # Along dim 0 the output is one row.
    pytest.param([[5], [7]], 0, [_FP32] * 2, marks=_FULL, id="rank-1"),
    # One input: a launch with one tensor slot.
    pytest.param([[4, 16]], 1, [_BF16], marks=_FULL, id="single"),
    # More inputs than one launch takes.
    pytest.param(
        [[3, 2], [3, 1], [3, 4], [3, 3], [3, 1], [3, 9]], -1, [_FP16] * 6, marks=_FULL, id="six"
    ),
    # An empty element inside the first launch, and one alone after it.
    pytest.param(
        [[4, 8], [4, 0], [4, 8], [4, 8], [4, 0]], 1, [_BF16] * 5, marks=_FULL, id="empty-element"
    ),
    # More rows than the grid's y extent holds, each a whole number of lines: a second launch
    # of column tiles takes the last row.
    pytest.param([[65536, 64], [65536, 64]], 1, [_BF16] * 2, marks=_FULL, id="rows-past-grid-y"),
    # Rows of whole 16-byte vectors that end inside a line, past one resident wave, copied by
    # CTAs over the flat output. A CTA holds at least one warp and each row takes one, so 16385
    # rows pass one wave on any device of up to 256 SMs holding 64 warps each.
    # Rows of one vector: a vector's row is its position.
    pytest.param([[16385, 8]], 1, [_FP16], marks=_FULL, id="flat-one-vector-rows"),
    # Runs that cross rows and all four inputs, the last run partial.
    pytest.param(
        [[16385, 4], [16385, 12], [16385, 8], [16385, 4]],
        1,
        [_FP32] * 4,
        marks=_FULL,
        id="flat-four-inputs",
    ),
    # bfloat16 and float16 rows past one wave that are not whole groups: per-row tiles of
    # 8-byte reads, not the widening program's 16-byte reads.
    pytest.param(
        [[16385, _GROUP // 2], [16385, 3 * _GROUP // 2]],
        1,
        [_BF16, _FP16],
        marks=_FULL,
        id="widen-part-groups",
    ),
    # Five bfloat16 and float16 inputs past one wave: more than one launch takes, so per-row
    # tiles in two launches.
    pytest.param(
        [[16385, _GROUP]] * 5, 1, [_BF16, _FP16, _BF16, _FP16, _BF16], marks=_FULL, id="widen-five"
    ),
    # Rows that end inside a line, with more output elements than a 32-bit offset reaches: the
    # flat program's element addresses.
    pytest.param(
        [[262144, 4104], [262144, 4096]],
        1,
        [_BF16] * 2,
        marks=pytest.mark.nightly,
        id="flat-past-32-bit-offsets",
    ),
)


@pytest.mark.parametrize("S, dim, D", _CASES)
def test_concat(S, dim, D) -> None:
    call = manifest_call("ConcatFwdOp", {"D": D}, S=S, dim=dim)
    test = ConcatTest(call)
    test.check(ConcatFwdOp(**call.arguments()), *test.gen_inputs())


def _checked_launches(call, monkeypatch) -> list[tuple[str, tuple]]:
    """Check the op on *call*'s inputs, and return each launch's program build and facts."""
    test = ConcatTest(call)
    launches = []
    program = ConcatKernel.program

    def record(self, build, *facts):
        launches.append((build.__name__, facts))
        return program(self, build, *facts)

    monkeypatch.setattr(ConcatKernel, "program", record)
    test.check(ConcatFwdOp(**call.arguments()), *test.gen_inputs())
    return launches


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "widths",
    [
        # Two inputs: each vector is read from the input its column picks.
        pytest.param([8, 24], marks=_SMOKE, id="two-inputs"),
        # One input: a vector's row is its position.
        pytest.param([8], marks=_FULL, id="one-input"),
    ],
)
def test_concat_flat_rows_within_one_wave_at_four(widths, monkeypatch) -> None:
    """float16 rows that end inside a line, one row past the flat program's default vectors a
    thread for every resident thread: past one wave at the default and within it at twice
    that, which every launch takes. The last run is partial."""
    props = torch.cuda.get_device_properties(run_device())
    resident_threads = props.multi_processor_count * props.max_threads_per_multi_processor
    vectors = sum(widths) // 8
    rows = resident_threads * ConcatKernel._FLAT_PER_THREAD // vectors + 1
    S = [[rows, w] for w in widths]
    call = manifest_call("ConcatFwdOp", {"D": [_FP16] * len(widths)}, S=S, dim=1)
    launches = _checked_launches(call, monkeypatch)
    doubled = ("_compile_flat", (len(widths), 2 * ConcatKernel._FLAT_PER_THREAD))
    assert launches and all((name, facts[:2]) == doubled for name, facts in launches), launches


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "dtypes, widths",
    [
        # bfloat16 around float16: each group is widened by the dtype of the input it is read from.
        # Rows of 2, 1 and 3 groups: each input's row offset scales by its own width.
        pytest.param([_BF16, _FP16, _BF16], [2, 1, 3], marks=_SMOKE, id="three-inputs"),
        # Four inputs, float16 first: every slot reads.
        pytest.param([_FP16, _BF16, _FP16, _BF16], [1, 1, 1, 1], marks=_FULL, id="four-inputs"),
        # An empty float16 input promotes one bfloat16 input: a group's row is its position.
        pytest.param([_BF16, _FP16], [1, 0], marks=_FULL, id="one-bf16-input"),
        # An empty bfloat16 input promotes one float16 input.
        pytest.param([_FP16, _BF16], [1, 0], marks=_FULL, id="one-f16-input"),
    ],
)
def test_concat_widen_rows(dtypes, widths, monkeypatch) -> None:
    """bfloat16 and float16 rows of *widths* whole groups, promoted to float32, one row past one
    resident wave of per-row tiles, which every launch widens. The last run is partial."""
    props = torch.cuda.get_device_properties(run_device())
    wave = props.multi_processor_count * props.max_threads_per_multi_processor // _THREADS
    rows = wave // sum(1 for w in widths if w) + 1
    S = [[rows, w * _GROUP] for w in widths]
    call = manifest_call("ConcatFwdOp", {"D": dtypes}, S=S, dim=1)
    launches = _checked_launches(call, monkeypatch)
    assert launches and all(name == "_compile_widen" for name, _ in launches), launches


# The bytes a row of bfloat16 and float16 inputs of 1024 elements each reads and writes: 2 x 1024
# 2-byte elements in, 2048 4-byte ones out.
_WIDE_ROW_BYTES = 2 * 1024 * 2 + 2048 * 4


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "rows, widths, program",
    [
        # Within half of L2: per-row tiles.
        pytest.param(
            lambda l2: l2 // 2 // _WIDE_ROW_BYTES + 1,
            [1024, 1024],
            "_compile",
            marks=_SMOKE,
            id="within-l2",
        ),
        # Inputs and output past the device's L2: the widening program.
        pytest.param(
            lambda l2: 2 * l2 // _WIDE_ROW_BYTES + 1,
            [1024, 1024],
            "_compile_widen",
            marks=_FULL,
            id="past-l2",
        ),
        # A bfloat16 input of 1024 elements fewer than 2**31: the widening program, its element
        # offsets at the top of the 32-bit range.
        pytest.param(
            lambda l2: 2**21 - 1,
            [1024, _GROUP],
            "_compile_widen",
            marks=_FULL,
            id="near-32-bit-offsets",
        ),
        # A bfloat16 input of more elements than a 32-bit offset reaches: per-row tiles.
        pytest.param(
            lambda l2: 2**21 + 1, [1024, _GROUP], "_compile", marks=_FULL, id="past-32-bit-offsets"
        ),
    ],
)
def test_concat_widen_dispatch(rows, widths, program, monkeypatch) -> None:
    """A bfloat16 and a float16 input of *widths*, promoted to float32, past one resident wave of
    per-row tiles, with ``rows(l2)`` rows on a device of ``l2`` bytes of L2; every launch takes
    *program*."""
    l2 = torch.cuda.get_device_properties(run_device()).L2_cache_size
    S = [[rows(l2), w] for w in widths]
    call = manifest_call("ConcatFwdOp", {"D": [_BF16, _FP16]}, S=S, dim=1)
    launches = _checked_launches(call, monkeypatch)
    assert launches and all(name == program for name, _ in launches), launches


@pytest.mark.smoke
def test_concat_follows_each_call_shape() -> None:
    """One op joins each call's own shapes: new rows and widths, then the first call again."""
    op = ConcatFwdOp(-1)
    for S in ([[3, 40], [3, 24]], [[4, 5120], [4, 5120]], [[3, 40], [3, 24]]):
        test = ConcatTest(manifest_call("ConcatFwdOp", {"D": [_FP16] * 2}, S=S, dim=-1))
        test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_concat_strided_elements() -> None:
    """Elements that are transposed views are joined by their logical elements."""
    call = manifest_call("ConcatFwdOp", {"D": [_BF16] * 2}, S=[[3, 40], [3, 24]], dim=1)
    test = TransposedConcatTest(call)
    test.check(ConcatFwdOp(1), *test.gen_inputs())


@pytest.mark.smoke
def test_concat_rejects_an_integer_element() -> None:
    """The signature takes float16, bfloat16 and float32 elements only."""
    tensors = [torch.zeros(2, 3, device=run_device(), dtype=torch.int32) for _ in range(2)]
    with pytest.raises(ValueError):
        ConcatFwdOp(1)(tensors)


register_compile_contract(ConcatFwdOp)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_concat_compiles_cold() -> None:
    """A cold ``fullgraph=True`` compile joins the elements as the eager op does."""
    call = manifest_call("ConcatFwdOp", {"D": [_BF16, _FP16]}, S=[[3, 40], [3, 24]], dim=-1)
    test = ConcatTest(call)
    op = ConcatFwdOp(-1)
    test.check(op, *test.gen_inputs(), runs=torch.compile(op, fullgraph=True))
