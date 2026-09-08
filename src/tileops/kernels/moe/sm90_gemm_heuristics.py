"""Template parameters and config selection for the SM90 grouped GEMM template.

Ported from DeepGEMM: the kernel template ``sm90_bf16_gemm_impl`` takes every
structural choice as a template parameter, and the host side (``SM90ArchSpec``
+ ``get_best_config``) enumerates the legal layouts for a call and scores them
with an L1/L2 cycle model instead of benchmarking. ``SM90GemmSpec`` is the
TileOPs counterpart of that template parameter pack, and ``get_best_config`` is
the counterpart of the selector.

Deviations from DeepGEMM, all deliberate:

* Only the grouped and batched GEMM types are carried; DeepGEMM's dense
  ``Normal`` type, and with it TMA multicast across a 2-CTA cluster and the
  stage merging of its single-warp-group NT schedule, are not. The MoE
  layouts this template serves never took a cluster (every expert streams
  its own ``B``), so the code stayed unexercised.
* ``block_m`` in {16, 32} is not offered. TileLang's WGMMA lowering is only
  exercised at 64-row multiples here.
* ``with_accumulation`` (TMA reduce-add into C) is absent: TileLang's TMA store
  lowering hard-codes ``need_reduce = 0``.
* Swizzle modes are not spec fields. TileLang derives the swizzle from the
  shared tile's inner extent, the same rule DeepGEMM applies, and with 64-wide
  ``block_n`` steps every inner extent is a whole 128-byte atom, so DeepGEMM's
  swizzle prune has nothing to reject and is not carried.
"""

import dataclasses
import enum
import functools
import math

__all__ = [
    "ACTIVATIONS",
    "PER_GROUP_TYPES",
    "PER_ROW_TYPES",
    "GemmDesc",
    "GemmType",
    "Major",
    "SM90GemmSpec",
    "get_best_config",
    "layout_candidates",
    "spec_from_config",
]

# Hopper opt-in shared memory per block, in bytes.
_SMEM_CAPACITY = 232448
_MAX_STAGES = 16
_BARRIER_BYTES = _MAX_STAGES * 8 * 2
# What TileLang lays out beyond DeepGEMM's accounting: 1024-byte alignment of
# each swizzled buffer (A per math warp-group, B, C per math warp-group) and
# the per-group tile prefix sum. Without it a budget within a few hundred
# bytes of the cap fails at launch ("Failed to set the allowed dynamic shared
# memory size").
_SMEM_ALIGN_SLACK = 6 * 1024
_WGMMA_M = 64
_ELEM_BYTES = 2  # bf16 / fp16 operands
_BLOCK_K = 128 // _ELEM_BYTES
# H200 calibration, 2026-09: block_n must hold whole 128-byte swizzle atoms.
_BLOCK_N_STEP = 64


class GemmType(str, enum.Enum):
    """Which rows of ``A`` and which ``B`` a tile reads.

    * ``M_GROUPED_ALIGNED_PER_ROW``: DeepGEMM's ``MGroupedContiguous``. Rows are
      packed per group into segments whose length is a multiple of ``block_m``,
      ``grouped_layout[row]`` names the group of each row, and rows past the
      last group carry ``num_groups``.
    * ``M_GROUPED_ALIGNED_PSUM``: DeepGEMM's ``MGroupedContiguousWithPsumLayout``.
      ``grouped_layout[g]`` is the prefix-sum end row of group ``g``, and each
      group starts at the previous end rounded up to ``block_m``.
    * ``M_GROUPED_TIGHT_PSUM``: TileOPs' ``tight_physical_psum``. As the aligned
      psum layout but each group starts exactly at the previous end, so a
      group's last tile is stored with a row mask.
    * ``M_GROUPED_TIGHT_PER_ROW``: TileOPs' ``tight_per_row``. The tight layout
      described by a non-decreasing group id per row instead of the ends; the
      kernel recovers the ends with one binary search per group at start and
      then runs the tight-psum schedule.
    * ``M_GROUPED_MASKED``: DeepGEMM's ``MGroupedMasked``. ``A`` and ``C`` carry a
      leading group dim of ``max_m`` rows each and ``grouped_layout[g]`` is the
      valid row count.
    * ``BATCHED``: independent GEMMs on a leading batch dim of ``A``, ``B``, ``C``.
    * ``K_GROUPED_CONTIGUOUS``: DeepGEMM's ``KGroupedContiguous``. The groups are
      packed along ``K``: ``A`` is ``[M, sum_k]``, ``B`` is ``[N, sum_k]``,
      ``grouped_layout[g]`` is group ``g``'s ``K`` and ``C`` is ``[G, M, N]``. The
      weight-gradient GEMM of an expert MLP: each group contracts over its own
      tokens, its last K block is masked in shared memory where it runs into the
      next group, and a group with no tokens stores zeros.
    """

    M_GROUPED_ALIGNED_PER_ROW = "m_grouped_aligned_per_row"
    M_GROUPED_ALIGNED_PSUM = "m_grouped_aligned_psum"
    M_GROUPED_TIGHT_PSUM = "m_grouped_tight_psum"
    M_GROUPED_TIGHT_PER_ROW = "m_grouped_tight_per_row"
    M_GROUPED_MASKED = "m_grouped_masked"
    BATCHED = "batched"
    K_GROUPED_CONTIGUOUS = "k_grouped_contiguous"


_ALIGNED_TYPES = (GemmType.M_GROUPED_ALIGNED_PER_ROW, GemmType.M_GROUPED_ALIGNED_PSUM)
PER_GROUP_TYPES = (
    GemmType.M_GROUPED_MASKED,
    GemmType.M_GROUPED_ALIGNED_PSUM,
    GemmType.M_GROUPED_TIGHT_PSUM,
    GemmType.M_GROUPED_TIGHT_PER_ROW,
)
# The types whose grouped_layout is one entry per row rather than per group.
PER_ROW_TYPES = (GemmType.M_GROUPED_ALIGNED_PER_ROW, GemmType.M_GROUPED_TIGHT_PER_ROW)
# The one type whose A carries no grouping in its rows: it may be MN-major and
# takes the widest tiles.
# One full tile grid per group, and no M-grouping to constrain the tile.
_FLAT_LIKE_TYPES = (GemmType.BATCHED, GemmType.K_GROUPED_CONTIGUOUS)


# Gated activations the epilogue can fuse: B stacks gate and up along N; a tile's B
# half-loads block_n / 2 gate columns and the matching up columns, one accumulator
# holds both, and the epilogue stores act(gate) * up, so C has N / 2 columns.
ACTIVATIONS = ("none", "silu_and_mul", "gelu_and_mul")


class Major(str, enum.Enum):
    """Which logical dim is contiguous in memory for an operand."""

    K = "k"
    MN = "mn"


@dataclasses.dataclass(frozen=True)
class SM90GemmSpec:
    """One instantiation of the kernel template.

    Every field is a compile-time parameter of the kernel; two specs that
    differ in any field are two distinct compiled kernels. ``shape_m`` /
    ``shape_n`` / ``shape_k`` are ``0`` for a dim left dynamic, which is
    DeepGEMM's ``SHAPE_* == 0`` convention.
    """

    gemm_type: GemmType
    major_a: Major
    major_b: Major
    ab_dtype: str
    cd_dtype: str
    num_groups: int
    shape_m: int
    shape_n: int
    shape_k: int
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_math_threads: int
    num_sms: int
    activation: str = "none"

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {self.activation!r}")
        if self.activation != "none" and self.major_b is not Major.K:
            raise ValueError(
                "a fused gated activation half-loads the B tile; an MN-major B would split "
                "the 128-byte swizzle atom, so it takes a K-major B"
            )
        if self.activation != "none" and self.gemm_type is GemmType.K_GROUPED_CONTIGUOUS:
            raise ValueError("a fused gated activation needs a per-group B; K-grouped has one B")
        if self.ab_dtype not in ("bfloat16", "float16"):
            raise ValueError(f"ab_dtype must be bfloat16 or float16, got {self.ab_dtype!r}")
        if self.cd_dtype not in (self.ab_dtype, "float32"):
            raise ValueError(
                f"cd_dtype must be the operand dtype or float32, got {self.cd_dtype!r} "
                f"for {self.ab_dtype} operands"
            )
        if self.block_m not in (64, 128, 256):
            raise ValueError(f"block_m must be 64, 128 or 256, got {self.block_m}")
        if self.block_n % 8 or not 8 <= self.block_n <= 256:
            raise ValueError(f"block_n must be a multiple of 8 in [8, 256], got {self.block_n}")
        if self.block_k not in (_BLOCK_K, 2 * _BLOCK_K):
            # One or two 128-byte swizzle atoms per operand row.
            raise ValueError(f"block_k must be {_BLOCK_K} or {2 * _BLOCK_K}, got {self.block_k}")
        if self.num_math_threads != (128 if self.block_m <= 64 else 256):
            raise ValueError(
                "num_math_threads is 128 for block_m <= 64 and 256 otherwise, "
                f"got {self.num_math_threads} for block_m={self.block_m}"
            )
        if self.num_stages < 2:
            # The mainloop keeps one WGMMA group in flight and releases a stage
            # one k-step late, so with a single stage producer and consumer wait
            # on each other for ever.
            raise ValueError("num_stages must be at least 2")
        if self.num_sms < 1 or min(self.shape_m, self.shape_n, self.shape_k) < 0:
            raise ValueError("num_sms must be positive and static dims non-negative")
        if self.gemm_type not in _FLAT_LIKE_TYPES and self.major_a is not Major.K:
            raise ValueError("m-grouped GEMM requires a K-major A")
        if self.num_groups < 1:
            raise ValueError("num_groups must be positive")
        if self.block_m > 128 and self.block_n > 128:
            raise ValueError("block_m and block_n cannot both exceed 128 (register budget)")
        if self.block_m > 128 and self.cd_dtype == "float32":
            raise ValueError("block_m=256 is only offered for a 16-bit output")

    @property
    def num_math_warpgroups(self) -> int:
        return self.num_math_threads // 128


@dataclasses.dataclass(frozen=True)
class GemmDesc:
    """What a call asks for; the selector's input.

    ``static_dims`` names the dims baked into the kernel (DeepGEMM's
    ``compiled_dims``); the others stay dynamic. ``expected_m`` feeds the cost
    model only and defaults to the call's own ``m``; for the per-group layouts
    it is the rows per group the model should plan for, which a caller who
    knows the routing is skewed sets to the large experts' size rather than
    the mean. ``m_alignment`` is the segment alignment of an m-grouped layout
    and fixes ``block_m`` for it.
    """

    gemm_type: GemmType
    m: int
    n: int
    k: int
    num_groups: int
    major_a: Major
    major_b: Major
    cd_dtype: str
    num_sms: int
    static_dims: str = "nk"
    ab_dtype: str = "bfloat16"
    m_alignment: int = 128
    expected_m: int = 0
    activation: str = "none"

    @property
    def fused(self) -> bool:
        return self.activation != "none"

    @property
    def c_cols(self) -> int:
        """Columns of C: half of N when the gated activation is fused."""
        return self.n // 2 if self.fused else self.n

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {self.activation!r}")
        if any(c not in "mnk" for c in self.static_dims):
            raise ValueError(f"static_dims may only name m, n, k; got {self.static_dims!r}")
        if self.num_sms < 1:
            raise ValueError("the persistent grid needs at least one SM")

    def get_expected_m(self) -> int:
        return self.expected_m if self.expected_m > 0 else self.m


@dataclasses.dataclass(frozen=True)
class _Layout:
    block_m: int
    block_n: int
    block_k: int


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _num_stages(desc: GemmDesc, layout: _Layout) -> int:
    cd_bytes = 4 if desc.cd_dtype == "float32" else 2
    c_width = layout.block_n // 2 if desc.fused else layout.block_n
    smem_cd = _align(layout.block_m * c_width * cd_bytes, 1024)
    # s_cum + s_total, plus the recovered ends for the tight per-row layout.
    prefix_ints = desc.num_groups + 2
    if desc.gemm_type is GemmType.M_GROUPED_TIGHT_PER_ROW:
        prefix_ints += desc.num_groups
    smem_prefix = _align(prefix_ints * 4, 128)
    per_stage = (layout.block_m + layout.block_n) * layout.block_k * _ELEM_BYTES
    budget = _SMEM_CAPACITY - smem_cd - _BARRIER_BYTES - _SMEM_ALIGN_SLACK - smem_prefix
    return min(budget // per_stage, _MAX_STAGES)


def _num_math_threads(block_m: int) -> int:
    return 128 if block_m <= 64 else 256


def layout_candidates(desc: GemmDesc) -> list[_Layout]:
    """Every tile layout the template accepts for ``desc``.

    Mirrors ``SM90ArchSpec::get_layout_candidates``, minus the 16/32-row tiles
    and the 2-CTA clusters.
    """
    if desc.gemm_type in _FLAT_LIKE_TYPES:
        block_m_candidates = [64, 128]
        if desc.cd_dtype != "float32":
            block_m_candidates.append(256)
    elif desc.gemm_type in _ALIGNED_TYPES:
        if desc.m_alignment not in (64, 128, 256):
            raise ValueError(
                f"m-grouped block_m follows the segment alignment, which must be 64, 128 or "
                f"256; got {desc.m_alignment}"
            )
        block_m_candidates = [desc.m_alignment]
    else:
        # Masked and tight rows have no alignment to honour; short groups want 64.
        block_m_candidates = [64, 128]

    # DeepGEMM steps block_n by 16. Measured on H200 (see docs), every block_n
    # that is not a multiple of 64 runs 1.5x or more behind the best tile in
    # this TileLang epilogue, so those widths are not offered.
    block_n_candidates = list(range(_BLOCK_N_STEP, 256 + 1, _BLOCK_N_STEP))

    # The selector enumerates the 64-wide K block only. A 128-wide block (still
    # accepted from a pinned config) halves the barrier round trips and, cold,
    # was 2-3% faster on MoE decode shapes; under the sustained 700 W power cap
    # an H200 sits at within a second it lost 5-7%, because the tile it fits in
    # shared memory is half as wide and doubles the L2 re-reads of A.
    candidates = []
    for block_m in block_m_candidates:
        for block_n in block_n_candidates:
            if block_m > 128 and block_n > 128:
                continue
            layout = _Layout(block_m, block_n, _BLOCK_K)
            stages = _num_stages(desc, layout)
            if stages < 3 or (block_m * block_n < 128 * 192 and stages < 4):
                continue
            candidates.append(layout)
    if not candidates:
        raise ValueError(f"no legal layout for {desc}")
    return candidates


def _num_m_blocks(desc: GemmDesc, block_m: int) -> int:
    """M tiles the schedule will run, counting each group's round-up separately.

    DeepGEMM divides the expected rows by ``block_m`` once. A per-group layout
    rounds every group up on its own, and for short groups that round-up is
    most of the work: 32-row groups cost a full 128-row tile each. Counting it
    is what lets the selector prefer ``block_m=64`` there.
    """
    if desc.gemm_type in PER_GROUP_TYPES:
        if desc.gemm_type is GemmType.M_GROUPED_MASKED or desc.expected_m > 0:
            rows_per_group = desc.expected_m if desc.expected_m > 0 else desc.m
        else:
            rows_per_group = desc.m / max(1, desc.num_groups)
        return desc.num_groups * math.ceil(rows_per_group / block_m)
    return math.ceil(desc.get_expected_m() / block_m)


def _num_cycles(desc: GemmDesc, layout: _Layout) -> tuple[int, int]:
    """DeepGEMM's L1/L2 cycle model; returns ``(num_waves, cycles)``."""
    # Only a batched GEMM runs one full tile grid per group; the grouped
    # layouts are counted per group by _num_m_blocks instead.
    # A fused tile covers block_n / 2 output columns, i.e. block_n columns of B.
    num_blocks = (
        _num_m_blocks(desc, layout.block_m)
        * math.ceil(desc.n / layout.block_n)
        * (desc.num_groups if desc.gemm_type in _FLAT_LIKE_TYPES else 1)
    )
    num_waves = math.ceil(num_blocks / desc.num_sms)
    if num_blocks == 0:  # a call with no rows or no columns runs nothing
        return 0, 0

    l2_bandwidth_per_cycle = int(min(64.0 * desc.num_sms, 8e6 / 1.3e3))
    l1_bandwidth_per_cycle = 128 * desc.num_sms
    elem_ab = _ELEM_BYTES
    elem_cd = 4 if desc.cd_dtype == "float32" else 2

    k = desc.k
    if desc.gemm_type is GemmType.K_GROUPED_CONTIGUOUS:
        # The groups split K between them; a tile runs the mean group's contraction.
        k = math.ceil(desc.k / desc.num_groups)
    c_width = layout.block_n // 2 if desc.fused else layout.block_n
    bytes_l2_ab = k * (layout.block_m + layout.block_n) * elem_ab
    bytes_l1_ab = k * (layout.block_m + layout.block_n) * elem_ab
    bytes_l1_tc = k * (max(_WGMMA_M, layout.block_m) + layout.block_n) * elem_ab
    bytes_l1_tc += layout.block_m * c_width * elem_cd
    bytes_cd = layout.block_m * c_width * elem_cd

    l2_cycles = (bytes_l2_ab + bytes_cd) * num_blocks // l2_bandwidth_per_cycle
    l1_cycles = (bytes_l1_ab + bytes_l1_tc + bytes_cd) * num_blocks // l1_bandwidth_per_cycle
    wave_efficiency = num_blocks / (num_waves * desc.num_sms)
    return num_waves, int(max(l1_cycles, l2_cycles) / wave_efficiency)


def _best_layout(desc: GemmDesc, candidates: list[_Layout]) -> _Layout:
    """The tile the cycle model prefers."""
    return min(candidates, key=lambda lay: _num_cycles(desc, lay)[1])


def _spec(desc: GemmDesc, layout: _Layout, num_stages: int) -> SM90GemmSpec:
    return SM90GemmSpec(
        gemm_type=desc.gemm_type,
        major_a=desc.major_a,
        major_b=desc.major_b,
        ab_dtype=desc.ab_dtype,
        cd_dtype=desc.cd_dtype,
        num_groups=desc.num_groups,
        shape_m=desc.m if "m" in desc.static_dims else 0,
        shape_n=desc.n if "n" in desc.static_dims else 0,
        shape_k=desc.k if "k" in desc.static_dims else 0,
        block_m=layout.block_m,
        block_n=layout.block_n,
        block_k=layout.block_k,
        num_stages=num_stages,
        num_math_threads=_num_math_threads(layout.block_m),
        num_sms=desc.num_sms,
        activation=desc.activation,
    )


@functools.lru_cache(maxsize=1024)
def get_best_config(desc: GemmDesc) -> SM90GemmSpec:
    """The spec DeepGEMM's selector would pick for ``desc``.

    Cached per descriptor: the enumeration is a few hundred candidates and a
    call site typically repeats a handful of shapes.
    """
    best = _best_layout(desc, layout_candidates(desc))
    return _spec(desc, best, _num_stages(desc, best))


def spec_from_config(desc: GemmDesc, config: dict) -> SM90GemmSpec:
    """A spec from an explicit ``config`` instead of the selector.

    ``config`` names ``block_m``, ``block_n`` and optionally ``block_k`` and
    ``num_stages``. A missing ``num_stages`` takes the shared-memory budget's
    maximum for that tile, so a tuning sweep can pin the tile and leave the
    pipeline depth alone; a pinned one past the budget is refused rather than
    launched.
    """
    unknown = set(config) - {"block_m", "block_n", "block_k", "num_stages"}
    if unknown:
        raise ValueError(f"config names no such template parameter: {sorted(unknown)}")
    layout = _Layout(config["block_m"], config["block_n"], config.get("block_k", _BLOCK_K))
    if desc.gemm_type in _ALIGNED_TYPES and layout.block_m != desc.m_alignment:
        raise ValueError(
            f"an aligned layout's block_m is its segment alignment: the kernel rounds group "
            f"starts up to block_m and reads a tile's group off its first row, so a block_m of "
            f"{layout.block_m} does not fit m_alignment={desc.m_alignment}"
        )
    max_stages = _num_stages(desc, layout)
    stages = config.get("num_stages", max_stages)
    if not 2 <= stages <= max_stages:
        raise ValueError(
            f"a {layout.block_m}x{layout.block_n}x{layout.block_k} tile fits at most "
            f"{max_stages} stages in shared memory, got num_stages={stages}; the pipeline "
            "needs at least 2"
        )
    return _spec(desc, layout, stages)
