"""Template parameters and config selection for the SM90 GEMM template.

Ported from DeepGEMM: the kernel template ``sm90_bf16_gemm_impl`` takes every
structural choice as a template parameter, and the host side (``SM90ArchSpec``
+ ``get_best_config``) enumerates the legal layouts for a call and scores them
with an L1/L2 cycle model instead of benchmarking. ``SM90GemmSpec`` is the
TileOPs counterpart of that template parameter pack, and ``get_best_config`` is
the counterpart of the selector.

Deviations from DeepGEMM, all deliberate:

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
    "PER_GROUP_TYPES",
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
# H200 calibration, 2026-09: a 2-CTA cluster (TMA multicast of one operand) is
# added to the tile the plain model picked, only for a dense GEMM, and only
# when that tile runs at least this many waves. DeepGEMM lets clusters compete
# for the tile choice and rejects them below 2 waves. Measured on H200, a
# cluster on the best plain dense tile gained 0.6% to 5% at 16 waves and more
# and lost 1% to 6% at 8 waves and fewer; letting it pick the tile instead
# moved the choice to 128x128 tiles whose halved L2 traffic the model
# overvalues, 8% to 15% behind. On grouped GEMMs, which stream every expert's
# B from HBM, the pairing gained at most 1.5% and lost 8.5% on an 8-expert
# 4096-row case, so they never take one.
_MIN_WAVES_FOR_CLUSTER = 16


class GemmType(str, enum.Enum):
    """Which rows of ``A`` and which ``B`` a tile reads.

    * ``NORMAL``: one dense GEMM.
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
    * ``M_GROUPED_MASKED``: DeepGEMM's ``MGroupedMasked``. ``A`` and ``C`` carry a
      leading group dim of ``max_m`` rows each and ``grouped_layout[g]`` is the
      valid row count.
    * ``BATCHED``: independent GEMMs on a leading batch dim of ``A``, ``B``, ``C``.
    """

    NORMAL = "normal"
    M_GROUPED_ALIGNED_PER_ROW = "m_grouped_aligned_per_row"
    M_GROUPED_ALIGNED_PSUM = "m_grouped_aligned_psum"
    M_GROUPED_TIGHT_PSUM = "m_grouped_tight_psum"
    M_GROUPED_MASKED = "m_grouped_masked"
    BATCHED = "batched"


_ALIGNED_TYPES = (GemmType.M_GROUPED_ALIGNED_PER_ROW, GemmType.M_GROUPED_ALIGNED_PSUM)
PER_GROUP_TYPES = (
    GemmType.M_GROUPED_MASKED,
    GemmType.M_GROUPED_ALIGNED_PSUM,
    GemmType.M_GROUPED_TIGHT_PSUM,
)
_FLAT_LIKE_TYPES = (GemmType.NORMAL, GemmType.BATCHED)
# Only a dense GEMM may take a 2-CTA cluster; see _MIN_WAVES_FOR_CLUSTER. Offering
# one to a per-group type would also need the CTA pair kept inside one group,
# i.e. an even ceil(N / block_n), the prune DeepGEMM applies there.
_CLUSTER_TYPES = (GemmType.NORMAL,)


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
    num_tma_multicast: int
    is_tma_multicast_on_a: bool
    num_sms: int

    def __post_init__(self) -> None:
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
        if self.block_k % 64:
            raise ValueError(f"block_k must be a multiple of 64, got {self.block_k}")
        if self.num_math_threads != (128 if self.block_m <= 64 else 256):
            raise ValueError(
                "num_math_threads is 128 for block_m <= 64 and 256 otherwise, "
                f"got {self.num_math_threads} for block_m={self.block_m}"
            )
        if self.num_tma_multicast not in (1, 2):
            raise ValueError("num_tma_multicast must be 1 or 2")
        if self.num_tma_multicast > 1 and self.num_sms % 2:
            raise ValueError("TMA multicast needs an even persistent grid")
        if self.gemm_type not in _FLAT_LIKE_TYPES and self.major_a is not Major.K:
            raise ValueError("m-grouped GEMM requires a K-major A")
        if self.gemm_type is GemmType.NORMAL and self.num_groups != 1:
            raise ValueError("a normal GEMM has exactly one group")
        if self.gemm_type is GemmType.BATCHED and self.num_tma_multicast > 1:
            raise ValueError("the batched scheduler does not multicast")
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

    def __post_init__(self) -> None:
        if any(c not in "mnk" for c in self.static_dims):
            raise ValueError(f"static_dims may only name m, n, k; got {self.static_dims!r}")
        if self.num_sms % 2:
            raise ValueError("the persistent grid must have an even SM count")

    def get_expected_m(self) -> int:
        return self.expected_m if self.expected_m > 0 else self.m


@dataclasses.dataclass(frozen=True)
class _Layout:
    block_m: int
    block_n: int
    block_k: int
    cluster_m: int
    cluster_n: int

    @property
    def cluster_size(self) -> int:
        return self.cluster_m * self.cluster_n


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _num_stages(desc: GemmDesc, layout: _Layout) -> int:
    cd_bytes = 4 if desc.cd_dtype == "float32" else 2
    smem_cd = _align(layout.block_m * layout.block_n * cd_bytes, 1024)
    smem_prefix = _align((desc.num_groups + 2) * 4, 128)
    per_stage = (layout.block_m + layout.block_n) * layout.block_k * _ELEM_BYTES
    budget = _SMEM_CAPACITY - smem_cd - _BARRIER_BYTES - _SMEM_ALIGN_SLACK - smem_prefix
    return min(budget // per_stage, _MAX_STAGES)


def _num_math_threads(block_m: int) -> int:
    return 128 if block_m <= 64 else 256


def layout_candidates(desc: GemmDesc) -> list[_Layout]:
    """Every tile/cluster layout the template accepts for ``desc``.

    Mirrors ``SM90ArchSpec::get_layout_candidates``, minus the 16/32-row tiles.
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
    block_k = _BLOCK_K
    max_cluster = 2 if desc.gemm_type in _CLUSTER_TYPES else 1

    candidates = []
    for cluster_m in (1, 2):
        for cluster_n in (1, 2):
            cluster = cluster_m * cluster_n
            if cluster > max_cluster or desc.num_sms % cluster:
                continue
            for block_m in block_m_candidates:
                for block_n in block_n_candidates:
                    if block_m > 128 and block_n > 128:
                        continue
                    layout = _Layout(block_m, block_n, block_k, cluster_m, cluster_n)
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
    num_blocks = (
        _num_m_blocks(desc, layout.block_m)
        * math.ceil(desc.n / layout.block_n)
        * (desc.num_groups if desc.gemm_type is GemmType.BATCHED else 1)
    )
    num_waves = math.ceil(num_blocks / desc.num_sms)

    l2_bandwidth_per_cycle = int(min(64.0 * desc.num_sms, 8e6 / 1.3e3))
    l1_bandwidth_per_cycle = 128 * desc.num_sms
    elem_ab = _ELEM_BYTES
    elem_cd = 4 if desc.cd_dtype == "float32" else 2

    k = desc.k
    bytes_l2_ab = k * (layout.block_m // layout.cluster_n + layout.block_n // layout.cluster_m)
    bytes_l2_ab *= elem_ab
    bytes_l1_ab = k * (layout.block_m + layout.block_n) * elem_ab
    bytes_l1_tc = k * (max(_WGMMA_M, layout.block_m) + layout.block_n) * elem_ab
    bytes_l1_tc += layout.block_m * layout.block_n * elem_cd
    bytes_cd = layout.block_m * layout.block_n * elem_cd

    l2_cycles = (bytes_l2_ab + bytes_cd) * num_blocks // l2_bandwidth_per_cycle
    l1_cycles = (bytes_l1_ab + bytes_l1_tc + bytes_cd) * num_blocks // l1_bandwidth_per_cycle
    wave_efficiency = num_blocks / (num_waves * desc.num_sms)
    cycles = int(max(l1_cycles, l2_cycles) / wave_efficiency)

    if layout.cluster_size > 1 and num_waves < _MIN_WAVES_FOR_CLUSTER:
        cycles = 2**62
    return num_waves, cycles


def _best_layout(desc: GemmDesc, candidates: list[_Layout]) -> _Layout:
    """The plain (single-CTA) tile the model prefers, with a cluster added on top.

    Tile and cluster are decided in two steps rather than one ranking: the
    cluster variants of the chosen tile compete among themselves only, so
    multicast never changes which tile runs.
    """
    plain = [lay for lay in candidates if lay.cluster_size == 1]
    best = min(plain, key=lambda lay: _num_cycles(desc, lay)[1])
    clustered = [
        lay
        for lay in candidates
        if lay.cluster_size > 1
        and (lay.block_m, lay.block_n, lay.block_k) == (best.block_m, best.block_n, best.block_k)
    ]
    if clustered:
        with_cluster = min(clustered, key=lambda lay: _num_cycles(desc, lay)[1])
        if _num_cycles(desc, with_cluster)[1] < _num_cycles(desc, best)[1]:
            return with_cluster
    return best


def _merge_stages(spec: SM90GemmSpec) -> SM90GemmSpec:
    """Widen ``block_k`` out of surplus stages, DeepGEMM's ``kDoMergeStages``.

    Fewer, longer stages cut the per-stage barrier traffic of a single math
    warp-group on a dense NT GEMM; kept as DeepGEMM has it, although this
    kernel already overlaps a k-step's drain with the next (``wait_group 1``).
    """
    merge = (
        spec.num_stages >= 10
        and spec.gemm_type is GemmType.NORMAL
        and spec.major_a is Major.K
        and spec.major_b is Major.K
        and spec.num_math_threads == 128
    )
    if not merge:
        return spec
    per_merge = spec.num_stages // 5
    return dataclasses.replace(
        spec, block_k=spec.block_k * per_merge, num_stages=spec.num_stages // per_merge
    )


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
        num_tma_multicast=layout.cluster_size,
        is_tma_multicast_on_a=layout.cluster_n > 1,
        num_sms=desc.num_sms,
    )


@functools.lru_cache(maxsize=1024)
def get_best_config(desc: GemmDesc) -> SM90GemmSpec:
    """The spec DeepGEMM's selector would pick for ``desc``.

    Cached per descriptor: the enumeration is a few hundred candidates and a
    call site typically repeats a handful of shapes.
    """
    best = _best_layout(desc, layout_candidates(desc))
    return _merge_stages(_spec(desc, best, _num_stages(desc, best)))


def spec_from_config(desc: GemmDesc, config: dict) -> SM90GemmSpec:
    """A spec from an explicit ``config`` instead of the selector.

    ``config`` names ``block_m``, ``block_n`` and optionally ``block_k``,
    ``num_stages``, ``cluster_m``, ``cluster_n``. Missing fields take the
    selector's derivation for that layout, so a tuning sweep can pin the tile
    and leave the pipeline depth to the shared-memory budget.
    """
    layout = _Layout(
        config["block_m"],
        config["block_n"],
        config.get("block_k", _BLOCK_K),
        config.get("cluster_m", 1),
        config.get("cluster_n", 1),
    )
    stages = config.get("num_stages")
    if stages is None:
        stages = _num_stages(desc, dataclasses.replace(layout, block_k=_BLOCK_K))
        stages = max(1, stages * _BLOCK_K // layout.block_k)
    return _spec(desc, layout, stages)
