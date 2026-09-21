"""Configuration and selection policy for ``GemmTemplate``.

``GroupedGemmSpec`` contains compile-time kernel parameters; ``GemmDesc``
describes one call and drives tile selection.
"""

import dataclasses
import enum
import functools
import math

from tileops.utils import is_h200_name

__all__ = [
    "ACTIVATIONS",
    "PER_GROUP_TYPES",
    "PER_ROW_TYPES",
    "GemmDesc",
    "GemmType",
    "Major",
    "GroupedGemmSpec",
    "get_best_config",
    "layout_candidates",
    "spec_from_config",
]


@dataclasses.dataclass(frozen=True)
class _HeuristicPolicy:
    """The constants the selector reads, in three kinds a reader must tell apart.

    Hardware facts describe an SM90 device. Fitted values are only as good as
    the measurement behind them; refitting one means forcing every legal config
    of a descriptor through ``spec_from_config``, checking it against a
    reference, and timing the candidate set interleaved in one repeat loop in a
    fresh order each round -- medians of seven rounds on an idle device. The
    exclusion is neither: it patches a tile the cost function misprices.
    """

    # Hardware, SM90.
    smem_capacity: int = 232448  # shared memory one CTA can be given
    wgmma_m: int = 64
    element_bytes: int = 2
    max_stages: int = 16
    smem_alignment_slack: int = 6 * 1024

    # The candidate set `layout_candidates` enumerates.
    block_k: int = 64
    block_n_step: int = 64
    block_n_max: int = 256

    # Fitted: depth to reach, depth past which not to bother, wave count past
    # which the store rounds are all that is left.
    staged_epilogue_stages: int = 4
    staged_epilogue_depth_cap: int = 6
    staged_epilogue_wave_limit: int = 20

    # Fitted: a shallow ring only costs where both hold -- few waves, and too
    # few tiles to fill the SMs.
    shallow_wave_limit: int = 4
    shallow_tiles_per_sm: float = 2.5
    hiding_stages: int = 4

    # Fitted: the band where one tile beats whatever the cost function scores.
    short_group_rows: int = 32
    short_group_min_k: int = 1024
    short_group_unfused_max_n: int = 5120
    short_group_tile: tuple[int, int, int] = (64, 128, 128)

    # A patch, not a fit: the cost function prices a tile by block_m + block_n
    # and so reads a 192-wide tile as cheap, while its WGMMA issues twice.
    block_n_excluded: tuple[int, ...] = (192,)

    @property
    def barrier_bytes(self) -> int:
        return self.max_stages * 8 * 2


#: One instance for the process: nothing varies it per call.
_POLICY = _HeuristicPolicy()


class GemmType(str, enum.Enum):
    """Which rows of ``A`` and which ``B`` a tile reads.

    * ``DENSE``: one ordinary 2-D GEMM, with no group metadata.
    * ``M_GROUPED_ALIGNED_PER_ROW``: rows are
      packed per group into segments whose length is a multiple of ``block_m``,
      ``grouped_layout[row]`` names the group of each row, and rows past the
      last group carry ``num_groups``.
    * ``M_GROUPED_ALIGNED_PSUM``:
      ``grouped_layout[g]`` is the prefix-sum end row of group ``g``, and each
      group starts at the previous end rounded up to ``block_m``.
    * ``M_GROUPED_TIGHT_PSUM``: TileOPs' ``tight_physical_psum``. As the aligned
      psum layout but each group starts exactly at the previous end, so a
      group's last tile is stored with a row mask.
    * ``M_GROUPED_TIGHT_PER_ROW``: tight rows described by a non-decreasing
      group id per row rather than prefix-sum ends.
    * ``M_GROUPED_MASKED``: ``A`` and ``C`` carry a
      leading group dim of ``max_m`` rows each and ``grouped_layout[g]`` is the
      valid row count.
    * ``BATCHED``: independent GEMMs on a leading batch dim of ``A``, ``B``, ``C``.
    * ``K_GROUPED_CONTIGUOUS``: the groups are
      packed along ``K``: ``A`` is ``[M, sum_k]``, ``B`` is ``[N, sum_k]``,
      ``grouped_layout[g]`` is group ``g``'s ``K`` and ``C`` is ``[G, M, N]``. The
      weight-gradient GEMM of an expert MLP: each group contracts over its own
      tokens, its last K block is masked in shared memory where it runs into the
      next group, and a group with no tokens stores zeros.
    """

    DENSE = "dense"
    M_GROUPED_ALIGNED_PER_ROW = "m_grouped_aligned_per_row"
    M_GROUPED_ALIGNED_PSUM = "m_grouped_aligned_psum"
    M_GROUPED_TIGHT_PSUM = "m_grouped_tight_psum"
    M_GROUPED_TIGHT_PER_ROW = "m_grouped_tight_per_row"
    M_GROUPED_MASKED = "m_grouped_masked"
    BATCHED = "batched"
    K_GROUPED_CONTIGUOUS = "k_grouped_contiguous"


_ALIGNED_TYPES = (GemmType.M_GROUPED_ALIGNED_PER_ROW, GemmType.M_GROUPED_ALIGNED_PSUM)
_TIGHT_TYPES = (GemmType.M_GROUPED_TIGHT_PER_ROW, GemmType.M_GROUPED_TIGHT_PSUM)
_CALIBRATED_PSUM_TYPES = (
    GemmType.M_GROUPED_TIGHT_PSUM,
    GemmType.M_GROUPED_ALIGNED_PSUM,
)
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
_FLAT_LIKE_TYPES = (GemmType.DENSE, GemmType.BATCHED, GemmType.K_GROUPED_CONTIGUOUS)


# Gated activations the epilogue can fuse: B stacks gate and up along N; a tile's B
# half-loads block_n / 2 gate columns and the matching up columns, one accumulator
# holds both, and the epilogue stores act(gate) * up, so C has N / 2 columns.
ACTIVATIONS = ("none", "silu_and_mul", "gelu_and_mul")


class Major(str, enum.Enum):
    """Which logical dim is contiguous in memory for an operand."""

    K = "k"
    MN = "mn"


@dataclasses.dataclass(frozen=True)
class GroupedGemmSpec:
    """One instantiation of the kernel template.

    Every field is a compile-time parameter of the kernel; two specs that
    differ in any field are two distinct compiled kernels. ``shape_m`` /
    ``shape_n`` / ``shape_k`` are ``0`` for a dimension left dynamic.
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
    epilogue_stage_n: int = 0
    swizzle_group_m: int = 0

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {self.activation!r}")
        c_tile_n = self.block_n // 2 if self.activation != "none" else self.block_n
        if self.epilogue_stage_n < 0:
            raise ValueError("epilogue_stage_n must be non-negative")
        if self.epilogue_stage_n:
            if self.activation != "none":
                raise ValueError(
                    "epilogue_stage_n takes an unfused GEMM: a fused epilogue writes the whole "
                    "tile into shared memory at once, so its output cannot leave in chunks"
                )
            if c_tile_n % self.epilogue_stage_n:
                raise ValueError("epilogue_stage_n must divide the output tile width")
        if self.swizzle_group_m and self.gemm_type is not GemmType.DENSE:
            raise ValueError("swizzle_group_m only supports dense GEMM")
        if self.swizzle_group_m < 0:
            raise ValueError("swizzle_group_m must be non-negative")
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
        if self.block_k not in (64, 128):
            raise ValueError(f"block_k must be 64 or 128, got {self.block_k}")
        if self.num_math_threads not in (128, 256):
            raise ValueError("num_math_threads must be 128 or 256")
        if self.block_m // self.num_math_warpgroups not in (64, 128):
            raise ValueError(
                f"num_math_threads={self.num_math_threads} makes each math warpgroup own "
                f"{self.block_m // self.num_math_warpgroups} rows; expected 64 or 128"
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

    ``static_dims`` names the dimensions baked into the kernel; the others stay
    dynamic. ``expected_m`` feeds the cost
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
    device_name: str = ""

    @property
    def policy(self) -> _HeuristicPolicy:
        """The selector's constants. A module singleton, not per-descriptor state.

        A field would join this frozen dataclass's equality, and so the key of
        ``get_best_config``'s cache and of anything comparing two descriptors --
        which silently makes descriptors from two builds of the selector
        incomparable.
        """
        return _POLICY

    @property
    def fused(self) -> bool:
        return self.activation != "none"

    @property
    def h200(self) -> bool:
        """Whether the bands fitted on H200 apply to this device.

        Read through :func:`tileops.utils.is_h200_name`, so a band and the
        selection that routes work to it agree on every H200 SKU.
        """
        return is_h200_name(self.device_name)

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


@dataclasses.dataclass(frozen=True)
class _LayoutFeatures:
    """What one candidate tile does with one call, before any of it is priced.

    Everything here follows from the descriptor and the tile; a cost model reads
    these and adds the constants. The byte fields are whole-call totals over
    ``num_blocks`` tiles, so a per-tile figure divides by that count.
    """

    num_blocks: int

    # Waves of the persistent grid, the share of a wave's block slots that are
    # filled, and the tiles an SM runs.
    num_waves: int
    wave_efficiency: float
    tiles_per_sm: float

    l1_bytes: int
    l2_bytes: int
    num_stages: int


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _num_stages(desc: GemmDesc, layout: _Layout, epilogue_stage_n: int = 0) -> int:
    cd_bytes = 4 if desc.cd_dtype == "float32" else 2
    c_width = epilogue_stage_n or (layout.block_n // 2 if desc.fused else layout.block_n)
    smem_cd = _align(layout.block_m * c_width * cd_bytes, 1024)
    prefix_ints = desc.num_groups + 2
    if desc.gemm_type is GemmType.M_GROUPED_TIGHT_PER_ROW:
        prefix_ints += desc.num_groups
    smem_prefix = _align(prefix_ints * 4, 128)
    policy = desc.policy
    per_stage = (layout.block_m + layout.block_n) * layout.block_k * policy.element_bytes
    budget = (
        policy.smem_capacity
        - smem_cd
        - policy.barrier_bytes
        - (
            0
            if epilogue_stage_n and desc.gemm_type in (GemmType.DENSE, GemmType.BATCHED)
            else policy.smem_alignment_slack
        )
        - smem_prefix
    )
    return min(budget // per_stage, policy.max_stages)


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

    policy = desc.policy
    block_n_candidates = list(
        range(policy.block_n_step, policy.block_n_max + 1, policy.block_n_step)
    )
    if desc.gemm_type in _CALIBRATED_PSUM_TYPES:
        block_n_candidates = [bn for bn in block_n_candidates if bn not in policy.block_n_excluded]

    candidates = []
    for block_m in block_m_candidates:
        for block_n in block_n_candidates:
            if block_m > 128 and block_n > 128:
                continue
            layout = _Layout(block_m, block_n, policy.block_k)
            stages = _num_stages(desc, layout)
            if stages < 3 or (block_m * block_n < 128 * 192 and stages < 4):
                continue
            candidates.append(layout)
    if not candidates:
        raise ValueError(f"no legal layout for {desc}")
    return candidates


def _num_m_blocks(desc: GemmDesc, block_m: int) -> int:
    """M tiles the schedule will run, counting each group's round-up separately.

    A per-group layout rounds every group independently because tiles cannot
    cross group boundaries.
    """
    if desc.gemm_type in PER_GROUP_TYPES:
        if desc.gemm_type is GemmType.M_GROUPED_MASKED or desc.expected_m > 0:
            rows_per_group = desc.expected_m if desc.expected_m > 0 else desc.m
        else:
            rows_per_group = desc.m / max(1, desc.num_groups)
        return desc.num_groups * math.ceil(rows_per_group / block_m)
    return math.ceil(desc.get_expected_m() / block_m)


def _layout_features(desc: GemmDesc, layout: _Layout) -> _LayoutFeatures:
    """What ``layout`` makes this call do. Counted, not priced: nothing here is fitted."""
    num_m_blocks = _num_m_blocks(desc, layout.block_m)
    num_n_blocks = math.ceil(desc.n / layout.block_n)
    num_blocks = (
        num_m_blocks * num_n_blocks * (desc.num_groups if desc.gemm_type in _FLAT_LIKE_TYPES else 1)
    )
    num_waves = math.ceil(num_blocks / desc.num_sms)

    effective_k = desc.k
    if desc.gemm_type is GemmType.K_GROUPED_CONTIGUOUS:
        # The groups split K between them; a tile runs the mean group's contraction.
        effective_k = math.ceil(desc.k / desc.num_groups)

    elem_ab = desc.policy.element_bytes
    elem_cd = 4 if desc.cd_dtype == "float32" else 2
    c_width = layout.block_n // 2 if desc.fused else layout.block_n
    bytes_l2_ab = effective_k * (layout.block_m + layout.block_n) * elem_ab
    bytes_l1_ab = effective_k * (layout.block_m + layout.block_n) * elem_ab
    bytes_l1_tc = effective_k * (max(desc.policy.wgmma_m, layout.block_m) + layout.block_n)
    bytes_l1_tc *= elem_ab
    bytes_l1_tc += layout.block_m * c_width * elem_cd
    bytes_cd = layout.block_m * c_width * elem_cd

    return _LayoutFeatures(
        num_blocks=num_blocks,
        num_waves=num_waves,
        wave_efficiency=num_blocks / (num_waves * desc.num_sms) if num_blocks else 0.0,
        tiles_per_sm=num_blocks / desc.num_sms,
        l1_bytes=(bytes_l1_ab + bytes_l1_tc + bytes_cd) * num_blocks,
        l2_bytes=(bytes_l2_ab + bytes_cd) * num_blocks,
        num_stages=_num_stages(desc, layout),
    )


def _tile_cycles(desc: GemmDesc, layout: _Layout, features: _LayoutFeatures) -> int:
    """Estimated cycles: the bandwidth a wave moves, over how full that wave is."""
    if features.num_blocks == 0:  # a call with no rows or no columns runs nothing
        return 0

    l2_bandwidth_per_cycle = int(min(64.0 * desc.num_sms, 8e6 / 1.3e3))
    l1_bandwidth_per_cycle = 128 * desc.num_sms
    l2_cycles = features.l2_bytes // l2_bandwidth_per_cycle
    l1_cycles = features.l1_bytes // l1_bandwidth_per_cycle
    cycles = max(l1_cycles, l2_cycles) / features.wave_efficiency

    # Over many waves a tile's TMA latency hides under the next tile, and the
    # bandwidth terms above decide. Over few waves *on a device the tiles do not
    # fill* there is no next tile: an SM runs one or two, and what it can overlap
    # is its own pipeline, so a shallow ring stalls however little it moves. The
    # model prices no pipeline, which is why it reads the widest tile -- the one
    # whose output buffer leaves room for three stages -- as cheapest exactly
    # where it measures slowest.
    if (
        desc.gemm_type in _CALIBRATED_PSUM_TYPES
        and features.num_waves < desc.policy.shallow_wave_limit
        and features.tiles_per_sm < desc.policy.shallow_tiles_per_sm
    ):
        deficit = max(0, desc.policy.hiding_stages - features.num_stages)
        cycles *= 1.0 + deficit / desc.policy.hiding_stages
    return int(cycles)


def _best_layout(desc: GemmDesc, candidates: list[_Layout]) -> _Layout:
    """The tile the cycle model prefers."""
    return min(candidates, key=lambda lay: _tile_cycles(desc, lay, _layout_features(desc, lay)))


def _short_group_layout(desc: GemmDesc) -> _Layout | None:
    """The tile pinned for short tight groups, or ``None`` outside that band."""
    policy = desc.policy
    rows_per_group = math.ceil(desc.m / desc.num_groups)
    if (
        desc.h200
        and desc.gemm_type is GemmType.M_GROUPED_TIGHT_PSUM
        and desc.ab_dtype == desc.cd_dtype
        and desc.activation in ("none", "silu_and_mul")
        and rows_per_group <= policy.short_group_rows
        and desc.k >= policy.short_group_min_k
        and (desc.activation != "none" or desc.n <= policy.short_group_unfused_max_n)
    ):
        return _Layout(*policy.short_group_tile)
    return None


def _spec(
    desc: GemmDesc,
    layout: _Layout,
    num_stages: int,
    *,
    epilogue_stage_n: int = 0,
    swizzle_group_m: int = 0,
    num_math_wgs: int = 0,
) -> GroupedGemmSpec:
    return GroupedGemmSpec(
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
        num_math_threads=(
            128 * num_math_wgs if num_math_wgs else _num_math_threads(layout.block_m)
        ),
        num_sms=desc.num_sms,
        activation=desc.activation,
        epilogue_stage_n=epilogue_stage_n,
        swizzle_group_m=swizzle_group_m,
    )


def _staged_epilogue(desc: GemmDesc, layout: _Layout) -> GroupedGemmSpec | None:
    """A spec that trades a narrower output staging buffer for a deeper mainloop.

    Returns ``None`` where the trade buys no stage, or where the mainloop is deep
    enough in waves not to need one, and so only costs the extra staging rounds.
    The widest chunk that reaches the policy's depth wins.
    """
    policy = desc.policy
    if desc.activation != "none" or not desc.h200:
        return None
    if desc.gemm_type in _TIGHT_TYPES:
        # A tight group's last tile is ragged, and those rows are stored under a
        # row mask rather than in one wide store. Chunking makes them pay a
        # staging round each, for a store that was never going to widen; how many
        # tiles are ragged is a property of the routing, not of the shape.
        return None
    if _num_stages(desc, layout) >= policy.staged_epilogue_depth_cap:
        return None
    if (
        desc.gemm_type is GemmType.M_GROUPED_ALIGNED_PSUM
        and _layout_features(desc, layout).num_waves >= policy.staged_epilogue_wave_limit
    ):
        return None
    base = _num_stages(desc, layout)
    for stage_n in (layout.block_n // 2, layout.block_n // 4):
        stages = _num_stages(desc, layout, epilogue_stage_n=stage_n)
        if stages > base and stages >= policy.staged_epilogue_stages:
            return _spec(desc, layout, stages, epilogue_stage_n=stage_n)
    return None


@functools.lru_cache(maxsize=1024)
def get_best_config(desc: GemmDesc) -> GroupedGemmSpec:
    """Return the selected kernel spec for ``desc``."""
    best = _short_group_layout(desc) or _best_layout(desc, layout_candidates(desc))
    return _staged_epilogue(desc, best) or _spec(desc, best, _num_stages(desc, best))


def spec_from_config(desc: GemmDesc, config: dict) -> GroupedGemmSpec:
    """A spec from an explicit ``config`` instead of the selector.

    ``config`` names ``block_m``, ``block_n`` and optionally ``block_k`` and
    ``num_stages``. A missing ``num_stages`` takes the shared-memory budget's
    maximum for that tile, so a tuning sweep can pin the tile and leave the
    pipeline depth alone; a pinned one past the budget is refused rather than
    launched.
    """
    unknown = set(config) - {
        "block_m",
        "block_n",
        "block_k",
        "num_stages",
        "epilogue_stage_n",
        "swizzle_group_m",
        "num_math_wgs",
    }
    if unknown:
        raise ValueError(f"config names no such template parameter: {sorted(unknown)}")
    layout = _Layout(
        config["block_m"], config["block_n"], config.get("block_k", desc.policy.block_k)
    )
    if desc.gemm_type in _ALIGNED_TYPES and layout.block_m != desc.m_alignment:
        raise ValueError(
            f"an aligned layout's block_m is its segment alignment: the kernel rounds group "
            f"starts up to block_m and reads a tile's group off its first row, so a block_m of "
            f"{layout.block_m} does not fit m_alignment={desc.m_alignment}"
        )
    epilogue_stage_n = config.get("epilogue_stage_n", 0)
    max_stages = _num_stages(desc, layout, epilogue_stage_n)
    stages = config.get("num_stages", max_stages)
    if not 2 <= stages <= max_stages:
        raise ValueError(
            f"a {layout.block_m}x{layout.block_n}x{layout.block_k} tile fits at most "
            f"{max_stages} stages in shared memory, got num_stages={stages}; the pipeline "
            "needs at least 2"
        )
    return _spec(
        desc,
        layout,
        stages,
        epilogue_stage_n=epilogue_stage_n,
        swizzle_group_m=config.get(
            "swizzle_group_m", 16 if desc.gemm_type is GemmType.DENSE else 0
        ),
        num_math_wgs=config.get("num_math_wgs", 0),
    )
