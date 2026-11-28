"""Select SM90 dense-GEMM configurations without autotuning.

Candidates cover basic, split-K, cooperative, ping-pong and stream structures. The
scorer enumerates, prunes, and ranks them with board-calibrated throughput,
workspace, launch, and issue costs. It returns ``None`` on an uncalibrated
board so the serving kernel can use its default configuration.
"""

import functools
import math
from dataclasses import asdict, dataclass

from tileops._kernels.constants import BLOCK_SHARED_BYTES_OPT_IN, QUANT_SCALE_BLOCK, WGMMA_ROWS

__all__ = [
    "COOP2_SWAP_STAGING",
    "FP8_MIN_SLICE_K_TILES",
    "SWAP_AB_MPAD",
    "best_config",
    "fp8_ws_config",
    "gemv_config",
    "gemv_registers",
    "gemv_ring_copies",
    "gemv_unrolled",
    "int8_pingpong_grid",
    "int8_w8a8_config",
    "small_batch_config",
    "small_m_splitk_config",
    "stream_width",
    "two_row_gemv_wins",
]

SWAP_AB_MPAD = 8

# Shortest K slice, in 128-wide K tiles, an FP8 split-K launch pays for: a slice ends in an
# fp32 round trip through the workspace and a second launch.
FP8_MIN_SLICE_K_TILES = 12

# One coop2_swap consumer's output staging buffer: both 64-column halves of its 128 rows,
# ``store_m = 32`` columns of C at a time.
COOP2_SWAP_STAGING = (2, 32, 64)
# The deepest coop2_swap ring; only a ring this deep hides mixed-width tiles, and a
# shallower one takes its own score factor.
_COOP2_SWAP_FULL_RING = 4
# The basic tile widths; ``_fill_cand`` offers only a width outside them.
_BASIC_WIDTHS = (64, 128, 192, 256)


# The static mbarriers sit before the 1024-aligned dynamic shared buffer, so they take its
# first KiB.
_STATIC_SMEM_BYTES = 1024


def stream_width(m: int) -> tuple[int, bool]:
    """The stream tile's width and whether it runs the helper's own-width WGMMA.

    The width is A's rows rounded up to the 8-row WGMMA N. A width that does not divide 256 runs
    the helper's WGMMA over K steps of whole 64-element chunks, and its accumulator reaches
    the transposed output through a second, untransposed shared tile.
    """
    width = -(-m // 8) * 8
    return width, 256 % width != 0


def _ring_budget(staging: int, block_n: int) -> int:
    """Shared memory a basic or fill tile's operand ring may take.

    The ring shares the opt-in budget with the static mbarriers, the output staging tile of
    *staging* bytes and the 16-bit bias row the producer stages for a biased call,
    ``block_n`` wide. Selection does not know whether a call has a bias, so every ring
    leaves room for that row.
    """
    return BLOCK_SHARED_BYTES_OPT_IN[90] - _STATIC_SMEM_BYTES - staging - 2 * block_n


@dataclass(frozen=True)
class _TinyMCalibration:
    """Ranking constants of :func:`_tiny_m_score_us`, the m <= 8 NT weight-stream model.

    ``latency_us`` and ``cta_gbps`` set the rate one CTA streams its rows of B at,
    ``dram_tbps`` the rate the whole grid does, ``split_k_us`` is a split-K call's
    reduce, and ``gemv_tbps`` is the rate the GEMV streams B at. ``cta_gbps`` is fixed
    and the rest fitted, under :class:`_Calibration`'s objective.
    """

    latency_us: float
    cta_gbps: float
    split_k_us: float
    gemv_tbps: float
    dram_tbps: float


@dataclass(frozen=True)
class _StreamCalibration:
    """Ranking constants of the stream tile's weight-stream model in :func:`_score_us`.

    The grid streams ``B`` at ``min(CTAs * cta_gbps, dram_tbps)``. These two replace the
    stream structure's factor and are fitted with every other constant fixed.
    """

    cta_gbps: float
    dram_tbps: float


@dataclass(frozen=True)
class _Calibration:
    """The scorer's ranking constants for one board.

    Fitted ranking constants, not physical rates: ``l1_tbps`` and ``l2_tbps`` are not
    the board's SMEM and L2 bandwidths, so they stay here and not in ``perf/profiles/``.

    ``structure_factor`` scales a structure's whole score; a per-structure tensor-core
    rate would drop out of the ``max`` wherever bytes bound the shape.

    The fit minimizes mean ``log(busy / fastest busy)`` over all :func:`_enumerate`
    candidates under ``softmax(-log score / tau)``, with ``tensor_core_tflops`` and the
    basic factor fixed. A new structure fits only its own constants.
    """

    l1_tbps: float
    l2_tbps: float
    reduce_tbps: float
    issue_ns: float
    tensor_core_tflops: float
    structure_factor: tuple
    tiny_m: _TinyMCalibration
    stream: _StreamCalibration

    def factor(self, structure: str) -> float:
        return dict(self.structure_factor)[structure]


def _calibration(calibration: str | None) -> _Calibration | None:
    """The calibration measured on the board keyed *calibration*, or ``None`` for a board
    without one."""
    # One entry per board the scorer has been calibrated on, by its calibration key
    # (``tileops._utils.calibration_key``). A board absent here is not ranked; see
    # :func:`best_config`.
    calibrations = {
        "h200": _Calibration(
            l1_tbps=6.93,
            l2_tbps=10.0,
            reduce_tbps=0.141,
            issue_ns=819.0,
            tensor_core_tflops=420.0,
            structure_factor=(
                ("basic", 1.0),
                ("coop2", 1.0),
                ("coop2s", 0.125),
                ("coop2_splitk", 1.0),
                ("coop2_swap", 1.2),
                ("coop2_swap_3stage", 1.36),
                ("fill", 0.58),
                ("fill_one", 0.524),
                ("simple", 0.763),
                ("splitk", 0.904),
            ),
            tiny_m=_TinyMCalibration(
                latency_us=1.54,
                cta_gbps=100.0,
                split_k_us=2.6,
                gemv_tbps=4.19,
                dram_tbps=5.04,
            ),
            stream=_StreamCalibration(cta_gbps=15.9, dram_tbps=1.96),
        ),
    }
    return calibrations.get(calibration)


@dataclass
class _Cand:
    structure: str
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    split_k: int = 1
    stage_n: int = 0
    panel_size: int = 16
    cluster_m: int = 1
    blocks: int = 1
    # A coop2 tile offered only as the one-wave tile; see ``_enumerate``.
    one_wave: bool = False

    def to_config(self) -> dict:
        fields = asdict(self) | {"threads": 128}
        flag = {} if self.structure in _UNFLAGGED else {self.structure: True}
        return flag | {key: fields[key] for key in _CONFIG_KEYS[self.structure]}


# The structures the basic or split-K launch path runs, which carry no structure flag.
_UNFLAGGED = ("basic", "splitk", "fill")
# The config keys each structure's launch reads; a fill tile is a cluster launch without a
# panel swizzle (its panel_size is 0).
_BASIC_KEYS = ("block_m", "block_n", "block_k", "num_stages", "panel_size", "split_k")
_CONFIG_KEYS = {
    "basic": _BASIC_KEYS,
    "splitk": _BASIC_KEYS,
    "fill": (*_BASIC_KEYS, "cluster_m"),
    "simple": (
        "block_m",
        "block_n",
        "block_k",
        "num_stages",
        "threads",
        "panel_size",
        "cluster_m",
    ),
    "coop2": ("block_n", "block_k", "num_stages", "stage_n"),
    "coop2s": ("block_n", "block_k", "num_stages"),
    "coop2_swap": ("block_n", "num_stages"),
    "stream": ("block_k", "num_stages", "blocks"),
    "coop2_splitk": ("block_n", "block_k", "num_stages", "split_k"),
}


def _enumerate(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, sm_count: int, out_bytes: int = 2
) -> list:
    """The candidates :func:`_score_us` ranks for a call whose output elements take
    *out_bytes*, which each structure's output staging tile holds."""
    # Accumulator registers per thread a basic tile may hold: ``bm * bn / 128``.
    max_accum_regs = 200
    # Ring caps below the SMEM budget. Basic and split-K tiles take the deepest ring the
    # SMEM holds; the one-wave 64x192 basic tile sets its own ring below.
    ns_cap = {"simple": 4, "coop2": 4, "coop2_splitk": 4}

    def _strip_width(bm: int, bn: int, sm_count: int) -> int:
        """The panel width ``g`` minimizing the resident-CTA footprint
        ``g*bm + ceil(SMs/g)*bn``."""
        return min((4, 8, 10, 16), key=lambda g: g * bm + math.ceil(sm_count / g) * bn)

    def _ns_basic(bm: int, bn: int, bk: int) -> int:
        ring = (bm + bn) * bk * 2
        return _ring_budget(bm * bn * out_bytes, bn) // ring

    def _coop2_ns_sn(bn: int, bk: int, exposed: bool, deep: bool):
        """The deepest ring the SMEM budget allows, under ``ns_cap`` unless *deep*, and its
        epilogue staging chunk (stage_n, 0 for the whole tile). A hidden epilogue takes the
        widest chunk, shrunk only to buy a stage; an *exposed* one takes the chunk of the widest
        TMA swizzle, in two chunks or more so one chunk's store overlaps the next one's copy."""
        ring = (128 + bn) * bk * 2
        best = None
        for sn in range(8, bn + 1, 8) if exposed else (bn, bn // 2, bn // 4):
            # The coop2 epilogue stores whole chunks of 8 columns.
            if bn % sn or sn % 8 or (sn < 32 and not exposed):
                continue
            staging = 2 * 64 * sn * out_bytes
            ns = (BLOCK_SHARED_BYTES_OPT_IN[90] - _STATIC_SMEM_BYTES - staging) // ring
            if not deep:
                ns = min(ns_cap["coop2"], ns)
            if ns < 3:
                continue
            # The swizzle span: the largest power of two up to 128 B dividing the row.
            key = (ns, math.gcd(sn * out_bytes, 128), sn < bn, sn) if exposed else (ns, sn)
            if best is None or key > best:
                best = key
        if best is None:
            return None
        return best[0], (0 if best[-1] == bn else best[-1])

    def _stage_rule_ok(bm: int, bn: int, ns: int) -> bool:
        if ns < 3:
            return False
        return not (bm * bn < 128 * 192 and ns < 4)

    nt = (not trans_a) and trans_b
    out = []

    for bm in (64, 128):
        for bn in _BASIC_WIDTHS:
            if (bm * bn) // 128 > max_accum_regs:
                continue
            # 64x192 is offered only unsplit, where its grid fits one wave and m >= 8.
            # NT runs it on a 3-deep ring of 128-wide K steps; the other layouts, whose B
            # or A tile is MN-major, on a 5-deep ring of 64-wide steps.
            one_wave_192 = bn == 192
            if one_wave_192 and (
                bm != 64 or m < 8 or math.ceil(m / bm) * math.ceil(n / bn) > sm_count
            ):
                continue
            if one_wave_192:
                # Offered only where that ring fits beside the output staging tile.
                bk, ns = (128, 3) if nt else (64, 5)
                rings = [(bk, ns)] if _ns_basic(bm, bn, bk) >= ns else []
            else:
                rings = [(bk, _ns_basic(bm, bn, bk)) for bk in (32, 64, 128)]
            for bk, ns in rings:
                if not (one_wave_192 or _stage_rule_ok(bm, bn, ns)):
                    continue
                ps = _strip_width(bm, bn, sm_count)
                out.append(_Cand("basic", bm, bn, bk, ns, panel_size=ps))
                if one_wave_192:
                    continue
                k_iters = math.ceil(k / bk)
                for sk in (2, 4, 8):
                    if k_iters % sk == 0 and k_iters // sk >= 4:
                        out.append(_Cand("splitk", bm, bn, bk, ns, split_k=sk, panel_size=ps))

    nn_tiles = (m // 128) * (n // 64)
    if (
        not trans_a
        and not trans_b
        and m % 128 == 0
        and n % 64 == 0
        and k % 64 == 0
        and nn_tiles <= 2 * sm_count
        and 6 <= k // 64 <= 32
    ):
        # coop2s runs an 8-deep ring in this band.
        out.append(_Cand("coop2s", 128, 64, 64, 8))

    if nt and m <= 128 and m % 64 == 0 and k % 128 == 0:
        cluster_m = m // 64
        for bn in (64, 128, 192, 256):
            if n % bn:
                continue
            ns = min(ns_cap["simple"], _ns_basic(64, bn, 128))
            if ns >= 3:
                out.append(
                    _Cand(
                        "simple",
                        64,
                        bn,
                        128,
                        ns,
                        panel_size=0 if cluster_m > 1 else 8,
                        cluster_m=cluster_m,
                    )
                )
    # The narrowest coop2 width (one WGMMA per 16 of K) whose grid of 128-row tiles fits one
    # wave. Offered only where the last row tile gives both 64-row consumers rows and the
    # grid takes at least ``wave_15_16`` SMs.
    rows = math.ceil(m / 128)
    wave_15_16 = -(-15 * sm_count // 16)
    one_wave = None
    if nt and (m % 128 == 0 or m % 128 > 64):
        widths = (64, 128, 144, 160, 176, 192, 224, 240, 256)
        fits = [w for w in widths if rows * math.ceil(n / w) <= sm_count]
        if fits and rows * math.ceil(n / fits[0]) >= wave_15_16:
            one_wave = fits[0]
    for bn in sorted({64, 128, 176, 192, 256, one_wave or 256}):
        for bk in (32, 64, 128):
            # 176 is offered as whole tiles only (an N tail's epilogues are not scored), unless
            # it is the one-wave tile; it is one WGMMA only over 64-wide K steps.
            if bn == 176 and (bk != 64 or (n % bn and bn != one_wave)):
                continue
            mn_tiles = rows * math.ceil(n / bn)
            # One tile a CTA leaves the epilogue unhidden; a float32 one with 16-byte rows goes
            # through the TMA store.
            exposed = mn_tiles <= sm_count and out_bytes == 4 and n * out_bytes % 16 == 0
            # A one-wave 128-wide coop2 tile beats the basic tile only on a ring at least as
            # deep, so it takes the deepest ring the SMEM holds (see ``score``).
            deep = bn == one_wave == 128 and rows > 1 and n > bn
            d = _coop2_ns_sn(bn, bk, exposed, deep)
            if d is None:
                continue
            ns, sn = d
            if not _stage_rule_ok(128, bn, ns):
                continue
            coop2_ok = bn == 256 or (nt and bn in (176, 192))
            # A width that does not divide 256 is one WGMMA only over 64-wide K steps.
            one_wave_ok = bn == one_wave and (deep or (bn > 128 and (bk == 64 or 256 % bn == 0)))
            full_wave_ok = coop2_ok and mn_tiles >= sm_count
            if full_wave_ok or one_wave_ok:
                out.append(_Cand("coop2", 128, bn, bk, ns, stage_n=sn, one_wave=not full_wave_ok))
            if nt:
                k_iters = math.ceil(k / bk)
                ns_sk = min(ns, ns_cap["coop2_splitk"])
                if ns_sk >= 3 and bn <= 128:
                    # 15/16 of a wave suffices when no 128-row tile runs past ``m``.
                    min_ctas = wave_15_16 if m % 128 == 0 else sm_count
                    for sk in (2, 4, 8):
                        if k_iters % sk == 0 and k_iters // sk >= 4 and mn_tiles * sk >= min_ctas:
                            out.append(_Cand("coop2_splitk", 128, bn, bk, ns_sk, split_k=sk))
    fill = _fill_cand(m, n, k, sm_count, out_bytes) if nt else None
    if fill is not None:
        out.append(fill)
    if nt and k > 0:
        # The stream tile computes C.T in 64-row blocks of B and stages its transposed output
        # tile, twice at a helper width. Under 6 stages it does not stream B faster than the
        # basic tiles.
        mpad, helper_width = stream_width(m)
        staging = mpad * 64 * out_bytes * (2 if helper_width else 1)
        ring = BLOCK_SHARED_BYTES_OPT_IN[90] - _STATIC_SMEM_BYTES - staging
        ns = ring // ((64 + mpad) * 128 * 2)
        if ns >= 6:
            blocks = math.ceil(math.ceil(n / 64) / sm_count)
            out.append(_Cand("stream", 64, mpad, 128, ns, blocks=blocks))
    # The coop2_swap tile's transposed store, stmatrix.trans, writes 16-bit elements only.
    if not nt and n % 8 == 0 and out_bytes == 2:
        # Reserve both consumers' fp16 output staging before sizing the operand ring.
        swap_epilogue = 2 * math.prod(COOP2_SWAP_STAGING) * 2
        for bn in (128, 160, 192):
            stage = (256 + bn) * 64 * 2 + 2 * 8  # operands plus full/empty barriers
            ns = min(
                _COOP2_SWAP_FULL_RING, (BLOCK_SHARED_BYTES_OPT_IN[90] - swap_epilogue) // stage
            )
            # Three stages cannot hide mixed-width tiles; require whole uniform tiles.
            if ns >= _COOP2_SWAP_FULL_RING or (ns >= 3 and m % bn == 0):
                out.append(_Cand("coop2_swap", 256, bn, 64, ns))
    return out


def _fill_cand(m: int, n: int, k: int, sm_count: int, out_bytes: int = 2) -> _Cand | None:
    """The one-wave 64-row NT tile at a width the basic tiles skip, or None.

    Its width is the narrowest multiple of 8 whose grid of 64-row tiles fits one wave,
    offered only where that width is none of the basic widths and ``m`` fills more than half
    of one 64-row tile or more than three quarters of two. A one-tile grid is scored under
    its own factor, ``fill_one``. The tile runs 64-wide K steps on the deepest ring the
    shared memory holds (rounded down to a power of two for one M tile) and launches the
    M tiles of each N column as one cluster.
    """
    m_tiles = 1 if 32 < m <= 64 else 2 if 96 < m <= 128 else 0
    # The tile's own-width WGMMA needs the TMA store, so 16-byte output rows; its ring needs
    # a K step to stream.
    if not m_tiles or n % 8 or k == 0:
        return None
    width = next((w for w in range(16, 257, 8) if m_tiles * -(-n // w) <= sm_count), None)
    if width is None or width in _BASIC_WIDTHS:
        return None
    # A stage's A and B tiles; the output staging tile.
    stage = (64 + width) * 64 * 2
    ns = min(-(-k // 64), _ring_budget(64 * width * out_bytes, width) // stage)
    if m_tiles == 1:
        ns = 1 << (ns.bit_length() - 1)
    return _Cand("fill", 64, width, 64, ns, panel_size=0, cluster_m=m_tiles)


def _tile_fill(m: int, n: int, block_n: int, sm_count: int) -> float:
    """Share of the tensor-core time 128 x ``block_n`` tiles spend on output elements.

    The grid runs ``ceil(tiles / sm_count)`` full waves; a partial last wave and the
    columns an N tail tile computes past ``n`` are both lost.
    """
    tiles = math.ceil(m / 128) * math.ceil(n / block_n)
    return m * n / (math.ceil(tiles / sm_count) * sm_count * 128 * block_n)


def _pingpong_config(m: int, n: int, k: int, coop2_block_n: int, sm_count: int) -> dict | None:
    """The ping-pong config to run in place of a coop2 tile, or None to keep coop2.

    The ping-pong tile is 176 or 192 wide, taken where it fills the waves at least as well
    as the coop2 width, by a fill margin past 64 K steps when the widths differ.
    Under 8 K steps or 8 row tiles coop2 stays.
    """
    k_steps = math.ceil(k / 64)
    # The ping-pong epilogue stores through TMA, which needs 16-byte output rows.
    if k_steps < 8 or math.ceil(m / 128) < 8 or n % 8:
        return None
    width = max((176, 192), key=lambda bn: (_tile_fill(m, n, bn, sm_count), bn))
    wider_margin = {176: 0.01, 192: 0.05}[width]
    margin = 0.0 if k_steps <= 64 or width == coop2_block_n else wider_margin
    if _tile_fill(m, n, width, sm_count) < _tile_fill(m, n, coop2_block_n, sm_count) + margin:
        return None
    # Widest chunk dividing the tile in 16-byte TMA rows that keeps 192 unspilled.
    stage_n = 16 if width == 176 else 32
    return {
        "pingpong": True,
        "block_n": width,
        "block_k": 64,
        # A 192-wide 5-stage ring leaves no room for the epilogue buffers.
        "num_stages": 5 if width == 176 else 4,
        "stage_n": stage_n,
        "epi_bufs": 2 if width == 176 else 1,
    }


def _narrow_widths(cfg: dict, n: int) -> tuple:
    """The narrow tile widths a ping-pong or coop2 config's kernel runs, or ().

    Each is one ``dense_wgmma_helper.h`` WGMMA; a ping-pong width is whole epilogue
    chunks, and coop2 narrow tiles need TMA-addressable (16-byte) output rows.
    """
    if cfg.get("pingpong"):
        return tuple(w for w in (160, 144) if w < cfg["block_n"] and w % cfg["stage_n"] == 0)
    if cfg.get("coop2") and cfg["block_n"] == 256 and cfg["block_k"] == 64 and n % 8 == 0:
        return (240,) if (cfg.get("stage_n") or 256) <= 240 else ()
    return ()


def _busiest_cta(wide: int, narrow: int, block_n: int, narrow_n: int, sm_count: int) -> int:
    """Columns the busiest CTA of a persistent grid computes when ``wide`` tiles
    ``block_n`` wide run before ``narrow`` tiles ``narrow_n`` wide."""
    total = wide + narrow
    worst = 0
    # A CTA's tile counts step down only at these pids.
    for pid in {0, wide % sm_count, total % sm_count}:
        tiles = total // sm_count + (pid < total % sm_count)
        wide_tiles = wide // sm_count + (pid < wide % sm_count)
        worst = max(worst, wide_tiles * block_n + (tiles - wide_tiles) * narrow_n)
    return worst


def _narrow_tiles(
    m: int, n: int, block_n: int, widths: tuple, group_size_m: int, sm_count: int
) -> dict:
    """The narrow tiles that cut the busiest CTA's columns on a persistent grid, or {}.

    The last ``narrow_rows`` row tiles start with ``narrow_cols`` tiles ``narrow_n``
    wide, run after every wide tile. The fewest busiest-CTA columns win, then the wider
    ``narrow_n``, then fewer narrow tiles. Narrow rows form their own raster groups, so
    both row segments span at least half a group and no split leaves a group under 4
    rows: smaller groups re-read ``B``. A group over 16 rows spans every row, so each
    segment is one whole group; it bounds the split as a 16-row group does.
    """
    group_size_m = min(group_size_m, 16)
    rows = math.ceil(m / 128)
    cols = math.ceil(n / block_n)

    def busiest(wide: int, narrow: int, narrow_n: int) -> int:
        return _busiest_cta(wide, narrow, block_n, narrow_n, sm_count)

    uniform = busiest(rows * cols, 0, 0)
    best = (uniform, 0, 0)
    choice: dict = {}
    min_group = 4
    for narrow_n in sorted(widths, reverse=True):
        for narrow_rows in range(1, rows + 1):
            full_rows = rows - narrow_rows
            if 0 < min(full_rows, narrow_rows) < group_size_m // 2:
                continue
            last_groups = (full_rows % group_size_m, narrow_rows % group_size_m)
            if any(0 < g < min(min_group, group_size_m) for g in last_groups):
                continue
            for narrow_cols in range(1, n // narrow_n + 1):
                tail_cols = math.ceil((n - narrow_cols * narrow_n) / block_n)
                wide = (rows - narrow_rows) * cols + narrow_rows * tail_cols
                narrow = narrow_rows * narrow_cols
                key = (busiest(wide, narrow, narrow_n), -narrow_n, narrow)
                if key[0] < uniform and key < best:
                    best = key
                    choice = {
                        "narrow_n": narrow_n,
                        "narrow_rows": narrow_rows,
                        "narrow_cols": narrow_cols,
                    }
    return choice


def _swap_row_mix(m: int, n: int, block_n: int, trans_a: bool, sm_count: int) -> dict:
    """The narrow tiles that cut the busiest CTA's columns on a ``coop2_swap`` grid, or {}.

    Its tiles span 256 rows of C.T and ``m`` columns; ``row_mix`` covers the first
    ``split`` row blocks with ``wide_a`` wide and ``narrow_a`` narrow tiles and the rest
    with ``wide_b`` and ``narrow_b``, narrow tiles run last. A block takes at most
    ``2 * block_n / gcd(block_n, narrow_n)`` narrow tiles: that many cover what fewer
    wide ones do. The fewest busiest-CTA columns win, then the wider ``narrow_n``, then
    fewer narrow tiles. With ``trans_a``, a ``block_n`` that is a multiple of 64 stages
    ``op(A).T`` in 64-column swizzle atoms, which a narrow load must fill whole.
    """
    rows = math.ceil(n / 256)
    uniform = _busiest_cta(rows * math.ceil(m / block_n), 0, block_n, 0, sm_count)
    best = (uniform, 0, 0)
    choice: dict = {}
    for narrow_n in (block_n - 32, block_n - 64):
        if narrow_n < 32 or (trans_a and block_n % 64 == 0 and narrow_n % 64):
            continue
        mixes = {
            (math.ceil(max(0, m - b * narrow_n) / block_n), b)
            for b in range(2 * block_n // math.gcd(block_n, narrow_n) + 1)
        }
        mixes = sorted((a, b) for a, b in mixes if a * block_n + b * narrow_n - m < narrow_n)
        for i, (wide_a, narrow_a) in enumerate(mixes):
            for wide_b, narrow_b in mixes[i:]:
                for split in range(1, rows + 1):
                    wide = split * wide_a + (rows - split) * wide_b
                    narrow = split * narrow_a + (rows - split) * narrow_b
                    busiest = _busiest_cta(wide, narrow, block_n, narrow_n, sm_count)
                    key = (busiest, -narrow_n, narrow)
                    if busiest < uniform and key < best:
                        best = key
                        choice = {
                            "narrow_n": narrow_n,
                            "row_mix": (split, wide_a, narrow_a, wide_b, narrow_b),
                        }
    return choice


def _group_size_m(m: int, n: int, k: int, block_n: int, l2_bytes: int, block_m: int = 128) -> int:
    """Row tiles one raster group of a persistent coop2 or ping-pong grid spans.

    - 4 where row tiles outnumber column tiles and ``B`` fits in one L2 partition.
    - Every row where ``B`` exceeds one L2 partition but ``A`` fits in one, so each ``B``
      tile is read from DRAM once.
    - 16 otherwise.

    ``coop2_swap`` computes ``C.T`` in 256-row tiles, so it passes ``n`` and ``m``
    swapped with ``block_m=256``, and ``A`` takes the place of ``B``.
    """
    # The dense family's operands are 16-bit.
    elem_bytes = 2
    # SM90's L2 is two partitions, each caching for the SMs attached to it, so data
    # every SM reads is held in both.
    l2_partitions = 2
    row_tiles = -(-m // block_m)
    col_tiles = -(-n // block_n)
    partition = l2_bytes // l2_partitions
    if row_tiles > col_tiles and n * k * elem_bytes <= partition:
        return 4
    if n * k * elem_bytes > partition >= m * k * elem_bytes:
        return max(row_tiles, 16)
    return 16


_GEMV = {"gemv": True}


def _tiny_m_candidates(k: int, gemv: bool) -> list[dict]:
    """The m <= 8 NT configs :func:`_tiny_m_score_us` ranks, fewest stages first.

    The operand-swapped kernel at each ring depth, the 64 x 128 tile at each split
    dividing the K-tile count, and with *gemv* the ``GemvKernel`` stand-in ``_GEMV``.
    """
    out = [{"swap_ab": True, "block_nn": 64, "block_k": 128, "num_stages": ns} for ns in (6, 8, 10)]
    out += [
        {
            "block_m": 64,
            "block_n": 128,
            "block_k": 128,
            "num_stages": 4,
            "panel_size": 16 if sk > 1 else 8,
            "split_k": sk,
        }
        for sk in (1, 2, 4, 8)
        if math.ceil(k / 128) % sk == 0
    ]
    return out + [_GEMV] if gemv else out


def _tiny_m_score_us(cfg: dict, n: int, k: int, sm_count: int, cal: _TinyMCalibration) -> float:
    """Predicted microseconds of an m <= 8 NT config: the time its grid streams B.

    Each CTA streams its rows of B at ``min(bytes in flight / latency, cta_gbps)``, a grid
    past the CTAs the device holds at once runs in waves, and no grid streams B faster than
    ``dram_tbps``. Split-K adds its reduce; the GEMV streams B at its own rate. Only the
    ordering is meaningful.
    """
    if cfg.get("gemv"):
        return n * k * 2 / (cal.gemv_tbps * 1e6)
    ns, bk = cfg["num_stages"], cfg["block_k"]
    if cfg.get("swap_ab"):
        rows, sk = cfg["block_nn"], 1
        stage_bytes = (rows + SWAP_AB_MPAD) * bk * 2
        per_sm = max(1, BLOCK_SHARED_BYTES_OPT_IN[90] // (ns * stage_bytes))
    else:
        rows, sk, per_sm = cfg["block_n"], cfg["split_k"], 1
    ctas = math.ceil(n / rows) * sk
    rate = min(ns * rows * bk * 2 / cal.latency_us, cal.cta_gbps * 1e3)
    us = math.ceil(ctas / (per_sm * sm_count)) * rows * (k / sk) * 2 / rate
    us = max(us, n * k * 2 / (cal.dram_tbps * 1e6))
    return us + (cal.split_k_us if sk > 1 else 0.0)


@functools.lru_cache(maxsize=512)  # lint: target-cache allow: facts are arguments
def _tiny_m_config(n: int, k: int, sm_count: int, gemv: bool, cal: _TinyMCalibration) -> dict:
    """The lowest-scoring of :func:`_tiny_m_candidates`; a tie keeps the fewer stages."""
    return min(_tiny_m_candidates(k, gemv), key=lambda c: _tiny_m_score_us(c, n, k, sm_count, cal))


def two_row_gemv_wins(n: int, k: int, sm_count: int, calibration: str | None) -> bool:
    """Whether ``GemvKernel`` outscores every tiny-m tile on an ``m == 2`` NT call.

    *calibration* is the board's calibration key; False on a board without a calibration.
    """
    cal = _calibration(calibration)
    return cal is not None and _tiny_m_config(n, k, sm_count, True, cal.tiny_m) is _GEMV


def _score_us(
    cd: _Cand, m: int, n: int, k: int, sm_count: int, cal: _Calibration, out_bytes: int = 2
) -> float:
    """Predicted microseconds of a call whose output elements take *out_bytes*; only the
    relative ordering is meaningful."""
    if cd.structure == "stream":
        # The time the grid streams B, at a rate growing with its CTAs up to the DRAM's.
        ctas = math.ceil(math.ceil(n / cd.block_m) / cd.blocks)
        rate = min(ctas * cal.stream.cta_gbps * 1e3, cal.stream.dram_tbps * 1e6)
        return n * k * 2 / rate
    if cd.structure == "coop2_swap":
        m, n = n, m
    bm, bn, sk = cd.block_m, cd.block_n, cd.split_k
    elem, elem_out = 2, out_bytes
    num_blocks = math.ceil(m / bm) * math.ceil(n / bn) * sk
    num_waves = math.ceil(num_blocks / sm_count)
    wave_eff = num_blocks / (num_waves * sm_count)
    k_exp = k / sk
    ws = 4 if sk > 1 else elem_out

    # The operands cross L2 and L1 once each.
    ab_bytes = k_exp * (bm + bn) * elem
    l1_tc = k_exp * (max(64, bm) + bn) * elem + bm * bn * ws
    cd_bytes = bm * bn * ws

    l2_us = (ab_bytes + cd_bytes) * num_blocks / (cal.l2_tbps * 1e6)
    l1_us = (ab_bytes + l1_tc + cd_bytes) * num_blocks / (cal.l1_tbps * 1e6)
    tc_us = 2.0 * m * n * k / (cal.tensor_core_tflops * 1e6)
    issue_us = (k_exp / cd.block_k) * num_waves * cal.issue_ns * 1e-3

    us = max(tc_us, l1_us, l2_us) / wave_eff + issue_us
    if sk > 1:
        red_bytes = sk * m * n * 4 + m * n * elem_out
        us += red_bytes / (cal.reduce_tbps * 1e6)
    # A 3-stage swap ring takes its own factor: it scores fewer bytes and K steps than a
    # 4-stage one but runs no faster per FLOP.
    if cd.structure == "coop2_swap" and cd.num_stages < _COOP2_SWAP_FULL_RING:
        return us * cal.factor("coop2_swap_3stage")
    # A one-M-tile fill grid has no cluster and the whole of each SM's B stream; see
    # ``_fill_cand``.
    if cd.structure == "fill" and cd.cluster_m == 1:
        return us * cal.factor("fill_one")
    return us * cal.factor(cd.structure)


@functools.lru_cache(maxsize=512)  # lint: target-cache allow: facts are arguments
def _best_config_cached(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    sm_count: int,
    l2_bytes: int,
    cal: _Calibration,
    out_bytes: int,
) -> dict:
    """Cached selection body of :func:`best_config` — do not mutate results."""
    if m <= 8 and not trans_a and trans_b:
        return _tiny_m_config(n, k, sm_count, False, cal.tiny_m)
    cands = _enumerate(m, n, k, trans_a, trans_b, sm_count, out_bytes)
    if not cands:
        raise AssertionError(
            f"no config candidate for {m}x{n}x{k} (trans_a={trans_a}, "
            f"trans_b={trans_b}, sm_count={sm_count})"
        )

    # Of two tiles scoring alike, the deeper ring wins: a one-wave coop2 tile on a ring
    # shallower than the basic tile's loses to it.
    def score(c: _Cand) -> tuple:
        return _score_us(c, m, n, k, sm_count, cal, out_bytes), -c.block_k, -c.num_stages

    best = min(cands, key=score)
    # In one row tile past 64 K steps a 128 x 128 basic grid streams B at the DRAM rate
    # however full its last wave is, which the score does not model.
    if best.one_wave and m <= 128 and math.ceil(k / 64) > 64:
        other = min((c for c in cands if not c.one_wave), key=score, default=best)
        if (other.structure, other.block_m, other.block_n) == ("basic", 128, 128):
            best = other
    if best.structure == "coop2":
        nt = not trans_a and trans_b
        cfg = (nt and _pingpong_config(m, n, k, best.block_n, sm_count)) or best.to_config()
        cfg["group_size_m"] = _group_size_m(m, n, k, cfg["block_n"], l2_bytes)
        # Under 16 K steps a tile's fixed cost outweighs the columns narrow tiles save.
        if nt and math.ceil(k / cfg["block_k"]) >= 16:
            cfg |= _narrow_tiles(
                m, n, cfg["block_n"], _narrow_widths(cfg, n), cfg["group_size_m"], sm_count
            )
        return cfg
    if best.structure == "coop2_swap":
        cfg = best.to_config()
        cfg["group_size_m"] = _group_size_m(n, m, k, best.block_n, l2_bytes, block_m=256)
        # A shallower ring runs uniform tiles only.
        if math.ceil(k / best.block_k) >= 16 and best.num_stages >= _COOP2_SWAP_FULL_RING:
            cfg |= _swap_row_mix(m, n, best.block_n, trans_a, sm_count)
        return cfg
    return best.to_config()


def best_config(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    sm_count: int,
    calibration: str | None,
    *,
    l2_bytes: int,
    out_bytes: int = 2,
) -> dict | None:
    """Return the analytically selected ``GEMMTMAKernel`` config for a shape.

    Args:
        m: Logical GEMM rows of the output.
        n: Logical GEMM columns of the output.
        k: Contraction dim.
        trans_a: ``A`` stored ``[k, m]`` when True.
        trans_b: ``B`` stored ``[n, k]`` when True.
        sm_count: Streaming multiprocessors of the target device.
        calibration: The target board's calibration key
            (``tileops._utils.calibration_key``), which selects the profile the ranking
            constants come from.
        l2_bytes: L2 cache of the target device, in bytes; it sets the raster group
            of a persistent coop2 or ping-pong tile.
        out_bytes: Bytes of one output element: 2 for an output in the operands' 16-bit
            dtype, 4 for a float32 one, whose staging tiles take twice the shared memory.

    Returns:
        ``None`` when no profile carries this board's ranking constants, so the
        caller takes its own default rather than a ranking measured elsewhere.
        Otherwise a config dict in ``GEMMTMAKernel`` schema — either the single-consumer
        form (``block_m/block_n/block_k/num_stages/panel_size/split_k``,
        optionally ``simple``) or a structure-flagged form (``coop2s`` /
        ``coop2`` / ``coop2_splitk`` / ``coop2_swap`` / ``pingpong`` / ``stream`` /
        ``swap_ab``). A fresh dict per call, which the caller may mutate.
    """
    cal = _calibration(calibration)
    if cal is None:
        return None
    return dict(_best_config_cached(m, n, k, trans_a, trans_b, sm_count, l2_bytes, cal, out_bytes))


def gemv_unrolled(aligned: bool, slices: int) -> bool:
    """Whether a register-body warp that loads ``slices`` 16-byte slices unrolls its bursts:
    over vector-aligned rows, up to 96 slices. Rolled, the burst loop more than doubles a
    warp's registers and so halves the blocks an SM fits; unrolled past that, or over the
    scalar loads of unaligned rows, the body outgrows the instruction cache."""
    return aligned and slices <= 96


def _reduce_threads(k: int) -> int:
    """The lanes that reduce a row of ``k`` values, on the ring and in registers alike, which
    fixes the row's bits: 64 over rows of 12288 values or more, 256 from 6144, else 128."""
    return 64 if k >= 12288 else 256 if k >= 6144 else 128


def _stage_values(k: int) -> int:
    """Values one step of a row of ``k`` covers, on the ring and in registers alike: a
    16-byte slice of 16-bit values for each reducing lane."""
    return _reduce_threads(k) * 8


def _ring_config(k: int) -> dict:
    """The staged ring's config for one-row bands: two rows a block over five stages at 64
    reducing lanes, else one row over four."""
    reduce_threads = _reduce_threads(k)
    if reduce_threads == 64:
        return {"block_n": 2, "reduce_threads": reduce_threads, "num_stages": 5}
    return {"block_n": 1, "reduce_threads": reduce_threads, "num_stages": 4}


def _ring_copy_bytes(k: int) -> int:
    """Bytes the staged ring copies at a time from rows of ``k`` 16-bit values: the largest
    power of two up to 16 that divides a row's bytes."""
    row_bytes = 2 * k
    return min(16, row_bytes & -row_bytes)


def _two_rows_a_warp_from(k: int) -> int:
    """Rows an SM from which odd rows of ``k`` values go two to a warp, where the ring keeps
    them from 2 rows an SM up to this below the two-row ring: 8 there, else 12."""
    return 8 if _ring_config(k)["block_n"] == 2 else 12


def gemv_ring_copies(k: int) -> bool:
    """Whether the staged ring can copy one-row bands of ``k`` values: its two-row blocks
    cannot copy odd rows, whose 2-byte width no asynchronous copy takes."""
    return not (k % 2 and _ring_config(k)["block_n"] == 2)


def _wave_window(k: int, sm_count: int) -> tuple[int, int]:
    """Rows over which bf16 default split rows of ``k`` values need a second wave while the
    staged ring's two-row blocks still fit one: an SM holds 12 split-row blocks and 9 ring
    blocks (their bf16 builds' register counts)."""
    return 12 * sm_count, 9 * _ring_config(k)["block_n"] * sm_count


def gemv_registers(n: int, k: int, sm_count: int, dtype: str) -> bool:
    """Whether ``GemvKernel``'s one-row bands read ``n`` rows of ``k`` values into registers
    (``_gemv_row_kernel``) rather than through the staged ring.

    Calibrated on SM90 against the ring at the same ``reduce_threads``. The ring keeps rows
    of up to 4096 values; odd rows below 12288 values over 2 rows an SM up to where they go
    two to a warp; rows past what the default split row unrolls that the ring copies 8 bytes
    or more at a time; and aligned bf16 rows over ``_wave_window``, where the split rows
    need a second wave and the ring one. The window skips the first two-row K, where the
    ring's shorter loop unrolls and its registers cut it to 3 blocks an SM, and every K off
    the ring's stage width, where each of its steady-state copies is predicated. Odd rows
    the two-row ring cannot copy always take registers.
    """
    if k <= 4096:
        return False
    if k % 2:
        return not gemv_ring_copies(k) or not (
            2 * sm_count < n < _two_rows_a_warp_from(k) * sm_count
        )
    stage = _stage_values(k)
    slices = -(-k // stage)
    if _ring_copy_bytes(k) >= 8 and not gemv_unrolled(True, slices):
        return False
    if dtype != "bfloat16" or k % stage or _ring_config(k - stage)["block_n"] == 1:
        return True
    low, high = _wave_window(k, sm_count)
    return not low < n <= high


def gemv_config(n: int, k: int, sm_count: int, registers: bool) -> dict:
    """SM90 ``GemvKernel`` config for its one-row bands (single-row / single-column GEMV).

    ``_reduce_threads`` fixes a row's bits. In registers a row is one block of a warp for
    each of the ring's warps, 32 slices a burst. Odd rows, whose loads are scalar, go two to
    a warp at bursts of 8 from ``_two_rows_a_warp_from`` rows an SM.
    """
    if not registers:
        return _ring_config(k)
    reduce_threads = _reduce_threads(k)
    if k % 2 and n >= _two_rows_a_warp_from(k) * sm_count:
        return {"reduce_threads": reduce_threads, "warps": 1, "rows": 2, "depth": 8, "split": 1}
    warps = reduce_threads // 32
    return {
        "reduce_threads": reduce_threads,
        "warps": warps,
        "rows": 1,
        "depth": 32,
        "split": warps,
    }


def small_batch_config(n: int, k: int, sm_count: int) -> dict:
    """``GemvKernel`` config for its ``lhs_rows`` band (m == 2 NT).

    One output column a block, a 64-lane reduction over K and a 6-deep ring; a 2-deep ring
    where the grid fills the device's CTA slots and the per-thread K loop is long.
    """
    cfg = {"block_n": 1, "reduce_threads": 64, "num_stages": 6}
    k_iters = math.ceil(k / (cfg["reduce_threads"] * 8))
    if n >= 28 * sm_count and k_iters >= 12:
        cfg["num_stages"] = 2
    return cfg


def small_m_splitk_config(
    m: int, n: int, k: int, sm_count: int, calibration: str | None
) -> dict | None:
    """Select the split-K basic GEMM band fitted on a calibrated board for a small-M NT call."""
    block_k = 128
    k_tiles = k // block_k
    # The row count the small-M split-K band was fitted at, by calibration key. A board
    # without an entry has no band.
    small_m_splitk_m = {"h200": 32}
    band_m = small_m_splitk_m.get(calibration)
    if m != band_m or k % block_k or n % 8 or k_tiles < 48:
        return None
    block_ns = [block_n for block_n in range(8, 129, 8) if n % block_n == 0]
    target_n_tiles = max(1, sm_count // 2)
    block_n = min(block_ns, key=lambda value: abs(n // value - target_n_tiles))
    n_tiles = n // block_n
    split_ks = [
        split_k for split_k in (2, 4) if k_tiles % split_k == 0 and k_tiles // split_k >= 12
    ]
    if not split_ks:
        return None
    split_k = min(split_ks, key=lambda value: abs(n_tiles * value - 2 * sm_count))
    return {
        "block_m": m,
        "block_n": block_n,
        "block_k": block_k,
        "num_stages": 2,
        "threads": 128,
        "split_k": split_k,
    }


def fp8_ws_config(m: int, n: int, k: int, sm_count: int, block_scaled: bool) -> dict:
    """Tile, ring depth and K split for the warp-specialized FP8 GEMM.

    ``block_m`` is one or two consumer warpgroups of 64 rows. ``block_n`` 256 would
    spill: its accumulator and scaled partial want 256 registers per thread.

    - ``block_n``: 128, or 64 where the 128-wide grid does not fill one wave and
      ``m > 8``. Per-tensor, 192 is taken where its whole waves cover fewer columns than
      128's, and 176 replaces 192 where it takes as many waves and pads ``n`` no further.
    - ring depth: the deepest the budget allows, except block128 with ``m > 8``, whose
      per-K-step promotion serializes with the next WGMMA: 5 stages with ``block_n`` 64
      or at most 12 K steps, 3 at 128 K steps or more, else 4.
    - split-K: only where the sliced grid still fits one wave and every slice keeps at
      least ``FP8_MIN_SLICE_K_TILES`` K steps.
    - one row tile (``m <= 128``): 64 rows and one consumer, with the narrowest
      ``block_n`` of 32, 64 and 128 whose grid fits one wave, where that unsplit grid
      holds at least as many CTAs as the 128-row one with its split. Under block128
      only an unsplit 128-row grid is replaced.
    """
    # One consumer warpgroup's rows; the K tile is the block128 scale grid's.
    half_m, block_k = WGMMA_ROWS, QUANT_SCALE_BLOCK

    def _fp8_ws_stages(block_m: int, block_n: int) -> int:
        """Deepest ring the SMEM budget allows for one warp-specialized FP8 tile.

        A stage holds the ``A`` and ``B`` tiles, the block128 scale vectors when staged,
        and its full and empty mbarriers; the epilogue staging tile exists only for ``m``
        over one consumer's rows.
        """
        mbarrier_bytes = 8
        per_stage = (
            (block_m + block_n) * block_k
            + block_scaled * (block_m + block_n) * 4
            + 2 * mbarrier_bytes
        )
        epilogue = block_m * block_n * 2 if m > half_m else 0
        return (BLOCK_SHARED_BYTES_OPT_IN[90] - epilogue) // per_stage

    def _waves(block_n: int) -> int:
        return -(-(-(-m // 128) * -(-n // block_n)) // sm_count)

    def _waved_columns(block_n: int) -> int:
        return _waves(block_n) * block_n

    k_iters = -(-k // block_k)
    block_n = 128 if m <= 8 or -(-m // 128) * -(-n // 128) >= sm_count else 64
    if not block_scaled and block_n == 128 and _waved_columns(192) < _waved_columns(128):
        narrow = _waves(176) == _waves(192) and -(-n // 176) * 176 <= -(-n // 192) * 192
        block_n = 176 if narrow else 192
    deepest = _fp8_ws_stages(2 * half_m, block_n)
    if not block_scaled or m <= 8:
        num_stages = deepest
    elif block_n == 64 or k_iters <= 12:
        num_stages = min(5, deepest)
    elif k_iters >= 128:
        num_stages = 3
    else:
        num_stages = 4
    tiles = -(-m // 128) * -(-n // block_n)
    split_k = 1
    for candidate in (4, 2):
        if (
            tiles * candidate <= sm_count
            and k_iters % candidate == 0
            and k_iters // candidate >= FP8_MIN_SLICE_K_TILES
        ):
            split_k = candidate
            break

    if m <= 2 * half_m:
        m_tiles = -(-m // half_m)
        narrow_n = next((bn for bn in (32, 64) if m_tiles * -(-n // bn) <= sm_count), 128)
        unsplit_ctas = min(m_tiles * -(-n // narrow_n), sm_count)
        if unsplit_ctas >= tiles * split_k and (not block_scaled or split_k == 1):
            return {
                "block_n": narrow_n,
                "num_stages": min(k_iters, _fp8_ws_stages(half_m, narrow_n)),
                "group_size_m": 8,
                "split_k": 1,
                "block_m": half_m,
            }
    return {
        "block_n": block_n,
        "num_stages": num_stages,
        "group_size_m": 8,
        "split_k": split_k,
        "block_m": 2 * half_m,
    }


def int8_w8a8_config(m: int, n: int, k: int, sm_count: int) -> dict:
    """Tile width, ring depth and K step for the 64-row warp-specialized INT8 GEMM.

    - ``block_n``: the narrowest listed width whose grid fits one wave, so each SM reads
      the fewest bytes.
    - ``block_k``: 256 where ``m`` is exactly 64 and the tile is at most 160 wide; 128
      otherwise.
    - ring depth: the deepest the shared memory holds, at most one stage a K step.
    """
    block_m = 64
    # Multiples of 16 from 32 to 256 that beat the next listed width where picked.
    widths = (32, 64, 80, 96, 112, 128, 160, 192, 256)
    tiles_m = -(-m // block_m)
    block_n = next((w for w in widths if tiles_m * -(-n // w) <= sm_count), widths[-1])
    block_k = 256 if m == block_m and block_n <= 160 else 128
    # A stage's A and B tiles and its full and empty mbarriers, 8 bytes each.
    stage = (block_m + block_n) * block_k + 16
    # The output staging tile, which exists only where a row tile can be full.
    staging = block_m * block_n * 2 if m >= block_m else 0
    num_stages = min(-(-k // block_k), (BLOCK_SHARED_BYTES_OPT_IN[90] - staging) // stage)
    return {"block_n": block_n, "num_stages": num_stages, "group_size_m": 8, "block_k": block_k}


def _pingpong_busiest(wide: int, narrow: int, block_n: int, narrow_n: int, sm_count: int) -> tuple:
    """Columns the busiest CTA and the busiest consumer of a persistent ping-pong grid compute,
    when ``wide`` tiles ``block_n`` wide run before ``narrow`` tiles ``narrow_n`` wide.

    CTA ``p`` takes tiles ``p``, ``p + sm_count``, ...; its consumer ``w % 2`` takes its
    ``w``-th tile and runs that tile's epilogue.
    """
    total = wide + narrow
    cta = consumer = 0
    for p in range(min(sm_count, total)):
        widths = [block_n if t < wide else narrow_n for t in range(p, total, sm_count)]
        cta = max(cta, sum(widths))
        consumer = max(consumer, sum(widths[0::2]), sum(widths[1::2]))
    return cta, consumer


def int8_pingpong_grid(m: int, n: int, sm_count: int, group_size_m: int) -> dict:
    """Tile width and narrow tiles of the persistent INT8 ping-pong GEMM's grid.

    The grid takes 128 x 192 tiles, ending in the narrow tiles :func:`_narrow_tiles` picks,
    where they leave the busiest CTA, then the busiest consumer, no more columns than
    128 x 128 tiles; 128 x 128 tiles otherwise.

    Returns:
        ``block_n``, ``narrow_n``, ``narrow_rows`` and ``narrow_cols``; ``narrow_n`` 0 is a
        uniform grid.
    """
    rows = -(-m // 128)
    uniform = {"block_n": 128, "narrow_n": 0, "narrow_rows": 0, "narrow_cols": 0}
    uniform_load = _pingpong_busiest(rows * -(-n // 128), 0, 128, 0, sm_count)
    # The widest multiple of the 32-column store box whose accumulators (block_n registers
    # a thread) leave the 240-register budget room for the rest.
    wide_n = 192
    # Every width of whole 32-column store boxes below the wide tile.
    narrow_widths = tuple(range(wide_n - 32, 0, -32))
    cols = -(-n // wide_n)
    narrow = {"narrow_n": 0, "narrow_rows": 0, "narrow_cols": 0} | _narrow_tiles(
        m, n, wide_n, narrow_widths, group_size_m, sm_count
    )
    narrow_n, narrow_rows, narrow_cols = narrow.values()
    tail_cols = -(-(n - narrow_cols * narrow_n) // wide_n)
    wide_tiles = (rows - narrow_rows) * cols + narrow_rows * tail_cols
    wide_load = _pingpong_busiest(wide_tiles, narrow_rows * narrow_cols, wide_n, narrow_n, sm_count)
    if wide_load > uniform_load:
        return uniform
    return {"block_n": wide_n, **narrow}
