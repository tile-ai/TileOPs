"""Analytic config selection for the dense GEMM kernel family (SM90).

Picks the kernel structure — single-consumer (``basic``), grid-z split-K
(``splitk``), 2-consumer cooperative (``coop2``), cooperative split-K
(``coop2_splitk``), or ping-pong in place of a coop2 tile with few K steps
(``pingpong``) — and its tile configuration for an arbitrary GEMM shape
in microseconds, without autotuning. The module also owns the measured
config bands of the bandwidth-mode kernels (``gemv_config``,
``small_batch_config``), so the family's per-shape configuration lives in one
place; which kernel serves a call is a separate question, answered by each
kernel's own ``refusal``. The scored path follows DeepGEMM's SM90
heuristics (enumerate -> prune -> score -> derive, see
``csrc/jit_kernels/heuristics/sm90.hpp``), with three extensions the
TileOPs kernel family needs:

- the structure choice itself is part of the candidate space, scored via
  per-structure effective tensor-throughput constants (DeepGEMM never ranks
  1-consumer against 2-consumer layouts; we must);
- a split-K penalty prices the fp32 workspace round trip, the reduce pass,
  and the extra launch (DeepGEMM SM90 has no split-K path);
- a per-K-iteration issue-overhead term separates ``block_k`` candidates
  the byte model cannot (measured: block_k=128 beats 32 consistently).

The ranking constants are fitted per board, and the scorer reads the current
device's from the table in :func:`_calibration`. A board absent from that table
is not ranked at all: :func:`best_config` returns ``None`` and the kernel takes
its own default, because ranking with another board's numbers is worse than not
ranking. Re-fit when the kernel structures change materially, or when a new
board is added.

Resource model mirrors ``tileops/kernels/gemm.py``:

- basic/splitk: SMEM ``ns*(bm+bn)*bk*2 + bm*bn*2``, accum regs
  ``bm*bn/128 <= 200``;
- coop2*: ``block_m`` fixed at 128 (two 64-row consumers), SMEM
  ``ns*(128+bn)*bk*2 + 2*64*sn*2``, NT only, persistent grid;
- split-K variants require ``ceildiv(k, bk) % split_k == 0``.
"""

import functools
import math
from dataclasses import dataclass
from typing import Optional

from tileops.kernels.constants import BLOCK_SHARED_BYTES_OPT_IN
from tileops.utils import calibration_key

__all__ = [
    "SWAP_AB_MPAD",
    "best_config",
    "fp8_ws_config",
    "gemv_config",
    "int8_w8a8_config",
    "small_batch_config",
    "small_m_splitk_config",
    "swap_ab_grid_underfills",
    "two_row_gemv_wins",
]

# SM90's opt-in shared memory per block: the dense candidate rings (``_enumerate``) and
# the FP8 warp-specialized ring (``fp8_ws_config``) are both sized against it.
_SMEM_BUDGET = BLOCK_SHARED_BYTES_OPT_IN[90]

# The swap-AB kernel's N tile: the grid-fill test, the stage pick and the config it
# returns must count the same CTAs.
_SWAP_AB_BLOCK_NN = 64

SWAP_AB_MPAD = 8


@dataclass(frozen=True)
class _Calibration:
    """The scorer's ranking constants for one board.

    Effective ranking constants, not physical rates: each was fitted so the
    model's ordering matches a measured sweep of this kernel family, not
    measured with a microbenchmark. ``l1_tbps`` and ``l2_tbps`` are not this
    board's SMEM and L2 bandwidths, and no other algorithm can read them as
    such — which is why they live here and not in ``perf/profiles/``.
    """

    l1_tbps: float
    l2_tbps: float
    reduce_tbps: float
    launch_us: float
    issue_ns: float
    tensor_core_tflops: tuple

    def tc_tflops(self, structure: str) -> float:
        return dict(self.tensor_core_tflops)[structure]


def _calibration(device_name: str) -> Optional[_Calibration]:
    """The calibration measured on *device_name*'s board, or ``None`` for a board without one."""
    # One entry per board the scorer has been calibrated on, by its calibration key
    # (``tileops.utils.calibration_key``). A board absent here is not ranked; see
    # :func:`best_config`.
    calibrations = {
        "h200": _Calibration(
            l1_tbps=33.0,
            l2_tbps=17.5,
            reduce_tbps=1.5,
            launch_us=2.25,
            issue_ns=150.0,
            tensor_core_tflops=(
                ("basic", 420.0),
                ("coop2", 525.0),
                ("coop2s", 600.0),
                ("coop2_splitk", 525.0),
                ("simple", 500.0),
                ("splitk", 420.0),
            ),
        ),
    }
    return calibrations.get(calibration_key(device_name))


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

    def to_config(self) -> dict:
        if self.structure == "simple":
            return {
                "simple": True,
                "block_m": self.block_m,
                "block_n": self.block_n,
                "block_k": self.block_k,
                "num_stages": self.num_stages,
                "threads": 128,
                "panel_size": self.panel_size,
                "cluster_m": self.cluster_m,
            }
        if self.structure == "coop2":
            return {
                "coop2": True,
                "block_n": self.block_n,
                "block_k": self.block_k,
                "num_stages": self.num_stages,
                "stage_n": self.stage_n,
            }
        if self.structure == "coop2s":
            return {
                "coop2s": True,
                "block_n": self.block_n,
                "block_k": self.block_k,
                "num_stages": self.num_stages,
            }
        if self.structure == "coop2_splitk":
            return {
                "coop2_splitk": True,
                "block_n": self.block_n,
                "block_k": self.block_k,
                "num_stages": self.num_stages,
                "split_k": self.split_k,
            }
        return {
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_k": self.block_k,
            "num_stages": self.num_stages,
            "panel_size": self.panel_size,
            "split_k": self.split_k,
        }


def _enumerate(m: int, n: int, k: int, trans_a: bool, trans_b: bool, sm_count: int) -> list:
    # Accumulator registers per thread a basic tile may hold: ``bm * bn / 128``.
    max_accum_regs = 200
    # Deepest ring each structure is built with, whatever the SMEM budget allows; the
    # one-wave 64x192 basic tile sets its own ring below.
    ns_cap = {"basic": 4, "splitk": 4, "coop2": 4, "coop2_splitk": 4}

    def _strip_width(bm: int, bn: int, sm_count: int) -> int:
        """L2-footprint-minimizing rasterization panel width.

        DeepGEMM's ``get_num_1d_blocks_per_group`` (scheduler/gemm.cuh): choose
        the group size whose resident-CTA footprint ``g*bm + ceil(SMs/g)*bn``
        is smallest.
        """
        return min((4, 8, 10, 16), key=lambda g: g * bm + math.ceil(sm_count / g) * bn)

    def _ns_basic(bm: int, bn: int, bk: int) -> int:
        ring = (bm + bn) * bk * 2
        return min(ns_cap["basic"], (_SMEM_BUDGET - bm * bn * 2) // ring)

    def _coop2_ns_sn(bn: int, bk: int):
        """Deepest ring under ``ns_cap`` the SMEM budget allows; shrink the epilogue
        staging chunk (stage_n) when that buys another pipeline stage."""
        ring = (128 + bn) * bk * 2
        best = None
        for sn in (bn, bn // 2, bn // 4):
            if sn < 32 or bn % sn:
                continue
            ns = min(ns_cap["coop2"], (_SMEM_BUDGET - 2 * 64 * sn * 2) // ring)
            if ns < 3:
                continue
            if best is None or (ns, sn) > best[:2]:
                best = (ns, sn)
        if best is None:
            return None
        ns, sn = best
        return ns, (0 if sn == bn else sn)

    def _stage_rule_ok(bm: int, bn: int, ns: int) -> bool:
        if ns < 3:
            return False
        return not (bm * bn < 128 * 192 and ns < 4)

    nt = (not trans_a) and trans_b
    out = []

    for bm in (64, 128):
        for bn in (64, 128, 192, 256):
            if (bm * bn) // 128 > max_accum_regs:
                continue
            # 64x192 is offered only unsplit, where its grid fits one wave and m holds at
            # least 8 rows: there it measured faster than every tile it displaced; split,
            # past one wave, or at m below 8 (where split tiles measured faster) it
            # measured slower. NT runs it on a 3-deep ring of 128-wide K steps; the other
            # layouts, whose B or A tile is MN-major, on a 5-deep ring of 64-wide steps,
            # which measured 1.020-1.027x geomean against 0.994-1.022x for the 3-deep one.
            one_wave_192 = bn == 192
            if one_wave_192 and (
                bm != 64 or m < 8 or math.ceil(m / bm) * math.ceil(n / bn) > sm_count
            ):
                continue
            if one_wave_192:
                rings = [(128, 3)] if nt else [(64, 5)]
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
        # An 8-deep ring measured 2-9% faster than 6 on every shape this band takes;
        # 7 is slower than both.
        out.append(_Cand("coop2s", 128, 64, 64, 8))

    if nt:
        if m <= 128 and m % 64 == 0 and k % 128 == 0:
            cluster_m = m // 64
            for bn in (64, 128, 192, 256):
                if n % bn:
                    continue
                ns = _ns_basic(64, bn, 128)
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
        for bn in (64, 128, 176, 192, 256):
            for bk in (32, 64, 128):
                # 176 fills the waves of some N that 192 and 256 leave a partial last
                # wave. Only as whole tiles: with an N tail its extra tiles each pay an
                # epilogue the score does not count (4096x7168x2048 measured a tie with
                # 256). It is one WGMMA only over 64-wide K steps.
                if bn == 176 and (n % bn or bk != 64):
                    continue
                d = _coop2_ns_sn(bn, bk)
                if d is None:
                    continue
                ns, sn = d
                if not _stage_rule_ok(128, bn, ns):
                    continue
                coop2_ok = bn >= 176
                mn_tiles = math.ceil(m / 128) * math.ceil(n / bn)
                if coop2_ok and mn_tiles >= sm_count:
                    out.append(_Cand("coop2", 128, bn, bk, ns, stage_n=sn))
                k_iters = math.ceil(k / bk)
                ns_sk = min(ns, ns_cap["coop2_splitk"])
                if ns_sk >= 3 and bn <= 128:
                    # 15/16 of a wave where no 128-row tile runs past ``m`` (128x4096x12288:
                    # 34.6 us on 128 CTAs against 36.7); tiles with rows past ``m`` measured
                    # slower on such a grid and need a full wave.
                    min_ctas = -(-15 * sm_count // 16) if m % 128 == 0 else sm_count
                    for sk in (2, 4, 8):
                        if k_iters % sk == 0 and k_iters // sk >= 4 and mn_tiles * sk >= min_ctas:
                            out.append(_Cand("coop2_splitk", 128, bn, bk, ns_sk, split_k=sk))
    return out


def swap_ab_grid_underfills(n: int, sm_count: int) -> bool:
    """Whether ``ceil(n / _SWAP_AB_BLOCK_NN)`` CTAs sit below three-eighths of a wave.

    The measured n boundary where the operand-swapped grid loses its width
    advantage, written once for its two consumers: ``_swap_ab_stages`` returns
    None below it (the tiny-m band falls to split-K or the plain tile), and
    :func:`two_row_gemv_wins` claims part of this underfilled band at ``m == 2``
    for ``GemvKernel``. Retune the two together.
    """
    return -(-n // _SWAP_AB_BLOCK_NN) * 8 < sm_count * 3


def _tile_fill(m: int, n: int, block_n: int, sm_count: int) -> float:
    """Share of the tensor-core time 128 x ``block_n`` tiles spend on output elements.

    The grid runs ``ceil(tiles / sm_count)`` full waves; a partial last wave and the
    columns an N tail tile computes past ``n`` are both lost.
    """
    tiles = math.ceil(m / 128) * math.ceil(n / block_n)
    return m * n / (math.ceil(tiles / sm_count) * sm_count * 128 * block_n)


def _pingpong_config(m: int, n: int, k: int, coop2_block_n: int, sm_count: int) -> Optional[dict]:
    """The ping-pong config to run in place of a coop2 tile, or None to keep coop2.

    A ping-pong consumer runs its tile's epilogue under the other consumer's
    mainloop, which buys most where a tile has few K steps; a coop2 tile pays its
    epilogue between two mainloops. A ping-pong tile is 176 or 192 wide (one
    consumer's accumulators hold 128 x 192 at most; 128 wide ran 10% slower than
    coop2 on every shape measured), so it is taken where that width fills the waves
    at least as well as the coop2 width. Measured on H200 over M 512-8192, N
    1536-24576, K 256-7168: up to 64 K steps that comparison decides; past 64 the
    ping-pong tile needs a fill 0.05 better to win. Under 8 K steps a tile's mainloop
    is shorter than its epilogue, and under 8 row tiles (M below 1024) coop2 kept
    the lead; both stay coop2.
    """
    k_steps = math.ceil(k / 64)
    # The ping-pong epilogue stores through TMA, which needs 16-byte output rows.
    if k_steps < 8 or math.ceil(m / 128) < 8 or n % 8:
        return None
    width = max((176, 192), key=lambda bn: (_tile_fill(m, n, bn, sm_count), bn))
    margin = 0.0 if k_steps <= 64 else 0.05
    if _tile_fill(m, n, width, sm_count) < _tile_fill(m, n, coop2_block_n, sm_count) + margin:
        return None
    return {
        "pingpong": True,
        "block_n": width,
        "block_k": 64,
        "num_stages": 4,
        # The widest epilogue chunk that divides the tile, is a whole number of 16-byte
        # TMA rows and leaves the 192-wide accumulators unspilled.
        "stage_n": 16 if width == 176 else 32,
    }


def _group_size_m(m: int, n: int, k: int, block_n: int, l2_bytes: int) -> int:
    """Row tiles one raster group of a persistent coop2 or ping-pong grid spans.

    A CTA takes its tiles a group of rows at a time, every column of the group before
    the next group. Groups of 16 rows give a wave 16 rows by a few columns. Where the
    grid has more row tiles than column tiles (the case CUTLASS's SM90 scheduler
    rasterizes along N) and the whole ``B`` fits in one L2 partition, groups of 4 rows
    give a wave whole rows, and every wave reads every ``B`` tile from L2. At the power
    limit fewer L2 reads leave a higher clock. A ``B`` larger than one L2 partition
    keeps 16-row groups, since 4-row groups would read more of it from DRAM.
    """
    # The dense family's operands are 16-bit.
    elem_bytes = 2
    # SM90's L2 is two partitions, each caching for the SMs attached to it, so data
    # every SM reads is held in both.
    l2_partitions = 2
    row_tiles = -(-m // 128)
    col_tiles = -(-n // block_n)
    if row_tiles > col_tiles and n * k * elem_bytes <= l2_bytes // l2_partitions:
        return 4
    return 16


def _tiny_m_split_k(k: int) -> int:
    """K slices of the m <= 8 split-K tile: 4 while each slice keeps 12 K-tiles, else 2, else 1.

    Shorter slices pay the warp-specialized pipeline's fill and drain.
    """
    k_iters = math.ceil(k / 128)
    return next((sk for sk in (4, 2) if k_iters % sk == 0 and k_iters // sk >= 12), 1)


def two_row_gemv_wins(n: int, k: int, sm_count: int) -> bool:
    """Whether ``GemvKernel`` beats the tiny-m tile on an ``m == 2`` NT call.

    Only where the operand-swapped grid underfills (:func:`swap_ab_grid_underfills`).
    There the tiny-m tile slices K four ways when that leaves every slice 12 K-tiles
    (:func:`_tiny_m_split_k`), and once the weight also holds ``2**23`` elements those
    slices stream it faster than the GEMV's per-column reduction (2x2112x7168: 13.3
    against 14.8 us); otherwise the GEMV wins.
    """
    if not swap_ab_grid_underfills(n, sm_count):
        return False
    return _tiny_m_split_k(k) < 4 or n * k < 1 << 23


def _swap_ab_stages(n: int, sm_count: int) -> Optional[int]:
    """``num_stages`` for the tiny-m swap_ab kernel, or None if it underfills.

    ``_gemm_swap_ab_kernel`` puts ``n`` on the 64-row WGMMA axis, so its grid
    is ``ceil(n / _SWAP_AB_BLOCK_NN)`` CTAs — the whole point being that this is
    twice the CTA count of the ``block_n = 128`` output tiling, with no padded
    ``A`` re-read. Below three-eighths of a wave
    (``swap_ab_grid_underfills``) that advantage is gone and the split-K path
    (which multiplies its own grid by ``split_k``) wins.

    The stage count falls as the grid grows: with more CTAs resident the device
    already has enough loads in flight, and a deeper ring only costs SMEM. Both
    ends are measured points; the boundary between them is interpolated.
    """
    if swap_ab_grid_underfills(n, sm_count):
        return None
    ctas = -(-n // _SWAP_AB_BLOCK_NN)
    return 4 if ctas * 4 >= sm_count * 3 else 8


@functools.lru_cache(maxsize=512)
def _best_config_cached(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    sm_count: int,
    l2_bytes: int,
    cal: _Calibration,
) -> dict:
    """Cached selection body of :func:`best_config` — do not mutate results."""

    def _score_us(cd: _Cand, m: int, n: int, k: int, sm_count: int, cal: _Calibration) -> float:
        """Predicted microseconds; only the relative ordering is meaningful."""
        bm, bn, sk = cd.block_m, cd.block_n, cd.split_k
        elem, elem_out = 2, 2
        num_blocks = math.ceil(m / bm) * math.ceil(n / bn) * sk
        num_waves = math.ceil(num_blocks / sm_count)
        wave_eff = num_blocks / (num_waves * sm_count)
        k_exp = k / sk
        ws = 4 if sk > 1 else elem_out

        l2_ab = k_exp * (bm + bn) * elem
        l1_ab = k_exp * (bm + bn) * elem
        l1_tc = k_exp * (max(64, bm) + bn) * elem + bm * bn * ws
        cd_bytes = bm * bn * ws

        l2_us = (l2_ab + cd_bytes) * num_blocks / (cal.l2_tbps * 1e6)
        l1_us = (l1_ab + l1_tc + cd_bytes) * num_blocks / (cal.l1_tbps * 1e6)
        tc_us = 2.0 * m * n * k / (cal.tc_tflops(cd.structure) * 1e6)
        issue_us = (k_exp / cd.block_k) * num_waves * cal.issue_ns * 1e-3

        us = max(tc_us, l1_us, l2_us) / wave_eff + issue_us
        if sk > 1:
            red_bytes = sk * m * n * 4 + m * n * elem_out
            us += red_bytes / (cal.reduce_tbps * 1e6) + cal.launch_us
        return us

    def _tiny_m_config(n: int, k: int, sm_count: int) -> dict:
        """m <= 8 NT band: bandwidth-regime rule, not the compute-regime score.

        The score above divides byte terms by wave efficiency, which is right when
        SMs bound the shape and wrong at m <= 8, where the kernel is a weight stream
        (arithmetic intensity ~= m) and DRAM is shared across the CTAs carrying it.

        - long K: grid-z split-K on the 64-row tile, ``split_k=4`` while the K-tile
          count leaves at least 12 iterations per slice; shorter slices pay the
          warp-specialized pipeline's fill and drain.
        - short K: the plain 64-row tile. Its slices are too short to amortize the
          reduce, and ``simple`` needs ``m % block_m == 0``, never true here.

        Both use ``block_n=128``: half the CTAs of ``bn=64`` still saturate the
        weight stream, and it halves the padded-``A`` re-read through L2. Wherever
        the operand-swapped kernel's grid fills the device it removes that re-read
        instead, and wins (:func:`_swap_ab_stages`); these rules serve what it
        leaves.
        """
        tiny_m_block_n = 128
        stages = _swap_ab_stages(n, sm_count)
        if stages is not None:
            return {
                "swap_ab": True,
                "block_nn": _SWAP_AB_BLOCK_NN,
                "block_k": 128,
                "num_stages": stages,
            }
        sk = _tiny_m_split_k(k)
        if sk > 1:
            return {
                "block_m": 64,
                "block_n": tiny_m_block_n,
                "block_k": 128,
                "num_stages": 4,
                "panel_size": 16,
                "split_k": sk,
            }
        return {
            "block_m": 64,
            "block_n": tiny_m_block_n,
            "block_k": 128,
            "num_stages": 4,
            "panel_size": 8,
            "split_k": 1,
        }

    if m <= 8 and not trans_a and trans_b:
        return _tiny_m_config(n, k, sm_count)
    cands = _enumerate(m, n, k, trans_a, trans_b, sm_count)
    if not cands:
        raise AssertionError(
            f"no config candidate for {m}x{n}x{k} (trans_a={trans_a}, "
            f"trans_b={trans_b}, sm_count={sm_count})"
        )
    best = min(cands, key=lambda c: (_score_us(c, m, n, k, sm_count, cal), -c.block_k))
    if best.structure == "coop2":
        cfg = _pingpong_config(m, n, k, best.block_n, sm_count) or best.to_config()
        cfg["group_size_m"] = _group_size_m(m, n, k, cfg["block_n"], l2_bytes)
        return cfg
    return best.to_config()


def best_config(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    sm_count: int,
    device_name: str,
    *,
    l2_bytes: int,
) -> Optional[dict]:
    """Return the analytically selected ``GemmTMAKernel`` config for a shape.

    Args:
        m: Logical GEMM rows of the output.
        n: Logical GEMM columns of the output.
        k: Contraction dim.
        trans_a: ``A`` stored ``[k, m]`` when True.
        trans_b: ``B`` stored ``[n, k]`` when True.
        sm_count: Streaming multiprocessors of the target device.
        device_name: The target device as CUDA names it, which selects the
            profile the ranking constants come from.
        l2_bytes: L2 cache of the target device, in bytes; it sets the raster group
            of a persistent coop2 or ping-pong tile.

    Returns:
        ``None`` when no profile carries this board's ranking constants, so the
        caller takes its own default rather than a ranking measured elsewhere.
        Otherwise a config dict in ``GemmTMAKernel`` schema — either the single-consumer
        form (``block_m/block_n/block_k/num_stages/panel_size/split_k``,
        optionally ``simple``) or a structure-flagged form (``coop2s`` /
        ``coop2`` / ``coop2_splitk`` / ``pingpong`` / ``swap_ab``). A fresh dict per
        call: the selection itself is
        cached, but callers hold (and may mutate) their own copy — sibling
        kernel families mutate ``self.config`` in place, and a shared cached
        dict would be poisoned by that idiom.
    """
    cal = _calibration(device_name)
    if cal is None:
        return None
    return dict(_best_config_cached(m, n, k, trans_a, trans_b, sm_count, l2_bytes, cal))


def gemv_config(k: int) -> dict:
    """SM90 ``GemvKernel`` config rule for its one-row bands (single-row / single-column GEMV).

    GEMV is HBM-bandwidth bound; the lever is memory-level parallelism per
    output row via ``reduce_threads > 32`` (cross-warp SMEM tree reduction,
    auto-lowered from ``tvm_thread_allreduce``). These bands read at close to the
    per-shape bandwidth ceiling, where a ``block_n=8 / reduce_threads=32`` default
    leaves most of it on the table. Bands:

    - very deep rows (k >= 12288, e.g. 7168x16384): 2 rows/block, 64
      threads/row — a 128-way reduction tree over a long row costs more than
      the bandwidth it buys, and 2 rows/block improves wave quantization;
    - mid-deep rows (k >= 6144, decode qkv-a / mlp-up): a 256-lane
      reduction still runs >= 3 pipeline iterations, and the extra per-row
      memory-level parallelism beats rt=128 on a cold read;
    - shorter rows degenerate to ~1 iteration at 256 lanes and stay on
      rt=128, which maximizes per-row MLP to saturate HBM.
    """
    if k >= 12288:
        return {"block_n": 2, "reduce_threads": 64, "num_stages": 5}
    if k >= 6144:
        return {"block_n": 1, "reduce_threads": 256, "num_stages": 4}
    return {"block_n": 1, "reduce_threads": 128, "num_stages": 4}


def small_batch_config(n: int, k: int, sm_count: int) -> dict:
    """``GemvKernel`` config rule for its ``lhs_rows`` band (m == 2 NT).

    Modal best across the dispatched band: one output column per block, a
    64-lane reduction over K, 4-deep cp.async ring. One exception: when the
    grid alone fills the device's CTA slots (block_n=1 -> n CTAs vs the
    32-per-SM cap) AND the per-thread K loop is long, resident warps already
    hide the load latency and the deep ring only adds sync overhead, so the
    2-stage ring wins.
    """
    cfg = {"block_n": 1, "reduce_threads": 64, "num_stages": 4}
    k_iters = math.ceil(k / (cfg["reduce_threads"] * 8))
    if n >= 28 * sm_count and k_iters >= 12:
        cfg["num_stages"] = 2
    return cfg


def small_m_splitk_config(
    m: int, n: int, k: int, sm_count: int, device_name: str
) -> Optional[dict]:
    """Select the split-K basic GEMM band fitted on a calibrated board for a small-M NT call."""
    block_k = 128
    k_tiles = k // block_k
    # The row count the small-M split-K band was fitted at, by calibration key. A board
    # without an entry has no band.
    small_m_splitk_m = {"h200": 32}
    band_m = small_m_splitk_m.get(calibration_key(device_name))
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

    ``block_m`` is one or two consumer warpgroups of 64 rows. ``block_n`` 256
    spills: its accumulator and scaled partial together want 256 registers per
    thread.

    - ``block_n``: 128, except where that grid does not fill one wave, which
      leaves SMs idle for the whole launch. The narrow tile doubles the grid,
      and the re-read of ``A`` it costs stays in L2. At ``m <= 8`` the ``A``
      tile is mostly padding and the ``B`` re-read decides instead, so the wide
      tile wins even under a short grid. Per-tensor, with no scaled
      partial beside the accumulator, 192 also fits; it is taken where its waves,
      counted whole, cover fewer columns than 128's, since a partial last wave
      costs a full one. 176 replaces 192 where it fills as many waves and pads
      ``n`` no further: each wave then multiplies 176 columns instead of 192.
    - ring depth: the deepest the budget allows, except in block128, where a
      K-step ends in a promotion the next WGMMA cannot start under. There the
      ring covers nothing beyond the fill, and 4 stages is enough — 3 over a
      long K axis, 5 over a short one or a weight stream.
    - split-K: only where the sliced grid still fits one wave and every slice
      keeps enough K-tiles to amortize the fill and the fp32 workspace.
    - one row tile (``m <= 128``): 64 rows and one consumer instead, where that
      grid, unsplit, holds at least as many CTAs as the 128-row one with its K
      split. It reads half the ``A`` rows a CTA, needs no fp32 workspace or
      reduction, and takes the narrowest ``block_n`` of 32, 64 and 128 whose grid
      fits one wave, under the deepest ring the budget holds. Under block128 the
      128-row grid keeps its split: each K-step's promotion runs in series with
      the next WGMMA, so a CTA's time follows its K-steps, and an unsplit grid
      would run four times as many of them.
    """
    half_m = 64

    def _fp8_ws_stages(block_m: int, block_n: int) -> int:
        """Deepest ring the SMEM budget allows for one warp-specialized FP8 tile.

        A stage holds the ``A`` rows of every consumer, one ``block_n`` ``B`` tile,
        the block128 scale vectors when staged, and its full and empty mbarriers.
        Each consumer's epilogue staging tile is live alongside the ring, and exists
        only for ``m`` over one consumer's rows.
        """
        # A stage's full and empty mbarriers, 8 bytes each, come out of the same budget.
        mbarrier_bytes = 8
        per_stage = (
            (block_m + block_n) * 128 + block_scaled * (block_m + block_n) * 4 + 2 * mbarrier_bytes
        )
        epilogue = block_m * block_n * 2 if m > half_m else 0
        return (_SMEM_BUDGET - epilogue) // per_stage

    def _waves(block_n: int) -> int:
        return -(-(-(-m // 128) * -(-n // block_n)) // sm_count)

    def _waved_columns(block_n: int) -> int:
        return _waves(block_n) * block_n

    k_iters = -(-k // 128)
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
    # Shortest K slice the FP8 split-K path pays for.
    fp8_min_slice_k_tiles = 12
    split_k = 1
    for candidate in (4, 2):
        if (
            tiles * candidate <= sm_count
            and k_iters % candidate == 0
            and k_iters // candidate >= fp8_min_slice_k_tiles
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

    - ``block_n``: the narrowest of 32, 64, 96, 128, 160 and 256 whose grid fits one wave.
      A decode shape is bound by the bytes each SM reads, ``(64 + block_n) * k`` a tile, so
      the narrowest tile whose grid still has an SM for every tile reads the fewest. The
      widths between are left out: TileLang issues a ``T.wgmma_gemm`` as
      ``gcd(block_n, 256)``-wide instructions, so 48 runs as three n16 WGMMAs, each
      re-reading A (measured: 48 no faster than 64 on 64 x 6144 x 4096 at two-thirds the
      grid). This width was the fastest measured on every decode row of the manifest.
    - ``block_k``: 256 where ``a`` is exactly one full 64-row tile and the tile is at most
      160 wide, which halves the barrier round trips a K axis takes (measured 0.5-6.3%
      faster on every ``m = 64`` row); 128 otherwise. At ``m = 1`` TileLang cannot lay out
      a one-row 256-byte box; at ``m = 128`` and on the 256-wide tile the 256-wide step
      measured slower.
    - ring depth: the deepest the shared memory holds, at most one stage a K step.
    """
    block_m = 64
    widths = (32, 64, 96, 128, 160, 256)
    tiles_m = -(-m // block_m)
    block_n = next((w for w in widths if tiles_m * -(-n // w) <= sm_count), widths[-1])
    block_k = 256 if m == block_m and block_n <= 160 else 128
    # A stage's A and B tiles and its full and empty mbarriers, 8 bytes each.
    stage = (block_m + block_n) * block_k + 16
    # The output staging tile, which exists only where a row tile can be full.
    staging = block_m * block_n * 2 if m >= block_m else 0
    num_stages = min(-(-k // block_k), (_SMEM_BUDGET - staging) // stage)
    return {"block_n": block_n, "num_stages": num_stages, "group_size_m": 8, "block_k": block_k}
