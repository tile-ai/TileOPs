"""Analytic config selection for the dense GEMM kernel family (SM90).

Picks the kernel structure — single-consumer (``basic``), grid-z split-K
(``splitk``), 2-consumer cooperative (``coop2``), or cooperative split-K
(``coop2_splitk``) — and its tile configuration for an arbitrary GEMM shape
in microseconds, without autotuning. The module also owns the measured
config bands of the bandwidth-mode kernels (``gemv_config``,
``small_batch_config``), so the family's per-shape configuration lives in one
place; which kernel serves a call is a separate question, answered by each
kernel's own ``applies``. The scored path follows DeepGEMM's SM90
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
device's from ``_CALIBRATIONS``. A board absent from that table is not ranked
at all: :func:`best_config` returns ``None`` and the kernel takes its own
default, because ranking with another board's numbers is worse than not
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

__all__ = [
    "SWAP_AB_MPAD",
    "best_config",
    "gemv_config",
    "small_batch_config",
    "swap_ab_grid_underfills",
]

_SMEM_BUDGET = 227 * 1024
_MAX_ACCUM_REGS = 200

TINY_M_BLOCK_N = 128

_SWAP_AB_BLOCK_NN = 64
SWAP_AB_MPAD = 8

_NS_CAP = {"basic": 4, "splitk": 4, "coop2": 6, "coop2_splitk": 4}


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


#: One entry per board the scorer has been calibrated on, by the name CUDA
#: reports. A board absent here is not ranked; see :func:`best_config`.
_CALIBRATIONS = {
    "NVIDIA H200": _Calibration(
        l1_tbps=33.0,
        l2_tbps=17.5,
        reduce_tbps=1.5,
        launch_us=2.25,
        issue_ns=150.0,
        tensor_core_tflops=(
            ("basic", 420.0),
            ("coop2", 525.0),
            ("coop2_splitk", 525.0),
            ("splitk", 420.0),
        ),
    ),
}


def _calibration(device_name: str) -> Optional[_Calibration]:
    """The calibration measured on *device_name*, or ``None`` for a board without one."""
    return _CALIBRATIONS.get(device_name)


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

    def to_config(self) -> dict:
        if self.structure == "coop2":
            return {
                "coop2": True,
                "block_n": self.block_n,
                "block_k": self.block_k,
                "num_stages": self.num_stages,
                "group_size_m": 16,
                "stage_n": self.stage_n,
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


def _strip_width(bm: int, bn: int, sm_count: int) -> int:
    """L2-footprint-minimizing rasterization panel width.

    DeepGEMM's ``get_num_1d_blocks_per_group`` (scheduler/gemm.cuh): choose
    the group size whose resident-CTA footprint ``g*bm + ceil(SMs/g)*bn``
    is smallest.
    """
    return min((4, 8, 10, 16), key=lambda g: g * bm + math.ceil(sm_count / g) * bn)


def _ns_basic(bm: int, bn: int, bk: int) -> int:
    ring = (bm + bn) * bk * 2
    return min(_NS_CAP["basic"], (_SMEM_BUDGET - bm * bn * 2) // ring)


def _coop2_ns_sn(bn: int, bk: int):
    """Deepest ring the SMEM budget allows; shrink the epilogue staging
    chunk (stage_n) when that buys another pipeline stage."""
    ring = (128 + bn) * bk * 2
    best = None
    for sn in (bn, bn // 2, bn // 4):
        if sn < 32 or bn % sn:
            continue
        ns = min(_NS_CAP["coop2"], (_SMEM_BUDGET - 2 * 64 * sn * 2) // ring)
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


def _enumerate(m: int, n: int, k: int, trans_a: bool, trans_b: bool, sm_count: int) -> list:
    nt = (not trans_a) and trans_b
    out = []

    for bm in (64, 128):
        for bn in (64, 128, 256):
            if (bm * bn) // 128 > _MAX_ACCUM_REGS:
                continue
            for bk in (32, 64, 128):
                ns = _ns_basic(bm, bn, bk)
                if not _stage_rule_ok(bm, bn, ns):
                    continue
                ps = _strip_width(bm, bn, sm_count)
                out.append(_Cand("basic", bm, bn, bk, ns, panel_size=ps))
                k_iters = math.ceil(k / bk)
                for sk in (2, 4, 8):
                    if k_iters % sk == 0 and k_iters // sk >= 4:
                        out.append(_Cand("splitk", bm, bn, bk, ns, split_k=sk, panel_size=ps))

    if nt:
        for bn in (64, 128, 192, 256):
            for bk in (32, 64, 128):
                d = _coop2_ns_sn(bn, bk)
                if d is None:
                    continue
                ns, sn = d
                if not _stage_rule_ok(128, bn, ns):
                    continue
                coop2_ok = bn >= 192
                mn_tiles = math.ceil(m / 128) * math.ceil(n / bn)
                if coop2_ok and mn_tiles >= sm_count:
                    out.append(_Cand("coop2", 128, bn, bk, ns, stage_n=sn))
                k_iters = math.ceil(k / bk)
                ns_sk = min(ns, _NS_CAP["coop2_splitk"])
                if ns_sk >= 3 and bn <= 128:
                    for sk in (2, 4, 8):
                        if k_iters % sk == 0 and k_iters // sk >= 4 and mn_tiles * sk >= sm_count:
                            out.append(_Cand("coop2_splitk", 128, bn, bk, ns_sk, split_k=sk))
    return out


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


def swap_ab_grid_underfills(n: int, sm_count: int) -> bool:
    """Whether ``ceil(n / _SWAP_AB_BLOCK_NN)`` CTAs sit below three-eighths of a wave.

    The measured n boundary where the operand-swapped grid loses its width
    advantage, written once for its two consumers: ``_swap_ab_stages`` returns
    None below it (the tiny-m band falls to split-K or the plain tile), and
    ``SmallBatchGemmKernel.applies`` claims exactly this underfilled band at
    ``m == 2`` — that handoff has no gap and no overlap. Retune the two
    together.
    """
    return -(-n // _SWAP_AB_BLOCK_NN) * 8 < sm_count * 3


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
    stages = _swap_ab_stages(n, sm_count)
    if stages is not None:
        return {
            "swap_ab": True,
            "block_nn": _SWAP_AB_BLOCK_NN,
            "block_k": 128,
            "num_stages": stages,
        }
    k_iters = math.ceil(k / 128)
    for sk in (4, 2):
        if k_iters % sk == 0 and k_iters // sk >= 12:
            return {
                "block_m": 64,
                "block_n": TINY_M_BLOCK_N,
                "block_k": 128,
                "num_stages": 4,
                "panel_size": 16,
                "split_k": sk,
            }
    return {
        "block_m": 64,
        "block_n": TINY_M_BLOCK_N,
        "block_k": 128,
        "num_stages": 4,
        "panel_size": 8,
        "split_k": 1,
    }


@functools.lru_cache(maxsize=512)
def _best_config_cached(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, sm_count: int, cal: _Calibration
) -> dict:
    """Cached selection body of :func:`best_config` — do not mutate results."""
    if m <= 8 and not trans_a and trans_b:
        return _tiny_m_config(n, k, sm_count)
    cands = _enumerate(m, n, k, trans_a, trans_b, sm_count)
    if not cands:
        raise AssertionError(
            f"no config candidate for {m}x{n}x{k} (trans_a={trans_a}, "
            f"trans_b={trans_b}, sm_count={sm_count})"
        )
    best = min(cands, key=lambda c: (_score_us(c, m, n, k, sm_count, cal), -c.block_k))
    return best.to_config()


def best_config(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool, sm_count: int, device_name: str
) -> Optional[dict]:
    """Return the analytically selected ``GemmKernel`` config for a shape.

    Args:
        m: Logical GEMM rows of the output.
        n: Logical GEMM columns of the output.
        k: Contraction dim.
        trans_a: ``A`` stored ``[k, m]`` when True.
        trans_b: ``B`` stored ``[n, k]`` when True.
        sm_count: Streaming multiprocessors of the target device.
        device_name: The target device as CUDA names it, which selects the
            profile the ranking constants come from.

    Returns:
        ``None`` when no profile carries this board's ranking constants, so the
        caller takes its own default rather than a ranking measured elsewhere.
        Otherwise a config dict in ``GemmKernel`` schema — either the single-consumer
        form (``block_m/block_n/block_k/num_stages/panel_size/split_k``,
        optionally ``simple``) or a structure-flagged form (``coop2`` /
        ``coop2_splitk``). A fresh dict per call: the selection itself is
        cached, but callers hold (and may mutate) their own copy — sibling
        kernel families mutate ``self.config`` in place, and a shared cached
        dict would be poisoned by that idiom.
    """
    cal = _calibration(device_name)
    if cal is None:
        return None
    return dict(_best_config_cached(m, n, k, trans_a, trans_b, sm_count, cal))


def gemv_config(k: int) -> dict:
    """SM90 ``GemvKernel`` config band (single-row / single-column GEMV).

    GEMV is HBM-bandwidth bound; the lever is memory-level parallelism per
    output row via ``reduce_threads > 32`` (cross-warp SMEM tree reduction,
    auto-lowered from ``tvm_thread_allreduce``). These bands read at close to the
    per-shape bandwidth ceiling, where a ``block_n=8 / reduce_threads=32`` default
    leaves most of it on the table. Bands:

    - very deep rows (k >= 12288, e.g. 7168x16384): 2 rows/block, 64
      threads/row — a 128-way reduction tree over a long row costs more than
      the bandwidth it buys, and 2 rows/block improves wave quantization;
    - mid-deep rows (k >= 6144, decode gate-up / attn-proj): a 256-lane
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
    """``SmallBatchGemmKernel`` (m == 2 NT band) config rule.

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
