"""Analytic config selection for the dense GEMM kernel family (SM90).

Picks the kernel structure — single-consumer (``basic``), grid-z split-K
(``splitk``), 2-consumer cooperative (``coop2``), or cooperative split-K
(``coop2_splitk``) — and its tile configuration for an arbitrary GEMM shape
in microseconds, without autotuning. The module also owns the measured
config bands of the bandwidth-mode kernels (``gemv_config``,
``small_batch_config``), so the family's per-shape configuration lives in one
place; which kernel serves a call is a separate question, answered by each
kernel's own region (``gemm_call.gemv_region`` / ``small_batch_region``). The scored path follows DeepGEMM's SM90
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

The constants below are *effective ranking constants*, not physical rates:
they were calibrated on H200 (132 SM) against a per-rep interleaved CUPTI
kernel-only sweep of 16 shapes x ~7 configs spanning all four structures
(geometric-mean regret 1.001, model top1 inside measured top2 on 15/16
shapes; every shape pinned in ``GemmKernel._TUNED_CONFIGS`` scored >= 99%
of its table entry). Re-calibrate when kernel structures change materially.

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

__all__ = ["best_config", "gemv_config", "small_batch_config"]

_SMEM_BUDGET = 227 * 1024  # SM90 per-CTA opt-in SMEM ceiling
_MAX_ACCUM_REGS = 200

# Validated num_stages ranges per structure in the shipped kernels.
_NS_CAP = {"basic": 4, "splitk": 4, "coop2": 6, "coop2_splitk": 4}

# Calibrated ranking constants (H200; see module docstring for protocol).
_L1_TBPS = 33.0      # aggregate SMEM<->core bandwidth proxy
_L2_TBPS = 17.5      # aggregate L2 bandwidth proxy
_TC_TFLOPS = {"basic": 420.0, "splitk": 420.0,
              "coop2": 525.0, "coop2_splitk": 525.0}
_RED_TBPS = 1.5      # split-K workspace round-trip bandwidth
_LAUNCH_US = 2.25    # reduce-pass launch + sync overhead
_ISSUE_NS = 150.0    # per-K-iteration issue/TMA overhead


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
            return {"coop2": True, "block_n": self.block_n,
                    "block_k": self.block_k, "num_stages": self.num_stages,
                    "group_size_m": 16, "stage_n": self.stage_n}
        if self.structure == "coop2_splitk":
            return {"coop2_splitk": True, "block_n": self.block_n,
                    "block_k": self.block_k, "num_stages": self.num_stages,
                    "split_k": self.split_k}
        return {"block_m": self.block_m, "block_n": self.block_n,
                "block_k": self.block_k, "num_stages": self.num_stages,
                "panel_size": self.panel_size, "split_k": self.split_k}


def _strip_width(bm: int, bn: int, sm_count: int) -> int:
    """L2-footprint-minimizing rasterization panel width.

    DeepGEMM's ``get_num_1d_blocks_per_group`` (scheduler/gemm.cuh): choose
    the group size whose resident-CTA footprint ``g*bm + ceil(SMs/g)*bn``
    is smallest.
    """
    return min((4, 8, 10, 16),
               key=lambda g: g * bm + math.ceil(sm_count / g) * bn)


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
    # DeepGEMM sm90.hpp stage rule: < 3 stages cannot hide TMA latency;
    # small tiles (bm*bn < 128*192) need at least 4.
    if ns < 3:
        return False
    return not (bm * bn < 128 * 192 and ns < 4)


def _enumerate(m: int, n: int, k: int, trans_a: bool, trans_b: bool,
               sm_count: int) -> list:
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
                        out.append(_Cand("splitk", bm, bn, bk, ns,
                                         split_k=sk, panel_size=ps))

    if nt:
        for bn in (64, 128, 192, 256):
            for bk in (32, 64, 128):
                d = _coop2_ns_sn(bn, bk)
                if d is None:
                    continue
                ns, sn = d
                if not _stage_rule_ok(128, bn, ns):
                    continue
                # Plain coop2 never measured a win below bn=192 (narrow tiles
                # cannot feed two consumer warpgroups); split-K coop2 is the
                # structure that owns the narrow-bn regime.
                coop2_ok = bn >= 192
                mn_tiles = math.ceil(m / 128) * math.ceil(n / bn)
                # Persistent coop2 below one full wave of tiles is measured
                # far slower than basic (idle SMs + static-wave overhead);
                # the same regime DeepGEMM guards by disabling multicast.
                if coop2_ok and mn_tiles >= sm_count:
                    out.append(_Cand("coop2", 128, bn, bk, ns, stage_n=sn))
                k_iters = math.ceil(k / bk)
                ns_sk = min(ns, _NS_CAP["coop2_splitk"])
                if ns_sk >= 3 and bn <= 128:
                    for sk in (2, 4, 8):
                        if (k_iters % sk == 0 and k_iters // sk >= 4
                                and mn_tiles * sk >= sm_count):
                            out.append(_Cand("coop2_splitk", 128, bn, bk,
                                             ns_sk, split_k=sk))
    return out


def _score_us(cd: _Cand, m: int, n: int, k: int, sm_count: int) -> float:
    """Predicted microseconds; only the relative ordering is meaningful."""
    bm, bn, sk = cd.block_m, cd.block_n, cd.split_k
    elem, elem_out = 2, 2
    num_blocks = math.ceil(m / bm) * math.ceil(n / bn) * sk
    num_waves = math.ceil(num_blocks / sm_count)
    wave_eff = num_blocks / (num_waves * sm_count)
    k_exp = k / sk
    ws = 4 if sk > 1 else elem_out  # split-K mainloop writes fp32 workspace

    l2_ab = k_exp * (bm + bn) * elem
    l1_ab = k_exp * (bm + bn) * elem
    l1_tc = k_exp * (max(64, bm) + bn) * elem + bm * bn * ws
    cd_bytes = bm * bn * ws

    l2_us = (l2_ab + cd_bytes) * num_blocks / (_L2_TBPS * 1e6)
    l1_us = (l1_ab + l1_tc + cd_bytes) * num_blocks / (_L1_TBPS * 1e6)
    tc_us = 2.0 * m * n * k / (_TC_TFLOPS[cd.structure] * 1e6)
    issue_us = (k_exp / cd.block_k) * num_waves * _ISSUE_NS * 1e-3

    us = max(tc_us, l1_us, l2_us) / wave_eff + issue_us
    if sk > 1:
        red_bytes = sk * m * n * 4 + m * n * elem_out
        us += red_bytes / (_RED_TBPS * 1e6) + _LAUNCH_US
    return us


def _tiny_m_config(k: int) -> dict:
    """m <= 8 NT band: bandwidth-regime rule, not the compute-regime score.

    The score above divides byte terms by wave efficiency — right when SMs
    bound the shape, wrong at m <= 8 where the kernel is a weight stream
    (arithmetic intensity ~= m) and DRAM is shared across however many CTAs
    carry it. Ranked that way, full-wave narrow tiles (bn=64 split-K) beat
    the measured-best wide tiles (bn=128, ~half wave), costing 3-8%.

    Measured rule (H200 per-rep interleaved sweep, {2112x7168, 7168x2048,
    4096x7168} x m{2,4,8} x {full sb grid, split-K/simple/basic variants},
    winner reproduced in every cell; dh/wi1_small_batch/tinym_fit.py):

    - long K: grid-z split-K on the 64-row tile — split_k=4 when the K-tile
      count allows >= 12 iterations per slice (shorter slices pay the WS
      pipeline fill/drain: split_k=8 at 7 iters/slice measured 20% slower,
      and split_k=2 at 8 iters/slice lost to the plain tile on the 16-tile K);
    - short K: the plain (split_k=1) 64-row tile — split-K slices of a
      16-tile K are too short to amortize, and the ``simple`` structure is
      unavailable here (``_gemm_simple_kernel`` requires ``m % block_m == 0``,
      never true at m <= 8).

    Both use block_n=128: half the CTAs of bn=64 still saturate the weight
    stream, and the padded-A reread (block_m=64 vs m <= 8) through L2 halves.
    """
    k_iters = math.ceil(k / 128)
    for sk in (4, 2):
        if k_iters % sk == 0 and k_iters // sk >= 12:
            return {"block_m": 64, "block_n": 128, "block_k": 128,
                    "num_stages": 4, "panel_size": 16, "split_k": sk}
    return {"block_m": 64, "block_n": 128, "block_k": 128,
            "num_stages": 4, "panel_size": 8, "split_k": 1}


@functools.lru_cache(maxsize=512)
def best_config(m: int, n: int, k: int, trans_a: bool, trans_b: bool,
                sm_count: int = 132) -> dict:
    """Return the analytically selected ``GemmKernel`` config for a shape.

    Args:
        m: Logical GEMM rows of the output.
        n: Logical GEMM columns of the output.
        k: Contraction dim.
        trans_a: ``A`` stored ``[k, m]`` when True.
        trans_b: ``B`` stored ``[n, k]`` when True.
        sm_count: Streaming multiprocessors of the target device.

    Returns:
        A config dict in ``GemmKernel`` schema — either the single-consumer
        form (``block_m/block_n/block_k/num_stages/panel_size/split_k``,
        optionally ``simple``) or a structure-flagged form (``coop2`` /
        ``coop2_splitk``).
    """
    if m <= 8 and not trans_a and trans_b:
        return _tiny_m_config(k)
    cands = _enumerate(m, n, k, trans_a, trans_b, sm_count)
    if not cands:  # degenerate shapes: fall back to the modal default
        return {"block_m": 128, "block_n": 128, "block_k": 64,
                "num_stages": 4, "panel_size": 16, "split_k": 1}
    # Residual score ties break toward larger block_k (fewer K iterations,
    # measured faster whenever the byte model cannot separate candidates).
    best = min(cands, key=lambda c: (_score_us(c, m, n, k, sm_count),
                                     -c.block_k))
    return best.to_config()


def gemv_config(k: int) -> dict:
    """SM90 ``GemvKernel`` config band (single-row / single-column GEMV).

    GEMV is HBM-bandwidth bound; the lever is memory-level parallelism per
    output row via ``reduce_threads > 32`` (cross-warp SMEM tree reduction,
    auto-lowered from ``tvm_thread_allreduce``). Measured on H200 these reach
    90-102% of the per-shape read-BW ceiling and beat cuBLAS 1.1-2.4x, vs the
    old ``block_n=8 / reduce_threads=32`` default (0.57-0.87x cuBLAS). Bands:

    - very deep rows (k >= 12288, e.g. 7168x16384): 2 rows/block, 64
      threads/row — a 128-way reduction tree over a long row costs more than
      the bandwidth it buys, and 2 rows/block improves wave quantization;
    - mid-deep rows (k >= 6144, decode gate-up / attn-proj): a 256-lane
      reduction still runs >= 3 pipeline iterations and the extra per-row
      memory-level parallelism beats rt=128 by 9-10% under the cold-read
      protocol (two independent 30-rep interleaved rounds, H200);
    - shorter rows degenerate to ~1 iteration at 256 lanes and stay on
      rt=128, which maximizes per-row MLP to saturate HBM.
    """
    if k >= 12288:
        return {"block_n": 2, "reduce_threads": 64, "num_stages": 5}
    if k >= 6144:
        return {"block_n": 1, "reduce_threads": 256, "num_stages": 4}
    return {"block_n": 1, "reduce_threads": 128, "num_stages": 4}


def small_batch_config(n: int, k: int, sm_count: int = 132) -> dict:
    """``SmallBatchGemmKernel`` (m == 2 NT band) config rule.

    Modal best across the dispatched band (H200 per-rep interleaved sweep):
    one output column per block, a 64-lane reduction over K, 4-deep cp.async
    ring. One measured exception: when the grid alone fills the device's CTA
    slots (block_n=1 -> n CTAs vs the 32-per-SM cap) AND the per-thread K
    loop is long, resident warps already hide the load latency and the deep
    ring only adds sync overhead — the 2-stage ring is ~9% faster there
    (attn-family 4096x7168 at m=2).
    """
    cfg = {"block_n": 1, "reduce_threads": 64, "num_stages": 4}
    k_iters = math.ceil(k / (64 * 8))  # reduce_threads * tile_k (fp16/bf16)
    if n >= 28 * sm_count and k_iters >= 12:
        cfg["num_stages"] = 2
    return cfg
