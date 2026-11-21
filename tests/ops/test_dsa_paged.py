import pytest
import torch

from tileops._kernels.attention.call_spec import DSAPagedCall as DSAPagedKernelCall
from tileops.attention import DSAPagedFwdOp
from tileops.backend import OpNotAvailableError
from workloads.attention.dsa import DSAPagedCall
from workloads.device import run_device
from workloads.numerics import Request, verify
from workloads.workload_base import manifest_call


def _call(**row):
    return manifest_call("DSAPagedFwdOp", {"KV": "uint8"}, DK=576, **row)


def _bf16_call(**row):
    return manifest_call("DSAPagedFwdOp", {"KV": "bfloat16"}, **row)


def _run(op, attn_sink):
    """The op as a function of the call's inputs, ``attn_sink`` last when the call has one."""
    if attn_sink is None:
        return op

    def run(*args):
        return op(*args[:5], attn_sink=args[5])

    return run


def _check(call, dead_request=False):
    """Compare the op with the reference; on the FP8 row the verification's negative controls
    must fail.

    Every other dead slot holds a position at or past its request's cache_seqlens instead
    of -1 where the request's pages have room past it: both are dead. With
    ``dead_request`` request 0 has no live slot at all.
    """
    workload = DSAPagedCall(call)
    q, kv_cache, block_table, cache_seqlens, indices, attn_sink = workload.gen_inputs()
    length = cache_seqlens.long()[:, None, None]
    room = block_table.shape[1] * kv_cache.shape[1] - length
    flat = torch.arange(indices.numel(), device=indices.device).view_as(indices)
    past = (length + flat // 2 % room.clamp_min(1)).to(indices.dtype)
    indices = torch.where((indices < 0) & (flat % 2 == 0) & (room > 0), past, indices)
    if dead_request:
        indices[0] = -1
    inputs = (q, kv_cache, block_table, cache_seqlens, indices)
    if attn_sink is not None:
        inputs = (*inputs, attn_sink)
    verify(
        workload.ref_program,
        inputs,
        evidence=workload.verification(*inputs),
        requests={"tileops": Request(_run(DSAPagedFwdOp(**call.arguments()), attn_sink), inputs)},
    )


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@pytest.mark.parametrize(
    "row",
    [
        # Cluster kernel with key splits.
        pytest.param(
            dict(S_q=1, H=128, K=256, NP=64, PS=64, W=8, cache_lens="repeat(500, 4)"),
            id="h128",
            marks=pytest.mark.smoke,
        ),
        # Single-CTA kernel, two query rows, dead trailing slots.
        pytest.param(
            dict(
                S_q=2,
                H=64,
                K=256,
                NP=40,
                PS=64,
                W=4,
                cache_lens="balanced_sizes(1001, 6)",
                sm_scale=0.05,
            ),
            id="h64-mtp-short",
            marks=pytest.mark.full,
        ),
        # No split; 8 blocks per CTA pair cycle every barrier phase.
        pytest.param(
            dict(S_q=1, H=128, K=512, NP=480, PS=64, W=10, cache_lens="repeat(600, 48)"),
            id="h128-unsplit",
            marks=pytest.mark.full,
        ),
        # Mostly empty splits; 32-row pages.
        pytest.param(
            dict(S_q=1, H=128, K=1024, NP=8, PS=32, W=4, cache_lens="repeat(100, 2)"),
            id="h128-empty-splits-page32",
            marks=pytest.mark.full,
        ),
    ],
)
def test_dsa_paged(row, arch) -> None:
    _check(_call(**row))


# GLM-5.3-Flash's latent rows: a topk off the 64-slot block, its last block's slots past topk
# dead; key splits.
_LATENT = dict(S_q=1, H=64, DK=512, K=2051, NP=80, PS=64, W=40, cache_lens=[2100, 300])
# DeepSeek-V4-Flash sink rows whose two key splits a CTA pair merges: request 0's five live
# blocks split 3 and 2, request 1's four 2 and 2, and request 2's one leaves the peer's half
# empty.
_PAIR = dict(
    S_q=1,
    H=64,
    DK=512,
    K=320,
    NP=45,
    PS=64,
    W=15,
    cache_lens=[900, 200, 40],
    some=["attn_sink"],
)
# A DeepSeek-V4-Flash sink row with a single split over two 64-head blocks.
_SINK_UNSPLIT = dict(
    S_q=1,
    H=128,
    DK=512,
    K=256,
    NP=216,
    PS=64,
    W=3,
    cache_lens="repeat(130, 72)",
    some=["attn_sink"],
)


@pytest.mark.sm90
@pytest.mark.parametrize(
    "row",
    [
        pytest.param(_LATENT, id="latent-topk-off-the-block", marks=pytest.mark.smoke),
        # A pair per 64-head block of two, its last key block past topk.
        pytest.param(
            dict(_PAIR, H=128, K=300, NP=33, W=11, cache_lens=[700, 200, 40]),
            id="pair-two-head-blocks-topk-off-the-block",
            marks=pytest.mark.full,
        ),
        # 64 key columns past the value, merged by a CTA pair; 32-row pages.
        pytest.param(
            dict(S_q=1, H=64, DK=576, K=256, NP=24, PS=32, W=8, cache_lens=[256, 100, 31]),
            id="key-tail",
            marks=pytest.mark.full,
        ),
        # Two 64-head blocks of a query row, each gathering every row.
        pytest.param(
            dict(S_q=1, H=128, DK=512, K=128, NP=16, PS=64, W=4, cache_lens=[200, 130]),
            id="h128",
            marks=pytest.mark.full,
        ),
    ],
)
def test_dsa_paged_bfloat16_rows(row) -> None:
    """bfloat16 rows on the SM90 kernel, the only one that reads them."""
    _check(_bf16_call(**row))


@pytest.mark.sm90
@pytest.mark.parametrize(
    "row",
    [
        pytest.param(dict(_LATENT, some=["attn_sink"]), id="split", marks=pytest.mark.smoke),
        pytest.param(_SINK_UNSPLIT, id="unsplit", marks=pytest.mark.full),
        pytest.param(_PAIR, id="pair", marks=pytest.mark.full),
    ],
)
def test_dsa_paged_sink_on_a_request_without_live_slots(row) -> None:
    """A query row with no live slot gives zeros and lse -inf whatever the sink, merged over
    key splits or not; SM90 only, as test_dsa_paged_bfloat16_rows."""
    _check(_bf16_call(**row), dead_request=True)


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "row, reason",
    [
        pytest.param(
            dict(S_q=1, H=32, K=128, NP=8, PS=64, W=2, cache_lens="repeat(128, 2)"),
            "64 heads",
            id="heads-below-64",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            dict(S_q=1, H=64, K=96, NP=8, PS=64, W=2, cache_lens="repeat(128, 2)"),
            "topk",
            id="topk-not-multiple-of-64",
            marks=pytest.mark.full,
        ),
        pytest.param(
            dict(
                S_q=1,
                H=64,
                K=128,
                NP=8,
                PS=64,
                W=2,
                cache_lens="repeat(128, 2)",
                some=["attn_sink"],
            ),
            "attn_sink",
            id="fp8-sink",
            marks=pytest.mark.full,
        ),
    ],
)
def test_dsa_paged_refuses(row, reason, arch) -> None:
    """The in-tree kernels serve 64-head blocks, and on the FP8 row 64-slot key blocks and no
    sink only."""
    _check_refused(_call(**row), reason)


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
def test_dsa_paged_kernels_refuse_a_cache_of_another_dtype() -> None:
    """The op's signature admits uint8 and bfloat16 caches; each in-tree kernel refuses any
    other dtype on its own."""
    call = DSAPagedKernelCall(
        batch=2, seq_len=1, heads=64, topk=128, page_size=64,
        cache_dtype=torch.float8_e4m3fn, device=torch.device(run_device()),
    )  # fmt: skip
    with pytest.raises(OpNotAvailableError, match="reads uint8 FP8 rows or bfloat16 rows only"):
        DSAPagedFwdOp().key_for("dsa_paged", call)


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
def test_dsa_paged_kernel_refuses_a_sink_on_another_device() -> None:
    """The op's signature keeps every tensor on one device; a kernel called directly checks
    ``attn_sink`` too."""
    call = _bf16_call(**_LATENT, some=["attn_sink"])
    *inputs, attn_sink = DSAPagedCall(call).gen_inputs()
    op = DSAPagedFwdOp(**call.arguments())
    op(*inputs, attn_sink=attn_sink)
    (kernel,) = op._entries_built("dsa_paged").values()
    with pytest.raises(ValueError, match="attn_sink on cpu"):
        kernel(*inputs, attn_sink.cpu())


def _check_refused(call, reason):
    inputs = DSAPagedCall(call).gen_inputs()
    with pytest.raises(Exception, match=reason):
        _run(DSAPagedFwdOp(**call.arguments()), inputs[5])(*(i for i in inputs if i is not None))


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "row, reason",
    [
        # The SM89 kernel reads the FP8 row only.
        pytest.param(
            dict(S_q=1, H=64, DK=512, K=128, NP=8, PS=64, W=2, cache_lens="repeat(128, 2)"),
            "FP8 656-byte",
            id="bfloat16-on-sm89",
            marks=[pytest.mark.sm89, pytest.mark.smoke],
        ),
        # The SM90 kernel's bfloat16 rows hold 512 value columns, and 0 or 64 key columns
        # past them.
        pytest.param(
            dict(
                S_q=1,
                H=64,
                K=128,
                NP=8,
                PS=64,
                W=2,
                cache_lens="repeat(128, 2)",
                DK=512,
                head_dim_v=448,
            ),
            "512 value columns",
            id="bfloat16-value-width",
            marks=[pytest.mark.sm90, pytest.mark.full],
        ),
        pytest.param(
            dict(
                S_q=1,
                H=64,
                K=128,
                NP=8,
                PS=64,
                W=2,
                cache_lens="repeat(128, 2)",
                DK=640,
                head_dim_v=512,
            ),
            "512 value columns",
            id="bfloat16-key-tail",
            marks=[pytest.mark.sm90, pytest.mark.full],
        ),
    ],
)
def test_dsa_paged_refuses_bfloat16_rows(row, reason) -> None:
    """bfloat16 rows the in-tree kernels do not read."""
    _check_refused(_bf16_call(**row), reason)
