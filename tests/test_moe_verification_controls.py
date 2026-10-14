"""Which wrong outputs the MoE verifications reject on the swigluoai and expert-bias rows.

Each case takes a manifest row of an MoE op that fuses ``swigluoai`` or passes an expert bias,
with its expert and token counts cut so it runs on the CPU; its hidden and ffn sizes, scales,
activation, biases and routing are the row's, so each output element is drawn as the row
draws it. A case constructs, from the workload's reference, an output that breaks one part of
the op's contract and asserts the workload's verification rejects it; one positive case
asserts an output the contract admits passes. Rows the cut makes equal are one case. No kernel
runs.
"""

import functools
import json

import pytest
import torch

from tileops.manifest import load_workloads
from workloads.moe import (
    FusedMoEWorkload,
    MoEExpertMLPWorkload,
    MoEExpertsWorkload,
    MoEGroupedGemmWorkload,
    ref_fused_topk,
    ref_moe_grouped_gemm,
)
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call


@pytest.fixture(autouse=True, scope="module")
def _eight_threads():
    """The cases' float32 products run on at most eight threads, so test workers running side
    by side do not oversubscribe the cores."""
    threads = torch.get_num_threads()
    torch.set_num_threads(min(threads, 8))
    yield
    torch.set_num_threads(threads)


_WORKLOADS = {
    "MoEGroupedGemmFwdOp": MoEGroupedGemmWorkload,
    "MoEExpertMLPFwdOp": MoEExpertMLPWorkload,
    "FusedMoEExpertsFwdOp": MoEExpertsWorkload,
    "FusedMoEFwdOp": FusedMoEWorkload,
}
_ROUTED = ("FusedMoEExpertsFwdOp", "FusedMoEFwdOp")


def _biases(row: dict) -> list[str]:
    return [name for name in ("bias", "b_gate_up", "b_down") if name in row.get("some", ())]


def _cut(op: str, row: dict) -> dict:
    """The row's call fields with its expert and token counts cut to a CPU size."""
    fields = {k: v for k, v in row.items() if k not in ("label", "dtype_cases")}
    return fields | ({"E": 4, "T": 16} if op in _ROUTED else {"E": 2, "P": 128})


def _rows(ops=tuple(_WORKLOADS), swigluoai_only=False, each=lambda row: [None]) -> list:
    """``pytest.param(op, label, x)`` for every distinct cut of a row of *ops* that fuses
    swigluoai or passes an expert bias (only the swigluoai ones with *swigluoai_only*), labelled
    by the first row with that cut, once per *x* in ``each(row)``; the first case is ``smoke``,
    the rest ``full``."""
    params, seen = [], set()
    for op in ops:
        for row in load_workloads(op):
            fused = row.get("activation") == "swigluoai"
            if not (fused or (_biases(row) and not swigluoai_only)):
                continue
            key = json.dumps([op, row["dtype_cases"][0], _cut(op, row)], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            for x in each(row):
                suffix = "" if x is None else f"-{x}"
                tier = pytest.mark.full if params else pytest.mark.smoke
                params.append(
                    pytest.param(op, row["label"], x, id=f"{op}-{row['label']}{suffix}", marks=tier)
                )
    return params


@functools.cache
def _case(op: str, label: str) -> tuple:
    """The row's workload with its expert and token counts cut, its inputs and reference."""
    row = next(r for r in load_workloads(op) if r["label"] == label)
    call = manifest_call(op, row["dtype_cases"][0], **_cut(op, row))
    workload = _WORKLOADS[op](call, device="cpu")
    inputs = workload.gen_inputs()
    reference = workload.ref_program(*inputs)
    # The construction without a fault passes, so a rejected construction is rejected for its
    # fault.
    compare_outputs(
        _BUILDERS[op](workload, inputs, None), reference, workload.verification(*inputs)
    )
    return workload, inputs, reference


def _activate(gate_up: torch.Tensor, activation: str, fault) -> torch.Tensor:
    """The gated activation of float32 gate/up columns as ``MoEGroupedGemmFwdOp`` defines it,
    with one *fault*."""
    if activation != "swigluoai":
        gate, up = gate_up.chunk(2, dim=-1)
        if fault == "gate-up-swapped":
            gate, up = up, gate
        act = torch.nn.functional.silu if activation == "silu_and_mul" else torch.nn.functional.gelu
        return act(gate) * up
    if fault == "stacked-read":
        gate, up = gate_up.chunk(2, dim=-1)
    elif fault == "gate-up-swapped":
        gate, up = gate_up[..., 1::2], gate_up[..., ::2]
    else:
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    alpha, limit = 1.702, 7.0  # the docstring's fixed alpha and limit
    if fault != "gate-clamp-missing":
        gate = gate.clamp(max=limit)
    up = up.clamp(
        min=None if fault == "up-lower-clamp-missing" else -limit,
        max=None if fault == "up-upper-clamp-missing" else limit,
    )
    sigmoid = torch.sigmoid((1.0 if fault == "alpha-1" else alpha) * gate)
    return (up if fault == "plus-one-missing" else up + 1) * gate * sigmoid


def _grouped(workload, inputs, fault, held=False):
    """``MoEGroupedGemmFwdOp``'s output by its definition, with one *fault*."""
    a, b, metadata, bias = inputs
    p = workload.call.params
    rows = torch.zeros_like(a) if fault == "product-dropped" else a
    kept = None if fault == "bias-dropped" else bias
    product = ref_moe_grouped_gemm(rows, b, metadata, p["layout"], torch.float32, bias=kept)
    if p["activation"] is not None:
        product = _activate(product, p["activation"], fault)
    return product.to(a.dtype if p["out_dtype"] is None else getattr(torch, p["out_dtype"]))


def _mlp(workload, inputs, fault, held=False):
    """``MoEExpertMLPFwdOp``'s output by its definition, with one *fault*."""
    x, w_gate_up, w_down, metadata, b_gate_up, b_down = inputs
    p = workload.call.params
    kept = None if fault == "b_gate_up-dropped" else b_gate_up
    gate_up = ref_moe_grouped_gemm(x, w_gate_up, metadata, p["layout"], torch.float32, bias=kept)
    activated = _activate(gate_up, p["activation"], fault).to(x.dtype)
    if fault == "product-dropped":
        activated = torch.zeros_like(activated)
    kept = None if fault == "b_down-dropped" else b_down
    return ref_moe_grouped_gemm(activated, w_down, metadata, p["layout"], bias=kept)


def _routed(workload, inputs, fault, held=False):
    """The routed experts' output by ``FusedMoEExpertsFwdOp``'s definition, with one *fault*;
    *held* keeps the activation and each route's down output in the operand dtype."""
    p = workload.call.params
    if isinstance(workload, FusedMoEWorkload):
        hidden, gating, w_gate_up, w_down, correction, b_gate_up, b_down = inputs
        weights, ids = ref_fused_topk(
            gating,
            correction,
            p["top_k"],
            p["scoring_func"],
            p["renormalize"],
            p["num_expert_group"],
            p["topk_group"],
        )
    else:
        _, hidden, w_gate_up, w_down, weights, ids, b_gate_up, b_down = inputs
    dtype = hidden.dtype

    def hold(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype).float() if held else t

    output = torch.zeros(hidden.shape, dtype=torch.float32)
    ids = ids.to(torch.int64)
    for e in range(w_gate_up.shape[0]):
        t_idx, k_idx = (ids == e).nonzero(as_tuple=True)
        if t_idx.numel() == 0:
            continue
        gate_up = hidden[t_idx].float() @ w_gate_up[e].float().T
        if b_gate_up is not None and fault != "b_gate_up-dropped":
            gate_up = gate_up + b_gate_up[e].float()
        down = hold(_activate(gate_up, p["activation"], fault)) @ w_down[e].float().T
        if fault == "product-dropped":
            down = torch.zeros_like(down)
        if b_down is not None and fault not in ("b_down-dropped", "b_down-after-weight"):
            down = down + b_down[e].float()
        weighted = hold(down) * weights[t_idx, k_idx].float().unsqueeze(-1)
        if fault == "b_down-after-weight":
            weighted = weighted + b_down[e].float()
        output.index_add_(0, t_idx, weighted)
    return (output * p["routed_scaling_factor"]).to(dtype)


_BUILDERS = {
    "MoEGroupedGemmFwdOp": _grouped,
    "MoEExpertMLPFwdOp": _mlp,
    "FusedMoEExpertsFwdOp": _routed,
    "FusedMoEFwdOp": _routed,
}


def _construct(op: str, label: str, fault=None, held=False) -> tuple:
    """An output of the cut row built by the op's definition with one *fault*, the reference
    and the verification."""
    workload, inputs, reference = _case(op, label)
    got = _BUILDERS[op](workload, inputs, fault, held)
    return got, reference, workload.verification(*inputs)


def _rejects(op: str, label: str, fault) -> None:
    got, reference, evidence = _construct(op, label, fault)
    with pytest.raises(AssertionError):
        compare_outputs(got, reference, evidence)


@pytest.mark.parametrize(("op", "label", "_"), _rows())
def test_rejects_a_zero_output(op, label, _):
    _, reference, evidence = _construct(op, label)
    with pytest.raises(AssertionError):
        compare_outputs(torch.zeros_like(reference), reference, evidence)


@pytest.mark.parametrize(("op", "label", "_"), _rows())
def test_rejects_an_output_without_the_product(op, label, _):
    """The grouped GEMM's ``a @ b^T``, or the expert MLP's down GEMM, left out: only the bias
    and what follows it remain."""
    _rejects(op, label, "product-dropped")


@pytest.mark.parametrize(("op", "label", "bias"), _rows(each=_biases))
def test_rejects_an_output_without_a_bias_the_call_passes(op, label, bias):
    _rejects(op, label, f"{bias}-dropped")


# Each fault breaks one clause of swigluoai's definition in MoEGroupedGemmFwdOp's docstring.
@pytest.mark.parametrize(
    ("op", "label", "fault"),
    _rows(
        swigluoai_only=True,
        each=lambda row: (
            "stacked-read",
            "gate-up-swapped",
            "gate-clamp-missing",
            "up-upper-clamp-missing",
            "up-lower-clamp-missing",
            "alpha-1",
            "plus-one-missing",
        ),
    ),
)
def test_rejects_an_output_off_the_swigluoai_definition(op, label, fault):
    """Gate and up read stacked or swapped, a clamp left out, alpha 1, or no ``+ 1`` on up.

    One case per clause for each op's distinct cut swigluoai row, 4 x 7: each op's verification
    is its own code path and each clause a different wrong output, so more than 20.
    """
    _rejects(op, label, fault)


@pytest.mark.parametrize(("op", "label", "_"), _rows(ops=_ROUTED))
def test_rejects_b_down_added_after_the_routing_weight(op, label, _):
    _rejects(op, label, "b_down-after-weight")


@pytest.mark.parametrize(("op", "label", "_"), _rows(ops=("MoEGroupedGemmFwdOp",)))
def test_rejects_an_output_in_another_dtype(op, label, _):
    """The output is written in the operand dtype unless ``out_dtype`` asks for fp32."""
    _, reference, evidence = _construct(op, label)
    with pytest.raises(AssertionError, match="dtype"):
        compare_outputs(reference.float(), reference, evidence)


@pytest.mark.parametrize(("op", "label", "_"), _rows(ops=_ROUTED))
def test_accepts_operand_dtype_intermediates(op, label, _):
    """An output holding the activation and each route's down output in the operand dtype,
    as the staged pipeline stores them, passes."""
    got, reference, evidence = _construct(op, label, held=True)
    compare_outputs(got, reference, evidence)
