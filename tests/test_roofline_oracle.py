"""Structural oracle for roofline ``bytes`` (docs/design/roofline.md §4.6).

Each case recomputes the minimum traffic from the tensors the workload binds
— every distinct input storage read once, every output written once — and
requires equality with ``eval_roofline()``. The oracle enumerates tensors
from the signature, so a formula that drops a term, double-counts a tensor,
or prices a broadcast operand at the output's shape breaks the equality.
"""

from math import prod

import pytest
import torch

pytestmark = pytest.mark.smoke


def _nbytes(*tensors: tuple[tuple[int, ...], torch.dtype]) -> int:
    return sum(prod(shape) * dtype.itemsize for shape, dtype in tensors)


# Ops a hand-written case recounted in this run, which the completeness test reads.
_RECOUNTED: set[str] = set()


def _ledger(op_name: str, **tensors: "tuple[tuple[int, ...], torch.dtype] | None") -> int:
    """Sum the named tensors a call binds, and require the names to be the signature's.

    A hand-written case states a tensor per name, ``None`` for an optional input the
    call does not pass, a ``<name>_write``
    entry for a write that is not an output's -- a ``mutated`` input's -- and
    ``<name>_unread=True`` for an input the call passes and the algorithm does not
    read. Every declared input and output has to appear,
    and a name the signature does not declare is rejected, so a case cannot quietly
    drop, duplicate or substitute one of them.
    """
    from tileops.manifest import load_manifest

    signature = load_manifest()[op_name]["signature"]
    inputs = signature.get("inputs") or {}
    outputs = signature.get("outputs") or {}
    declared = (
        set(inputs)
        | set(outputs)
        | {f"{name}_write" for name in inputs}
        | {f"{name}_unread" for name in inputs}
    )
    unknown = sorted(set(tensors) - declared)
    assert not unknown, f"{op_name}: {unknown} are not in the signature"
    _RECOUNTED.add(op_name)
    written = {name for name in inputs if tensors.get(f"{name}_write") is not None}
    undeclared = sorted(name for name in written if not (inputs[name] or {}).get("mutated"))
    assert not undeclared, (
        f"{op_name}: {undeclared} are written by the case and the signature does not "
        "mark them mutated; the write half a read half is taken off reads that marker"
    )
    unread = {name for name in inputs if tensors.get(f"{name}_unread")}
    accounted = set(tensors) | unread
    missing = sorted((set(inputs) | set(outputs)) - accounted)
    assert not missing, f"{op_name}: the case says nothing about {missing}"
    for name, spec in inputs.items():
        if name in unread:
            # Declared and passed, and the algorithm does not read it: no traffic.
            continue
        if tensors[name] is not None:
            continue
        assert (spec or {}).get("optional"), (
            f"{op_name}: {name} is not optional and the case passes None"
        )
    return _nbytes(
        *(
            entry
            for name, entry in tensors.items()
            if entry is not None and not name.endswith("_unread")
        )
    )


def _manifest_rows(op_name: str) -> list:
    from tileops.manifest import load_manifest

    return load_manifest()[op_name]["workloads"]


def _manifest_call(op_name: str, row: "dict | None" = None):
    """The call a workload row of *op_name* states, its metadata generated; the first row
    and dtype case unless *row* is given."""
    from tileops.manifest import load_adts, load_manifest
    from tileops.manifest.plan import entry_plan
    from tileops.manifest.workload import instantiate

    plan = entry_plan(op_name, load_manifest()[op_name], load_adts())
    row = row if row is not None else _manifest_rows(op_name)[0]
    return instantiate(plan, row, (row.get("dtype_cases") or [{}])[0])


class TestBytesOracle:
    # __new__ + attribute binding keeps the oracle CUDA-free; each case binds
    # exactly the state the op's eval_roofline reads after a forward().

    @staticmethod
    def _priced(op, tensors, **stages):
        """``(flops, bytes)`` of *op*'s formula on the checked call of *tensors* (CPU)."""
        import dataclasses

        plan = type(op)._signature
        call = plan.check(op, tensors)
        return plan.roofline(dataclasses.replace(call, stages=stages))

    @staticmethod
    def _routed_tensors(tokens, experts, top_k, hidden, ffn, ids):
        bf16 = torch.bfloat16
        return {
            "output": torch.empty(tokens, hidden, dtype=bf16),
            "hidden_states": torch.empty(tokens, hidden, dtype=bf16),
            "w_gate_up": torch.empty(experts, 2 * ffn, hidden, dtype=bf16),
            "w_down": torch.empty(experts, hidden, ffn, dtype=bf16),
            "topk_weights": torch.empty(tokens, top_k),
            "topk_ids": torch.tensor(ids, dtype=torch.int32),
        }

    def test_routed_expert_mlp_counts_active_experts_and_the_routing(self):
        from tileops.moe import FusedMoEExpertsFwdOp, IndexedExpertMLPFwdOp

        tokens, experts, top_k, hidden, ffn = 2, 8, 2, 64, 32
        # Only experts 0, 3 and 7 receive rows.
        tensors = self._routed_tensors(tokens, experts, top_k, hidden, ffn, [[0, 3], [3, 7]])
        for cls in (FusedMoEExpertsFwdOp, IndexedExpertMLPFwdOp):
            oracle = _ledger(
                cls.__name__,
                hidden_states=((tokens, hidden), torch.bfloat16),
                w_gate_up=((3, 2 * ffn, hidden), torch.bfloat16),  # active experts only
                w_down=((3, hidden, ffn), torch.bfloat16),  # active experts only
                topk_ids=((tokens, top_k), torch.int32),
                topk_weights=((tokens, top_k), torch.float32),
                # the caller's buffer is the output, written once
                output=((tokens, hidden), torch.bfloat16),
            )
            assert self._priced(cls(), tensors)[1] == oracle, cls.__name__

    def test_fused_moe_counts_the_experts_its_stage_read_and_the_bias(self):
        from tileops.moe import FusedMoEExpertsFwdOp, FusedMoeFwdOp

        tokens, experts, top_k, hidden, ffn = 2, 8, 2, 64, 32
        routed = self._routed_tensors(tokens, experts, top_k, hidden, ffn, [[0, 3], [3, 7]])
        stage = FusedMoEExpertsFwdOp()._signature.check(FusedMoEExpertsFwdOp(), routed)
        for has_bias in (True, False):
            op = FusedMoeFwdOp(top_k, scoring_func="sigmoid")
            tensors = {
                "hidden_states": routed["hidden_states"],
                "gating_output": torch.empty(tokens, experts),
                "w_gate_up": routed["w_gate_up"],
                "w_down": routed["w_down"],
                "correction_bias": torch.empty(experts) if has_bias else None,
            }

            def ledger(active, has_bias=has_bias):
                return _ledger(
                    "FusedMoeFwdOp",
                    hidden_states=((tokens, hidden), torch.bfloat16),
                    gating_output=((tokens, experts), torch.float32),
                    w_gate_up=((active, 2 * ffn, hidden), torch.bfloat16),  # read experts
                    w_down=((active, hidden, ffn), torch.bfloat16),  # read experts
                    correction_bias=(((experts,), torch.float32) if has_bias else None),
                    output=((tokens, hidden), torch.bfloat16),
                )

            # Experts 0, 3 and 7, from the routed-experts stage's checked call.
            assert self._priced(op, tensors, routed_experts=(stage,))[1] == ledger(3)
            # No stage call: each token's top_k experts, the bound for every routing.
            assert self._priced(op, tensors)[1] == ledger(top_k), f"has_bias={has_bias}"

    def test_shared_expert_adds_its_shard_to_the_routed_cost(self):
        from tileops.moe import FusedMoeSharedExpertFwdOp

        tokens, experts, top_k, hidden, ffn, shared_ffn, tp = 2, 8, 2, 64, 32, 32, 2
        bf16 = torch.bfloat16
        op = FusedMoeSharedExpertFwdOp(top_k, tp_size=tp, tp_rank=1)
        tensors = {
            "hidden_states": torch.empty(tokens, hidden, dtype=bf16),
            "gating_output": torch.empty(tokens, experts),
            "w_gate_up": torch.empty(experts, 2 * ffn, hidden, dtype=bf16),
            "w_down": torch.empty(experts, hidden, ffn, dtype=bf16),
            "correction_bias": None,
            "shared_w_gate_up": None,
            "shared_w_down": None,
        }
        routed = _ledger(
            "FusedMoeSharedExpertFwdOp",
            hidden_states=((tokens, hidden), bf16),
            gating_output=((tokens, experts), torch.float32),
            w_gate_up=((top_k, 2 * ffn, hidden), bf16),  # the bound: top_k experts
            w_down=((top_k, hidden, ffn), bf16),
            correction_bias=None,
            shared_w_gate_up=None,
            shared_w_down=None,
            routed_output=((tokens, hidden), bf16),
            shared_output=None,
        )
        assert self._priced(op, tensors)[1] == routed

        tensors["shared_w_gate_up"] = torch.empty(2 * shared_ffn, hidden, dtype=bf16)
        tensors["shared_w_down"] = torch.empty(hidden, shared_ffn, dtype=bf16)
        shared = _nbytes(
            ((3 * shared_ffn // tp, hidden), bf16),  # this rank's shared weights
            ((tokens, hidden), bf16),  # its own read of the hidden states
            ((tokens, hidden), bf16),  # shared_output, returned separately
        )
        assert self._priced(op, tensors)[1] == routed + shared

    def test_nsa_forward_counts_the_blocks_its_selection_kept(self):
        """How much this call reads follows `block_counts`, so the case reads the
        selection the manifest row generates rather than inventing one of its own."""
        from tileops.perf.formulas import nsa_fwd_varlen_roofline

        call = _manifest_call("NSAVarlenFwdOp")
        ix = call.ix
        c_seq_len, heads, head_kv, dim = ix["T_q"], ix["H"], ix["H_kv"], ix["D"]
        block_size, selected = ix["block_size"], ix["SEL"]
        # The blocks the kernel reads: for each token and KV head, the kept picks
        # whose block starts at or before that token. Counted here from the
        # tensors, not from the formula's own walk of them.
        kept = [n for row in call.values("block_counts") for n in row]
        picks = [p for row in call.values("block_indices") for p in row]
        positions = [position for _request, position in call.values("token_indices")]
        tiles = sum(
            sum(1 for start in row[:n] if 0 <= start * block_size <= positions[i // head_kv])
            for i, (n, row) in enumerate(zip(kept, picks, strict=True))
        )
        gathered = tiles * block_size * dim
        oracle = _ledger(
            "NSAVarlenFwdOp",
            q=((c_seq_len, heads, dim), torch.float16),
            # k and v are read through the selection, not end to end
            k=((gathered,), torch.float16),
            v=((gathered,), torch.float16),
            block_indices=((c_seq_len, head_kv, selected), torch.int32),
            block_counts=((c_seq_len, head_kv), torch.int32),
            offsets=((len(call.values("offsets")),), torch.int32),
            token_indices=((c_seq_len, 2), torch.int32),
            o_slc=((c_seq_len, heads, dim), torch.float16),
        )
        assert nsa_fwd_varlen_roofline(call)[1] == oracle

    def test_nsa_topk_does_not_charge_the_lse_it_recomputes(self):
        """`lse_in` is declared and passed, and the top-k kernel recomputes the lse
        and discards the argument. A declared input the algorithm does not read
        produces no traffic, and the contract does not
        say which inputs those are."""
        from tileops.perf.formulas import nsa_topk_varlen_roofline

        call = _manifest_call("NSATopkVarlenFwdOp")
        ix = call.ix
        c_seq_len, heads, head_kv, dim = ix["T_q"], ix["H"], ix["H_kv"], ix["D"]
        seq_num, chunk_num, selected = ix["N"], ix["C"], ix["selected_block_num"]
        oracle = _ledger(
            "NSATopkVarlenFwdOp",
            q=((c_seq_len, heads, dim), torch.float16),
            k_cmp=((chunk_num, head_kv, dim), torch.float16),
            lse_in_unread=True,
            offsets=((seq_num + 1,), torch.int32),
            chunk_offsets=((seq_num + 1,), torch.int32),
            token_indices=((c_seq_len, 2), torch.int32),
            block_indices=((c_seq_len, head_kv, selected), torch.int32),
        )
        assert nsa_topk_varlen_roofline(call)[1] == oracle

    def test_gqa_prefill_paged_reads_the_pages_the_block_table_selects(self):
        """The cache is one pool and the call touches the pages its block table
        names, so the recount prices that subset rather than the pool. The scales
        travel with every call and the kernel reads them only for fp8 pages."""
        from tileops.perf.formulas import gqa_prefill_paged_with_kv_cache_fwd_roofline

        name = "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp"
        base = next(r for r in _manifest_rows(name) if "cache_dtype" not in r)
        # One token against an empty or single-page cache names one or two entries of
        # its block-table row; a length that does not divide by the page size is
        # rounded up to a page.
        short = dict(base, T_q=len(base["q_lens"]))
        short["q_lens"] = [1] * len(base["q_lens"])
        short["cache_lens"] = [0, 64] * (len(base["q_lens"]) // 2)
        rows = {
            "cached": base,
            "fp8 cache": dict(base, cache_dtype="float8_e4m3fn"),
            "short": short,
        }
        for label, row in rows.items():
            call = _manifest_call(name, row)
            ix = call.ix
            heads, heads_kv, dim, page_size = ix["H"], ix["H_kv"], ix["D"], ix["page_size"]
            q_lens, cache_lens = row["q_lens"], row["cache_lens"]
            total_q, cached, batch = sum(q_lens), sum(cache_lens), len(q_lens)
            cache = torch.float8_e4m3fn if "cache_dtype" in row else torch.float16
            pages_named = sum(
                -(-(q + c) // page_size) for q, c in zip(q_lens, cache_lens, strict=True)
            )
            new_kv = ((total_q, heads_kv, dim), torch.float16)
            fp8 = cache is torch.float8_e4m3fn
            oracle = _ledger(
                name,
                q=((total_q, heads, dim), torch.float16),
                k_new=new_kv,
                v_new=new_kv,
                # the cached tokens the block table points at, not the whole pool
                k_pages=((cached, heads_kv, dim), cache),
                v_pages=((cached, heads_kv, dim), cache),
                # the new tokens are appended into those same pages
                k_pages_write=((total_q, heads_kv, dim), cache),
                v_pages_write=((total_q, heads_kv, dim), cache),
                k_scale=((1,), torch.float32) if fp8 else None,
                v_scale=((1,), torch.float32) if fp8 else None,
                k_scale_unread=not fp8,
                v_scale_unread=not fp8,
                cu_seqlens_q=((batch + 1,), torch.int32),
                cache_seqlens=((batch,), torch.int32),
                block_table=((pages_named,), torch.int32),
                o=((total_q, heads, dim), torch.float16),
            )
            assert gqa_prefill_paged_with_kv_cache_fwd_roofline(call)[1] == oracle, label

    def test_dropout_short_circuits_read_and_write_what_they_touch(self):
        """The generated case covers the masking path its workloads state. The three
        short-circuit paths have no row -- the manifest keeps them out of the
        release-facing rows -- so they are recounted here: eval mode and `p == 0`
        copy, and `p == 1` writes zeros without reading the input."""
        from tileops.elementwise import DropoutFwdOp

        n = 1024 * 4096
        x = torch.empty((n,), dtype=torch.float16, device="meta")

        def priced(**params):
            op = DropoutFwdOp(**params)
            # The formula prices the op's last completed call; this one is that call.
            op._signature_call = type(op)._signature.check(op, {"input": x})
            return op.eval_roofline()[1]

        copied = _ledger(
            "DropoutFwdOp",
            input=((n,), torch.float16),
            output=((n,), torch.float16),
        )
        assert priced(p=0.5, training=False) == copied
        assert priced(p=0.0) == copied

        zeroed = _ledger("DropoutFwdOp", input_unread=True, output=((n,), torch.float16))
        assert priced(p=1.0) == zeroed

    def test_grouped_gemm_does_not_charge_the_padding_offsets_it_ignores(self):
        """`batch_padded_offsets` is declared and passed, and no kernel indexes it:
        the templates pad nothing. A declared input the algorithm does not read
        produces no traffic, and the contract does not say which inputs those are."""
        from tileops.perf.formulas import grouped_gemm_roofline

        batch_sum, batch_count, n, k = 4096, 16, 4096, 4096
        op = type("_Bound", (), {})()
        op.batch_sum, op.batch_count = batch_sum, batch_count
        op.n, op.k, op.N, op.K = n, k, None, None
        op.transpose_a, op.transpose_b = False, True
        op.dtype = torch.float16
        groups = ((batch_count,), torch.int32)
        oracle = _ledger(
            "GroupedGemmFwdOp",
            a=((batch_sum, k), torch.float16),
            b=((batch_count, n, k), torch.float16),
            batch_sizes=groups,
            batch_offsets=groups,
            batch_padded_offsets_unread=True,
            output=((batch_sum, n), torch.float16),
        )
        assert grouped_gemm_roofline(op)[1] == oracle


# Coverage levels. Every implemented op sits at
# exactly one, and the level says what an independent recount rests on.
#
#   one   The binder builds the case from the manifest: signature, one workload
#         row, dtypes, mutation. It never reads the `roofline` block, and what it
#         does share with the formula is written down: the minimum-traffic
#         definition, the op's own `_infer_output_shapes`, and the manifest's
#         output-dtype resolution. Computed, not
#         listed -- adding an op earns this level or fails the completeness test
#         below.
#   two   The binder cannot build the call and a case above does it by hand,
#         with what the case shares written next to it.
#   three No independent recount is available yet. Marked with what is missing,
#         and asserted against nothing.
#
# Some level-one ops also keep a case above. Those cover a branch one workload
# row does not reach -- an optional input present and absent, a second dtype
# pairing -- and do not change the op's level.

# Level two: a case above recounts these by hand. The value says why the
# generated case cannot, which is what the hand-written one supplies.
HAND_WRITTEN = {
    "FusedMoEExpertsFwdOp": "the routed weight reads follow the values in `topk_ids`",
    "FusedMoeFwdOp": "the routed weight reads follow the routing its experts stage receives",
    "FusedMoeSharedExpertFwdOp": "the routed weight reads follow the routing its experts stage receives",
    "GroupedGemmFwdOp": "`batch_padded_offsets` is passed and no kernel indexes it",
    "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp": "it reads the pages its block table names, not the pool",
    "NSAVarlenFwdOp": "how much it reads follows the values in `block_counts`",
    "NSATopkVarlenFwdOp": "`lse_in` is passed and the kernel recomputes the lse instead of reading it",
    "IndexedExpertMLPFwdOp": "the routed weight reads follow the values in `topk_ids`",
}

# Level three: no independent recount is available. Empty, and an entry here has
# to say what is missing rather than that nobody has got to it.
NOT_RECOUNTABLE: dict[str, str] = {}


def _implemented_ops() -> list[str]:
    from tileops.manifest import load_manifest

    return sorted(
        name for name, entry in load_manifest().items() if entry.get("status") == "implemented"
    )


def _draws_metadata(op_name: str) -> bool:
    """Whether a parametric entry generates some metadata tensor at random."""
    from tileops.manifest import load_manifest
    from tileops.manifest.primitives import RANDOM_GENERATORS
    from tileops.manifest.signature import is_legacy

    entry = load_manifest()[op_name]
    if is_legacy(entry):
        return False
    inputs = entry["signature"].get("inputs") or {}
    return any(
        str(spec.get("values", "")).split("(")[0] in RANDOM_GENERATORS for spec in inputs.values()
    )


def _binder_builds(op_name: str) -> bool:
    """Whether the manifest alone builds a case for *op_name*.

    The formula is not called here. Whether it agrees, or even returns, is a
    separate question: a formula that raises is a defect, and treating that as
    "the binder cannot build this" would let it qualify for level three.
    """
    from tests.roofline_binder import NotBindableError, manifest_cases

    try:
        cases = list(manifest_cases(op_name))
    except NotBindableError:
        return False
    # Anything else -- a broken supplement, a constructor regression, a binder
    # defect -- is a failure to report, not a reason to call an op unrecountable.
    return bool(cases)


def _binder_agrees(op_name: str) -> bool:
    """Whether the formula returns what the binder's recount implies."""
    from tests.roofline_binder import manifest_cases

    try:
        return all(
            op.eval_roofline()[1] == oracle for _l, _d, op, oracle, _r in manifest_cases(op_name)
        )
    except Exception:
        return False


class TestCoverageLevels:
    """Every implemented op sits at exactly one level, and the level is the truth."""

    def test_a_generated_case_equals_its_op(self):
        from tests.roofline_binder import NotBindableError, manifest_cases

        checked = 0
        for op_name in _implemented_ops():
            if op_name in HAND_WRITTEN or op_name in NOT_RECOUNTABLE:
                continue
            try:
                cases = list(manifest_cases(op_name))
            except NotBindableError as exc:  # pragma: no cover - the next test names it
                raise AssertionError(f"{op_name} is level one but does not bind: {exc}") from exc
            for label, dtype, op, oracle, _reads in cases:
                assert op.eval_roofline()[1] == oracle, f"{op_name} {label} {dtype}"
                checked += 1
        assert checked > 0

    def test_a_generated_case_agrees_on_the_read_half(self):
        """The audit judges the read side alone, and an op derives it by taking the
        write side the signature settles off its `bytes`. Where the
        binder recounts the op, the two halves have to be the same halves."""
        from tests.roofline_binder import NotBindableError, manifest_cases

        checked = 0
        for op_name in _implemented_ops():
            if op_name in HAND_WRITTEN or op_name in NOT_RECOUNTABLE:
                continue
            try:
                cases = list(manifest_cases(op_name))
            except NotBindableError:  # pragma: no cover - another test names it
                continue
            for label, dtype, op, _oracle, reads in cases:
                declared = op.eval_roofline_read_bytes()
                assert declared is not None, f"{op_name} {label} {dtype}"
                assert declared == reads, f"{op_name} {label} {dtype}"
                checked += 1
        assert checked > 0

    def test_every_implemented_op_sits_at_one_level(self):
        both = sorted(set(HAND_WRITTEN) & set(NOT_RECOUNTABLE))
        assert not both, f"declared at two levels: {both}"
        unknown = sorted((set(HAND_WRITTEN) | set(NOT_RECOUNTABLE)) - set(_implemented_ops()))
        assert not unknown, f"declared but not implemented: {unknown}"

    def test_a_declared_op_is_one_the_manifest_does_not_already_check(self):
        """Level two and three are for ops the manifest cannot recount, not a queue.

        An op whose rows draw metadata at random stays at level two however its rows fall:
        one row's draw agreeing with the recount says nothing of another's
        (docs/design/roofline.md §4.7).
        """
        promotable = sorted(
            name
            for name in {**HAND_WRITTEN, **NOT_RECOUNTABLE}
            if not _draws_metadata(name) and _binder_builds(name) and _binder_agrees(name)
        )
        assert not promotable, (
            f"the binder now recounts {promotable} and the formula agrees; move them out "
            "of HAND_WRITTEN / NOT_RECOUNTABLE so the generated case is what checks them"
        )

    def test_level_three_is_for_a_call_the_binder_cannot_build(self):
        """A recount the binder can build and the formula disagrees with is a defect.

        Level three says no independent recount is available. If the binder builds one,
        one is available, and a disagreement is then the formula's, not a coverage gap:
        it belongs at level two with the condition the contract omits written next to it.
        """
        buildable = sorted(name for name in NOT_RECOUNTABLE if _binder_builds(name))
        assert not buildable, (
            f"the binder builds a recount for {buildable}; they are not level three, and "
            "a disagreement there is a formula defect"
        )

    def test_every_level_two_op_has_a_case_that_names_its_tensors(self):
        """Level two is a hand-written reference, so the reference has to be here,
        and it has to account for the signature's tensors by name.

        `_ledger` is what makes that mechanical: a case that drops, duplicates or
        substitutes one of them fails there rather than agreeing with a formula
        that made the same mistake. It records the op it recounted, so what this
        reads is the cases that ran, not the text of the file they live in.
        """
        if not _RECOUNTED:
            pytest.skip("this selection ran no hand-written case, so none is recorded")
        missing = sorted(set(HAND_WRITTEN) - _RECOUNTED)
        assert not missing, (
            f"declared level two with no _ledger case above: {missing}; a case that "
            "sums anonymous tuples cannot be checked against the signature"
        )

    def test_a_reason_says_what_is_missing(self):
        for level in (HAND_WRITTEN, NOT_RECOUNTABLE):
            for name, reason in level.items():
                assert reason and not reason.endswith("."), name
                assert len(reason.split()) >= 5, f"{name}: {reason!r} says too little"
