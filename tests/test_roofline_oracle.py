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

# The coverage check reads the ledger populated by the preceding cases in this file.
pytestmark = [pytest.mark.smoke, pytest.mark.xdist_group("roofline-oracle")]


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
    from tileops.manifest._signature import members

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
        assert set(members((spec or {}).get("type", "Tensor"))) - {"Tensor", "list[Tensor]"}, (
            f"{op_name}: {name} holds only tensors and the case passes none"
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
    from tileops.manifest._plan import entry_plan
    from tileops.manifest._workload import instantiate

    plan = entry_plan(op_name, load_manifest()[op_name], load_adts())
    row = row if row is not None else _manifest_rows(op_name)[0]
    return instantiate(plan, row, (row.get("dtype_cases") or [{}])[0])


_F16, _BF16, _F16_KV = {"T": "float16"}, {"T": "bfloat16"}, {"T": "float16", "KV": "float16"}
_VARLEN, _MHA, _DSA = "GQAVarlenFwdOp", "MHAPagedFwdOp", "DSADenseFwdOp"
_INDEXER, _GG = "FP8LightningIndexerFwdOp", "GroupedGEMMFwdOp"
_ONE_HEAD = {"H": 1, "H_kv": 1, "D": 64}
_ROPE = {"pos_encoding_mode": "rope", "some": ["rope_cos", "rope_sin"]}
_DENSE_WINDOW = {**_ONE_HEAD, "B": 1, "S_q": 1, "S_kv": 8, "is_causal": True, "window_size_left": 0}
_PAGED_ROPE = {
    **_ONE_HEAD,
    **_ROPE,
    "T_q": 1,
    "NP": 2,
    "PS": 4,
    "W": 1,
    "P": 8,
    "q_lens": [1, 0],
    "cache_lens": [3, 4],
    "is_causal": True,
}
_NSA = {"H": 16, "H_kv": 1, "D": 64, "SEL": 1, "block_size": 32, "seq_lens": [1], "kv_lens": [1]}
_NSA_CMP = {"C": 3, "H": 16, "H_kv": 1, "DK": 64, "DV": 64, "block_size": 32, "seq_lens": [16, 40]}
_SENTINEL = {"indices": [[[[1]]]]}
_INDEXER_ROW = {"B": 1, "S": 2, "H": 1, "D": 64, "S_kv": 2, "G": 1, "seq_lens": [2], "first": 0}
_WINDOWS = {"starts": [0, 0], "ends": [1, 0]}
_GROUPS = {"batch_sizes": [1, 0], "batch_offsets": [0, 1]}


def _varlen(q_lens, kv_lens, is_causal, **extra):
    return {
        **_ONE_HEAD,
        "T_q": sum(q_lens),
        "T_kv": sum(kv_lens),
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "is_causal": is_causal,
        **extra,
    }


def _mha_paged(s_q, kv_lens, is_causal):
    row = {"S_q": s_q, "H": 1, "D": 64, "N_kv": 16, "W": 1, "kv_lens": kv_lens, "page_size": 16}
    return {**row, "is_causal": is_causal}


def _dsa(s_kv, is_causal):
    row = {**_ONE_HEAD, "B": 1, "S": 1, "S_kv": s_kv, "K": 1, "dim_tail": 0, "stride_kv": 1}
    return {**row, "q_start_index_s": 0, "is_causal": is_causal}


def _grouped(trans_a, trans_b):
    return {"M": 1, "N": 2, "K": 2, "sizes": [1, 0], "trans_a": trans_a, "trans_b": trans_b}


class TestBytesOracle:
    # __new__ + attribute binding keeps the oracle CUDA-free; each case binds
    # exactly the state the op's eval_roofline reads after a forward().

    @staticmethod
    def _priced(op, tensors, **stages):
        """``(flops, bytes)`` of *op*'s formula on the checked call of *tensors* (CPU)."""
        import dataclasses

        plan = type(op)._signature
        call = plan.check(op, tensors)
        return plan.eval_roofline(dataclasses.replace(call, stages=stages))

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
        # FusedMoEExpertsFwdOp takes the optional expert biases; IndexedExpertMLPFwdOp none.
        unbiased = {"b_gate_up": None, "b_down": None}
        for cls, biases in ((FusedMoEExpertsFwdOp, unbiased), (IndexedExpertMLPFwdOp, {})):
            oracle = _ledger(
                cls.__name__,
                hidden_states=((tokens, hidden), torch.bfloat16),
                w_gate_up=((3, 2 * ffn, hidden), torch.bfloat16),  # active experts only
                w_down=((3, hidden, ffn), torch.bfloat16),  # active experts only
                topk_ids=((tokens, top_k), torch.int32),
                topk_weights=((tokens, top_k), torch.float32),
                # the caller's buffer is the output, written once
                output=((tokens, hidden), torch.bfloat16),
                **biases,
            )
            assert self._priced(cls(), tensors)[1] == oracle, cls.__name__
        # The biases are read with their experts' weights: the active experts' rows only.
        biased = {
            **tensors,
            "b_gate_up": torch.empty(experts, 2 * ffn, dtype=torch.bfloat16),
            "b_down": torch.empty(experts, hidden, dtype=torch.bfloat16),
        }
        oracle = _ledger(
            "FusedMoEExpertsFwdOp",
            hidden_states=((tokens, hidden), torch.bfloat16),
            w_gate_up=((3, 2 * ffn, hidden), torch.bfloat16),
            w_down=((3, hidden, ffn), torch.bfloat16),
            topk_ids=((tokens, top_k), torch.int32),
            topk_weights=((tokens, top_k), torch.float32),
            output=((tokens, hidden), torch.bfloat16),
            b_gate_up=((3, 2 * ffn), torch.bfloat16),
            b_down=((3, hidden), torch.bfloat16),
        )
        assert self._priced(FusedMoEExpertsFwdOp(), biased)[1] == oracle

    def test_fused_moe_counts_the_experts_its_stage_read_and_the_bias(self):
        from tileops.moe import FusedMoEExpertsFwdOp, FusedMoEFwdOp

        tokens, experts, top_k, hidden, ffn = 2, 8, 2, 64, 32
        routed = self._routed_tensors(tokens, experts, top_k, hidden, ffn, [[0, 3], [3, 7]])
        stage = FusedMoEExpertsFwdOp()._signature.check(FusedMoEExpertsFwdOp(), routed)
        for has_bias in (True, False):
            op = FusedMoEFwdOp(top_k, scoring_func="sigmoid")
            tensors = {
                "hidden_states": routed["hidden_states"],
                "gating_output": torch.empty(tokens, experts),
                "w_gate_up": routed["w_gate_up"],
                "w_down": routed["w_down"],
                "correction_bias": torch.empty(experts) if has_bias else None,
            }

            def ledger(active, has_bias=has_bias):
                return _ledger(
                    "FusedMoEFwdOp",
                    hidden_states=((tokens, hidden), torch.bfloat16),
                    gating_output=((tokens, experts), torch.float32),
                    w_gate_up=((active, 2 * ffn, hidden), torch.bfloat16),  # read experts
                    w_down=((active, hidden, ffn), torch.bfloat16),  # read experts
                    correction_bias=(((experts,), torch.float32) if has_bias else None),
                    output=((tokens, hidden), torch.bfloat16),
                    b_gate_up=None,
                    b_down=None,
                )

            # Experts 0, 3 and 7, from the routed-experts stage's checked call.
            assert self._priced(op, tensors, routed_experts=(stage,))[1] == ledger(3)
            # No stage call: each token's top_k experts, the bound for every routing.
            assert self._priced(op, tensors)[1] == ledger(top_k), f"has_bias={has_bias}"

    def test_shared_expert_adds_its_shard_to_the_routed_cost(self):
        from tileops.moe import FusedMoESharedExpertFwdOp

        tokens, experts, top_k, hidden, ffn, shared_ffn, tp = 2, 8, 2, 64, 32, 32, 2
        bf16 = torch.bfloat16
        op = FusedMoESharedExpertFwdOp(top_k, tp_size=tp, tp_rank=1)
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
            "FusedMoESharedExpertFwdOp",
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
            ((tokens, hidden), bf16),  # shared_output, returned separately
        )
        assert self._priced(op, tensors)[1] == routed + shared

    def test_masked_grouped_gemm_moves_only_valid_rows_and_active_weights(self):
        """A sparse masked slab, as a decode shard holds: of 4 x 64 slab rows, 3 are valid and
        two experts are empty, so the rows of ``a``, ``a_scale`` and ``output`` past each
        count and the empty experts' weights move nothing."""
        from tileops._ops.moe import MaskedLayoutSpec, MoEGroupedGEMMFP8FwdOp, MoEGroupedGEMMFwdOp

        experts, max_m, n, k, counts = 4, 64, 256, 512, [2, 0, 1, 0]
        valid, active = sum(counts), 2
        fp8, bf16 = torch.float8_e4m3fn, torch.bfloat16
        metadata = torch.tensor(counts, dtype=torch.int32)
        layout = MaskedLayoutSpec(max_m=max_m)
        fp8_tensors = {
            "a": torch.empty(experts, max_m, k, dtype=fp8),
            "a_scale": torch.empty(experts, max_m, k // 128),
            "b": torch.empty(experts, n, k, dtype=fp8),
            "b_scale": torch.empty(experts, n // 128, k // 128),
            "layout_metadata": metadata,
            "out": None,
        }
        oracle = _ledger(
            "MoEGroupedGEMMFP8FwdOp",
            a=((valid, k), fp8),  # valid rows only
            a_scale=((valid, k // 128), torch.float32),  # valid rows only
            b=((active, n, k), fp8),  # active experts only
            b_scale=((active, n // 128, k // 128), torch.float32),  # active experts only
            layout_metadata=((experts,), torch.int32),
            output=((valid, n), bf16),  # valid rows only
        )
        assert self._priced(MoEGroupedGEMMFP8FwdOp(layout), fp8_tensors)[1] == oracle

        bf16_tensors = {
            "a": torch.empty(experts, max_m, k, dtype=bf16),
            "b": torch.empty(experts, n, k, dtype=bf16),
            "layout_metadata": metadata,
            "bias": None,
            "out": None,
        }
        oracle = _ledger(
            "MoEGroupedGEMMFwdOp",
            a=((valid, k), bf16),  # valid rows only
            b=((active, n, k), bf16),  # active experts only
            layout_metadata=((experts,), torch.int32),
            bias=None,
            output=((valid, n), bf16),  # valid rows only
        )
        assert self._priced(MoEGroupedGEMMFwdOp(layout), bf16_tensors)[1] == oracle

    def test_masked_grouped_gemm_without_slab_rows_moves_only_the_metadata(self):
        """A masked layout of ``max_m = 0`` holds no row, so no expert is active: the row
        tensors and the weights move nothing, and only ``layout_metadata`` is read."""
        from tileops._ops.moe import MaskedLayoutSpec, MoEGroupedGEMMFP8FwdOp, MoEGroupedGEMMFwdOp

        experts, n, k = 2, 256, 512
        fp8, bf16 = torch.float8_e4m3fn, torch.bfloat16
        metadata = torch.zeros(experts, dtype=torch.int32)
        layout = MaskedLayoutSpec(max_m=0)
        fp8_tensors = {
            "a": torch.empty(experts, 0, k, dtype=fp8),
            "a_scale": torch.empty(experts, 0, k // 128),
            "b": torch.empty(experts, n, k, dtype=fp8),
            "b_scale": torch.empty(experts, n // 128, k // 128),
            "layout_metadata": metadata,
            "out": None,
        }
        oracle = _ledger(
            "MoEGroupedGEMMFP8FwdOp",
            a=((0, k), fp8),
            a_scale=((0, k // 128), torch.float32),
            b=((0, n, k), fp8),  # no active expert
            b_scale=((0, n // 128, k // 128), torch.float32),  # no active expert
            layout_metadata=((experts,), torch.int32),
            output=((0, n), bf16),
        )
        assert self._priced(MoEGroupedGEMMFP8FwdOp(layout), fp8_tensors) == (0, oracle)

        bf16_tensors = {
            "a": torch.empty(experts, 0, k, dtype=bf16),
            "b": torch.empty(experts, n, k, dtype=bf16),
            "layout_metadata": metadata,
            "bias": None,
            "out": None,
        }
        oracle = _ledger(
            "MoEGroupedGEMMFwdOp",
            a=((0, k), bf16),
            b=((0, n, k), bf16),  # no active expert
            layout_metadata=((experts,), torch.int32),
            bias=None,
            output=((0, n), bf16),
        )
        assert self._priced(MoEGroupedGEMMFwdOp(layout), bf16_tensors) == (0, oracle)

    def test_nsa_forward_reads_the_rows_its_selection_kept(self):
        """How much this call reads follows `block_counts`, so the case reads the
        selection the manifest row generates rather than inventing one of its own."""
        row = _manifest_rows("NSAVarlenFwdOp")[0]
        for is_causal in (True, False):
            self._nsa_forward_case(_manifest_call("NSAVarlenFwdOp", dict(row, is_causal=is_causal)))
        # Decode over cached keys: each request's queries are the last of its keys.
        decode = {
            "T_kv": 22, "H": 16, "H_kv": 1, "D": 64, "SEL": 2, "block_size": 4,
            "seq_lens": [1, 2], "kv_lens": [9, 13], "some": ["cu_seqlens_k"],
            "label": "decode", "dtype_cases": [{"T": "float16"}],
        }  # fmt: skip
        for is_causal in (True, False):
            call = _manifest_call("NSAVarlenFwdOp", dict(decode, is_causal=is_causal))
            self._nsa_forward_case(call)

    @staticmethod
    def _nsa_forward_case(call):
        from tileops._perf.formulas import nsa_fwd_varlen_roofline

        indices = call.indices
        c_seq_len, heads, head_kv, dim = indices["T_q"], indices["H"], indices["H_kv"], indices["D"]
        block_size = indices["block_size"]
        # Key rows some token scores, per KV head: each kept block cut at the token (causal)
        # or the sequence end; a row several tokens score is read once.
        cu_seqlens = call.metadata_values("cu_seqlens")
        separate = call.present("cu_seqlens_k")
        keys = call.metadata_values("cu_seqlens_k") if separate else cu_seqlens

        def bound(request, position):
            length = keys[request + 1] - keys[request]
            queries = cu_seqlens[request + 1] - cu_seqlens[request]
            return length - queries + position + 1 if indices["is_causal"] else length

        rows = {
            (h, keys[request] + r)
            for (request, position), counts, picks in zip(
                call.metadata_values("token_indices"),
                call.metadata_values("block_counts"),
                call.metadata_values("block_indices"),
                strict=True,
            )
            for h in range(head_kv)
            for start in picks[h][: counts[h]]
            for r in range(
                start * block_size,
                min((start + 1) * block_size, bound(request, position)),
            )
        }
        gathered = len(rows) * dim
        consulted = sum(n for counts in call.metadata_values("block_counts") for n in counts)
        oracle = _ledger(
            "NSAVarlenFwdOp",
            q=((c_seq_len, heads, dim), torch.float16),
            # k and v are read through the selection, not end to end
            k=((gathered,), torch.float16),
            v=((gathered,), torch.float16),
            block_indices=((consulted,), torch.int32),
            block_counts=((c_seq_len, head_kv), torch.int32),
            cu_seqlens=((len(cu_seqlens),), torch.int32),
            token_indices_unread=True,
            cu_seqlens_k=((len(keys),), torch.int32) if separate else None,
            o=((c_seq_len, heads, dim), torch.float16),
        )
        assert nsa_fwd_varlen_roofline(call)[1] == oracle

    def test_gqa_paged_reads_the_rows_its_page_table_names(self):
        """The cache is one pool, and the call reads the rows its page table names as far
        as each request's cached length; the pages it never names move nothing."""
        from tileops._perf.formulas import gqa_paged_fwd_roofline

        name = "GQAPagedFwdOp"
        call = _manifest_call(name)
        indices = call.indices
        heads, heads_kv, dim, page = indices["H"], indices["H_kv"], indices["D"], indices["PS"]
        table, cached = call.metadata_values("block_table"), call.metadata_values("cache_seqlens")
        rows = {(table[b][r // page], r % page) for b, n in enumerate(cached) for r in range(n)}
        kv = ((len(rows), heads_kv, dim), torch.float16)
        oracle = _ledger(
            name,
            q=((indices["T_q"], heads, dim), torch.float16),
            k_cache=kv,
            v_cache=kv,
            block_table=((sum(-(-n // page) for n in cached),), torch.int32),
            cache_seqlens=((len(cached),), torch.int32),
            cu_seqlens_q=((len(cached) + 1,), torch.int32),
            q_scale=None,
            k_scale=None,
            v_scale=None,
            rope_cos=None,
            rope_sin=None,
            sinks=None,
            o=((indices["T_q"], heads, dim), torch.float16),
        )
        assert gqa_paged_fwd_roofline(call)[1] == oracle

    def test_gqa_varlen_sink_adds_no_key_or_value_contraction(self):
        row = {
            "T_q": 3,
            "T_kv": 4,
            "H": 8,
            "H_kv": 2,
            "D": 64,
            "q_lens": [2, 1],
            "kv_lens": [4, 0],
            "is_causal": True,
        }
        plain, _ = _evaluated("GQAVarlenFwdOp", row, {"T": "float16"})
        sink, _ = _evaluated(
            "GQAVarlenFwdOp", {**row, "some": [*row.get("some", []), "sinks"]}, {"T": "float16"}
        )
        # Two nonempty rows per head each add subtract/exp/add, and one FP32 logit
        # per head is read. The last request stays zero regardless of its sink.
        assert sink == (plain[0] + 8 * 2 * 3, plain[1] + 8 * 4)

    @pytest.mark.parametrize(
        ("op_name", "row", "case", "values", "nbytes"),
        [
            # Query rows that see no key read no q.
            pytest.param(_VARLEN, _varlen([4], [1], True), _F16, {}, 912, id="varlen-unseen-q"),
            # Empty key storage reads nothing and divides by nothing.
            pytest.param(_VARLEN, _varlen([1], [0], False), _F16, {}, 144, id="varlen-no-keys"),
            # RoPE positions clamp to the table's last row.
            pytest.param(
                _VARLEN, _varlen([1], [8], True, P=2, **_ROPE), _F16, {}, 2576, id="rope-clamp"
            ),
            # A window reads only the keys some query sees.
            pytest.param("GQADenseFwdOp", _DENSE_WINDOW, _F16, {}, 512, id="dense-window"),
            # RoPE rows are read where some request's keys reach.
            pytest.param("GQAPagedFwdOp", _PAGED_ROPE, _F16_KV, {}, 1432, id="paged-rope-rows"),
            # Query rows that see no key read no q; a call without queries reads no cache.
            pytest.param(_MHA, _mha_paged(4, [1], True), _F16, {}, 904, id="mha-unseen-q"),
            pytest.param(_MHA, _mha_paged(0, [16], False), _F16, {}, 4, id="mha-no-queries"),
            # A token with no selected block reads no q, k, v or slot; token_indices is derived.
            pytest.param("NSAVarlenFwdOp", _NSA, _F16, {"block_counts": [[0]]}, 2060, id="nsa"),
            # Tokens before a chunk boundary and unclosed chunks are not read.
            pytest.param("NSACompressedVarlenFwdOp", _NSA_CMP, _F16, {}, 136972, id="nsa-cmp"),
            # A sentinel slot selects nothing; a non-causal call keeps a slot past the query.
            pytest.param(_DSA, _dsa(1, True), _F16, _SENTINEL, 132, id="dsa-sentinel"),
            pytest.param(_DSA, _dsa(2, False), _F16, _SENTINEL, 388, id="dsa-non-causal"),
            # Keys no window covers and queries with an empty window are not read.
            pytest.param(_INDEXER, _INDEXER_ROW, _BF16, _WINDOWS, 292, id="indexer"),
            # An empty group reads none of b, but its split-contraction output is written.
            pytest.param(_GG, _grouped(False, True), _F16, _GROUPS, 24, id="grouped-rows"),
            pytest.param(_GG, _grouped(True, False), _F16, _GROUPS, 32, id="grouped-contraction"),
        ],
    )
    def test_partial_reads(self, op_name, row, case, values, nbytes):
        (_, priced), _ = _evaluated(op_name, row, case, **values)
        assert priced == nbytes

    def test_the_oracle_clamps_rope_positions_to_a_short_table(self):
        """The generated cases rely on the oracle's own RoPE clamp."""
        from tests.roofline_binder import _unseen_attention_rows

        row = _varlen([1], [8], True, P=2, window_size_left=0, **_ROPE)
        row |= {"label": "short-table", "dtype_cases": [_F16]}
        unseen = _unseen_attention_rows(_VARLEN, _manifest_call(_VARLEN, row))
        assert (unseen["rope_cos"], unseen["rope_sin"]) == (64, 64)

    def test_paged_attention_prices_an_empty_pool(self):
        from tileops.attention import GQAPagedFwdOp, MHAPagedFwdOp

        f16, i32 = torch.float16, torch.int32
        mha = {
            "q": torch.empty(1, 1, 1, 64, dtype=f16),
            "k_cache": torch.empty(0, 1, 64, dtype=f16),
            "v_cache": torch.empty(0, 1, 64, dtype=f16),
            "block_table": torch.empty(1, 0, dtype=i32),
            "cache_seqlens": torch.zeros(1, dtype=i32),
        }
        assert self._priced(MHAPagedFwdOp(page_size=16), mha) == (0, 132)
        gqa = {
            "q": torch.empty(0, 1, 64, dtype=f16),
            "k_cache": torch.empty(0, 4, 1, 64, dtype=f16),
            "v_cache": torch.empty(0, 4, 1, 64, dtype=f16),
            "block_table": torch.empty(1, 0, dtype=i32),
            "cache_seqlens": torch.zeros(1, dtype=i32),
            "cu_seqlens_q": torch.zeros(2, dtype=i32),
        }
        assert self._priced(GQAPagedFwdOp(), gqa) == (0, 12)

    def test_topk_select_reads_only_its_windows(self):
        """The manifest rows select from whole rows; a narrower window reads and compares
        only the scores inside it."""
        from tileops.attention import TopKSelectFwdOp

        batch, seq, extent, top_k = 1, 4, 16, 2
        starts = torch.tensor([[0, 4, 8, 12]], dtype=torch.int32)
        ends = torch.tensor([[4, 8, 16, 12]], dtype=torch.int32)
        tensors = {
            "logits": torch.empty(batch, seq, extent, 1),
            "starts": starts,
            "ends": ends,
        }
        scores = int((ends - starts).clamp(min=0).sum())
        oracle = _ledger(
            "TopKSelectFwdOp",
            logits=((scores,), torch.float32),
            starts=((batch, seq), torch.int32),
            ends=((batch, seq), torch.int32),
            indices=((batch, seq, 1, top_k), torch.int32),
        )
        assert self._priced(TopKSelectFwdOp(top_k), tensors) == (scores, oracle)

    def test_windowed_pools_read_the_positions_some_window_reaches(self):
        """A stride past the kernel span, or a dilation, leaves input positions no window
        reads; the case recounts the positions from the row's geometry."""
        from tileops._perf.formulas import pool_roofline

        for op_name in ("AvgPool1dFwdOp", "MaxPool1dFwdOp", "MaxPool1dIndicesFwdOp"):
            for row in _manifest_rows(op_name):
                call = _manifest_call(op_name, row)
                indices = call.indices
                read = {
                    o * indices["sW"] - indices["pW"] + j * indices.get("dW", 1)
                    for o in range(indices["L_out"])
                    for j in range(indices["kW"])
                } & set(range(indices["L_in"]))
                dtype = getattr(torch, call.tensors["input"][1])
                out = ((indices["N"], indices["C"], indices["L_out"]), dtype)
                tensors = {"input": ((indices["N"], indices["C"], len(read)), dtype), "output": out}
                if op_name == "MaxPool1dIndicesFwdOp":
                    tensors["indices"] = (
                        (indices["N"], indices["C"], indices["L_out"]),
                        torch.int64,
                    )
                assert pool_roofline(call)[1] == _ledger(op_name, **tensors), row["label"]

    def test_dsa_decode_reads_the_kv_rows_some_query_selects(self):
        """Top-k slots past the causal bound or repeated select nothing more, so `kv` is
        read at the rows the row's generated indices reach."""
        from tileops._perf.formulas import dsa_decode_roofline

        op_name = "DSADenseFwdOp"
        for row in _manifest_rows(op_name):
            call = _manifest_call(op_name, row)
            indices = call.indices
            rows = {
                (b, g, j)
                for b, batch in enumerate(call.metadata_values("indices"))
                for s, heads in enumerate(batch)
                for g, slots in enumerate(heads)
                for j in slots
                if 0 <= j < indices["S_kv"]
                and (
                    not indices["is_causal"]
                    or (j + 1) * indices["stride_kv"] - 1 <= indices["q_start_index_s"] + s
                )
            }
            dtype = getattr(torch, call.tensors["q"][1])
            width = indices["D"] + indices["dim_tail"]
            sink = ((indices["H"],), torch.float32) if call.present("attn_sink") else None
            oracle = _ledger(
                op_name,
                q=((indices["B"], indices["S"], indices["H"], width), dtype),
                kv=((len(rows), width), dtype),
                indices=((indices["B"], indices["S"], indices["H_kv"], indices["K"]), torch.int32),
                attn_sink=sink,
                o=((indices["B"], indices["S"], indices["H"], indices["D"]), dtype),
            )
            assert dsa_decode_roofline(call)[1] == oracle, row["label"]

    def test_deltanet_bwd_reads_the_strict_lower_triangle_of_aw_and_au(self):
        """Each chunk's C x C block of Aw and Au has a unit diagonal and a zero upper
        triangle, so only the strict-lower C * (C - 1) / 2 entries are read."""
        from tests.roofline_binder import manifest_cases

        op_name = "DeltaNetChunkBwdOp"
        rows = {row["label"]: row for row in _manifest_rows(op_name)}
        for label, dtype_name, op, _oracle, _reads in manifest_cases(op_name):
            indices = _manifest_call(op_name, rows[label]).indices
            b, h, n, dk, dv, c = (
                indices["B"],
                indices["H"],
                indices["L"],
                indices["DK"],
                indices["DV"],
                indices["chunk_size"],
            )
            dtype = getattr(torch, dtype_name)
            triangle = ((b, h, n // c, c * (c - 1) // 2), dtype)
            oracle = _ledger(
                op_name,
                do=((b, h, n, dv), dtype),
                q=((b, h, n, dk), dtype),
                k=((b, h, n, dk), dtype),
                v=((b, h, n, dv), dtype),
                beta=((b, h, n), dtype),
                h=((b, h, n // c + 1, dk, dv), torch.float32),
                Aw=triangle,
                Au=triangle,
                w=((b, h, n, dk), dtype),
                u=((b, h, n, dv), dtype),
                dq=((b, h, n, dk), dtype),
                dk=((b, h, n, dk), dtype),
                dv=((b, h, n, dv), dtype),
                dbeta=((b, h, n), dtype),
            )
            assert op.eval_roofline()[1] == oracle, label

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
            op._last_call = type(op)._signature.check(op, {"input": x})
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


def _evaluated(op_name: str, row: dict, case: dict, **values):
    """``(flops, bytes)`` the generated evaluator prices for a row of a spec-only entry, and the
    call; *values* replace the named metadata tensors' generated contents."""
    import dataclasses

    from tests.roofline_binder import signature_class
    from tileops.manifest import load_adts, load_manifest
    from tileops.manifest._plan import entry_plan
    from tileops.manifest._workload import instantiate

    entry = load_manifest()[op_name]
    plan = entry_plan(op_name, entry, load_adts())
    call = instantiate(plan, {**row, "label": "recount"}, case)
    specs = {
        **call.specs,
        **{n: dataclasses.replace(call.specs[n], values=v) for n, v in values.items()},
    }
    call = dataclasses.replace(call, specs=specs)
    tensors = call.materialize("meta")
    cls = signature_class(op_name, entry)
    op = cls(**call.arguments())
    checked = cls._signature.check(op, {t: tensors[t] for t in plan.sig.inputs})
    metadata = {n: torch.tensor(call.metadata_values(n)) for n in checked.metadata}
    op._last_call = dataclasses.replace(checked, metadata=metadata)
    return op.eval_roofline(), call


def _attention_flops(heads, qk, v, scores, rows):
    # Per score two contractions and the softmax (5); per output element its divide.
    return heads * (scores * (2 * qk + 2 * v + 5) + rows * v)


def _attention_bwd_flops(heads, qk, v, scores, rows):
    # Per score three contractions over qk and two over v, P (3) and dS (2); per row the delta.
    return heads * (scores * (2 * (3 * qk + 2 * v) + 5) + rows * 2 * v)


_BF16, _F32, _FP8, _I32, _I64 = (
    torch.bfloat16,
    torch.float32,
    torch.float8_e4m3fn,
    torch.int32,
    torch.int64,
)


class TestSpecOnlyRecounts:
    """Spec-only entries whose traffic or arithmetic follows their metadata values, recounted
    by walking the call. Each case picks metadata that reaches the branches the values decide."""

    @pytest.mark.parametrize(
        "row,case",
        [
            # Two requests with disjoint pages; two causal query rows each.
            (
                {"S_q": 2, "NP": 8, "W": 4, "cache_lens": [5, 9], "is_causal": True},
                {"T": "bfloat16", "KV": "bfloat16"},
            ),
            # A pool smaller than the requests' pages, so requests share rows; an FP8 cache.
            (
                {"S_q": 1, "NP": 3, "W": 3, "cache_lens": [5, 9, 12], "some": ["kv_scale"]},
                {"T": "bfloat16", "KV": "float8_e4m3fn"},
            ),
        ],
    )
    def test_mla_paged_reads_the_rows_its_block_table_reaches(self, row, case):
        name = "MLAPagedFwdOp"
        row = {"H": 3, "DK": 12, "PS": 4, "kv_lora_rank": 8, **row}
        (flops, moved), call = _evaluated(name, row, case)
        heads, dk, rank, page, s_q = row["H"], row["DK"], row["kv_lora_rank"], row["PS"], row["S_q"]
        table, lengths = call.metadata_values("block_table"), call.metadata_values("cache_seqlens")
        fp8 = case["KV"] == "float8_e4m3fn"
        scores = rows = 0
        for c in lengths:
            for i in range(s_q):
                seen = c - s_q + i + 1
                scores, rows = scores + seen, rows + 1
        cache_rows = {
            (table[b][j // page], j % page) for b, c in enumerate(lengths) for j in range(c)
        }
        pages = {(b, j // page) for b, c in enumerate(lengths) for j in range(c)}
        batch = len(lengths)
        assert moved == _ledger(
            name,
            q=((batch, s_q, heads, dk), _BF16),
            kv_cache=((len(cache_rows), dk), _FP8 if fp8 else _BF16),
            block_table=((len(pages),), _I32),
            cache_seqlens=((batch,), _I32),
            kv_scale=((1,), _F32) if fp8 else None,
            o=((batch, s_q, heads, rank), _BF16),
            lse=((batch, s_q, heads), _F32),
        )
        assert flops == _attention_flops(heads, dk, rank, scores, rows) + (
            heads * rows if fp8 else 0
        )

    @pytest.mark.parametrize(
        "extra,case",
        [
            ({"D": 512}, {"KV": "uint8"}),
            # A bfloat16 row of 8 values, 2 of them past the value; a sink per head.
            ({"D": 6, "dim_tail": 2, "some": ["attn_sink"]}, {"KV": "bfloat16"}),
        ],
    )
    def test_dsa_paged_scores_each_valid_slot_and_reads_the_rows_they_name(self, extra, case):
        name = "DSAPagedFwdOp"
        row = {"S_q": 2, "H": 2, "K": 4, "NP": 6, "PS": 4, "W": 3, "cache_lens": [5, 10], **extra}
        # Request 0: query 0 selects nothing, query 1 three positions on two pages.
        # Request 1: a repeated slot, then nothing.
        indices = [[[-1, -1, -1, -1], [0, 4, 3, -1]], [[3, 3, -1, -1], [-1, -1, -1, -1]]]
        (flops, moved), call = _evaluated(name, row, case, indices=indices)
        table, lengths, page = (
            call.metadata_values("block_table"),
            call.metadata_values("cache_seqlens"),
            row["PS"],
        )
        valid = [
            [[j for j in slots if 0 <= j < lengths[b]] for slots in per_q]
            for b, per_q in enumerate(indices)
        ]
        scores = sum(len(v) for per_q in valid for v in per_q)
        rows = sum(1 for per_q in valid for v in per_q if v)
        cache_rows = {
            (table[b][j // page], j % page)
            for b, per_q in enumerate(valid)
            for v in per_q
            for j in v
        }
        pages = {(b, j // page) for b, per_q in enumerate(valid) for v in per_q for j in v}
        fp8 = case["KV"] == "uint8"
        value = row["D"]
        width = value + (64 if fp8 else row["dim_tail"])
        cache = ((len(cache_rows), 656), torch.uint8) if fp8 else ((len(cache_rows), width), _BF16)
        assert moved == _ledger(
            name,
            q=((rows, row["H"], width), _BF16),  # the query rows that score something
            kv_cache=cache,
            block_table=((len(pages),), _I32),
            cache_seqlens=((2,), _I32),
            indices=((2, 2, 4), _I32),
            attn_sink=None if fp8 else ((row["H"],), _F32),
            o=((2, 2, row["H"], value), _BF16),
            lse=((2, 2, row["H"]), _F32),
        )
        # Each distinct FP8 row's 512 latent values are dequantized once; a sink adds its
        # subtract, exponential and add per head of a row with a score.
        extra_flops = 512 * len(cache_rows) if fp8 else 3 * row["H"] * rows
        assert flops == _attention_flops(row["H"], width, value, scores, rows) + extra_flops

    @pytest.mark.parametrize(
        "row,slots,emitted,state_reads,state_writes,ape_rows",
        [
            # Request 0 prefills 5 tokens, closing windows 0 (2 slots) and 1 (4 slots);
            # request 1 resumes at position 5 and closes window 2, reading position 4 from
            # the state for the current window and positions 2 and 3 for the previous one.
            ({"T_q": 7, "seq_lens": [5, 2], "positions": [0, 5]}, 2 + 4 + 4, 3, 3, 4 + 2, 2),
            # One decode token at position 7 takes one ape row and writes one state row.
            ({"T_q": 1, "seq_lens": [1], "positions": [7]}, 4, 1, 3, 1, 1),
        ],
    )
    def test_gated_pool_compress_moves_the_state_rows_its_windows_use(
        self, row, slots, emitted, state_reads, state_writes, ape_rows
    ):
        name = "GatedPoolCompressFwdOp"
        tokens, requests = row["T_q"], len(row["seq_lens"])
        row = {"NS": 4, "D": 2, "ratio": 2, "overlap": True, **row}
        (flops, moved), _call = _evaluated(name, row, {"T": "float32", "A": "float32"})
        read = ((state_reads, 2), _F32)  # D channels per role
        write = ((state_writes, 4), _F32)  # whole rows
        assert moved == _ledger(
            name,
            kv=((tokens, 4), _F32),
            score=((tokens, 4), _F32),
            ape=((ape_rows, 4), _F32),
            kv_state=read,
            kv_state_write=write,
            score_state=read,
            score_state_write=write,
            cu_seqlens=((requests + 1,), _I32),
            start_pos=((requests,), _I32),
            state_slots=((requests,), _I32),
            entries=((emitted, 2), _BF16),
            valid=((tokens,), torch.bool),
        )
        # The ape add per token and channel; per slot and channel max, subtract,
        # exponential, sum, divide, multiply and add.
        assert flops == tokens * 4 + 2 * 7 * slots

    def test_paged_kv_cache_write_moves_only_the_tokens_with_a_slot(self):
        name = "PagedKVCacheWriteFwdOp"
        row = {"N": 5, "H_kv": 2, "D": 4, "NP": 4, "PS": 3, "some": ["k_scale", "v_scale"]}
        slots = [7, -1, 2, -1, 11]
        (flops, moved), _call = _evaluated(
            name, row, {"T": "bfloat16", "KV": "float8_e4m3fn"}, slot_mapping=slots
        )
        written = ((3, 2, 4), _FP8)
        assert moved == _ledger(
            name,
            k=((3, 2, 4), _BF16),
            v=((3, 2, 4), _BF16),
            k_cache_unread=True,
            k_cache_write=written,
            v_cache_unread=True,
            v_cache_write=written,
            slot_mapping=((5,), _I64),
            k_scale=((1,), _F32),
            v_scale=((1,), _F32),
        )
        # Per stored FP8 value, the scale and the saturating cast.
        assert flops == 2 * 2 * 3 * 2 * 4

    def test_mla_kv_cache_write_rotates_and_moves_only_the_tokens_with_a_slot(self):
        name = "MLAKVCacheWriteFwdOp"
        row = {
            "DC": 6,
            "PE": 4,
            "NP": 4,
            "PS": 3,
            "P": 16,
            "seq_lens": [3, 2],
            "fuse_rope": True,
            "some": ["kv_scale", "cos_sin_cache", "positions"],
        }
        slots = [5, -1, 0, 8, -1]  # tokens 0, 2 and 3, at positions 0, 2 and 0
        case = {"T": "bfloat16", "KV": "float8_e4m3fn", "C": "float32"}
        (flops, moved), _call = _evaluated(name, row, case, slot_mapping=slots)
        assert moved == _ledger(
            name,
            kv_c=((3, 6), _BF16),
            k_pe=((3, 4), _BF16),
            kv_cache_unread=True,
            kv_cache_write=((3, 10), _FP8),
            slot_mapping=((5,), _I64),
            kv_scale=((1,), _F32),
            positions=((3,), _I64),
            cos_sin_cache=((2, 4), _F32),  # positions 0 and 2
        )
        assert flops == 3 * (2 * 10 + 3 * 4)

    @pytest.mark.parametrize(
        "row,case",
        [
            ({"seq_lens": [3, 4], "starts": [2, 5], "some": ["seq_starts"]}, {"KV": "bfloat16"}),
            (
                {"seq_lens": [3, 4], "out_dtype": "float16", "some": ["kv_scale"]},
                {"KV": "float8_e4m3fn"},
            ),
        ],
    )
    def test_paged_kv_cache_gather_reads_the_rows_each_range_reaches(self, row, case):
        name = "PagedKVCacheGatherFwdOp"
        row = {"T_q": 7, "NP": 6, "PS": 4, "W": 3, "E": [2, 3], **row}
        (flops, moved), call = _evaluated(name, row, case)
        table, page = call.metadata_values("block_table"), row["PS"]
        starts = row.get("starts", [0, 0])
        ranges = [range(s, s + n) for s, n in zip(starts, row["seq_lens"], strict=True)]
        cache_rows = {(table[b][j // page], j % page) for b, r in enumerate(ranges) for j in r}
        pages = {(b, j // page) for b, r in enumerate(ranges) for j in r}
        fp8 = case["KV"] == "float8_e4m3fn"
        assert moved == _ledger(
            name,
            out_unread=True,
            out_write=((7, 2, 3), torch.float16 if fp8 else _BF16),
            output=None,  # the `out` written above
            kv_cache=((len(cache_rows), 2, 3), _FP8 if fp8 else _BF16),
            block_table=((len(pages),), _I32),
            cu_seqlens=((3,), _I32),
            seq_starts=None if fp8 else ((2,), _I32),
            kv_scale=((1,), _F32) if fp8 else None,
        )
        assert flops == (7 * 2 * 3 if fp8 else 0)

    def test_fused_qk_norm_rope_touches_the_q_and_k_columns_and_the_named_rows(self):
        name = "FusedQKNormRoPEFwdOp"
        row = {"D": 8, "P": 16, "R": 4, "num_heads": 3, "num_kv_heads": 1, "seq_lens": [3, 2]}
        (flops, moved), _call = _evaluated(name, row, {"T": "bfloat16", "C": "float32"})
        qk = ((5, 4 * 8), _BF16)  # 5 tokens, 3 q heads and 1 k head of width 8
        assert moved == _ledger(
            name,
            qkv=qk,
            qkv_write=qk,
            q_weight=((8,), _BF16),
            k_weight=((8,), _BF16),
            cos_sin_cache=((3, 4), _F32),  # positions 0, 1, 2
            positions=((5,), _I64),
        )
        assert flops == 5 * 4 * (4 * 8 + 3 * 4)

    def test_fused_qk_norm_rope_mrope_reads_each_sections_columns_of_its_rows(self):
        name = "FusedQKNormRoPEFwdOp"
        row = {
            "D": 8,
            "P": 16,
            "R": 6,
            "num_heads": 3,
            "num_kv_heads": 1,
            "seq_lens": [3, 2],
            "image": [],
            "zero_centered": True,
            "mrope_section": [1, 1, 1],
            "mrope_interleaved": True,
        }
        # Temporal rows 0, 1, 2; height rows 4, 5; width row 7.
        positions = [[0, 1, 2, 0, 1], [4, 4, 5, 5, 4], [7, 7, 7, 7, 7]]
        (flops, moved), _call = _evaluated(
            name, row, {"T": "bfloat16", "C": "float32"}, positions=positions
        )
        qk = ((5, 4 * 8), _BF16)
        assert moved == _ledger(
            name,
            qkv=qk,
            qkv_write=qk,
            q_weight=((8,), _BF16),
            k_weight=((8,), _BF16),
            # One cos and one sin column a section: 3 + 2 + 1 rows.
            cos_sin_cache=((6, 2), _F32),
            positions=((3, 5), _I64),
        )
        assert flops == 5 * 4 * (4 * 8 + 3 * 6) + 2 * 8

    @pytest.mark.parametrize("interleaved,widths", [(True, (2, 1, 0)), (False, (0, 3, 0))])
    def test_fused_qk_norm_rope_mrope_widths_follow_the_assignment(self, interleaved, widths):
        name = "FusedQKNormRoPEFwdOp"
        row = {
            "D": 8,
            "P": 16,
            "R": 6,
            "num_heads": 1,
            "num_kv_heads": 1,
            "seq_lens": [2],
            "image": [],
            "mrope_section": [0, 3, 0],
            "mrope_interleaved": interleaved,
        }
        # Temporal rows 0, 1; height row 4; width rows 6, 7.
        positions = [[0, 1], [4, 4], [6, 7]]
        (_flops, moved), _call = _evaluated(
            name, row, {"T": "bfloat16", "C": "float32"}, positions=positions
        )
        rows = 2 * widths[0] + widths[1] + 2 * widths[2]
        qk = ((2, 2 * 8), _BF16)
        assert moved == _ledger(
            name,
            qkv=qk,
            qkv_write=qk,
            q_weight=((8,), _BF16),
            k_weight=((8,), _BF16),
            cos_sin_cache=((rows, 2), _F32),  # a cos and a sin column a frequency
            positions=((3, 2), _I64),
        )

    @pytest.mark.parametrize("name", ["GDNFwdOp", "KDAFwdOp"])
    def test_gated_delta_rule_moves_only_the_pool_slots_its_sequences_name(self, name):
        row = {"B": 2, "S": 1, "H": 1, "HV": 2, "K": 4, "V": 4, "P": 5}
        row["some"] = ["initial_state", "state_indices"]
        (flops, moved), _call = _evaluated(name, row, {"T": "bfloat16"})
        slots = ((2, 2, 4, 4), _F32)  # one slot of 2 value heads a sequence
        gate = (2, 1, 2, 4) if name == "KDAFwdOp" else (2, 1, 2)
        assert moved == _ledger(
            name,
            q=((2, 1, 1, 4), _BF16),
            k=((2, 1, 1, 4), _BF16),
            v=((2, 1, 2, 4), _BF16),
            g=(gate, _BF16),
            beta=((2, 1, 2), _BF16),
            initial_state=slots,
            initial_state_write=slots,
            cu_seqlens=None,
            cu_seqlens_cpu=None,
            A_log=None,
            dt_bias=None,
            state_indices=((2,), _I32),
            num_accepted_tokens=None,
            o=((2, 1, 2, 4), _BF16),
            final_state=None,  # the pool written above
        )
        # Per token and value head: 7 per state value, plus KDA's decay and step terms.
        per_head = 7 * 16 + (4 + 2 * 4 if name == "KDAFwdOp" else 0)
        assert flops == 2 * 2 * per_head + (2 * 4 if name == "KDAFwdOp" else 0)

    @pytest.mark.parametrize("name", ["GDNFwdOp", "KDAFwdOp"])
    def test_gated_delta_rule_rollback_writes_a_slot_a_token(self, name):
        row = {"B": 1, "S": 6, "H": 1, "HV": 2, "K": 4, "V": 4, "P": 9, "L": 3}
        row["seq_lens"] = [3, 1, 2]
        row["some"] = ["initial_state", "cu_seqlens", "state_indices", "num_accepted_tokens"]
        (_flops, moved), _call = _evaluated(name, row, {"T": "bfloat16"})
        gate = (1, 6, 2, 4) if name == "KDAFwdOp" else (1, 6, 2)
        assert moved == _ledger(
            name,
            q=((1, 6, 1, 4), _BF16),
            k=((1, 6, 1, 4), _BF16),
            v=((1, 6, 2, 4), _BF16),
            g=(gate, _BF16),
            beta=((1, 6, 2), _BF16),
            initial_state=((3, 2, 4, 4), _F32),  # each sequence's accepted slot
            initial_state_write=((6, 2, 4, 4), _F32),  # one slot a token
            cu_seqlens=((4,), _I64),
            cu_seqlens_cpu=None,
            A_log=None,
            dt_bias=None,
            state_indices=((3, 3), _I32),
            num_accepted_tokens=((3,), _I32),
            o=((1, 6, 2, 4), _BF16),
            final_state=None,  # the pool written above
        )

    def test_causal_conv1d_touches_the_dilated_window_of_each_named_state(self):
        name = "CausalConv1dFwdOp"
        row = {
            "D": 4,
            "W": 3,
            "L": 5,
            "P": 6,
            "Tt": 5,
            "seq_lens": [3, 2],
            "dilation": 2,
            "bias": None,
        }
        row["some"] = ["cache_indices"]
        (flops, moved), _call = _evaluated(name, row, {"T": "bfloat16", "Tw": "float32"})
        window = ((2, 4, 4), _BF16)  # 2 sequences, 4 channels, (W - 1) * dilation tokens
        assert moved == _ledger(
            name,
            x=((5, 4), _BF16),
            weight=((4, 3), _F32),
            conv_state_unread=True,  # without has_initial_state every sequence starts from zeros
            conv_state_write=window,
            cu_seqlens=((3,), _I32),
            bias=None,
            has_initial_state=None,
            cache_indices=((2,), _I32),
            output=((5, 4), _BF16),
        )
        assert flops == 2 * 4 * 5 * 3

    def test_causal_conv1d_update_rollback_rewrites_the_window_and_the_drafts(self):
        name = "CausalConv1dUpdateFwdOp"
        row = {"D": 4, "W": 3, "L": 6, "P": 6, "Tt": 5, "seq_lens": [3, 2], "max_query_len": 3}
        row["some"] = ["conv_state_indices", "num_accepted_tokens", "query_start_loc"]
        (flops, moved), _call = _evaluated(name, row, {"T": "bfloat16", "Tw": "bfloat16"})
        assert moved == _ledger(
            name,
            x=((5, 4), _BF16),
            weight=((4, 3), _BF16),
            conv_state=((2, 4, 2), _BF16),  # each sequence's W - 1 history tokens
            conv_state_write=((7, 4), _BF16),  # W - 1 + q - 1 tokens a sequence
            bias=None,
            conv_state_indices=((2,), _I32),
            num_accepted_tokens=((2,), _I32),
            query_start_loc=((3,), _I32),
            output=((5, 4), _BF16),
        )
        assert flops == 2 * 5 * 4 * 3

    @pytest.mark.parametrize(
        "max_norm,norm_type,flops_per_row",
        [
            (None, 2.0, 0),
            (1.0, 2.0, 2 * 4 + 2),
            (1.0, 1.0, 2 * 4 + 1),
            (1.0, 3.0, 3 * 4 + 2),
        ],
    )
    def test_embedding_reads_the_named_rows_once(self, max_norm, norm_type, flops_per_row):
        name = "EmbeddingFwdOp"
        row = {"S": [2, 3], "V": 10, "D": 4, "max_norm": max_norm, "norm_type": norm_type}
        ids = [[1, 7, 1], [7, 2, 2]]  # rows 1, 2 and 7
        (flops, moved), _call = _evaluated(name, row, {"I": "int64", "T": "float32"}, input=ids)
        assert moved == _ledger(
            name,
            input=((2, 3), _I64),
            weight=((3, 4), _F32),
            output=((2, 3, 4), _F32),
        )
        # With max_norm, a norm per distinct row; no row is written back at the cheaper outcome.
        assert flops == 3 * flops_per_row

    def test_embedding_prices_a_scalar_index(self):
        from types import SimpleNamespace

        from tileops._perf.formulas import embedding_roofline

        sizes = {"input": 8, "weight": 10 * 4 * 4, "output": 4 * 4}
        call = SimpleNamespace(
            indices={"D": 4, "norm_type": 2.0, "max_norm": None},
            tensors={"input": ((), "int64"), "weight": ((10, 4), "float32")},
            metadata_values=lambda name: 3,
            bytes=sizes.__getitem__,
        )
        assert embedding_roofline(call) == (0, 8 + 4 * 4 + 4 * 4)

    @pytest.mark.parametrize("broadcast", [False, True])
    def test_masked_scatter_prices_the_cheaper_mask_outcome(self, broadcast):
        name = "MaskedScatterFwdOp"
        a = [1, 4] if broadcast else [6, 4]
        row = {"A": a, "M": [6, 1], "R": [6, 4]}
        (flops, moved), _call = _evaluated(name, row, {"T": "bfloat16"})
        assert moved == _ledger(
            name,
            input=((*a,), _BF16),  # the cheaper outcome: an all-false mask
            mask=((6, 1), torch.bool),
            source_unread=True,
            output=((6, 4), _BF16),
        )
        assert flops == 0

    @pytest.mark.parametrize(
        "row,case",
        [
            (
                {"B": 2, "S": 3, "H": 2, "D": 8, "P": 6, "Rh": 2},
                {"T": "bfloat16", "C": "float32", "Cs": "float32"},
            ),
            (
                {
                    "Tt": 5,
                    "H": 2,
                    "D": 8,
                    "P": 6,
                    "Rh": 2,
                    "seq_lens": [3, 2],
                    "max_seqlen": 3,
                    "inplace": True,
                    "some": ["cu_seqlens"],
                },
                {"T": "bfloat16", "C": "float32", "Cs": "bfloat16", "I": "int32"},
            ),
        ],
    )
    def test_rope_cos_sin_reads_the_rows_the_sequences_reach(self, row, case):
        name = "RoPECosSinFwdOp"
        (flops, moved), _call = _evaluated(name, row, case)
        packed = "cu_seqlens" in row.get("some", [])
        tokens, rows = (5, 3) if packed else (6, 3)
        x = ((tokens, 2, 4 if packed else 8), _BF16)  # in place, only the 4 rotated columns
        sin = torch.bfloat16 if packed else _F32
        assert moved == _ledger(
            name,
            x=x,
            x_write=x if packed else None,
            cos=((rows, 2), _F32),
            sin=((rows, 2), sin),
            cu_seqlens=((3,), _I32) if packed else None,
            seqlen_offsets=None,
            output=None if packed else x,  # in place, the output is x
        )
        assert flops == 3 * tokens * 2 * 4

    def test_rope_cos_sin_reads_the_union_of_the_rows_each_offset_reaches(self):
        name = "RoPECosSinFwdOp"
        row = {"B": 2, "S": 3, "H": 2, "D": 8, "P": 6, "Rh": 2, "some": ["seqlen_offsets"]}
        case = {"T": "bfloat16", "C": "float32", "Cs": "float32", "O": "int32"}
        (flops, moved), _call = _evaluated(name, row, case, seqlen_offsets=[0, 2])
        x = ((2, 3, 2, 8), _BF16)
        # Rows 0..2 and 2..4: five distinct rows.
        assert moved == _ledger(
            name,
            x=x,
            cos=((5, 2), _F32),
            sin=((5, 2), _F32),
            cu_seqlens=None,
            seqlen_offsets=((2,), _I32),
            output=x,
        )
        assert flops == 3 * 6 * 2 * 4

    @pytest.mark.parametrize(
        "name, extra",
        [
            ("RoPEFwdOp", {}),
            (
                "RoPELlama31FwdOp",
                {
                    "scaling_factor": 8.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "orig_max_position": 8,
                },
            ),
            ("YaRNFwdOp", {"scaling_factor": 2.0}),
            (
                "LongRoPEFwdOp",
                {"original_max_position_embeddings": 8, "rescale_factors": [1.0, 2.0]},
            ),
        ],
    )
    def test_rope_reads_only_the_cache_rows_the_positions_name(self, name, extra):
        tokens = 5
        row = {
            "head_size": 8,
            "rotary_dim": 4,
            "max_position_embeddings": 16,
            "base": 10000.0,
            "is_neox_style": True,
            "dtype": "bfloat16",
            "Hq": 2,
            "Hk": 1,
            "some": ["key"],
            **extra,
            **({"N": tokens} if name == "LongRoPEFwdOp" else {"S": [tokens]}),
        }
        case = {"T": "bfloat16", "I": "int64"}
        positions = [3, 3, 7, 0, 7]
        (flops, moved), _call = _evaluated(name, row, case, positions=positions)
        q, k = ((tokens, 16), _BF16), ((tokens, 8), _BF16)
        ledger = {"offsets": None} if name == "LongRoPEFwdOp" else {}
        # Three distinct rows of 4 cos/sin values each.
        assert (
            moved
            == _ledger(
                name,
                positions=((tokens,), _I64),
                query=q,
                key=k,
                query_out=q,
                key_out=k,
                **ledger,
            )
            + 3 * 4 * 2
        )
        assert flops == 3 * tokens * 3 * 4

    def test_longrope_reads_the_rows_positions_plus_offsets_name(self):
        name, tokens = "LongRoPEFwdOp", 5
        row = {
            "head_size": 8,
            "rotary_dim": 4,
            "max_position_embeddings": 16,
            "original_max_position_embeddings": 8,
            "base": 10000.0,
            "is_neox_style": True,
            "dtype": "bfloat16",
            "rescale_factors": None,
            "N": tokens,
            "Hq": 2,
            "Hk": 1,
            "some": ["key", "offsets"],
        }
        case = {"T": "bfloat16", "I": "int64", "O": "int32"}
        # Rows 3, 4, 7, 0, 9: five distinct, where the positions alone name three.
        positions, offsets = [3, 3, 7, 0, 7], [0, 1, 0, 0, 2]
        (flops, moved), _call = _evaluated(name, row, case, positions=positions, offsets=offsets)
        q, k = ((tokens, 16), _BF16), ((tokens, 8), _BF16)
        assert (
            moved
            == _ledger(
                name,
                positions=((tokens,), _I64),
                query=q,
                key=k,
                offsets=((tokens,), _I32),
                query_out=q,
                key_out=k,
            )
            + 5 * 4 * 2
        )
        assert flops == 3 * tokens * 3 * 4

    def test_rope_cos_sin_reads_no_table_row_when_nothing_rotates(self):
        name = "RoPECosSinFwdOp"
        row = {"B": 0, "S": 3, "H": 2, "D": 8, "P": 6, "Rh": 2}
        case = {"T": "bfloat16", "C": "float32", "Cs": "float32"}
        assert _evaluated(name, row, case)[0] == (0, 0)

    @pytest.mark.parametrize(
        "scoring_func,renormalize,read",
        [
            ("sqrtsoftplus", True, 2),
            ("sigmoid", False, 2),
            ("softmax", True, 2),
            ("softmax", False, 16),
        ],
    )
    def test_fused_topk_with_given_experts_reads_their_logits(
        self, scoring_func, renormalize, read
    ):
        name = "FusedTopKFwdOp"
        row = {
            "T": 4,
            "E": 16,
            "top_k": 2,
            "scoring_func": scoring_func,
            "renormalize": renormalize,
            "some": ["correction_bias", "expert_ids"],
        }
        (_flops, moved), _call = _evaluated(name, row, {"G": "float32"})
        # Only an unrenormalized softmax needs the whole row; the bias selects nothing.
        assert moved == _ledger(
            name,
            gating_output=((4, read), _F32),
            correction_bias_unread=True,
            expert_ids=((4, 2), _I32),
            topk_weights=((4, 2), _F32),
            topk_ids=((4, 2), _I32),
        )

    @pytest.mark.parametrize("renormalize", [False, True])
    def test_fused_topk_softmax_selecting_with_a_bias_normalizes_the_whole_row(self, renormalize):
        row = {"T": 4, "E": 16, "top_k": 2, "scoring_func": "softmax", "renormalize": renormalize}
        (flops, _moved), _call = _evaluated(
            "FusedTopKFwdOp", {**row, "some": ["correction_bias"]}, {"G": "float32"}
        )
        # Softmax (3 per logit) and its row sum and divide (2 per logit), the bias add, top_k
        # compare-and-selects per logit, and 2 per kept weight to renormalize.
        assert flops == 4 * (16 * 5 + 16 + 2 * 16 + (4 if renormalize else 0))

    def test_ngram_hash_leaves_an_empty_sequence_untouched(self):
        name = "NGramHashFwdOp"
        row = {
            "Tt": 3,
            "seq_lens": [0, 3],
            "N": 3,
            "Hd": 4,
            "S": 5,
            "W": 4,
            "eos_token_id": 0,
            "some": ["cu_seqlens", "context_indices", "num_accepted_tokens"],
        }
        (flops, moved), _call = _evaluated(name, row, {})
        # Only the second sequence reads its 2 history columns and rewrites 1 + 3.
        assert moved == _ledger(
            name,
            input_ids=((3,), _I64),
            context=((1, 2), _I64),
            context_write=((1, 4), _I64),
            multipliers=((3,), _I64),
            primes=((4,), _I64),
            offsets=((4,), _I64),
            cu_seqlens=((3,), _I32),
            context_indices=((2,), _I32),
            num_accepted_tokens=((2,), _I32),
            ngram_ids=((3, 4), _I64),
        )
        assert flops == 0

    @pytest.mark.parametrize("verify", [False, True])
    def test_ngram_hash_moves_the_context_columns_each_sequence_reaches(self, verify):
        name = "NGramHashFwdOp"
        some = ["context_indices", *(["num_accepted_tokens"] if verify else [])]
        row = {"B": 2, "L": 3, "N": 3, "Hd": 4, "S": 5, "W": 4, "eos_token_id": 0, "some": some}
        (flops, moved), _call = _evaluated(name, row, {})
        # Each sequence reads its N - 1 = 2 history columns of the pool's 5 x 4, and rewrites
        # 2, or under verification the history after its first column and its 3 tokens.
        assert moved == _ledger(
            name,
            input_ids=((2, 3), _I64),
            context=((2, 2), _I64),
            context_write=((2, 4 if verify else 2), _I64),
            multipliers=((3,), _I64),
            primes=((4,), _I64),
            offsets=((4,), _I64),
            cu_seqlens=None,
            context_indices=((2,), _I32),
            num_accepted_tokens=((2,), _I32) if verify else None,
            ngram_ids=((2, 3, 4), _I64),
        )
        assert flops == 0

    def test_chain_speculative_sampling_prices_the_cheaper_outcome(self):
        name = "ChainSpeculativeSamplingFwdOp"
        batch, n, vocab = 2, 3, 10
        (flops, moved), _call = _evaluated(name, {"B": batch, "N": n, "V": vocab}, {})
        # With N < V, every draft accepted is cheaper than a rejection at the first: the N
        # ratio tests and a draw from target row N; the ids, 2 N probabilities, one row.
        assert moved == _ledger(
            name,
            draft_probs=((batch * n,), _F32),
            draft_token_ids=((batch, n), _I32),
            target_probs=((batch * (n + vocab),), _F32),
            seed=((1,), _I64),
            offset=((1,), _I64),
            output_token_ids=((batch, n + 1), _I32),
            num_accepted=((batch,), _I32),
        )
        assert flops == batch * (2 * n + 3 * vocab + 1)

    def test_top_k_masks_pay_nothing_on_a_row_k_leaves_whole(self):
        vocab, ks = 10, [3, 10, 12, 1]
        row = {"V": vocab, "k_list": ks}
        (flops, _moved), _call = _evaluated("TopKMaskFwdOp", row, {"T": "float32"})
        assert flops == sum(2 * vocab for k in ks if k < vocab)
        (flops, _moved), _call = _evaluated("TopKTopPMaskFwdOp", row, {"T": "float32"})
        # Top-k where k restricts; over the survivors max, subtract, exp, sum, compare and
        # accumulate; p times the sum; the final mask.
        assert flops == sum((vocab if k < vocab else 0) + 6 * min(k, vocab) + 1 + vocab for k in ks)

    def test_mla_varlen_scores_each_request_under_its_causal_mask(self):
        row = {"T_q": 7, "H": 2, "DN": 4, "PE": 2, "DV": 3, "seq_lens": [3, 4], "is_causal": True}
        (flops, _moved), _call = _evaluated("MLAVarlenFwdOp", row, {"T": "bfloat16"})
        scores = sum(i + 1 for n in row["seq_lens"] for i in range(n))
        assert flops == _attention_flops(2, 6, 3, scores, 7)

    def test_mla_dense_bwd_scores_each_row_under_bottom_right_alignment(self):
        row = {"B": 2, "S_q": 5, "S_kv": 3, "H": 2, "D_qk": 6, "D_v": 4, "is_causal": True}
        (flops, _moved), _call = _evaluated("MLADenseBwdOp", row, {"T": "bfloat16"})
        # Query i sits at key position i - 2: the first two see no key, the rest 1, 2, 3.
        assert flops == 2 * _attention_bwd_flops(2, 6, 4, 1 + 2 + 3, 3)

    def test_mla_varlen_bwd_scores_each_request_under_its_mask(self):
        row = {
            "T_q": 7,
            "T_kv": 9,
            "H": 2,
            "D_qk": 6,
            "D_v": 4,
            "q_lens": [3, 4],
            "kv_lens": [5, 4],
            "is_causal": True,
        }
        (flops, _moved), _call = _evaluated("MLAVarlenBwdOp", row, {"T": "bfloat16"})
        # Query i of the first request sits at key position i + 2, of the second at i.
        scores = sum(i + 3 for i in range(3)) + sum(i + 1 for i in range(4))
        assert flops == _attention_bwd_flops(2, 6, 4, scores, 7)

    def test_gqa_paged_sink_adds_only_normalization_work(self):
        row = {
            "T_q": 3,
            "NP": 4,
            "PS": 16,
            "W": 2,
            "H": 8,
            "H_kv": 2,
            "D": 64,
            "q_lens": [2, 1],
            "cache_lens": [4, 1],
        }
        dtypes = {"T": "float16", "KV": "float16"}
        plain, _ = _evaluated("GQAPagedFwdOp", row, dtypes)
        sink, _ = _evaluated(
            "GQAPagedFwdOp", {**row, "some": [*row.get("some", []), "sinks"]}, dtypes
        )
        assert sink == (plain[0] + 8 * 3 * 3, plain[1] + 8 * 4)


# Coverage levels. Every op, implemented or spec-only, sits at
# exactly one, and the level says what an independent recount rests on.
#
#   one   The binder builds the case from the manifest: signature, one workload
#         row, dtypes, mutation. It never reads the `roofline` block, and what it
#         does share with the formula is written down: the minimum-traffic
#         definition and the checked call. Computed, not listed -- adding an op
#         earns this level or fails the completeness test below.
#   two   The binder cannot build the call and a case above does it by hand,
#         with what the case shares written next to it.
#
# Some level-one ops also keep a case above. Those cover a branch one workload
# row does not reach -- an optional input present and absent, a second dtype
# pairing -- and do not change the op's level.

# Level two: a case above recounts these by hand. The value says why the
# generated case cannot, which is what the hand-written one supplies.
HAND_WRITTEN = {
    "AvgPool1dFwdOp": "a stride past the kernel leaves input positions no window reads",
    "DSADenseFwdOp": "it reads the kv rows its top-k indices select, not the cache",
    "DeltaNetChunkBwdOp": "it reads only the strict-lower triangle of each Aw and Au chunk block",
    "MaxPool1dFwdOp": "a dilated or strided window leaves input positions no window reads",
    "MaxPool1dIndicesFwdOp": "a dilated or strided window leaves input positions no window reads",
    "FusedMoEExpertsFwdOp": "the routed weight reads follow the values in `topk_ids`",
    "FusedMoEFwdOp": "the routed weight reads follow the routing its experts stage receives",
    "FusedMoESharedExpertFwdOp": "the routed weight reads follow the routing its experts stage receives",
    "NSAVarlenFwdOp": "how much it reads follows the values in `block_counts`",
    "IndexedExpertMLPFwdOp": "the routed weight reads follow the values in `topk_ids`",
    "GQAPagedFwdOp": "it reads the rows its page table names, not the pool",
    "MLAPagedFwdOp": "it reads the cache rows its block table reaches, not the pool",
    "DSAPagedFwdOp": "it reads the cache rows its valid index slots name",
    "PagedKVCacheWriteFwdOp": "only the tokens `slot_mapping` gives a slot are read and written",
    "MLAKVCacheWriteFwdOp": "only the tokens `slot_mapping` gives a slot are read and written",
    "PagedKVCacheGatherFwdOp": "it reads the cache rows each request's range reaches, not the pool",
    "FusedQKNormRoPEFwdOp": "it leaves the v columns untouched and reads only the named cos/sin rows",
    "GDNFwdOp": "it reads and writes only the pool slots its indices name",
    "KDAFwdOp": "it reads and writes only the pool slots its indices name",
    "CausalConv1dFwdOp": "it touches the dilated window of the states its sequences name",
    "CausalConv1dUpdateFwdOp": "it touches the dilated window of the states its sequences name",
    "EmbeddingFwdOp": "it reads the weight rows its indices name, not the table",
    "MaskedScatterFwdOp": "each output element reads input or source, as the mask decides",
    "RoPECosSinFwdOp": "it reads the cos/sin rows the sequences reach, and in place only the rotated columns",
    "RoPEFwdOp": "it reads only the cos/sin cache rows the positions name",
    "RoPELlama31FwdOp": "it reads only the cos/sin cache rows the positions name",
    "YaRNFwdOp": "it reads only the cos/sin cache rows the positions name",
    "LongRoPEFwdOp": "it reads only the cos/sin cache rows the positions and offsets name",
    "ChainSpeculativeSamplingFwdOp": "where the chain stops is drawn at run time, so it prices the cheaper outcome",
    "GatedPoolCompressFwdOp": "the state rows it reads and the entries it writes follow the positions",
    "NGramHashFwdOp": "it reads and rewrites only the context columns each sequence's slot reaches",
    "FusedTopKFwdOp": "with caller-given experts it reads only their logits and no bias",
}


def _entries() -> list[str]:
    """Every entry: a spec-only one is recounted from its signature, needing no implementation."""
    from tileops.manifest import load_manifest

    return sorted(load_manifest())


def _draws_metadata(op_name: str) -> bool:
    """Whether an entry generates some metadata tensor at random."""
    from tileops.manifest import load_manifest
    from tileops.manifest.primitives import RANDOM_GENERATORS

    entry = load_manifest()[op_name]
    inputs = entry["signature"].get("inputs") or {}
    return any(
        str(spec.get("values", "")).split("(")[0] in RANDOM_GENERATORS for spec in inputs.values()
    )


def _binder_builds(op_name: str) -> bool:
    """Whether the manifest alone builds a case for *op_name*.

    The formula is not called here: whether it agrees, or even returns, is a
    separate question, and a formula that raises is a defect.
    """
    from tests.roofline_binder import manifest_cases

    return bool(list(manifest_cases(op_name)))


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
    """Every op sits at exactly one level, and the level is the truth."""

    def test_a_generated_case_equals_its_op(self):
        from tests.roofline_binder import manifest_cases

        checked = 0
        for op_name in _entries():
            if op_name in HAND_WRITTEN:
                continue
            for label, dtype, op, oracle, _reads in manifest_cases(op_name):
                assert op.eval_roofline()[1] == oracle, f"{op_name} {label} {dtype}"
                checked += 1
        assert checked > 0

    def test_a_generated_case_agrees_on_the_read_half(self):
        """The audit judges the read side alone, and an op derives it by taking the
        write side the signature settles off its `bytes`. Where the
        binder recounts the op, the two halves have to be the same halves."""
        from tests.roofline_binder import manifest_cases

        checked = 0
        for op_name in _entries():
            if op_name in HAND_WRITTEN:
                continue
            for label, dtype, op, _oracle, reads in manifest_cases(op_name):
                declared = op.eval_roofline_read_bytes()
                assert declared is not None, f"{op_name} {label} {dtype}"
                assert declared == reads, f"{op_name} {label} {dtype}"
                checked += 1
        assert checked > 0

    def test_a_declared_op_is_one_the_manifest_does_not_already_check(self):
        """Level two is for ops the manifest cannot recount, not a queue.

        An op whose rows draw metadata at random stays at level two however its rows fall:
        one row's draw agreeing with the recount says nothing of another's
        (docs/design/roofline.md §4.7).
        """
        promotable = sorted(
            name
            for name in HAND_WRITTEN
            if not _draws_metadata(name) and _binder_builds(name) and _binder_agrees(name)
        )
        assert not promotable, (
            f"the binder now recounts {promotable} and the formula agrees; move them out "
            "of HAND_WRITTEN so the generated case is what checks them"
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
            "sums anonymous tuples cannot be checked against the signature. A case "
            "another xdist worker ran is not recorded here: run this file in one "
            "process, or pass --dist loadfile"
        )
