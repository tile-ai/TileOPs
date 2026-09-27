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
            ((tokens, hidden), bf16),  # shared_output, returned separately
        )
        assert self._priced(op, tensors)[1] == routed + shared

    def test_nsa_forward_reads_the_rows_its_selection_kept(self):
        """How much this call reads follows `block_counts`, so the case reads the
        selection the manifest row generates rather than inventing one of its own."""
        row = _manifest_rows("NSAVarlenFwdOp")[0]
        for is_causal in (True, False):
            self._nsa_forward_case(_manifest_call("NSAVarlenFwdOp", dict(row, is_causal=is_causal)))

    @staticmethod
    def _nsa_forward_case(call):
        from tileops.perf.formulas import nsa_fwd_varlen_roofline

        ix = call.ix
        c_seq_len, heads, head_kv, dim = ix["T_q"], ix["H"], ix["H_kv"], ix["D"]
        block_size, selected = ix["block_size"], ix["SEL"]
        # Key rows some token scores, per KV head: each kept block cut at the token (causal)
        # or the sequence end; a row several tokens score is read once.
        offsets = call.values("offsets")
        rows = {
            (h, offsets[request] + r)
            for (request, position), counts, picks in zip(
                call.values("token_indices"),
                call.values("block_counts"),
                call.values("block_indices"),
                strict=True,
            )
            for h in range(head_kv)
            for start in picks[h][: counts[h]]
            for r in range(
                start * block_size,
                min(
                    (start + 1) * block_size,
                    position + 1 if ix["is_causal"] else offsets[request + 1] - offsets[request],
                ),
            )
        }
        gathered = len(rows) * dim
        oracle = _ledger(
            "NSAVarlenFwdOp",
            q=((c_seq_len, heads, dim), torch.float16),
            # k and v are read through the selection, not end to end
            k=((gathered,), torch.float16),
            v=((gathered,), torch.float16),
            block_indices=((c_seq_len, head_kv, selected), torch.int32),
            block_counts=((c_seq_len, head_kv), torch.int32),
            offsets=((len(offsets),), torch.int32),
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
        batch = len(_manifest_call(name, base).values("cache_seqlens"))
        short = dict(base, T_q=batch, q_lens=[1] * batch, cache_lens=[0, 64] * (batch // 2))
        rows = {
            "cached": base,
            "fp8 cache": dict(base, cache_dtype="float8_e4m3fn"),
            "short": short,
        }
        for label, row in rows.items():
            call = _manifest_call(name, row)
            ix = call.ix
            heads, heads_kv, dim, page_size = ix["H"], ix["H_kv"], ix["D"], ix["page_size"]
            offsets, cache_lens = call.values("cu_seqlens_q"), call.values("cache_seqlens")
            q_lens = [b - a for a, b in zip(offsets, offsets[1:], strict=False)]
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

    def test_gqa_paged_reads_the_rows_its_page_table_names(self):
        """The cache is one pool, and the call reads the rows its page table names as far
        as each request's cached length; the pages it never names move nothing."""
        from tileops.perf.formulas import gqa_paged_fwd_roofline

        name = "GroupedQueryAttentionPagedFwdOp"
        call = _manifest_call(name)
        ix = call.ix
        heads, heads_kv, dim, page = ix["H"], ix["H_kv"], ix["D"], ix["PS"]
        table, cached = call.values("page_table"), call.values("cache_seqlens")
        rows = {(table[b][r // page], r % page) for b, n in enumerate(cached) for r in range(n)}
        kv = ((len(rows), heads_kv, dim), torch.float16)
        oracle = _ledger(
            name,
            q=((ix["T_q"], heads, dim), torch.float16),
            k_pages=kv,
            v_pages=kv,
            page_table=((sum(-(-n // page) for n in cached),), torch.int32),
            cache_seqlens=((len(cached),), torch.int32),
            cu_seqlens_q=((len(cached) + 1,), torch.int32),
            q_scale=None,
            k_scale=None,
            v_scale=None,
            rope_cos=None,
            rope_sin=None,
            o=((ix["T_q"], heads, dim), torch.float16),
        )
        assert gqa_paged_fwd_roofline(call)[1] == oracle

    def test_topk_selector_reads_only_its_windows(self):
        """The manifest rows select from whole rows; a narrower window reads and compares
        only the scores inside it."""
        from tileops.ops import TopkSelectorFwdOp

        batch, seq, extent, topk = 1, 4, 16, 2
        starts = torch.tensor([[0, 4, 8, 12]], dtype=torch.int32)
        ends = torch.tensor([[4, 8, 16, 12]], dtype=torch.int32)
        tensors = {
            "index_score": torch.empty(batch, seq, extent, 1),
            "starts": starts,
            "ends": ends,
        }
        scores = int((ends - starts).clamp(min=0).sum())
        oracle = _ledger(
            "TopkSelectorFwdOp",
            index_score=((scores,), torch.float32),
            starts=((batch, seq), torch.int32),
            ends=((batch, seq), torch.int32),
            indexes=((batch, seq, 1, topk), torch.int32),
        )
        assert self._priced(TopkSelectorFwdOp(topk), tensors) == (scores, oracle)

    def test_windowed_pools_read_the_positions_some_window_reaches(self):
        """A stride past the kernel span, or a dilation, leaves input positions no window
        reads; the case recounts the positions from the row's geometry."""
        from tileops.perf.formulas import pool_roofline

        for op_name in ("AvgPool1dFwdOp", "MaxPool1dFwdOp", "MaxPool1dIndicesFwdOp"):
            for row in _manifest_rows(op_name):
                call = _manifest_call(op_name, row)
                ix = call.ix
                read = {
                    o * ix["sW"] - ix["pW"] + j * ix.get("dW", 1)
                    for o in range(ix["L_out"])
                    for j in range(ix["kW"])
                } & set(range(ix["L_in"]))
                dtype = getattr(torch, call.tensors["input"][1])
                out = ((ix["N"], ix["C"], ix["L_out"]), dtype)
                tensors = {"input": ((ix["N"], ix["C"], len(read)), dtype), "output": out}
                if op_name == "MaxPool1dIndicesFwdOp":
                    tensors["indices"] = ((ix["N"], ix["C"], ix["L_out"]), torch.int64)
                assert pool_roofline(call)[1] == _ledger(op_name, **tensors), row["label"]

    def test_dsa_decode_reads_the_kv_rows_some_query_selects(self):
        """Top-k slots past the causal bound or repeated select nothing more, so `kv` is
        read at the rows the row's generated indices reach."""
        from tileops.perf.formulas import dsa_decode_roofline

        op_name = "DeepSeekSparseAttentionDecodeWithKVCacheFwdOp"
        for row in _manifest_rows(op_name):
            call = _manifest_call(op_name, row)
            ix = call.ix
            rows = {
                (b, g, j)
                for b, batch in enumerate(call.values("indices"))
                for s, heads in enumerate(batch)
                for g, slots in enumerate(heads)
                for j in slots
                if 0 <= j < ix["S_kv"]
                and (j + 1) * ix["stride_kv"] - 1 <= ix["q_start_index_s"] + s
            }
            dtype = getattr(torch, call.tensors["q"][1])
            width = ix["D"] + ix["dim_tail"]
            oracle = _ledger(
                op_name,
                q=((ix["B"], ix["S"], ix["H"], width), dtype),
                kv=((len(rows), width), dtype),
                indices=((ix["B"], ix["S"], ix["H_kv"], ix["K"]), torch.int32),
                o=((ix["B"], ix["S"], ix["H"], ix["D"]), dtype),
            )
            assert dsa_decode_roofline(call)[1] == oracle, row["label"]

    def test_deltanet_bwd_reads_the_strict_lower_triangle_of_aw_and_au(self):
        """Each chunk's C x C block of Aw and Au has a unit diagonal and a zero upper
        triangle, so only the strict-lower C * (C - 1) / 2 entries are read."""
        from tests.roofline_binder import manifest_cases

        op_name = "DeltaNetBwdOp"
        rows = {row["label"]: row for row in _manifest_rows(op_name)}
        for label, dtype_name, op, _oracle, _reads in manifest_cases(op_name):
            ix = _manifest_call(op_name, rows[label]).ix
            b, h, n, dk, dv, c = ix["B"], ix["H"], ix["L"], ix["DK"], ix["DV"], ix["chunk_size"]
            dtype = getattr(torch, dtype_name)
            triangle = ((b, h, n // c, c * (c - 1) // 2), dtype)
            oracle = _ledger(
                op_name,
                do=((b, h, n, dv), dtype),
                q=((b, h, n, dk), dtype),
                k=((b, h, n, dk), dtype),
                v=((b, h, n, dv), dtype),
                beta=((b, h, n), dtype),
                S=((b, h, n // c + 1, dk, dv), torch.float32),
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
        from tileops.ops import GroupedGemmFwdOp

        batch_sum, batch_count, n, k = 64, 4, 32, 16
        f16, groups = torch.float16, ((batch_count,), torch.int32)
        tensors = {
            "a": torch.empty(batch_sum, k, dtype=f16),
            "b": torch.empty(batch_count, n, k, dtype=f16),
            **{
                name: torch.zeros(batch_count, dtype=torch.int32)
                for name in ("batch_sizes", "batch_offsets", "batch_padded_offsets")
            },
        }
        oracle = _ledger(
            "GroupedGemmFwdOp",
            a=((batch_sum, k), f16),
            b=((batch_count, n, k), f16),
            batch_sizes=groups,
            batch_offsets=groups,
            batch_padded_offsets_unread=True,
            output=((batch_sum, n), f16),
        )
        assert self._priced(GroupedGemmFwdOp(), tensors)[1] == oracle


def _evaluated(op_name: str, row: dict, case: dict, **values):
    """``(flops, bytes)`` the generated evaluator prices for a row of a spec-only entry, and the
    call; *values* replace the named metadata tensors' generated contents."""
    import dataclasses

    from tests.roofline_binder import signature_class
    from tileops.manifest import load_adts, load_manifest
    from tileops.manifest.plan import entry_plan
    from tileops.manifest.workload import instantiate

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
    op = cls(**call.arguments(tensors))
    checked = cls._signature.check(op, {t: tensors[t] for t in plan.sig.inputs})
    metadata = {n: torch.tensor(call.values(n)) for n in checked.metadata}
    op._signature_call = dataclasses.replace(checked, metadata=metadata)
    return op.eval_roofline(), call


def _attention_flops(heads, qk, v, scores, rows):
    # Per score two contractions and the softmax (5); per output element its divide.
    return heads * (scores * (2 * qk + 2 * v + 5) + rows * v)


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
                {"S_q": 2, "NP": 8, "W": 4, "cache_lens": [5, 9]},
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
        name = "MultiHeadLatentAttentionPagedFwdOp"
        row = {"H": 3, "DK": 12, "PS": 4, "kv_lora_rank": 8, **row}
        (flops, moved), call = _evaluated(name, row, case)
        heads, dk, rank, page, s_q = row["H"], row["DK"], row["kv_lora_rank"], row["PS"], row["S_q"]
        table, lengths = call.values("block_table"), call.values("cache_seqlens")
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

    def test_dsa_paged_scores_each_valid_slot_and_reads_the_rows_they_name(self):
        name = "DeepSeekSparseAttentionPagedFwdOp"
        row = {"S_q": 2, "H": 2, "K": 4, "NP": 6, "PS": 4, "W": 3, "cache_lens": [5, 10]}
        # Request 0: query 0 selects nothing, query 1 three positions on two pages.
        # Request 1: a repeated slot, then nothing.
        indices = [[[-1, -1, -1, -1], [0, 4, 3, -1]], [[3, 3, -1, -1], [-1, -1, -1, -1]]]
        (flops, moved), call = _evaluated(name, row, {}, indices=indices)
        table, lengths, page = call.values("block_table"), call.values("cache_seqlens"), row["PS"]
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
        assert moved == _ledger(
            name,
            q=((rows, row["H"], 576), _BF16),  # the query rows that score something
            kv_cache=((len(cache_rows), 656), torch.uint8),
            block_table=((len(pages),), _I32),
            cache_seqlens=((2,), _I32),
            indices=((2, 2, 4), _I32),
            o=((2, 2, row["H"], 512), _BF16),
            lse=((2, 2, row["H"]), _F32),
        )
        # Each distinct row's 512 latent values are dequantized once.
        assert flops == _attention_flops(row["H"], 576, 512, scores, rows) + 512 * len(cache_rows)

    def test_paged_kv_cache_write_moves_only_the_tokens_with_a_slot(self):
        name = "PagedKVCacheWriteFwdOp"
        row = {"N": 5, "H_kv": 2, "D": 4, "NP": 4, "PS": 3, "some": ["k_scale"]}
        slots = [7, -1, 2, -1, 11]
        (flops, moved), _call = _evaluated(
            name, row, {"T": "bfloat16", "KV": "float8_e4m3fn"}, slot_mapping=slots
        )
        written = ((3, 2, 4), _FP8)
        assert moved == _ledger(
            name,
            k=((3, 2, 4), _BF16),
            v=((3, 2, 4), _BF16),
            k_pages_unread=True,
            k_pages_write=written,
            v_pages_unread=True,
            v_pages_write=written,
            slot_mapping=((5,), _I64),
            k_scale=((1,), _F32),
            v_scale=((1,), _F32),
        )
        # Per stored FP8 value, the scale and the saturating cast.
        assert flops == 2 * 2 * 3 * 2 * 4

    def test_mla_kv_cache_write_rotates_and_moves_only_the_tokens_with_a_slot(self):
        name = "MultiHeadLatentAttentionKVCacheWriteFwdOp"
        row = {
            "DC": 6,
            "PE": 4,
            "NP": 4,
            "PS": 3,
            "P": 16,
            "seq_lens": [3, 2],
            "fuse_rope": True,
            "some": ["scale"],
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
            scale=((1,), _F32),
            positions=((3,), _I64),
            cos_sin_cache=((2, 4), _F32),  # positions 0 and 2
        )
        assert flops == 3 * (2 * 10 + 3 * 4)

    @pytest.mark.parametrize(
        "row,case",
        [
            ({"seq_lens": [3, 4], "starts": [2, 5], "some": ["seq_starts"]}, {"KV": "bfloat16"}),
            (
                {"seq_lens": [3, 4], "out_dtype": "float16", "some": ["scale"]},
                {"KV": "float8_e4m3fn"},
            ),
        ],
    )
    def test_paged_kv_cache_gather_reads_the_rows_each_range_reaches(self, row, case):
        name = "PagedKVCacheGatherFwdOp"
        row = {"T_q": 7, "NP": 6, "PS": 4, "W": 3, "E": [2, 3], **row}
        (flops, moved), call = _evaluated(name, row, case)
        table, page = call.values("block_table"), row["PS"]
        starts = row.get("starts", [0, 0])
        ranges = [range(s, s + n) for s, n in zip(starts, row["seq_lens"], strict=True)]
        cache_rows = {(table[b][j // page], j % page) for b, r in enumerate(ranges) for j in r}
        pages = {(b, j // page) for b, r in enumerate(ranges) for j in r}
        fp8 = case["KV"] == "float8_e4m3fn"
        assert moved == _ledger(
            name,
            dst_unread=True,
            dst_write=((7, 2, 3), torch.float16 if fp8 else _BF16),
            cache=((len(cache_rows), 2, 3), _FP8 if fp8 else _BF16),
            block_table=((len(pages),), _I32),
            cu_seq_lens=((3,), _I32),
            seq_starts=None if fp8 else ((2,), _I32),
            scale=((1,), _F32) if fp8 else None,
        )
        assert flops == (7 * 2 * 3 if fp8 else 0)

    def test_fused_qk_norm_rope_touches_the_q_and_k_columns_and_the_named_rows(self):
        name = "FusedQKNormRopeFwdOp"
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
        row = {"T_q": 7, "H": 2, "DN": 4, "PE": 2, "DV": 3, "seq_lens": [3, 4]}
        (flops, _moved), _call = _evaluated(
            "MultiHeadLatentAttentionVarlenFwdOp", row, {"T": "bfloat16"}
        )
        scores = sum(i + 1 for n in row["seq_lens"] for i in range(n))
        assert flops == _attention_flops(2, 6, 3, scores, 7)


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
#   three No independent recount is available yet. Marked with what is missing,
#         and asserted against nothing.
#
# Some level-one ops also keep a case above. Those cover a branch one workload
# row does not reach -- an optional input present and absent, a second dtype
# pairing -- and do not change the op's level.

# Level two: a case above recounts these by hand. The value says why the
# generated case cannot, which is what the hand-written one supplies.
HAND_WRITTEN = {
    "AvgPool1dFwdOp": "a stride past the kernel leaves input positions no window reads",
    "DeepSeekSparseAttentionDecodeWithKVCacheFwdOp": "it reads the kv rows its top-k indices select, not the cache",
    "DeltaNetBwdOp": "it reads only the strict-lower triangle of each Aw and Au chunk block",
    "MaxPool1dFwdOp": "a dilated or strided window leaves input positions no window reads",
    "MaxPool1dIndicesFwdOp": "a dilated or strided window leaves input positions no window reads",
    "FusedMoEExpertsFwdOp": "the routed weight reads follow the values in `topk_ids`",
    "FusedMoeFwdOp": "the routed weight reads follow the routing its experts stage receives",
    "FusedMoeSharedExpertFwdOp": "the routed weight reads follow the routing its experts stage receives",
    "GroupedGemmFwdOp": "`batch_padded_offsets` is passed and no kernel indexes it",
    "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp": "it reads the pages its block table names, not the pool",
    "NSAVarlenFwdOp": "how much it reads follows the values in `block_counts`",
    "NSATopkVarlenFwdOp": "`lse_in` is passed and the kernel recomputes the lse instead of reading it",
    "IndexedExpertMLPFwdOp": "the routed weight reads follow the values in `topk_ids`",
    "GroupedQueryAttentionPagedFwdOp": "it reads the rows its page table names, not the pool",
    "MultiHeadLatentAttentionPagedFwdOp": "it reads the cache rows its block table reaches, not the pool",
    "DeepSeekSparseAttentionPagedFwdOp": "it reads the cache rows its valid index slots name",
    "PagedKVCacheWriteFwdOp": "only the tokens `slot_mapping` gives a slot are read and written",
    "MultiHeadLatentAttentionKVCacheWriteFwdOp": "only the tokens `slot_mapping` gives a slot are read and written",
    "PagedKVCacheGatherFwdOp": "it reads the cache rows each request's range reaches, not the pool",
    "FusedQKNormRopeFwdOp": "it leaves the v columns untouched and reads only the named cos/sin rows",
    "ChainSpeculativeSamplingFwdOp": "where the chain stops is drawn at run time, so it prices the cheaper outcome",
}

# Level three: no independent recount is available. Empty, and an entry here has
# to say what is missing rather than that nobody has got to it.
NOT_RECOUNTABLE: dict[str, str] = {}


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

    The formula is not called here. Whether it agrees, or even returns, is a
    separate question: a formula that raises is a defect, and treating that as
    "the binder cannot build this" would let it qualify for level three.
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
            if op_name in HAND_WRITTEN or op_name in NOT_RECOUNTABLE:
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
            if op_name in HAND_WRITTEN or op_name in NOT_RECOUNTABLE:
                continue
            for label, dtype, op, _oracle, reads in manifest_cases(op_name):
                declared = op.eval_roofline_read_bytes()
                assert declared is not None, f"{op_name} {label} {dtype}"
                assert declared == reads, f"{op_name} {label} {dtype}"
                checked += 1
        assert checked > 0

    def test_every_op_sits_at_one_level(self):
        both = sorted(set(HAND_WRITTEN) & set(NOT_RECOUNTABLE))
        assert not both, f"declared at two levels: {both}"
        unknown = sorted((set(HAND_WRITTEN) | set(NOT_RECOUNTABLE)) - set(_entries()))
        assert not unknown, f"declared but not in the manifest: {unknown}"

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
