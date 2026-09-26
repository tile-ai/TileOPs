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


class TestBytesOracle:
    # __new__ + attribute binding keeps the oracle CUDA-free; each case binds
    # exactly the state the op's eval_roofline reads after a forward().

    def test_conv2d_counts_input_weight_output_and_bias(self):
        from tileops.ops.convolution import Conv2dFwdOp

        n, c_in, h, w = 8, 64, 56, 56
        c_out, c_in_g, kh, kw = 128, 64, 3, 3
        out_h = out_w = 54  # stride 1, no padding
        for has_bias in (True, False):
            op = Conv2dFwdOp.__new__(Conv2dFwdOp)
            op.input_shape = (n, c_in, h, w)
            op.weight_shape = (c_out, c_in_g, kh, kw)
            op.bias_shape = (c_out,) if has_bias else None
            op.bias = None
            op.dtype = torch.float16
            op.stride, op.padding, op.dilation, op.groups = 1, 0, 1, 1
            oracle = _nbytes(
                ((n, c_in, h, w), torch.float16),
                ((c_out, c_in_g, kh, kw), torch.float16),
                ((n, c_out, out_h, out_w), torch.float16),
                *((((c_out,), torch.float16),) if has_bias else ()),
            )
            assert op.eval_roofline()[1] == oracle, f"has_bias={has_bias}"

    def test_gemm_fp8_counts_fp8_inputs_fp32_scales_and_out_dtype(self):
        from tileops.ops.gemm.gemm import GemmFp8FwdOp

        m, n, k = 4096, 4096, 8192
        for has_bias in (True, False):
            op = GemmFp8FwdOp.__new__(GemmFp8FwdOp)
            op.m, op.n, op.k = m, n, k
            op.dtype = torch.float8_e4m3fn
            op.out_dtype = torch.bfloat16
            op.scale_a_shape = (m, 1)
            op.scale_b_shape = (1, n)
            op.has_bias = has_bias
            oracle = _ledger(
                "GemmFp8FwdOp",
                a=((m, k), torch.float8_e4m3fn),
                b=((k, n), torch.float8_e4m3fn),
                scale_a=((m, 1), torch.float32),
                scale_b=((1, n), torch.float32),
                bias=(((n,), torch.bfloat16) if has_bias else None),
                d=((m, n), torch.bfloat16),
            )
            assert op.eval_roofline()[1] == oracle, f"has_bias={has_bias}"

    def test_w4a16_counts_packed_weights_and_group_metadata(self):
        from tileops.ops.gemm.gemm import GemmW4A16FwdOp

        m, n, k, group_size = 4096, 8192, 8192, 128
        op = GemmW4A16FwdOp.__new__(GemmW4A16FwdOp)
        op.m, op.n, op.k = m, n, k
        op.dtype = torch.float16
        op.group_size = group_size
        groups = k // group_size
        oracle = _ledger(
            "GemmW4A16FwdOp",
            activation=((m, k), torch.float16),
            # int4 weights, two per byte, stated as the bytes they occupy
            packed_weight=((n, k // 2), torch.int8),
            weight_scale=((n, groups), torch.float16),
            weight_zero=((n, groups), torch.int8),
            output=((m, n), torch.float16),
        )
        assert op.eval_roofline()[1] == oracle

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

    def test_gqa_dense_counts_qkv_output_and_the_optional_inputs(self):
        from tileops.ops.attention.gqa import GroupedQueryAttentionDenseFwdOp

        batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 256, 1792, 8, 2, 128
        q_shape = (batch, seq_len_q, heads, dim)
        kv_shape = (batch, seq_len_kv, heads_kv, dim)
        scales = (((batch, heads_kv), torch.float32),) * 3
        tables = (((seq_len_kv, dim // 4), torch.float16),) * 2
        # (q/k/v dtype, output dtype, the optional tensors the call passed)
        calls = {
            "16-bit": (torch.float16, torch.float16, ()),
            "fused RoPE": (torch.float16, torch.float16, tables),
            "FP8": (torch.float8_e4m3fn, torch.float16, scales),
        }
        for label, (dtype, out_dtype, optional) in calls.items():
            op = GroupedQueryAttentionDenseFwdOp.__new__(GroupedQueryAttentionDenseFwdOp)
            op._roofline_kwargs = {
                "q_shape": q_shape,
                "k_shape": kv_shape,
                "is_causal": True,
                "dtype": dtype,
                "out_dtype": out_dtype,
                "optional_shapes": optional,
            }
            scales, tables = (optional, ()) if len(optional) == 3 else ((), optional)
            oracle = _ledger(
                "GroupedQueryAttentionDenseFwdOp",
                q=(q_shape, dtype),
                k=(kv_shape, dtype),
                v=(kv_shape, dtype),
                q_scale=(scales[0] if scales else None),
                k_scale=(scales[1] if scales else None),
                v_scale=(scales[2] if scales else None),
                rope_cos=(tables[0] if tables else None),
                rope_sin=(tables[1] if tables else None),
                o=(q_shape, out_dtype),
            )
            assert op.eval_roofline()[1] == oracle, label

    def test_mha_backward_counts_o_and_lse(self):
        from tileops.ops.attention.mha import MultiHeadAttentionBwdOp

        batch, seq_len, heads, dim = 2, 2048, 16, 128
        shape = (batch, seq_len, heads, dim)
        op = MultiHeadAttentionBwdOp.__new__(MultiHeadAttentionBwdOp)
        op.batch, op.seq_len, op.heads, op.dim = batch, seq_len, heads, dim
        op.is_causal = True
        op.dtype = torch.float16
        oracle = _nbytes(
            *((shape, torch.float16),) * 5,  # q, k, v, o, do
            ((batch, heads, seq_len), torch.float32),  # lse
            *((shape, torch.float16),) * 3,  # dq, dk, dv
        )
        assert op.eval_roofline()[1] == oracle

    def test_gqa_backward_prices_the_kv_tensors_at_the_kv_head_count(self):
        from tileops.ops.attention.gqa import GroupedQueryAttentionBwdOp

        batch, seq_len, heads, heads_kv, dim = 2, 2048, 16, 4, 128
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        op = GroupedQueryAttentionBwdOp.__new__(GroupedQueryAttentionBwdOp)
        op.batch, op.seq_len, op.heads, op.heads_kv, op.dim = batch, seq_len, heads, heads_kv, dim
        op.is_causal = True
        op.dtype = torch.float16
        oracle = _nbytes(
            *((q_shape, torch.float16),) * 3,  # q, o, do
            *((kv_shape, torch.float16),) * 2,  # k, v
            ((batch, heads, seq_len), torch.float32),  # lse
            (q_shape, torch.float16),  # dq
            *((kv_shape, torch.float16),) * 2,  # dk, dv
        )
        assert op.eval_roofline()[1] == oracle

    def test_fp8_indexer_prices_each_input_at_its_own_dtype_and_the_scale_by_presence(self):
        from tileops.ops.fp8_lightning_indexer import FP8LightningIndexerFwdOp

        batch, seq_len, heads, index_dim = 1, 4096, 32, 128
        seq_len_kv, kv_group = 4096, 1
        scale_shape = (batch, seq_len_kv, kv_group)
        fp8, bf16 = torch.float8_e4m3fn, torch.bfloat16
        # Handed bf16, the op quantizes and produces the scale itself: both are
        # intermediates. Only the pre-quantized call reads a caller's scale, and
        # only it requires index_q and index_k to share a dtype.
        calls = {
            "bf16 inputs": (bf16, bf16, None),
            "pre-quantized": (fp8, fp8, scale_shape),
            "fp8 inputs, quantized k": (fp8, fp8, None),
            "mixed dtypes": (fp8, bf16, None),
        }
        for label, (q_dtype, k_dtype, supplied_scale) in calls.items():
            op = FP8LightningIndexerFwdOp.__new__(FP8LightningIndexerFwdOp)
            op.batch, op.seq_len, op.heads, op.index_dim = batch, seq_len, heads, index_dim
            op.seq_len_kv, op.kv_group = seq_len_kv, kv_group
            op.dtype, op.index_k_dtype = q_dtype, k_dtype
            op.index_k_scale_shape = supplied_scale
            oracle = _nbytes(
                ((batch, seq_len, heads, index_dim), q_dtype),  # index_q
                ((batch, seq_len_kv, kv_group, index_dim), k_dtype),  # index_k
                ((seq_len, heads), torch.float32),  # weights
                ((seq_len,), torch.int32),  # cu_seqlen_ks
                ((seq_len,), torch.int32),  # cu_seqlen_ke
                *(((supplied_scale, torch.float32),) if supplied_scale else ()),
                ((batch, seq_len, seq_len_kv, kv_group), torch.float32),  # logits
            )
            assert op.eval_roofline()[1] == oracle, label

    def test_gqa_prefill_varlen_counts_its_packed_tensors_and_bounds(self):
        from tileops.ops.attention.gqa import GroupedQueryAttentionPrefillVarlenFwdOp

        batch, heads, heads_kv, dim = 4, 32, 8, 128
        q_lens = [512] * batch
        total_q = total_kv = sum(q_lens)
        bounds = [0]
        for length in q_lens:
            bounds.append(bounds[-1] + length)
        cu = torch.tensor(bounds, dtype=torch.int32)
        op = GroupedQueryAttentionPrefillVarlenFwdOp.__new__(
            GroupedQueryAttentionPrefillVarlenFwdOp
        )
        # The formula derives per-request lengths from these cumulative bounds.
        op._roofline_kwargs = {
            "q_shape": (total_q, heads, dim),
            "k_shape": (total_kv, heads_kv, dim),
            "batch": batch,
            "max_seqlen_q": max(q_lens),
            "max_seqlen_kv": max(q_lens),
            "is_causal": True,
            "dtype": "float16",
            "cu_seqlens_q": cu,
            "cu_seqlens_kv": cu,
        }
        oracle = _ledger(
            "GroupedQueryAttentionPrefillVarlenFwdOp",
            q=((total_q, heads, dim), torch.float16),
            k=((total_kv, heads_kv, dim), torch.float16),
            v=((total_kv, heads_kv, dim), torch.float16),
            cu_seqlens_q=((batch + 1,), torch.int32),
            cu_seqlens_kv=((batch + 1,), torch.int32),
            o=((total_q, heads, dim), torch.float16),
        )
        assert op.eval_roofline()[1] == oracle

    def test_gqa_sliding_window_varlen_counts_its_packed_tensors_and_bounds(self):
        from tileops.ops.attention.gqa import GroupedQueryAttentionSlidingWindowVarlenFwdOp

        batch, heads, heads_kv, dim = 4, 32, 8, 128
        q_lens = [512] * batch
        total_q = total_k = sum(q_lens)
        bounds = [0]
        for length in q_lens:
            bounds.append(bounds[-1] + length)
        cu = torch.tensor(bounds, dtype=torch.int32)
        op = GroupedQueryAttentionSlidingWindowVarlenFwdOp.__new__(
            GroupedQueryAttentionSlidingWindowVarlenFwdOp
        )
        # The window narrows which keys each query attends, which moves the flops and
        # leaves the traffic alone: the call still reads every packed tensor once.
        op._roofline_kwargs = {
            "q_shape": (total_q, heads, dim),
            "k_shape": (total_k, heads_kv, dim),
            "batch": batch,
            "heads": heads,
            "heads_kv": heads_kv,
            "dim": dim,
            "total_q": total_q,
            "total_k": total_k,
            "is_causal": True,
            "window_size_left": 256,
            "window_size_right": -1,
            "dtype": torch.float16,
            "cu_seqlens_q": cu,
            "cu_seqlens_kv": cu,
        }
        oracle = _ledger(
            "GroupedQueryAttentionSlidingWindowVarlenFwdOp",
            q=((total_q, heads, dim), torch.float16),
            k=((total_k, heads_kv, dim), torch.float16),
            v=((total_k, heads_kv, dim), torch.float16),
            cu_seqlens_q=((batch + 1,), torch.int32),
            cu_seqlens_k=((batch + 1,), torch.int32),
            o=((total_q, heads, dim), torch.float16),
        )
        assert op.eval_roofline()[1] == oracle

    def test_nsa_forward_counts_the_blocks_its_selection_kept(self):
        """How much this call reads follows `block_counts`, so the case runs the
        workload that builds it rather than inventing a selection of its own."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required to build the workload")
        from tileops.perf.formulas import nsa_fwd_varlen_roofline
        from workloads.attention.deepseek import NsaFwdWorkload

        batch, heads, head_kv, dim = 4, 16, 1, 64
        c_seq_len, block_size, selected = 8192, 32, 4
        workload = NsaFwdWorkload(
            batch=batch, heads=heads, c_seq_len=c_seq_len, dim=dim, is_causal=True,
            scale=0.1, block_size=block_size, groups=heads, selected_blocks=selected,
            dtype=torch.float16, accum_dtype=torch.float32, seq_lens=[c_seq_len // batch] * batch,
        )  # fmt: skip
        q, k, v, block_indices, block_counts, offsets, token_indices = workload.gen_inputs()[:7]

        # The blocks the kernel reads: for each token and KV head, the kept picks
        # whose block starts at or before that token. Counted here from the
        # tensors, not from the formula's own walk of them.
        kept = block_counts.reshape(-1).tolist()
        picks = block_indices.reshape(-1, selected).tolist()
        positions = token_indices[:, 1].tolist()
        tiles = sum(
            sum(1 for start in row[:n] if 0 <= start * block_size <= positions[i // head_kv])
            for i, (n, row) in enumerate(zip(kept, picks, strict=True))
        )
        gathered = tiles * block_size * dim

        bound = {
            f"{name}_shape": tuple(tensor.shape)
            for name, tensor in (
                ("q", q), ("k", k), ("v", v), ("block_indices", block_indices),
                ("block_counts", block_counts), ("offsets", offsets),
                ("token_indices", token_indices),
            )
        }  # fmt: skip
        bound.update(
            block_indices=block_indices, block_counts=block_counts, offsets=offsets,
            token_indices=token_indices, block_size=block_size, is_causal=True,
            dtype="float16",
        )  # fmt: skip
        oracle = _ledger(
            "NSAFwdVarlenOp",
            q=((c_seq_len, heads, dim), torch.float16),
            # k and v are read through the selection, not end to end
            k=((gathered,), torch.float16),
            v=((gathered,), torch.float16),
            block_indices=(tuple(block_indices.shape), torch.int32),
            block_counts=(tuple(block_counts.shape), torch.int32),
            offsets=(tuple(offsets.shape), torch.int32),
            token_indices=(tuple(token_indices.shape), torch.int32),
            o_slc=((c_seq_len, heads, dim), torch.float16),
        )
        assert nsa_fwd_varlen_roofline(bound)[1] == oracle

    def test_nsa_topk_does_not_charge_the_lse_it_recomputes(self):
        """`lse_in` is declared and passed, and the top-k kernel recomputes the lse
        and discards the argument. A declared input the algorithm does not read
        produces no traffic, and the contract does not
        say which inputs those are."""
        from tileops.perf.formulas import nsa_topk_varlen_roofline

        seq_num, c_seq_len, heads, head_kv, dim = 8, 8192, 32, 2, 128
        chunk_num, selected, block = 256, 16, 32
        lengths = [c_seq_len // seq_num] * seq_num
        bounds = [0]
        for length in lengths:
            bounds.append(bounds[-1] + length)
        offsets = torch.tensor(bounds, dtype=torch.int32)
        bound = {
            "q_shape": (c_seq_len, heads, dim),
            "k_cmp_shape": (chunk_num, head_kv, dim),
            "offsets_shape": (seq_num + 1,),
            "offsets": offsets,
            "bs": block,
            "selected_block_num": selected,
            "dtype": "float16",
        }
        oracle = _ledger(
            "NSATopkVarlenOp",
            q=((c_seq_len, heads, dim), torch.float16),
            k_cmp=((chunk_num, head_kv, dim), torch.float16),
            lse_in_unread=True,
            offsets=((seq_num + 1,), torch.int32),
            chunk_offsets=((seq_num + 1,), torch.int32),
            token_indices=((c_seq_len, 2), torch.int32),
            block_indices=((c_seq_len, head_kv, selected), torch.int32),
        )
        assert nsa_topk_varlen_roofline(bound)[1] == oracle

    def test_gqa_prefill_paged_reads_the_pages_the_block_table_selects(self):
        """The cache is one pool and the call touches the pages its block table
        names, so the recount prices that subset rather than the pool. The scales
        travel with every call and the kernel reads them only for fp8 pages."""
        from tileops.perf.formulas import gqa_prefill_paged_with_kv_cache_fwd_roofline

        batch, heads, heads_kv, dim = 8, 32, 8, 256
        q_lens = [1024] * batch
        cache_lens = [32768] * batch
        total_q, cached = sum(q_lens), sum(cache_lens)
        page_size, max_pages_per_req = 64, 528
        # The call indexes the block table as far as each request's pages reach;
        # the rest of the row is capacity it never reads.
        pages_named = sum(-(-(q + c) // page_size) for q, c in zip(q_lens, cache_lens, strict=True))
        bound = {
            "total_q": total_q,
            "batch": batch,
            "q_lens": q_lens,
            "cache_lens": cache_lens,
            "heads": heads,
            "heads_kv": heads_kv,
            "dim": dim,
            "page_size": page_size,
            "max_pages_per_req": max_pages_per_req,
            "max_seqlen_q": max(q_lens),
            "is_causal": True,
            "dtype": "float16",
        }
        new_kv = ((total_q, heads_kv, dim), torch.float16)
        oracle = _ledger(
            "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp",
            q=((total_q, heads, dim), torch.float16),
            k_new=new_kv,
            v_new=new_kv,
            # the cached tokens the block table points at, not the whole pool
            k_pages=((cached, heads_kv, dim), torch.float16),
            v_pages=((cached, heads_kv, dim), torch.float16),
            # the new tokens are appended into those same pages
            k_pages_write=new_kv,
            v_pages_write=new_kv,
            k_scale_unread=True,
            v_scale_unread=True,
            cu_seqlens_q=((batch + 1,), torch.int32),
            cache_seqlens=((batch,), torch.int32),
            block_table=((pages_named,), torch.int32),
            o=((total_q, heads, dim), torch.float16),
        )
        assert gqa_prefill_paged_with_kv_cache_fwd_roofline(bound)[1] == oracle

        # An fp8 pool stores one byte per element, and the kernel reads both
        # scales to dequantize what it loads and to quantize what it appends.
        fp8 = torch.float8_e4m3fn
        quantized = _ledger(
            "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp",
            q=((total_q, heads, dim), torch.float16),
            k_new=new_kv,
            v_new=new_kv,
            k_pages=((cached, heads_kv, dim), fp8),
            v_pages=((cached, heads_kv, dim), fp8),
            k_pages_write=((total_q, heads_kv, dim), fp8),
            v_pages_write=((total_q, heads_kv, dim), fp8),
            k_scale=((1,), torch.float32),
            v_scale=((1,), torch.float32),
            cu_seqlens_q=((batch + 1,), torch.int32),
            cache_seqlens=((batch,), torch.int32),
            block_table=((pages_named,), torch.int32),
            o=((total_q, heads, dim), torch.float16),
        )
        assert (
            gqa_prefill_paged_with_kv_cache_fwd_roofline(dict(bound, cache_dtype="float8_e4m3fn"))[
                1
            ]
            == quantized
        )

        # One token against an empty or single-page cache: each request names one
        # or two entries, most of the table names nothing, and the page count is
        # a ceiling of a length that does not divide by the page size.
        short_q = [1] * batch
        short_cache = [0, 64] * (batch // 2)
        short_named = sum(
            -(-(q + c) // page_size) for q, c in zip(short_q, short_cache, strict=True)
        )
        short = dict(
            bound,
            total_q=sum(short_q),
            q_lens=short_q,
            cache_lens=short_cache,
            max_seqlen_q=max(short_q),
        )
        short_new_kv = ((sum(short_q), heads_kv, dim), torch.float16)
        oracle_short = _ledger(
            "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp",
            q=((sum(short_q), heads, dim), torch.float16),
            k_new=short_new_kv,
            v_new=short_new_kv,
            k_pages=((sum(short_cache), heads_kv, dim), torch.float16),
            v_pages=((sum(short_cache), heads_kv, dim), torch.float16),
            k_pages_write=short_new_kv,
            v_pages_write=short_new_kv,
            k_scale_unread=True,
            v_scale_unread=True,
            cu_seqlens_q=((batch + 1,), torch.int32),
            cache_seqlens=((batch,), torch.int32),
            block_table=((short_named,), torch.int32),
            o=((sum(short_q), heads, dim), torch.float16),
        )
        assert gqa_prefill_paged_with_kv_cache_fwd_roofline(short)[1] == oracle_short

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
#
# Two kinds sit here. For most, the binder cannot build the call at all. For
# the two GQA entries it builds one and counts something that is not this call's
# traffic, because the op translates the call before the formula sees it. Those
# two are the ones where a formula
# defect would look like the stated reason, so their cases are what check them
# and `_ledger` is what checks the cases.
HAND_WRITTEN = {
    "FusedMoEExpertsFwdOp": "the routed weight reads follow the values in `topk_ids`",
    "FusedMoeFwdOp": "the routed weight reads follow the routing its experts stage receives",
    "FusedMoeSharedExpertFwdOp": "the routed weight reads follow the routing its experts stage receives",
    "GemmFp8FwdOp": "the scale tensors' extents follow the scaling mode, not the dims",
    "GemmW4A16FwdOp": "the packed weight and its group metadata have a quantized layout",
    "GroupedQueryAttentionDenseFwdOp": "which optional tensors the call passed decides the traffic",
    "GroupedQueryAttentionPrefillVarlenFwdOp": "the per-request lengths the call packed decide the traffic",
    "GroupedQueryAttentionSlidingWindowVarlenFwdOp": "the per-request lengths the call packed decide the traffic",
    "GroupedGemmFwdOp": "`batch_padded_offsets` is passed and no kernel indexes it",
    "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp": "it reads the pages its block table names, not the pool",
    "NSAFwdVarlenOp": "how much it reads follows the values in `block_counts`",
    "NSATopkVarlenOp": "`lse_in` is passed and the kernel recomputes the lse instead of reading it",
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
        """Level two and three are for ops the manifest cannot recount, not a queue."""
        promotable = sorted(
            name
            for name in {**HAND_WRITTEN, **NOT_RECOUNTABLE}
            if _binder_builds(name) and _binder_agrees(name)
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


class TestValueDeterminedTraffic:
    """Ops whose `bytes` follows an input's values must build that input the same
    way every time. The global stream does not give
    that: a draw added anywhere earlier moves every draw after it."""

    def test_the_nsa_forward_workload_prices_the_same_call_twice(self):
        pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA required to build the workload")
        from tileops.perf.formulas import nsa_fwd_varlen_roofline
        from workloads.attention.deepseek import NsaFwdWorkload

        def priced(extra_draw: bool, *, builds: int = 1) -> int:
            torch.manual_seed(1235)
            if extra_draw:
                torch.randn(7, device="cuda")
            workload = NsaFwdWorkload(
                batch=4, heads=16, c_seq_len=8192, dim=64, is_causal=True, scale=0.1,
                block_size=32, groups=16, selected_blocks=16, dtype=torch.float16,
                accum_dtype=torch.float32, seq_lens=[2048] * 4,
            )  # fmt: skip
            for _ in range(builds - 1):
                workload.gen_inputs()
            q, k, v, block_indices, block_counts, offsets, token_indices = workload.gen_inputs()[:7]
            bound = {
                f"{name}_shape": tuple(tensor.shape)
                for name, tensor in (
                    ("q", q), ("k", k), ("v", v), ("block_indices", block_indices),
                    ("block_counts", block_counts), ("offsets", offsets),
                    ("token_indices", token_indices),
                )
            }  # fmt: skip
            bound.update(
                block_indices=block_indices,
                block_counts=block_counts,
                offsets=offsets,
                token_indices=token_indices,
                block_size=32,
                is_causal=True,
                dtype="float16",
            )
            return nsa_fwd_varlen_roofline(bound)[1]

        # A draw added upstream must not move it, and neither must building the
        # same workload a second time.
        assert priced(False) == priced(True)
        assert priced(False) == priced(False, builds=3)
