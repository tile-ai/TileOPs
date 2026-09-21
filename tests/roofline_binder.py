"""Build a bytes-oracle case for an op from its manifest entry alone.

The binder reads ``signature`` and one ``workloads`` row, creates every declared
input on the meta device, and asks the op for its ``(flops, bytes)``. In parallel
it counts the traffic the contract implies -- one read per distinct input storage
the call binds, one write per public output, both for a ``mutated`` input -- and
the caller requires the two to be equal.

Shared with the formula, and nothing beyond it: the minimum-traffic definition,
the op's ``_infer_output_shapes`` for the output extents, and ``_output_dtype``
for an output's dtype. The formula reaches those last two through codegen, so a
defect in either moves both sides together. Shapes otherwise come from the
workload row and the signature's ``shape`` strings; the ``roofline`` block is
never read.

One rule the contract does not settle, which the binder assumes: a ``mutated``
input is written in addition to the outputs, unless it is itself an output name
or the signature declares an ``inplace`` param, where that write is the
output's.

An op the binder cannot drive raises `NotBindableError` naming what the contract
did not settle, which is the entry condition for the other two coverage levels.
"""

from __future__ import annotations

import importlib
import inspect
from math import prod
from typing import Any

import torch

from tileops.manifest import load_manifest, load_workloads
from tileops.manifest.dtype_rules import parse_tokens, promote_int_to_float_ref, same_as_ref
from tileops.ops._output_dtype import output_dtype

__all__ = ["NotBindableError", "bind_case", "manifest_cases", "op_class"]


class NotBindableError(Exception):
    """The manifest contract does not settle what this call reads and writes."""


def op_class(op_name: str, entry: dict) -> type:
    """The op class a manifest entry names."""
    module = entry["source"]["op"].replace("/", ".").removesuffix(".py")
    return getattr(importlib.import_module(module), op_name)


def _torch_dtype(name: str) -> torch.dtype | None:
    resolved = getattr(torch, name, None)
    return resolved if isinstance(resolved, torch.dtype) else None


def _resolve_dtype(expr: Any, call_dtype: torch.dtype, inputs: dict, depth: int = 0) -> torch.dtype:
    """The dtype a signature expression resolves to for a call at *call_dtype*.

    A union is the call's own dtype -- the workload row picks from it. ``same_as``
    and ``promote_int_to_float`` follow the input they name.
    """
    if depth > 4 or not isinstance(expr, str):
        return call_dtype
    tokens = parse_tokens(expr)
    if len(tokens) != 1:
        # A union the row's dtype is a member of is that dtype; a union it is not --
        # an index tensor declared ``int32 | int64`` next to a float call -- takes the
        # declaration's first member.
        resolved = [_torch_dtype(token) for token in tokens]
        if call_dtype in resolved:
            return call_dtype
        return next((d for d in resolved if d is not None), call_dtype)
    referenced = same_as_ref(tokens[0]) or promote_int_to_float_ref(tokens[0])
    if referenced:
        return _resolve_dtype(
            (inputs.get(referenced) or {}).get("dtype"), call_dtype, inputs, depth + 1
        )
    return _torch_dtype(tokens[0]) or call_dtype


def _names_a_dtype(declared: Any) -> bool:
    """Whether a param's declared type is a set of dtype names."""
    if not isinstance(declared, str):
        return False
    # ``int``, ``float``, ``bool`` and ``complex`` name a torch dtype and a Python
    # type both; a param declared with them is the Python one.
    ambiguous = {"int", "float", "bool", "complex", "None"}
    tokens = [token.strip() for token in declared.split("|")]
    named = [token for token in tokens if token not in ambiguous]
    return bool(named) and all(_torch_dtype(token) is not None for token in named)


def _param_value(spec: dict, row: dict, name: str) -> Any:
    """A param's value for this row, with a dtype-typed one resolved to a torch dtype.

    A dtype-typed param the row does not name follows the row's own dtype, which is
    what a benchmark passes: ``DaCumsumFwdOp(out_dtype=dtype)`` writes ``dt_out`` in
    the dtype the case runs at, not in the declaration's fallback.
    """
    value = row.get(name, (spec or {}).get("default"))
    if name not in row and _names_a_dtype((spec or {}).get("type")):
        called_with = (row.get("dtypes") or [None])[0]
        if _torch_dtype(called_with or "") is not None:
            return _torch_dtype(called_with)
    if isinstance(value, str) and _torch_dtype(value) is not None:
        return _torch_dtype(value)
    if isinstance(value, list):
        return tuple(value)
    return value


def _instance(cls: type, params: dict) -> Any:
    """An op instance carrying *params*, built through ``__init__`` where it accepts them.

    The constructor is what normalizes a scalar ``kernel_size`` or ``stride`` into the
    per-axis form the shape inference reads, so going through it keeps that logic in
    the op. An op whose constructor wants more than its manifest params falls back to
    a bare instance with the params set directly.
    """
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
        signature = None
    if signature is not None:
        accepted = set(signature.parameters) - {"self"}
        required = {
            name
            for name, p in signature.parameters.items()
            if name != "self"
            and p.default is inspect.Parameter.empty
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        }
        if set(params) <= accepted and required <= set(params):
            try:
                return cls(**params)
            except Exception:  # the op rejects this row; fall through to a bare instance
                pass
    op = cls.__new__(cls)
    for name, value in params.items():
        setattr(op, name, value)
    return op


# Shapes a workload row implies but does not spell out, per op. A row states the
# dims a benchmark needs; each entry restates them as the shapes the signature
# declares, and states nothing about what those tensors cost.
def _packed_bounds(lengths: "list[int]") -> torch.Tensor:
    """The cumulative bounds a packed batch carries, from the row's own lengths."""
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + int(length))
    return torch.tensor(bounds, dtype=torch.int32)


def _kv_pair(row: dict) -> dict:
    kv = tuple(row["kv_shape"])
    return {"k_shape": kv, "v_shape": kv}


def _attention_bwd(row: dict) -> dict:
    q = tuple(row["q_shape"])
    batch, seq_len, heads, _ = q
    return {**_kv_pair(row), "o_shape": q, "do_shape": q, "lse_shape": (batch, heads, seq_len)}


def _elementwise_peers(*names: str):
    def supplement(row: dict) -> dict:
        shape = tuple(row["input_shape"])
        return {f"{name}_shape": shape for name in names}

    return supplement


def _normalized_pair(*names: str):
    def supplement(row: dict) -> dict:
        shape = tuple(row.get("normalized_shape") or (row["x_shape"][-1],))
        return {f"{name}_shape": shape for name in names}

    return supplement


def _elementwise_peers_of(source: str, *names: str):
    def supplement(row: dict) -> dict:
        shape = tuple(row[f"{source}_shape"])
        return {f"{name}_shape": shape for name in names}

    return supplement


def _channel_vectors(*names: str):
    def supplement(row: dict) -> dict:
        channels = (tuple(row["x_shape"])[-1],)
        return {f"{name}_shape": channels for name in names}

    return supplement


_ROW_SUPPLEMENT = {
    # Attention: the row names one KV shape for both k and v, and the backward
    # pass reads the forward's output and its log-sum-exp alongside them.
    "MultiHeadAttentionBwdOp": _attention_bwd,
    "GroupedQueryAttentionBwdOp": _attention_bwd,
    "GroupedQueryAttentionDenseFwdOp": _kv_pair,
    "MultiHeadLatentAttentionDecodeWithKVCacheFwdOp": lambda row: {
        "k_shape": tuple(row["kv_shape"])
    },
    # Dense GEMM and BMM: the row gives the logical dims.
    "GemmFwdOp": lambda row: {
        "a_shape": (row["k"], row["m"]) if row.get("trans_a") else (row["m"], row["k"]),
        "b_shape": (row["n"], row["k"]) if row.get("trans_b") else (row["k"], row["n"]),
    },
    # Normalization: the affine pair spans the normalized axes.
    "LayerNormFwdOp": _normalized_pair("weight", "bias"),
    "RMSNormFwdOp": _normalized_pair("weight"),
    "FusedAddLayerNormFwdOp": lambda row: {
        "residual_shape": tuple(row["x_shape"]),
        "weight_shape": (row["x_shape"][-1],),
        "bias_shape": (row["x_shape"][-1],),
    },
    "FusedAddRMSNormFwdOp": lambda row: {
        "residual_shape": tuple(row["x_shape"]),
        "weight_shape": (row["x_shape"][-1],),
    },
    "AdaLayerNormFwdOp": _elementwise_peers_of("x", "scale", "shift"),
    "AdaLayerNormZeroFwdOp": _elementwise_peers_of("x", "scale", "shift", "gate"),
    "BatchNormFwdOp": lambda row: {
        f"{name}_shape": (row["x_shape"][1],)
        for name in ("running_mean", "running_var", "weight", "bias")
    },
    "BatchNormBwdOp": lambda row: {
        "grad_out_shape": tuple(row["x_shape"]),
        **{f"{name}_shape": (row["x_shape"][1],) for name in ("weight", "mean", "rstd")},
    },
    # Broadcast-free elementwise rows: every operand has the output's shape.
    "WhereFwdOp": _elementwise_peers("condition", "other"),
    "MaskedFillScalarFwdOp": _elementwise_peers("mask"),
    "MaskedFillFwdOp": _elementwise_peers("mask"),
    "LerpTensorFwdOp": _elementwise_peers("end", "weight"),
    # The broadcast-binary base names its second operand ``other`` whatever the
    # signature calls it, and that is the name its formula reads.
    "LerpFwdOp": lambda row: {"other_shape": tuple(row["end_shape"])},
    "PowFwdOp": lambda row: {"other_shape": tuple(row["exponent_shape"])},
    # RoPE rows give the extents; the layout says how they lay out.
    **{
        name: (
            lambda row: {
                "x_shape": (
                    (row["batch"], row["num_heads"], row["seq_len"], row["head_dim"])
                    if row.get("layout") == "2d"
                    else (row["seq_len"], row["head_dim"])
                )
            }
        )
        for name in (
            "RopeNeoxFwdOp",
            "RopeNonNeoxFwdOp",
            "RopeLlama31FwdOp",
            "RopeYarnFwdOp",
            "RopeLongRopeFwdOp",
        )
    },
    # Dims a func-mode formula reads off the instance, named as the row names them.
    "BmmFwdOp": lambda row: {
        "batch": row["b"],
        "a_shape": (row["b"], row["m"], row["k"]),
        "b_shape": (row["b"], row["k"], row["n"]),
    },
    # Conv rows give the kernel extents and the output channel count; the weight's
    # input-channel extent is the input's divided by the groups.
    **{
        name: (
            lambda row: {
                "weight_shape": (
                    row["C_out"],
                    row["input_shape"][1] // row.get("groups", 1),
                    *(row[axis] for axis in ("kD", "kH", "kW") if axis in row),
                )
            }
        )
        for name in ("Conv1dFwdOp", "Conv2dFwdOp", "Conv3dFwdOp")
    },
    # NSA compression and top-k read the request bounds out of `offsets`. The row
    # states the lengths, so the tensor is their running sum, not an invention.
    "NSACmpFwdVarlenOp": lambda row: {
        "q_shape": (row["c_seq_len"], row["heads"], row["dim_k"]),
        "k_cmp_shape": (row["chunk_num"], row["head_kv"], row["dim_k"]),
        "v_cmp_shape": (row["chunk_num"], row["head_kv"], row["dim_v"]),
        "chunk_offsets_shape": (row["seq_num"] + 1,),
        "token_indices_shape": (row["c_seq_len"], 2),
        "offsets": _packed_bounds(row["seq_lens"]),
        "offsets_shape": (row["seq_num"] + 1,),
    },
    "NSATopkVarlenOp": lambda row: {
        "q_shape": (row["c_seq_len"], row["heads"], row["dim"]),
        "k_cmp_shape": (row["chunk_num"], row["head_kv"], row["dim"]),
        "lse_in_shape": (row["c_seq_len"], row["heads"]),
        "chunk_offsets_shape": (row["seq_num"] + 1,),
        "token_indices_shape": (row["c_seq_len"], 2),
        "offsets": _packed_bounds(row["seq_lens"]),
        "offsets_shape": (row["seq_num"] + 1,),
    },
    # Per-tensor scales: two fp32 scalars, which is what the formula's trailing 8
    # bytes are.
    "BmmFp8FwdOp": lambda row: {
        "batch": row["b"],
        "a_shape": (row["b"], row["m"], row["k"]),
        "b_shape": (row["b"], row["k"], row["n"]),
        "scale_a_shape": (),
        "scale_b_shape": (),
    },
    # The row gives the packed row total, the group count and the two inner dims;
    # the three int32 metadata tensors hold one entry per group.
    "GroupedGemmFwdOp": lambda row: {
        # transpose_a is the form whose output keeps the group axis: the packed rows
        # are the contraction, and b is two-dimensional.
        "a_shape": (
            (row["batch_sum"], row["n"]) if row.get("transpose_a") else (row["batch_sum"], row["k"])
        ),
        "b_shape": (
            (
                (row["k"], row["batch_sum"])
                if row.get("transpose_b")
                else (row["batch_sum"], row["k"])
            )
            if row.get("transpose_a")
            else (
                (row["batch_count"], row["n"], row["k"])
                if row.get("transpose_b")
                else (row["batch_count"], row["k"], row["n"])
            )
        ),
        "batch_sizes_shape": (row["batch_count"],),
        "batch_offsets_shape": (row["batch_count"],),
        "batch_padded_offsets_shape": (row["batch_count"],),
    },
    # A physical-psum layout's metadata holds one segment end per expert, and the
    # expert count is the weight tensor's leading extent.
    "MoeExpertMLPFwdOp": lambda row: {
        "layout_metadata_shape": (row["w_gate_up_shape"][0],),
        "input_shapes": [
            tuple(row["expert_input_shape"]),
            tuple(row["w_gate_up_shape"]),
            tuple(row["w_down_shape"]),
            (row["w_gate_up_shape"][0],),
        ],
    },
    "MoeGroupedGemmFwdOp": lambda row: {
        "layout_metadata_shape": (row["b_shape"][0],),
        "input_shapes": [
            tuple(row["a_shape"]),
            tuple(row["b_shape"]),
            (row["b_shape"][0],),
        ],
    },
    # Paged decode: the cache is one page pool, and the call carries a length per
    # request plus the pages that request's tokens sit in.
    "MultiHeadAttentionDecodePagedWithKVCacheFwdOp": lambda row: {
        **_kv_pair(row),
        "real_seqlen_kv_shape": (row["q_shape"][0],),
        "block_table_shape": (
            row["q_shape"][0],
            max(1, -(-row["kv_shape"][0] // row["page_size"])),
        ),
    },
    "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp": lambda row: {
        **_kv_pair(row),
        "real_seqlen_kv_shape": (row["q_shape"][0],),
        "block_table_shape": (
            row["q_shape"][0],
            max(1, -(-row["kv_shape"][0] // row["page_size"])),
        ),
    },
    "MoePermuteAlignFwdOp": lambda row: {"topk_ids_shape": (row["total_tokens"], row["top_k"])},
    "MeanPoolingFwdOp": lambda row: {
        "x_shape": (row["batch"], row["seq_len"], row["heads"], row["dim"])
    },
    # Packed-batch attention: the row gives the totals and the request count, and
    # the cumulative-length tensors hold one bound per request plus the zero.
    "GroupedQueryAttentionPrefillVarlenFwdOp": lambda row: {
        "q_shape": (row["total_q"], row["heads"], row["dim"]),
        "k_shape": (row["total_kv"], row["heads_kv"], row["dim"]),
        "v_shape": (row["total_kv"], row["heads_kv"], row["dim"]),
        "cu_seqlens_q_shape": (row["batch"] + 1,),
        "cu_seqlens_kv_shape": (row["batch"] + 1,),
    },
    "GroupedQueryAttentionSlidingWindowVarlenFwdOp": lambda row: {
        "q_shape": (row["total_q"], row["heads"], row["dim"]),
        "k_shape": (row["total_k"], row["heads_kv"], row["dim"]),
        "v_shape": (row["total_k"], row["heads_kv"], row["dim"]),
        "cu_seqlens_q_shape": (row["batch"] + 1,),
        "cu_seqlens_k_shape": (row["batch"] + 1,),
    },
    "DropoutFwdOp": lambda row: {"N_total": prod(row["input_shape"])},
    "FFTC2CFwdOp": lambda row: {"n": row["input_shape"][-1]},
    "FusedTopKOp": lambda row: {"gating_output_shape": (row["num_tokens"], row["num_experts"])},
    "DaCumsumFwdOp": lambda row: {
        "batch": row["dt_shape"][0],
        "seq_len": row["dt_shape"][1],
        "n_heads": row["dt_shape"][2],
    },
    "SSDChunkStateFwdOp": lambda row: {
        "batch": row["x_shape"][0],
        "seq_len": row["x_shape"][1],
        "n_heads": row["x_shape"][2],
        "d_head": row["x_shape"][3],
        "d_state": row["Bmat_shape"][3],
        "n_groups": row["Bmat_shape"][2],
        "chunk_len": row["dt_shape"][3],
        "num_chunks": row["dt_shape"][2],
    },
    "SSDChunkScanFwdOp": lambda row: {
        "batch": row["x_shape"][0],
        "seq_len": row["x_shape"][1],
        "n_heads": row["x_shape"][2],
        "d_head": row["x_shape"][3],
        "d_state": row["C_shape"][3],
        "n_groups": row["C_shape"][2],
        "chunk_len": row["dt_shape"][3],
        "num_chunks": row["dt_shape"][2],
    },
    "SSDStatePassingFwdOp": lambda row: {
        "batch": row["states_shape"][0],
        "num_chunks": row["states_shape"][1],
        "n_heads": row["dA_chunk_cumsum_shape"][1],
        "d_state": row["states_shape"][3],
    },
    "Mamba2FwdOp": lambda row: {
        "batch": row["x_shape"][0],
        "seqlen": row["x_shape"][1],
        "n_heads": row["x_shape"][2],
        "d_head": row["x_shape"][3],
        "d_state": row["B_shape"][3],
        "n_groups": row["B_shape"][2],
        "num_chunks": row["x_shape"][1] // 256,
    },
    "SSDDecodeFwdOp": lambda row: {
        "batch": row["x_shape"][0],
        "n_heads": row["x_shape"][1],
        "d_head": row["x_shape"][2],
        "d_state": row["state_shape"][3],
        "n_groups": row["B_in_shape"][1],
    },
}


def _dim_symbols(declaration: str) -> list[str]:
    """The per-axis expressions a signature ``shape`` string states."""
    body = declaration.strip()
    if not (body.startswith("[") and body.endswith("]")):
        return []
    return [part.strip() for part in body[1:-1].split(",") if part.strip()]


def _declared_shapes(inputs: dict, row: dict, params: dict) -> dict:
    """Shapes the signature's own ``shape`` strings give, read against the row.

    A declaration names its axes -- ``"[M, seq_len, d]"``, ``"[C_out, C_in_g, kH, kW]"``
    -- and a row states those dims by name. Where a tensor's shape is in the row, its
    declaration binds the names it uses; where it is not, the declaration is evaluated
    against everything bound so far. This reads the signature, not the roofline.
    """
    scope: dict[str, Any] = {
        key: value
        for key, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    for name, spec in params.items():
        value = row.get(name, (spec or {}).get("default"))
        if isinstance(value, int) and not isinstance(value, bool):
            scope.setdefault(name, value)
    for name, attrs in inputs.items():
        shape = row.get(f"{name}_shape")
        axes = _dim_symbols(((attrs or {}).get("shape") or ""))
        if shape is None or len(axes) != len(shape):
            continue
        for axis, extent in zip(axes, shape, strict=True):
            if axis.isidentifier():
                scope.setdefault(axis, extent)
    derived: dict[str, tuple[int, ...]] = {}
    for name, attrs in inputs.items():
        if f"{name}_shape" in row:
            continue
        if (attrs or {}).get("optional"):
            # A row omits an optional input because the call does not pass it.
            # Deriving its shape would price a tensor the call never binds.
            continue
        axes = _dim_symbols(((attrs or {}).get("shape") or ""))
        if not axes:
            continue
        try:
            derived[f"{name}_shape"] = tuple(
                int(eval(axis, {"__builtins__": {}}, scope))
                for axis in axes  # noqa: S307
            )
        except Exception:
            continue
    return derived


def bind_case(
    op_name: str, entry: dict, row: dict, call_dtype: torch.dtype
) -> tuple[Any, int, int]:
    """Return ``(bound op, oracle bytes, oracle read bytes)`` for one workload row.

    The read side is separate because the NCU audit judges that half alone, and
    an op derives it by subtracting the write side the signature settles.

    Raises:
        NotBindableError: The row does not give a required input's shape, the op's shape
            inference does not take exactly the declared inputs, or it refuses the row.
    """
    signature = entry.get("signature") or {}
    inputs = signature.get("inputs") or {}
    outputs = signature.get("outputs") or {}
    params = signature.get("params") or {}

    row = {**_declared_shapes(inputs, row, params), **row}
    supplement = _ROW_SUPPLEMENT.get(op_name)
    if supplement is not None:
        try:
            row = {**supplement(row), **row}
        except (KeyError, IndexError, TypeError) as exc:
            raise NotBindableError(f"the row supplement does not fit this row: {exc}") from exc

    order = list(inputs)
    shapes: list[tuple[int, ...] | None] = []
    for name in order:
        shape = row.get(f"{name}_shape")
        if shape is None:
            if not (inputs[name] or {}).get("optional"):
                raise NotBindableError(f"the workload row does not give {name}_shape")
            shapes.append(None)
            continue
        shapes.append(tuple(shape))

    cls = op_class(op_name, entry)
    infer = getattr(cls, "_infer_output_shapes", None)
    if infer is None:
        raise NotBindableError("the op declares no _infer_output_shapes")
    taken = [p for p in inspect.signature(infer).parameters if p != "self"]
    if taken != [f"{name}_shape" for name in order]:
        raise NotBindableError(f"_infer_output_shapes takes {taken}, not the declared inputs")

    op = _instance(cls, {n: _param_value(spec, row, n) for n, spec in params.items()})
    # A row also carries the dims a func-mode formula reads off the instance --
    # ``m``, ``batch``, ``kv_shape``. Setting them under their own names is what
    # the op's own forward does; nothing here reads the roofline block.
    for key, value in row.items():
        if key in ("dtypes", "label") or getattr(op, key, None) is not None:
            continue
        setattr(op, key, tuple(value) if isinstance(value, list) else value)
    for name, shape in zip(order, shapes, strict=True):
        dtype = _resolve_dtype((inputs[name] or {}).get("dtype"), call_dtype, inputs)
        # A bulk tensor is meta; metadata a formula reads the values of is
        # built for real. A supplement hands the built one back under the input's
        # own name; its values restate what the row already says.
        built = row.get(name)
        if isinstance(built, torch.Tensor):
            setattr(op, name, built)
        else:
            setattr(
                op, name, None if shape is None else torch.empty(shape, dtype=dtype, device="meta")
            )
        setattr(op, f"{name}_shape", shape)
        # An op whose inputs may differ in dtype reads them one per tensor.
        if getattr(op, f"{name}_dtype", None) is None:
            setattr(op, f"{name}_dtype", None if shape is None else dtype)
    op.dtype = call_dtype

    try:
        out_shapes = op._infer_output_shapes(*shapes)
    except Exception as exc:
        raise NotBindableError(f"_infer_output_shapes rejected the row: {exc}") from exc

    # One read per input the call binds, one write per public output. A ``mutated``
    # input is written too, unless that write is already an output's: an op with an
    # ``inplace`` param may write into the input it read, and an input that is also
    # an output name is that output.
    reads = 0
    writes = 0
    has_inplace = "inplace" in params
    for name, shape in zip(order, shapes, strict=True):
        if shape is None:
            continue
        dtype = _resolve_dtype((inputs[name] or {}).get("dtype"), call_dtype, inputs)
        nbytes = prod(shape) * torch.empty((), dtype=dtype).element_size()
        reads += nbytes
        if (inputs[name] or {}).get("mutated") and name not in outputs and not has_inplace:
            writes += nbytes
    for name, shape in out_shapes.items():
        # Through the op, so an output the entry marks ``caller_stated`` follows the
        # dtype this call asked for rather than the declaration's fallback.
        dtype = output_dtype(op, name, call_dtype)
        writes += prod(shape) * torch.empty((), dtype=dtype).element_size()
    return op, reads + writes, reads


def manifest_cases(op_name: str):
    """Yield ``(label, dtype, op, oracle bytes, oracle read bytes)`` per row and dtype."""
    entry = load_manifest()[op_name]
    rows = load_workloads(op_name)
    if not rows:
        raise NotBindableError("the entry declares no workloads")
    for row in rows:
        for dtype_name in row.get("dtypes") or ["float16"]:
            call_dtype = _torch_dtype(dtype_name) or torch.float16
            op, oracle, reads = bind_case(op_name, entry, row, call_dtype)
            yield row.get("label", "workload"), dtype_name, op, oracle, reads
