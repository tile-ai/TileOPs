"""Every implemented op, served by a target for another kind of device.

What a backend is promised, checked on the whole manifest rather than on a sample: its
builder is described with the op's ``forward`` inputs, the kernel it returns is called
with exactly those tensors, and nothing on the way asks a CUDA device anything. A new op
joins by having a manifest entry; one the workloads cannot build is listed in ``_CASES``.
"""

import math

import pytest
import torch

from tests import roofline_binder as rb
from tileops.backend import TensorSpec, registry
from tileops.manifest import forward_signature, load_adts, load_manifest, load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.signature import is_legacy
from tileops.manifest.workload import instantiate
from tileops.ops._output_dtype import output_dtype

pytestmark = pytest.mark.smoke

F16, BF16, F32, I32, U8 = torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.uint8
FP8 = torch.float8_e4m3fn


def _t(*shape: int, dtype: torch.dtype = F16) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype)


def _fused_moe(**extra):
    return dict(num_tokens=8, num_experts=4, top_k=2, hidden_size=64, ffn_size=128, **extra)


# Ops whose construction arguments or input shapes no workload row states. Each builds the
# op and the positional ``forward`` arguments; only dtypes and presence matter here.
_CASES = {
    "BmmFp8FwdOp": lambda c: (
        c(),
        (_t(2, 32, 32, dtype=FP8), _t(2, 32, 32, dtype=FP8), _t(dtype=F32), _t(dtype=F32)),
    ),
    "DeepSeekSparseAttentionDecodeWithKVCacheFwdOp": lambda c: (
        c(1, 16, 2, 8, 512, 64, 4, 1, 1, 0),
        (_t(1, 2, 16, 576), _t(1, 8, 1, 576), _t(1, 2, 1, 4, dtype=I32)),
    ),
    "GroupedQueryAttentionBwdOp": lambda c: (
        c(1, 4, 2, 16, 64),
        (
            _t(1, 16, 4, 64),
            *[_t(1, 16, 2, 64)] * 2,
            *[_t(1, 16, 4, 64)] * 2,
            _t(1, 4, 16, dtype=F32),
        ),
    ),
    "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp": lambda c: (
        c(2, 4, 2, 64, 64, 16),
        (_t(2, 4, 64), *[_t(64, 2, 64)] * 2, _t(2, dtype=I32), _t(2, 4, dtype=I32)),
    ),
    "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp": lambda c: (
        c(1, 4, 2, 4, 16, 64, 8),
        (
            _t(8, 4, 64),
            *[_t(8, 2, 64)] * 2,
            *[_t(64, 2, 64)] * 2,
            *[_t(1, dtype=F32)] * 2,
            _t(2, dtype=I32),
            _t(1, dtype=I32),
            _t(1, 4, dtype=I32),
        ),
    ),
    "GroupedQueryAttentionSlidingWindowVarlenFwdOp": lambda c: (
        c(2, 4, 2, 64, 8),
        (_t(16, 4, 64), *[_t(16, 2, 64)] * 2, *[_t(3, dtype=I32)] * 2),
    ),
    "MultiHeadAttentionBwdOp": lambda c: (
        c(1, 4, 16, 64),
        (*[_t(1, 16, 4, 64)] * 5, _t(1, 4, 16, dtype=F32)),
    ),
    "MultiHeadAttentionDecodePagedWithKVCacheFwdOp": lambda c: (
        c(1, 4, 1, 64, 64, 16),
        (_t(1, 1, 4, 64), *[_t(64, 4, 64)] * 2, _t(1, dtype=I32), _t(1, 4, dtype=I32)),
    ),
    "MultiHeadLatentAttentionDecodeWithKVCacheFwdOp": lambda c: (
        c(2, 4, 1, 64, 64, 32),
        (_t(2, 4, 64), _t(2, 4, 32), _t(2, 64, 1, 64), _t(2, 64, 1, 32)),
    ),
    "NSAFwdVarlenOp": lambda c: (
        c(True, 0.1, 32, F32),
        (
            _t(64, 16, 64),
            *[_t(64, 1, 64)] * 2,
            _t(64, 1, 1, dtype=I32),
            _t(64, 1, dtype=I32),
            _t(2, dtype=I32),
            _t(64, 2, dtype=I32),
        ),
    ),
    "GemmFp8FwdOp": lambda c: (
        c(),
        (_t(16, 32, dtype=FP8), _t(16, 32, dtype=FP8), _t(1, 1, dtype=F32), _t(1, 1, dtype=F32)),
    ),
    "GemmW4A16FwdOp": lambda c: (
        c(),
        (_t(16, 128), _t(16, 64, dtype=U8), _t(16, 1), _t(16, 1, dtype=U8)),
    ),
    "FusedMoEExpertsFwdOp": lambda c: (
        c(**_fused_moe()),
        (
            *[_t(8, 64)] * 2,
            _t(4, 256, 64),
            _t(4, 64, 128),
            _t(8, 2, dtype=F32),
            _t(8, 2, dtype=I32),
            *[_t(4096)] * 2,
        ),
    ),
    "IndexedExpertMLPFwdOp": lambda c: (
        c(1, 4, 2, 128, 256),
        (
            *[_t(1, 128)] * 2,
            _t(4, 512, 128),
            _t(4, 128, 256),
            _t(1, 2, dtype=F32),
            _t(1, 2, dtype=I32),
            _t(1 * 2 * 256),
            _t(1 * 2 * 128),
        ),
    ),
    "FusedMoeFwdOp": lambda c: (
        c(**_fused_moe()),
        (_t(8, 64), _t(8, 4, dtype=F32), _t(4, 256, 64), _t(4, 64, 128)),
    ),
    "FusedMoeSharedExpertFwdOp": lambda c: (
        c(**_fused_moe(shared_ffn_size=128)),
        (
            _t(8, 64),
            _t(8, 4, dtype=F32),
            _t(4, 256, 64),
            _t(4, 64, 128),
            None,
            _t(256, 64),
            _t(64, 128),
        ),
    ),
}


def _from_workload(cls: type, name: str, entry: dict) -> tuple:
    """The op and its ``forward`` arguments, from the smallest workload row that states them."""
    signature = forward_signature(entry)
    inputs, params = signature.get("inputs") or {}, signature.get("params") or {}
    best = None
    for row in load_workloads(name) or [{}]:
        dtype = rb._torch_dtype((row.get("dtypes") or ["float16"])[0]) or F16
        row = {**rb._declared_shapes(inputs, row, params), **row}
        supplement = rb._ROW_SUPPLEMENT.get(name)
        if supplement is not None:
            row = {**supplement(row), **row}
        shapes = [row.get(f"{n}_shape") for n in inputs]
        if any(
            s is None and not (inputs[n] or {}).get("optional")
            for n, s in zip(inputs, shapes, strict=True)
        ):
            continue
        size = sum(math.prod(s) for s in shapes if s)
        if best is None or size < best[0]:
            best = (size, row, dtype, shapes)
    assert best is not None or not inputs, f"no workload row of {name} states every input"
    _, row, dtype, shapes = best if best else (0, {}, F16, [])
    kwargs = {n: rb._param_value(spec, row, n) for n, spec in params.items()}
    op = cls(**{k: v for k, v in kwargs.items() if v is not None})
    args = []
    for n, shape in zip(inputs, shapes, strict=True):
        built = row.get(n)
        if isinstance(built, torch.Tensor):
            args.append(built.cpu())
        elif shape is None:
            args.append(None)
        else:
            args.append(
                _t(*shape, dtype=rb._resolve_dtype((inputs[n] or {}).get("dtype"), dtype, inputs))
            )
    while args and args[-1] is None:
        args.pop()
    return op, tuple(args)


def _from_call(cls: type, name: str, entry: dict) -> tuple:
    """A converted op, its ``forward`` arguments and the outputs its call declares, from the
    manifest call with the smallest inputs."""
    plan = entry_plan(name, entry, load_adts(), resolve=False)
    calls = [
        instantiate(plan, row, case)
        for row in entry["workloads"]
        for case in row.get("dtype_cases") or [{}]
    ]
    call = min(calls, key=lambda c: sum(math.prod(c.tensors[t][0]) for t in c.tensors))
    tensors = {
        n: None if spec is None else torch.empty(spec.shape, dtype=getattr(torch, spec.dtype))
        for n, spec in call.specs.items()
    }
    args = [tensors[n] for n in plan.sig.inputs]
    while args and args[-1] is None:
        args.pop()
    outputs = [tensors[o] for o in plan.sig.outputs]
    return cls(**call.arguments(tensors)), tuple(args), outputs


def _implemented() -> list[str]:
    return sorted(n for n, e in load_manifest().items() if e.get("status") == "implemented")


@pytest.fixture(autouse=True)
def isolated_registry():
    state = registry.snapshot()
    registry.DETECTORS.clear()
    registry.BUILDERS.clear()
    registry.LOAD_FAILURES.clear()
    registry.default_target = None
    registry._loaded = True
    registry.register_detector("elsewhere", lambda device: device.type == "cpu")
    registry.default_target = "elsewhere"
    yield
    registry.restore(state)


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    """A host with no CUDA: any device query fails the call that makes it."""
    from tileops.utils import forget_device_properties

    def refuse(*args, **kwargs):
        raise AssertionError("queried a CUDA device")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for name in (
        "current_device",
        "get_device_capability",
        "get_device_properties",
        "get_device_name",
    ):
        monkeypatch.setattr(torch.cuda, name, refuse)
    forget_device_properties()
    yield
    forget_device_properties()


@pytest.mark.parametrize("name", _implemented())
def test_a_target_is_described_and_called_with_the_forward_inputs(name):
    entry = load_manifest()[name]
    cls = rb.op_class(name, entry)
    parametric = not is_legacy(entry)
    if parametric:
        op, args, declared_outputs = _from_call(cls, name, entry)
    else:
        make = _CASES.get(name)
        op, args = make(cls) if make else _from_workload(cls, name, entry)
    declared = tuple(forward_signature(entry).get("inputs") or {})
    passed = args + (None,) * (len(declared) - len(args))
    described = tuple(None if t is None else TensorSpec.of(t) for t in passed)
    seen, returned = [], []

    def build_kernel(*specs, **params):
        assert specs == described
        assert params == op._manifest_params()

        def kernel(*tensors, **writes):
            seen.append(tensors)
            if parametric:
                result = declared_outputs
                returned.append(result[0] if len(result) == 1 else tuple(result))
                return returned[-1]
            shapes = [None if t is None else tuple(t.shape) for t in tensors]
            try:
                out_shapes = op._infer_output_shapes(*shapes)
            except Exception:
                out_shapes = {}
            outputs = entry["signature"]["outputs"]
            dtype = next((t.dtype for t in tensors if t is not None), F16)
            result = [
                torch.empty(out_shapes.get(o, (0,)), dtype=output_dtype(op, o, dtype))
                for o in outputs
            ]
            returned.append(
                None if not result else result[0] if len(result) == 1 else tuple(result)
            )
            return returned[-1]

        return kernel

    registry.register_kernel_builder(name, "elsewhere", build_kernel)

    result = op(*args)

    assert len(seen) == 1, "the kernel the target built runs the call"
    assert all(a is b for a, b in zip(seen[0], passed, strict=True)), (
        "called with what it was described"
    )
    outputs = tuple(entry["signature"]["outputs"])
    if all(o in declared for o in outputs):
        assert result is None, "an op that writes its caller's buffer returns nothing"
    elif isinstance(returned[0], tuple):
        assert all(a is b for a, b in zip(result, returned[0], strict=True))
    else:
        assert result is returned[0], "the op returns what the target's kernel returned"
    if parametric:
        assert hasattr(cls, "_signature"), "a target is held to its signature"
    else:
        assert not declared or hasattr(cls, "_validate_manifest_dtypes"), (
            "a target is held to its dtypes"
        )
    inputs = forward_signature(entry).get("inputs") or {}
    assert all((inputs[o] or {}).get("mutated") for o in outputs if o in inputs), (
        "an output passed in as an input is one the call writes"
    )
