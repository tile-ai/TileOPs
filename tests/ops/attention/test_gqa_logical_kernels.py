import pytest
import torch

from tileops.kernels.attention import gqa_logical

pytestmark = pytest.mark.smoke


class _FakeKernel:
    supported_archs = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> tuple[str, tuple[object, ...]]:
        self.calls.append((args, kwargs))
        return type(self).__name__, args


class _FakeDenseKernel(_FakeKernel):
    pass


class _FakeVarlenKernel(_FakeKernel):
    pass


class _FakeSlidingWindowKernel(_FakeKernel):
    pass


class _FakeAppendKernel(_FakeKernel):
    pass


class _FakeFwdKernel(_FakeKernel):
    pass


class _FakeFP8Kernel(_FakeKernel):
    pass


def test_gqa_prefill_kernel_dispatches_dense(monkeypatch) -> None:
    monkeypatch.setattr(gqa_logical, "GQAPrefillFwdKernel", _FakeDenseKernel)

    kernel = gqa_logical.GQAPrefillKernel(
        batch=1,
        heads=8,
        heads_kv=2,
        dim=64,
        seq_len_q=32,
        seq_len_kv=64,
        dtype=torch.float16,
    )

    assert isinstance(kernel.impl, _FakeDenseKernel)
    assert kernel.impl.kwargs["seq_len_q"] == 32
    assert kernel.impl.kwargs["seq_len_kv"] == 64


def test_gqa_prefill_kernel_dispatches_ragged(monkeypatch) -> None:
    monkeypatch.setattr(gqa_logical, "GQAPrefillVarlenFwdKernel", _FakeVarlenKernel)

    kernel = gqa_logical.GQAPrefillKernel(
        batch=2,
        heads=8,
        heads_kv=2,
        dim=64,
        layout="ragged",
        dtype=torch.float16,
    )

    assert isinstance(kernel.impl, _FakeVarlenKernel)
    assert kernel.impl.kwargs["batch"] == 2
    assert kernel.impl.kwargs["dim"] == 64


def test_gqa_prefill_kernel_dispatches_sliding_window(monkeypatch) -> None:
    monkeypatch.setattr(gqa_logical, "is_hopper", lambda: False)
    monkeypatch.setattr(gqa_logical, "GQASlidingWindowFwdKernel", _FakeSlidingWindowKernel)

    kernel = gqa_logical.GQAPrefillKernel(
        batch=1,
        heads=8,
        heads_kv=2,
        dim=64,
        seq_len_q=128,
        seq_len_kv=128,
        use_swa=True,
        window_size_left=64,
        dtype=torch.float16,
    )

    assert isinstance(kernel.impl, _FakeSlidingWindowKernel)
    assert kernel.impl.kwargs["window_size_left"] == 64


def test_gqa_prefill_with_kv_cache_kernel_dispatches_rope_pair(monkeypatch) -> None:
    monkeypatch.setattr(
        gqa_logical, "GQAPrefillWithKVCacheRopeAppendKernel", _FakeAppendKernel)
    monkeypatch.setattr(gqa_logical, "GQAPrefillWithKVCacheRopeFwdKernel", _FakeFwdKernel)

    kernel = gqa_logical.GQAPrefillWithKVCacheKernel(
        batch=1,
        heads=8,
        heads_kv=2,
        seq_len_new=32,
        seqlen_kv=128,
        dim=64,
        fuse_rope=True,
        max_position=128,
        rotary_dim=64,
        dtype=torch.float16,
    )
    result = kernel("q", "k_new", "v_new", "k_cache", "v_cache", "lens", "cos", "sin")

    assert isinstance(kernel.append_impl, _FakeAppendKernel)
    assert isinstance(kernel.impl, _FakeFwdKernel)
    assert len(kernel.append_impl.calls) == 1
    assert result[0] == "_FakeFwdKernel"


def test_gqa_prefill_paged_kernel_dispatches_fp8_cache(monkeypatch) -> None:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        return
    monkeypatch.setattr(gqa_logical, "GQAPrefillPagedWithFP8KVCacheFwdKernel", _FakeFP8Kernel)

    kernel = gqa_logical.GQAPrefillPagedKernel(
        batch=1,
        heads=8,
        heads_kv=2,
        max_pages_per_req=4,
        page_size=64,
        dim=64,
        cache_dtype=fp8_dtype,
        dtype=torch.float16,
    )

    assert isinstance(kernel.impl, _FakeFP8Kernel)
    assert kernel.append_impl is None


def test_gqa_prefill_paged_kernel_dispatches_rope_pair(monkeypatch) -> None:
    monkeypatch.setattr(
        gqa_logical, "GQAPrefillPagedWithKVCacheRopeAppendKernel", _FakeAppendKernel)
    monkeypatch.setattr(gqa_logical, "GQAPrefillPagedWithKVCacheRopeFwdKernel", _FakeFwdKernel)

    kernel = gqa_logical.GQAPrefillPagedKernel(
        batch=1,
        heads=8,
        heads_kv=2,
        max_pages_per_req=4,
        page_size=64,
        dim=64,
        fuse_rope=True,
        max_position=128,
        rotary_dim=64,
        dtype=torch.float16,
    )
    result = kernel(
        "q",
        "k_new",
        "v_new",
        "k_pages",
        "v_pages",
        "cu_q",
        "cache_lens",
        "block_table",
        "max_q",
        "cos",
        "sin",
    )

    assert isinstance(kernel.append_impl, _FakeAppendKernel)
    assert isinstance(kernel.impl, _FakeFwdKernel)
    assert len(kernel.append_impl.calls) == 1
    assert result[0] == "_FakeFwdKernel"
