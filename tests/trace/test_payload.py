"""Unit tests for trace payload support."""
import pytest
import tilelang
import tilelang.language as T
import torch

from tileops.trace import trace

# Mark all tests in this file as 'full' tier
pytestmark = pytest.mark.full


@pytest.fixture
def preserve_trace_state():
    """Save and restore trace state to avoid breaking global --trace-kernel."""
    original_enabled = trace.enabled
    original_output = trace.output
    try:
        yield
    finally:
        # Restore original state
        if original_enabled:
            trace.enable(output=original_output)
        else:
            trace.disable()


def test_payload_api_signature(preserve_trace_state):
    """Test that payload parameter is accepted by trace APIs."""
    trace.disable()

    # Test range() accepts payload
    @tilelang.jit(out_idx=trace.out_idx(1))
    def build1():
        @T.prim_func
        def kernel(out: T.Buffer((16,), "float32")):
            with T.Kernel(1, threads=16), trace.range("test", payload=0):
                pass
        return trace.finalize(kernel)

    # Should compile without error
    kernel = build1()
    assert kernel is not None


def test_payload_with_range_start_end(preserve_trace_state):
    """Test payload with explicit range_start/range_end."""
    trace.disable()

    @tilelang.jit(out_idx=trace.out_idx(1))
    def build():
        @T.prim_func
        def kernel(out: T.Buffer((16,), "float32")):
            with T.Kernel(1, threads=16):
                # Test range_start accepts payload
                tok = trace.range_start("test_range", payload=42)
                # Simple work
                trace.range_end(tok)

        return trace.finalize(kernel)

    kernel = build()
    assert kernel is not None


def test_payload_backward_compatibility(preserve_trace_state):
    """Test that existing code without payload still works."""
    trace.disable()

    @tilelang.jit(out_idx=trace.out_idx(1))
    def build():
        @T.prim_func
        def kernel(out: T.Buffer((16,), "float32")):
            with T.Kernel(1, threads=16):
                # Old code without payload should still work
                with trace.range("test"):
                    pass

                # With lane but no payload
                with trace.range("test2", lane="compute"):
                    pass

        return trace.finalize(kernel)

    kernel = build()
    assert kernel is not None


def test_implicit_thread_blocks_with_payload_e2e(preserve_trace_state, tmp_path):
    """End-to-end test: implicit thread blocks + payload lowering + decode.

    This test verifies:
    1. trace.range(..., payload=...) actually lowers to CUDA markers
    2. Payload is written to slots and can be decoded
    3. Implicit thread block fallback (__tl_thread_idx_x) compiles and runs
    4. Range begin/end pairs correctly into a slice
    """
    trace.enable(output=str(tmp_path))

    @tilelang.jit(out_idx=trace.out_idx(1))
    def build():
        @T.prim_func
        def kernel(out: T.Buffer((16,), "float32")):
            # Use simple T.Kernel(..., threads=16) - no explicit threadIdx.x binding
            # This triggers the __tl_thread_idx_x() fallback
            with T.Kernel(1, threads=16):
                tx = T.get_thread_binding()
                with trace.range("test_range", payload=42):
                    out[tx] = T.float32(tx)

        return trace.finalize(kernel)

    kernel = build()

    # When tracing is enabled, kernel returns (output, slots)
    # out_idx indicates output is at index 1, so kernel takes no inputs
    result = kernel()
    assert isinstance(result, (tuple, list)), "Expected (output, slots) tuple"

    output_tensor, slots = result

    # Verify kernel output
    expected = torch.arange(16, dtype=torch.float32, device="cuda")
    assert torch.allclose(output_tensor, expected)

    # Decode and verify payload
    events = trace.decode(kernel, slots)
    slices = [e for e in events if e.name == "test_range"]

    assert len(slices) >= 1, "Expected at least one test_range slice"

    # Check that payload 42 appears
    payloads = [s.payload for s in slices if hasattr(s, "payload")]
    assert 42 in payloads, f"Expected payload 42 in decoded slices, got {payloads}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
