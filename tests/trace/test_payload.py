"""Unit tests for trace payload support."""
import pytest
import tilelang
import tilelang.language as T

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
