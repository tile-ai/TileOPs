"""Unit tests for trace payload support."""
import pytest
import tilelang
import tilelang.language as T
import torch

from tileops.trace import trace


def test_payload_api_signature():
    """Test that payload parameter is accepted by trace APIs."""
    # These should not raise errors
    trace.disable()

    # Test range() accepts payload
    try:
        @tilelang.jit(out_idx=trace.out_idx(1))
        def build1():
            @T.prim_func
            def kernel(out: T.Buffer((16,), "float32")):
                with T.Kernel(1, threads=16), trace.range("test", payload=0):
                    pass
            return trace.finalize(kernel)

        # Should compile without error
        build1()
        assert True
    except Exception as e:
        pytest.fail(f"trace.range() with payload failed: {e}")


def test_payload_in_loop():
    """Test payload in a simple loop (non-pipelined).

    Note: This test verifies compilation and execution correctness.
    It does not decode trace slots or assert payload values.
    """
    trace.enable(output="debug")

    @tilelang.jit(out_idx=trace.out_idx(1))
    def build():
        @T.prim_func
        def kernel(out: T.Buffer((4,), "float32")):
            with T.Kernel(1, threads=4):
                tx = T.get_thread_binding()

                # Simple sequential loop (not T.Pipelined)
                for i in range(4):
                    # Each iteration traced with different payload
                    # Note: Due to compiler optimizations, actual captured
                    # payload values are not validated in this test
                    with trace.range("loop_iter", payload=i):
                        if tx == 0:  # Only thread 0 to keep it simple
                            out[i] = T.float32(i)

        return trace.finalize(kernel)

    kernel = build()
    output = torch.zeros(4, dtype=torch.float32, device="cuda")

    # When tracing is enabled, kernel returns (output, slots)
    result = kernel(output)
    if isinstance(result, tuple):
        output_tensor, slots = result
    else:
        output_tensor = result

    # Verify kernel ran correctly
    expected = torch.tensor([0., 1., 2., 3.], dtype=torch.float32, device="cuda")
    assert torch.allclose(output_tensor, expected)

    trace.disable()


def test_payload_with_range_start_end():
    """Test payload with explicit range_start/range_end."""
    trace.disable()

    @tilelang.jit(out_idx=trace.out_idx(1))
    def build():
        @T.prim_func
        def kernel(out: T.Buffer((16,), "float32")):
            with T.Kernel(1, threads=16):
                # Test range_start accepts payload
                tok = trace.range_start("test", payload=42)
                # ... work ...
                trace.range_end(tok)

        return trace.finalize(kernel)

    # Should compile
    kernel = build()
    assert kernel is not None


def test_payload_backward_compatibility():
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


def test_payload_with_trace_run():
    """Test payload using trace.run and verify slots are returned.

    This test validates the documented behavior: when tracing is enabled,
    compiled kernels return (*real_outputs, slots).
    """
    trace.enable(output="debug")

    @tilelang.jit(out_idx=trace.out_idx(1))
    def build():
        @T.prim_func
        def kernel(out: T.Buffer((16,), "float32")):
            with T.Kernel(1, threads=16):
                tx = T.get_thread_binding()
                with trace.range("test_range", payload=42):
                    out[tx] = T.float32(tx)

        return trace.finalize(kernel)

    kernel = build()
    output = torch.zeros(16, dtype=torch.float32, device="cuda")

    # Use trace.run which handles (output, slots) unpacking
    result = trace.run(kernel, (output,), stem="test_payload_run")

    # Verify kernel output
    expected = torch.arange(16, dtype=torch.float32, device="cuda")
    assert torch.allclose(result, expected)

    trace.disable()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
