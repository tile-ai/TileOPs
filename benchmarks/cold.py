"""The device side of nvbench's cold measurement on CUDA, ported to Triton.

- :class:`StreamBlocker` is nvbench's ``blocking_kernel`` (``nvbench/blocking_kernel.cu``):
  work queued behind :meth:`StreamBlocker.block` waits until :meth:`StreamBlocker.unblock`,
  and a timeout releases it on its own.
- :class:`SmClocks` is nvbench's ``gpu_frequency`` over ``timestamps_kernel``
  (``nvbench/detail/timestamps_kernel.cu``): SM 0 writes the global timer and its own cycle
  counter, and two such records bracket a sample to give its SM clock.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _global_ns():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _block_stream(released, timed_out, timeout_ns):
    start = _global_ns()
    now = start
    flag = tl.load(released, volatile=True)
    while (flag == 0) & (now - start < timeout_ns):
        flag = tl.load(released, volatile=True)
        now = _global_ns()
    if flag == 0:
        # Published system-wide, as nvbench fences it, so the host sees it while the stream runs.
        tl.atomic_xchg(timed_out, 1, sem="release", scope="sys")


@triton.jit
def _sm0_timestamps(out):
    sm = tl.inline_asm_elementwise(
        "mov.u32 $0, %smid;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )
    if sm == 0:
        tl.store(out, _global_ns())
        cycles = tl.inline_asm_elementwise(
            "mov.u64 $0, %clock64;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
        )
        tl.store(out + 1, cycles)


class StreamBlocker:
    """Hold the current stream until the host releases it, or until *timeout_s* passes.

    The two flags live in pinned host memory, which the kernel reads and writes directly.
    """

    def __init__(self, timeout_s: float):
        self._released = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self._timed_out = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        # Host views of the same memory, written and read without a torch dispatch.
        self._released_flag = self._released.numpy()
        self._timed_out_flag = self._timed_out.numpy()
        self._timeout_ns = int(timeout_s * 1e9)

    def block(self) -> None:
        self._released_flag[0] = 0
        self._timed_out_flag[0] = 0
        _block_stream[(1,)](self._released, self._timed_out, self._timeout_ns, num_warps=1)

    def unblock(self) -> None:
        self._released_flag[0] = 1

    def timed_out(self) -> bool:
        """Whether the last block ended by its timeout rather than by :meth:`unblock`."""
        return bool(self._timed_out_flag[0])


class SmClocks:
    """SM 0's clock across each of *n* samples on a CUDA *device*, read as nvbench's
    ``gpu_frequency`` reads it: a timestamp record before the call and one after.
    """

    def __init__(self, n: int, device: torch.device):
        # Per sample: SM 0's (global ns, cycles) before the call, then after it.
        self._stamps = torch.zeros(n, 2, 2, dtype=torch.int64, device=device)
        self._sms = torch.cuda.get_device_properties(device).multi_processor_count

    def clear(self) -> None:
        """Zero every record, so a sample SM 0 never reached reads as unknown."""
        self._stamps.zero_()

    def _record(self, out: torch.Tensor) -> None:
        # One program per SM, as nvbench launches it, so one of them lands on SM 0.
        _sm0_timestamps[(self._sms,)](out, num_warps=1)

    def start(self, i: int) -> None:
        self._record(self._stamps[i, 0])

    def stop(self, i: int) -> None:
        self._record(self._stamps[i, 1])

    def read_hz(self, n: int) -> list[float | None]:
        """The first *n* samples' clocks, as ``gpu_frequency::get_clock_frequency``
        computes them; ``None`` where SM 0 left a record unset."""
        clocks = []
        for before, after in self._stamps[:n].tolist():
            elapsed_ns, elapsed_cycles = after[0] - before[0], after[1] - before[1]
            written = before[0] != 0 and after[0] != 0
            ok = written and elapsed_ns > 0 and elapsed_cycles > 0
            clocks.append(elapsed_cycles / elapsed_ns * 1e9 if ok else None)
        return clocks
