#!/usr/bin/env python3
"""Run each benchmark file in its own pytest process and merge the results.

One native failure (hang, segfault, OOM kill) then costs a single file's
results: every other file's junit fragment survives into the merged report,
a hung child leaves a py-spy stack dump before it is killed, and the
per-file profile_run.log reports are concatenated into the usual single
file.

The next few files' processes start importing while the current file still
owns the GPU and each begins testing only when this parent releases it, so
per-file isolation hides almost all of the per-process import cost. This
parent must stay free of torch/tilelang imports: each child creates its
CUDA context in a fresh process.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

LOG_TAIL_LINES = 80
PY_SPY_TIMEOUT_S = 120
COLLECT_TIMEOUT_S = 1800
TEARDOWN_TIMEOUT_S = 120

# Child preamble; sys.argv[1] is the status pipe fd, sys.argv[2:] the pytest
# arguments, sys.argv[3] the bench file. Constraints the code cannot show:
# - prctl(PR_SET_PTRACER, parent): without it, yama ptrace_scope=1 blocks
#   the parent's py-spy.
# - The collect-only pass runs while another file's benchmark owns the GPU;
#   it must stay GPU-silent (imports only — no CUDA context, no
#   allocations, no kernel launches). CUDA initialization happens in the
#   real pytest run, after the stdin line grants GPU ownership.
# - The exit code goes through the status pipe as soon as the junit
#   fragment is written; interpreter teardown (seconds of torch/tilelang
#   atexit work) stays off the critical path.
_CHILD = """\
import ctypes, os, sys

ctypes.CDLL(None).prctl(0x59616D61, os.getppid(), 0, 0, 0)

import pytest

pytest.main(["--collect-only", "-q", sys.argv[3]])
sys.stdin.readline()
rc = int(pytest.main(sys.argv[2:]))
os.write(int(sys.argv[1]), str(rc).encode())
sys.exit(rc)
"""

_COLLECT = """\
import os, sys

import pytest


class Collector:
    def pytest_collection_finish(self, session):
        files = dict.fromkeys(str(item.path) for item in session.items)
        with open(sys.argv[2], "w") as out:
            out.write("".join(f + "\\n" for f in files))


sys.stdin.readline()
rc = int(pytest.main(["--collect-only", "-q", *sys.argv[3:]], plugins=[Collector()]))
os.write(int(sys.argv[1]), str(rc).encode())
sys.exit(rc)
"""


class Child:
    """One bench child process plus its status pipe."""

    def __init__(self, code: str, argv: list[str], log_path: Path):
        read_fd, write_fd = os.pipe()
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-c", code, str(write_fd), *argv],
                stdin=subprocess.PIPE,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(write_fd,),
            )
        finally:
            os.close(log_fd)
            os.close(write_fd)
        self.status_fd = read_fd

    def release(self) -> None:
        """Unblock the child waiting on stdin; a child already dead is fine."""
        stdin = self.proc.stdin
        assert stdin is not None
        with contextlib.suppress(BrokenPipeError):
            stdin.write(b"\n")
            stdin.flush()
        with contextlib.suppress(BrokenPipeError):
            stdin.close()

    def wait_result(self, timeout_s: float) -> int | None:
        """Return the pytest exit code, negated signal on death, None on timeout.

        Returns as soon as the child reports through the status pipe;
        interpreter teardown continues in the background (reap()).
        """
        readable, _, _ = select.select([self.status_fd], [], [], timeout_s)
        if not readable:
            return None
        data = os.read(self.status_fd, 16)
        if data:
            return int(data)
        # EOF without a status report: the child died; reap the reason.
        return self.proc.wait()

    def reap(self) -> None:
        """Wait out teardown; kill a child stuck in native teardown."""
        with contextlib.suppress(OSError):
            os.close(self.status_fd)
        try:
            self.proc.wait(timeout=TEARDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            print(f"WARNING: pid {self.proc.pid} stuck in teardown; killed", flush=True)
            self.kill()

    def kill(self) -> None:
        """Kill the child's whole process group and reap it."""
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc.wait()
        with contextlib.suppress(OSError):
            os.close(self.status_fd)


def _dump_stack(pid: int, dump_path: Path) -> None:
    """Write a py-spy stack dump of the still-running pid to dump_path."""
    py_spy = shutil.which("py-spy")
    if py_spy is None:
        dump_path.write_text("py-spy is not installed; no stack dump taken\n")
        return
    # --native shows the blocked C/CUDA frames; fall back to a plain dump.
    errors = []
    for extra in (["--native"], []):
        try:
            proc = subprocess.run(
                [py_spy, "dump", "--pid", str(pid), *extra],
                capture_output=True,
                text=True,
                timeout=PY_SPY_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            errors.append(f"py-spy dump {' '.join(extra)} timed out")
            continue
        if proc.returncode == 0:
            dump_path.write_text(proc.stdout)
            return
        errors.append(proc.stderr.strip())
    dump_path.write_text("py-spy dump failed:\n" + "\n".join(errors) + "\n")


def _collect_bench_files(
    targets: list[str], log_path: Path, lingering: list[Child]
) -> list[str]:
    """Return the absolute bench file paths pytest collects under targets."""
    out_file = log_path.with_suffix(".files")
    child = Child(_COLLECT, [str(out_file), *targets], log_path)
    child.release()
    rc = child.wait_result(COLLECT_TIMEOUT_S)
    if rc is None:
        child.kill()
    else:
        lingering.append(child)
    if rc != 0 or not out_file.exists():
        sys.stdout.write(log_path.read_text(errors="replace"))
        raise SystemExit(f"benchmark collection failed (pytest exit {rc})")
    return out_file.read_text().splitlines()


def _synthetic_suite(bench_file: str, message: str, log_tail: str) -> ET.Element:
    """Build a one-entry junit testsuite standing in for a dead child's results."""
    suite = ET.Element(
        "testsuite",
        {"name": "pytest", "tests": "1", "errors": "1", "failures": "0", "skipped": "0"},
    )
    classname = os.path.relpath(bench_file)[: -len(".py")].replace(os.sep, ".")
    case = ET.SubElement(suite, "testcase", {"classname": classname, "name": "whole_file"})
    error = ET.SubElement(case, "error", {"message": f"{os.path.relpath(bench_file)}: {message}"})
    error.text = log_tail
    return suite


def _fragment_suites(fragment: Path) -> list[ET.Element]:
    root = ET.parse(fragment).getroot()
    return list(root) if root.tag == "testsuites" else [root]


def _log_tail(log_path: Path) -> str:
    return "".join(log_path.read_text(errors="replace").splitlines(keepends=True)[-LOG_TAIL_LINES:])


def _absorb_profile_log(parts: list[str]) -> None:
    report = Path("profile_run.log")
    if report.exists():
        parts.append(report.read_text())
        report.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets", nargs="+", help="bench files or directories, e.g. benchmarks/ops"
    )
    parser.add_argument("--junit-xml", required=True, help="merged junit report path")
    parser.add_argument(
        "--timeout-per-file",
        type=float,
        default=1800.0,
        help="seconds before a bench file's process is dumped and killed",
    )
    parser.add_argument("--dump-dir", default="bench_stack_dumps", help="stack dump directory")
    parser.add_argument(
        "--prewarm",
        type=int,
        default=4,
        help="upcoming files importing in advance while the current file runs",
    )
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    suites: list[ET.Element] = []
    profile_parts: list[str] = []
    failed: list[str] = []
    lingering: list[Child] = []

    with tempfile.TemporaryDirectory(prefix="bench_runner_") as work:
        work_dir = Path(work)
        bench_files = _collect_bench_files(args.targets, work_dir / "collect.log", lingering)
        if not bench_files:
            raise SystemExit("no benchmark files collected")
        print(f"collected {len(bench_files)} benchmark files", flush=True)

        def spawn_at(index: int) -> Child:
            fragment = work_dir / f"{index:03d}.xml"
            argv = ["-q", bench_files[index], f"--junit-xml={fragment}"]
            return Child(_CHILD, argv, work_dir / f"{index:03d}.log")

        pending: deque[Child] = deque()
        spawned = 0

        def top_up() -> None:
            nonlocal spawned
            while len(pending) <= args.prewarm and spawned < len(bench_files):
                pending.append(spawn_at(spawned))
                spawned += 1

        top_up()
        try:
            for index, bench_file in enumerate(bench_files):
                child = pending.popleft()
                top_up()
                rel = os.path.relpath(bench_file)
                print(f"\n=== [{index + 1}/{len(bench_files)}] {rel} ===", flush=True)
                fragment = work_dir / f"{index:03d}.xml"
                log_path = work_dir / f"{index:03d}.log"

                start = time.monotonic()
                child.release()
                rc = child.wait_result(args.timeout_per_file)
                elapsed = time.monotonic() - start

                if rc is None:
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    dump_path = dump_dir / f"{Path(bench_file).stem}.txt"
                    _dump_stack(child.proc.pid, dump_path)
                    child.kill()
                    sys.stdout.write(log_path.read_text(errors="replace"))
                    message = (
                        f"timed out after {args.timeout_per_file:.0f}s; "
                        f"killed, stack dump at {dump_path}"
                    )
                    print(f"TIMEOUT: {rel}: {message}", flush=True)
                    suites.append(_synthetic_suite(bench_file, message, _log_tail(log_path)))
                    failed.append(rel)
                else:
                    lingering.append(child)
                    sys.stdout.write(log_path.read_text(errors="replace"))
                    if rc < 0:
                        sig = signal.Signals(-rc)
                        message = f"killed by signal {sig.value} ({sig.name})"
                        print(f"CRASH: {rel}: {message}", flush=True)
                        suites.append(_synthetic_suite(bench_file, message, _log_tail(log_path)))
                        failed.append(rel)
                    elif fragment.exists():
                        suites.extend(_fragment_suites(fragment))
                        # 0 = all passed, 5 = nothing collected (e.g. all skipped).
                        if rc not in (0, 5):
                            failed.append(rel)
                    else:
                        message = f"pytest exited with {rc} without writing results"
                        suites.append(_synthetic_suite(bench_file, message, _log_tail(log_path)))
                        failed.append(rel)
                print(f"--- {rel} finished in {elapsed:.0f}s ---", flush=True)
                _absorb_profile_log(profile_parts)
        finally:
            for leftover in pending:
                leftover.kill()
            for child in lingering:
                child.reap()

    merged = ET.Element("testsuites")
    merged.extend(suites)
    ET.ElementTree(merged).write(args.junit_xml, encoding="utf-8", xml_declaration=True)

    if profile_parts:
        Path("profile_run.log").write_text("\n".join(profile_parts))

    if failed:
        print(f"\n{len(failed)} benchmark file(s) failed:", flush=True)
        for rel in failed:
            print(f"  {rel}", flush=True)
        return 1
    print("\nall benchmark files passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
