#!/usr/bin/env python3
"""Run each benchmark file in its own pytest process and merge the results.

A native failure (hang, segfault, OOM kill) costs one file: other fragments
survive into the merged report and a hung child leaves a py-spy stack dump.
Upcoming children import while the current file owns the GPU, hiding startup
cost. This parent must never import torch: children need fresh processes.
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
TEARDOWN_TIMEOUT_S = 120

# Child preamble; argv[1] = status pipe fd, argv[2:] = pytest args, argv[3]
# = the bench file. prctl(PR_SET_PTRACER, parent) lets py-spy attach under
# yama ptrace_scope=1. The collect-only pass runs while another file owns
# the GPU, so it must stay GPU-silent; CUDA init happens after the stdin
# grant. TILEOPS_COLLECT_STATUS holds "started", then the collect exit code,
# so a child that dies mid-collect stays distinguishable from a clean one.
# The run exit code leaves via the pipe before interpreter teardown.
_CHILD = """\
import ctypes, os, sys

status = os.environ["TILEOPS_COLLECT_STATUS"]
open(status, "w").write("started")

ctypes.CDLL(None).prctl(0x59616D61, os.getppid(), 0, 0, 0)

import pytest

rc_collect = int(pytest.main(["--collect-only", "-q", sys.argv[3]]))
open(status, "w").write(str(rc_collect))
sys.stdin.readline()
rc = int(pytest.main(sys.argv[2:]))
os.write(int(sys.argv[1]), str(rc).encode())
sys.exit(rc)
"""

class Child:
    """One bench child process plus its status pipe."""

    def __init__(self, code: str, argv: list[str], log_path: Path, status_path: Path):
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
                env={**os.environ, "TILEOPS_COLLECT_STATUS": str(status_path)},
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
        interpreter teardown continues in the background.
        """
        readable, _, _ = select.select([self.status_fd], [], [], timeout_s)
        if not readable:
            return None
        data = os.read(self.status_fd, 16)
        os.close(self.status_fd)
        if data:
            return int(data)
        # EOF without a status report: the child died; reap the reason.
        return self.proc.wait()

    def poll_teardown(self, deadline: float) -> str | None:
        """Return the teardown outcome once the child is gone, else None.

        Outcomes: "" for a normal exit, otherwise a message describing a
        teardown crash (unexpected signal) or a deadline kill.
        """
        rc = self.proc.poll()
        if rc is not None:
            if rc < 0:
                sig = signal.Signals(-rc)
                return f"died in teardown: signal {sig.value} ({sig.name})"
            return ""
        if time.monotonic() < deadline:
            return None
        self.kill()
        return "stuck in teardown; killed at deadline"

    def kill(self) -> None:
        """Kill the child's whole process group and reap it."""
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc.wait()
        with contextlib.suppress(OSError):
            os.close(self.status_fd)


def _reap_lingering(
    lingering: list[tuple[Child, float, str]], block: bool
) -> list[tuple[str, str]]:
    """Drop finished children, killing any past its teardown deadline.

    Returns (bench_file, message) for every abnormal teardown. With
    block=True, waits out the remaining deadlines (end of run).
    """
    anomalies: list[tuple[str, str]] = []
    while True:
        remaining = []
        for child, deadline, bench_file in lingering:
            outcome = child.poll_teardown(deadline)
            if outcome is None:
                remaining.append((child, deadline, bench_file))
            elif outcome:
                anomalies.append((bench_file, outcome))
        lingering[:] = remaining
        if not lingering or not block:
            return anomalies
        time.sleep(1.0)


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


def _discover_bench_files(targets: list[str]) -> list[str]:
    """Return bench_*.py files under targets by filesystem walk.

    Discovery must not import benchmark modules: a module-level crash or
    hang would take down the whole run before any isolation exists. A
    spawned file with nothing to run exits with pytest code 5 (skipped).
    """
    files: dict[str, None] = {}
    for target in targets:
        path = Path(target)
        if path.is_file():
            files[str(path)] = None
        elif path.is_dir():
            for f in sorted(path.rglob("bench_*.py")):
                files[str(f)] = None
        else:
            raise SystemExit(f"no such benchmark target: {target}")
    return list(files)


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


def _collect_outcome(status_path: Path) -> str | None:
    """Describe the collect-only pass, or None if it exited 0 or 5 (nothing collected)."""
    try:
        raw = status_path.read_text().strip()
    except OSError:
        return "no status written; the child died before collection"
    if raw == "started":
        return "died during collection"
    if raw.isdigit() and int(raw) not in (0, 5):
        return f"collect-only exited with {raw}"
    return None


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
        "--teardown-timeout",
        type=float,
        default=TEARDOWN_TIMEOUT_S,
        help="seconds a reported child may spend in interpreter teardown",
    )
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
    lingering: list[tuple[Child, float, str]] = []

    def note_anomalies(anomalies: list[tuple[str, str]]) -> None:
        for anomaly_file, message in anomalies:
            rel_a = os.path.relpath(anomaly_file)
            print(f"TEARDOWN: {rel_a}: {message}", flush=True)
            suites.append(_synthetic_suite(anomaly_file, message, ""))
            failed.append(rel_a)

    with tempfile.TemporaryDirectory(prefix="bench_runner_") as work:
        work_dir = Path(work)
        bench_files = _discover_bench_files(args.targets)
        if not bench_files:
            raise SystemExit("no benchmark files found")
        print(f"discovered {len(bench_files)} benchmark files", flush=True)

        def spawn_at(index: int) -> Child:
            fragment = work_dir / f"{index:03d}.xml"
            argv = ["-q", bench_files[index], f"--junit-xml={fragment}"]
            return Child(
                _CHILD, argv, work_dir / f"{index:03d}.log", work_dir / f"{index:03d}.collect"
            )

        pending: deque[Child] = deque()
        collect_failed: list[tuple[str, str]] = []
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
                # Poll in short steps so lingering teardown deadlines are
                # enforced while this file runs, not after it.
                file_deadline = start + args.timeout_per_file
                while True:
                    rc = child.wait_result(min(1.0, max(0.0, file_deadline - time.monotonic())))
                    note_anomalies(_reap_lingering(lingering, block=False))
                    if rc is not None or time.monotonic() >= file_deadline:
                        break
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
                    if rc >= 0:
                        # A pre-status signal death (rc < 0) is already reaped
                        # and recorded below; enqueue only status-reported
                        # children so post-status deaths stay observable.
                        lingering.append(
                            (child, time.monotonic() + args.teardown_timeout, bench_file)
                        )
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
                outcome = _collect_outcome(work_dir / f"{index:03d}.collect")
                if outcome is not None:
                    collect_failed.append((rel, outcome))
                print(f"--- {rel} finished in {elapsed:.0f}s ---", flush=True)
                _absorb_profile_log(profile_parts)
        finally:
            for leftover in pending:
                leftover.kill()
            note_anomalies(_reap_lingering(lingering, block=True))

    merged = ET.Element("testsuites")
    merged.extend(suites)
    ET.ElementTree(merged).write(args.junit_xml, encoding="utf-8", xml_declaration=True)

    if profile_parts:
        Path("profile_run.log").write_text("\n".join(profile_parts))

    if collect_failed:
        scope = (
            "every file" if len(collect_failed) == len(bench_files)
            else f"{len(collect_failed)} of {len(bench_files)} files"
        )
        print(f"\ncollection did not complete cleanly for {scope}:", flush=True)
        for rel, outcome in collect_failed:
            print(f"  {rel}: {outcome}", flush=True)
        print(
            "  Collection runs while another file owns the GPU; a broken import is the"
            " usual cause, and one that initialises CUDA perturbs the measured file.",
            flush=True,
        )

    if failed:
        print(f"\n{len(failed)} benchmark file(s) failed:", flush=True)
        for rel in failed:
            print(f"  {rel}", flush=True)
        return 1
    print("\nall benchmark files passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
