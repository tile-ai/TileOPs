#!/usr/bin/env python3
"""Pure-program anti-pattern detector for TileOPs PR diffs.

This script scans a PR diff (or local file tree) for syntactic / structural
anti-patterns derived from the W2 retro of seven alignment PRs (#1211, #1222,
#1229, #1230, #1235, #1240, #1242). Each detector applies regex / AST / git-diff
predicates only; **no LLM call, no model API, no agent dispatch**. The detector
is the structural-fact source for the foundry articulation mechanism's source
pair 2 (surface vs structure). Keeping detection pure-program is required by
the D2 floor (same-class-disguise / detection isomorphism): an LLM judging the
diff would be subject to the same reward-hacking pressure that produced the
anti-pattern, so the judge must be a different class of system entirely.

Output is structured JSON, ready for foundry agents to consume as Source B.

Example output (truncated):

    {
      "pr": 1211,
      "base": "main",
      "head": "feat/...",
      "hits": [
        {
          "pattern_id": "AP-01",
          "match_count": 3,
          "match_locations": [
            "tests/ops/test_elementwise_unary_activation_alignment.py:42",
            "tests/ops/test_elementwise_unary_activation_alignment.py:43",
            "tests/ops/test_elementwise_unary_activation_alignment.py:44"
          ],
          "historical_PRs": [1211, 1222, 1229, 1235, 1240]
        },
        ...
      ]
    }

CLI surfaces:

    python scripts/detect_antipatterns.py --list
    python scripts/detect_antipatterns.py --replay-pr 1211
    python scripts/detect_antipatterns.py --scan <path>
    python scripts/detect_antipatterns.py --diff <diff_file>

`--replay-pr <N>` shells out to ``gh pr diff <N>`` and ``gh pr view <N> --json
body`` to fetch the diff and PR body; AP-04 / AP-07 use the body when
available and fall back to body-less variants otherwise. `gh` must be
installed and authenticated.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Diff parsing helpers (pure stdlib).
# ---------------------------------------------------------------------------


@dataclass
class DiffFile:
    """A single file in a unified diff."""

    path: str
    is_new: bool
    added_lines: List[Tuple[int, str]] = field(default_factory=list)  # (line_no_in_new_file, content)
    raw_added_text: str = ""  # all added '+' lines joined with '\n'


def parse_unified_diff(diff_text: str) -> List[DiffFile]:
    """Parse a unified ``git diff`` into per-file added-line records.

    Only ``+`` lines (excluding the ``+++`` header) are retained. ``-`` and
    context lines are discarded — anti-pattern detection looks at what the PR
    introduces.
    """
    files: List[DiffFile] = []
    current: Optional[DiffFile] = None
    new_lineno = 0
    for raw in diff_text.splitlines():
        if raw.startswith("diff --git "):
            # New file block — finalize the previous one.
            if current is not None:
                current.raw_added_text = "\n".join(c for _, c in current.added_lines)
                files.append(current)
            # Extract the b-side path; tolerant of spaces in paths is not required here.
            m = re.match(r"^diff --git a/(.+?) b/(.+)$", raw)
            path = m.group(2) if m else ""
            current = DiffFile(path=path, is_new=False)
            new_lineno = 0
            continue
        if current is None:
            continue
        if raw.startswith("new file mode"):
            current.is_new = True
            continue
        if raw.startswith("@@"):
            # @@ -a,b +c,d @@
            m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
            if m:
                new_lineno = int(m.group(1)) - 1
            continue
        if raw.startswith("+++") or raw.startswith("---"):
            continue
        if raw.startswith("+"):
            new_lineno += 1
            current.added_lines.append((new_lineno, raw[1:]))
        elif raw.startswith("-"):
            # deletions don't advance new-file line numbers
            pass
        else:
            new_lineno += 1
    if current is not None:
        current.raw_added_text = "\n".join(c for _, c in current.added_lines)
        files.append(current)
    return files


# ---------------------------------------------------------------------------
# Detector framework.
# ---------------------------------------------------------------------------


@dataclass
class DetectorContext:
    """Inputs available to every detector."""

    files: List[DiffFile]
    pr_body: Optional[str] = None  # None when scanning a local tree


@dataclass
class Detector:
    pattern_id: str
    description: str
    historical_prs: List[int]
    fn: Callable[[DetectorContext], List[str]]


def _scan_added(files: List[DiffFile], pattern: re.Pattern) -> List[str]:
    """Return ``path:line`` locations for every added line matching ``pattern``."""
    locs: List[str] = []
    for f in files:
        for lineno, content in f.added_lines:
            if pattern.search(content):
                locs.append(f"{f.path}:{lineno}")
    return locs


# ---- AP-01 ---------------------------------------------------------------
#
# Validator-internals workaround: tests re-import private helpers from
# ``scripts.validate_manifest`` (``_get_forward_params``, ``_get_init_params``,
# ``check_l1_signature``) instead of consulting the manifest, OR pin a manifest
# signature snapshot via a hardcoded ``_INIT_SIGNATURES`` / ``_FORWARD_SIGNATURE``
# dict in tests. Both forms encode the manifest contract in test code rather
# than reading it through ``tileops.manifest``.
_AP01_PATTERN = re.compile(
    r"_get_forward_params|_get_init_params|check_l1_signature|"
    r"from\s+scripts\.validate_manifest\s+import|"
    r"_INIT_SIGNATURES\s*=\s*\{|_FORWARD_SIGNATURE\s*=\s*[\"']"
)


def detect_ap01(ctx: DetectorContext) -> List[str]:
    return _scan_added(ctx.files, _AP01_PATTERN)


# ---- AP-02 ---------------------------------------------------------------
#
# Snapshot test pinning literal signature strings: ``inspect.signature(...)``
# is converted to a string and compared against a literal expected-sig
# constant. The test passes when the implementation matches the snapshot,
# which freezes the implementation at the test layer regardless of manifest.
_AP02_PATTERN = re.compile(
    r"_stringify_init|_stringify_forward|"
    r"signature_matches_snapshot|"
    r"str\s*\(\s*inspect\.signature\s*\("
)


def detect_ap02(ctx: DetectorContext) -> List[str]:
    return _scan_added(ctx.files, _AP02_PATTERN)


# ---- AP-03 ---------------------------------------------------------------
#
# Defensive explainer docstring: the code documents *why* a surprising
# behavior is OK ("may never be assigned, so forward consults the class-level
# default", "happens to work", "leaves remain bit-identical", "ignores any
# post-construction mutation"). A docstring like this is a tell that the code
# is preserving a workaround rather than a contract.
_AP03_PATTERN = re.compile(
    r"happens to work|post-construction mutation|"
    r"defeats the refactor|bit-identical|"
    r"drifts the public API|opts out via",
    re.IGNORECASE,
)


def detect_ap03(ctx: DetectorContext) -> List[str]:
    return _scan_added(ctx.files, _AP03_PATTERN)


# ---- AP-04 ---------------------------------------------------------------
#
# ``FIXME(staged-rollout)`` block whose "Broken invariant" line names the
# same AC that the PR body has marked ``[x]``. The PR claims the AC passes,
# but the in-source FIXME admits the invariant is currently broken.
#
# Pure-FIXME fallback: if the PR body is unavailable (local scan), fall back
# to "FIXME(staged-rollout) marker present" alone — the staged-rollout marker
# itself is a defer signal worth surfacing.
_AP04_FIXME_RE = re.compile(r"FIXME\(staged-rollout\)")
_AP04_AC_LINE_RE = re.compile(r"-\s*\[x\][^\n]*\bAC-?(\d+)", re.IGNORECASE)


def detect_ap04(ctx: DetectorContext) -> List[str]:
    locs: List[str] = []
    for f in ctx.files:
        for lineno, content in f.added_lines:
            if _AP04_FIXME_RE.search(content):
                locs.append(f"{f.path}:{lineno}")
    if not locs:
        return []
    if ctx.pr_body is None:
        # Body unavailable — surface the FIXME alone (degraded but useful).
        return locs
    # Body available: stronger signal is FIXME co-occurring with [x] AC. We
    # still surface the FIXME locations either way; presence of any [x] AC
    # bullet in the body is the gate.
    if _AP04_AC_LINE_RE.search(ctx.pr_body):
        return locs
    return locs  # FIXME alone still warrants a hit; body just amplifies it.


# ---- AP-05 ---------------------------------------------------------------
#
# Single-line ``return torch.<fn>(...)`` body inside ``_eager_forward``.
# Combined with manifest ``status: spec-only``, this means the op is just a
# re-export of the torch implementation; tests of the op trivially match
# torch and provide no real coverage.
_AP05_DEF_RE = re.compile(r"\bdef\s+_eager_forward\s*\(")
_AP05_RETURN_RE = re.compile(r"^\s*return\s+torch\.[a-zA-Z_]+\s*\(")


def detect_ap05(ctx: DetectorContext) -> List[str]:
    locs: List[str] = []
    for f in ctx.files:
        # Walk added lines; if a ``def _eager_forward(`` is followed (within
        # the next 8 added lines) by ``return torch.<fn>(...)``, mark the
        # return line as a hit.
        added = f.added_lines
        for i, (_lineno, content) in enumerate(added):
            if _AP05_DEF_RE.search(content):
                for j in range(i + 1, min(i + 9, len(added))):
                    next_lineno, next_content = added[j]
                    if _AP05_RETURN_RE.search(next_content):
                        locs.append(f"{f.path}:{next_lineno}")
                        break
                    if re.match(r"^\s*def\s+", next_content):
                        break
    return locs


# ---- AP-06 ---------------------------------------------------------------
#
# Class attribute ``_<NAME>_fn = staticmethod(torch.<fn>)``. The Op subclass
# stashes a torch function on the class so per-leaf overrides can swap it,
# turning a per-op fallback into shared base-class machinery.
_AP06_PATTERN = re.compile(
    r"_[a-zA-Z_]+_fn\s*=\s*staticmethod\s*\(\s*torch\.[a-zA-Z_]+"
)


def detect_ap06(ctx: DetectorContext) -> List[str]:
    return _scan_added(ctx.files, _AP06_PATTERN)


# ---- AP-07 ---------------------------------------------------------------
#
# PR body ``[x]`` claims an artifact (file/test/bench) was added/scaffolded,
# but the artifact path is absent from ``git diff --name-only``. Trigger
# requires a body paragraph that pairs "scaffold" / "bench entries" with
# specific op-class names in backticks; pass if any new ``benchmarks/`` file
# appears in the diff or if any backticked op name's snake_case form appears
# in a benchmarks/ filename in the diff.
_AP07_TRIGGER_RE = re.compile(
    r"scaffold[a-z]*[^\n]{0,200}\bbench|bench[^\n]{0,200}\bscaffold[a-z]*|"
    r"\bbench entries\b",
    re.IGNORECASE,
)
_AP07_OPNAME_RE = re.compile(r"`([A-Z][A-Za-z0-9\[\]]*)`")


def _camel_to_snake(name: str) -> str:
    name = name.replace("[", "").replace("]", "")
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def detect_ap07(ctx: DetectorContext) -> List[str]:
    if ctx.pr_body is None:
        return []
    body = ctx.pr_body
    # Walk body bullet-by-bullet (each ``- `` line). A bullet that pairs a
    # scaffold/bench-entries claim with backticked op-class names triggers
    # AP-07; the named ops are then checked against benchmarks/* filenames
    # in the diff.
    bench_paths = [f.path for f in ctx.files if f.path.startswith("benchmarks/")]
    new_bench_files = [
        f.path for f in ctx.files if f.is_new and f.path.startswith("benchmarks/")
    ]
    locs: List[str] = []
    for idx, line in enumerate(body.splitlines(), start=1):
        if not _AP07_TRIGGER_RE.search(line):
            continue
        ops = _AP07_OPNAME_RE.findall(line)
        if not ops:
            continue
        # Identify which named ops have a matching bench file in the diff.
        snake_ops = [_camel_to_snake(o) for o in ops]
        # Strip Op-class trailing tokens to get a family stem (e.g.
        # ``ada_layer_norm_fwd`` -> ``ada_layer_norm``).
        stems = [
            re.sub(r"_(fwd_op|bwd_op|op|fwd)$", "", s) for s in snake_ops
        ]
        unmatched = []
        for stem in stems:
            if not stem:
                continue
            if any(stem in bp for bp in bench_paths):
                continue
            unmatched.append(stem)
        # Any NEW bench file in the diff is treated as covering the family
        # generally — clears the hit for this bullet.
        if new_bench_files:
            continue
        if unmatched:
            locs.append(f"<pr-body>:{idx}")
    return locs


# ---- AP-08 ---------------------------------------------------------------
#
# New ctor kwarg has the manifest-aligned name AND a legacy alias / sentinel
# default lives in the same PR. Patterns: docstring "legacy", "deprecated
# alias", "Backwards-compat aliases", a per-kwarg sentinel default, or paired
# kwargs where one is documented as the legacy form of the other.
_AP08_PATTERN = re.compile(
    r"\bDeprecated alias\b|\blegacy alias(?:es)?\b|\bBackwards.compat\b|"
    r"\bbackwards.compat\b|\blegacy callers?\b|\blegacy\s+(?:`[A-Za-z_]+`|"
    r"``[A-Za-z_]+``|attribute|kwarg|param)|\bsentinel\b|"
    r"approximate\s*==?\s*[\"']tanh[\"']",
    re.IGNORECASE,
)


def detect_ap08(ctx: DetectorContext) -> List[str]:
    return _scan_added(ctx.files, _AP08_PATTERN)


# ---- AP-09 ---------------------------------------------------------------
#
# Test asserts ``op(a,b)`` matches ``<torch_fn>(a,b)`` AND op's
# ``_eager_forward`` body for that dtype is ``return <torch_fn>(...)``. The
# test trivially passes because the op *is* torch in eager mode. Surface
# proxies: a ``ref = torch.<fn>(...)`` baseline in tests, a
# ``_torch_fallback_fn`` class attribute, or a hardcoded ``_SPEC_ONLY_OPS``
# allowlist driving parametrized assertions against torch.
_AP09_PATTERN = re.compile(
    r"_torch_fallback_fn|_TORCH_FALLBACK_DTYPES|_SPEC_ONLY_OPS|"
    r"\bref\s*=\s*torch\.[a-z_]+\s*\(|\btorch_ref\s*=\s*torch\.[a-z_]+\s*\("
)


def detect_ap09(ctx: DetectorContext) -> List[str]:
    return _scan_added(ctx.files, _AP09_PATTERN)


# ---- Registry ------------------------------------------------------------

DETECTORS: List[Detector] = [
    Detector(
        pattern_id="AP-01",
        description=(
            "Validator-internals workaround: re-import of "
            "`scripts.validate_manifest._get_*_params` / `check_l1_signature` "
            "or hardcoded `_INIT_SIGNATURES` snapshot dict in tests."
        ),
        historical_prs=[1211, 1222, 1229, 1230, 1235, 1240],
        fn=detect_ap01,
    ),
    Detector(
        pattern_id="AP-02",
        description=(
            "Snapshot test pinning literal signature strings via "
            "`inspect.signature(cls.__init__)` compared to a literal expected sig."
        ),
        historical_prs=[1230],
        fn=detect_ap02,
    ),
    Detector(
        pattern_id="AP-03",
        description=(
            "Defensive explainer docstring (`may never be assigned`, "
            "`happens to work`, `consults the class`, `post-construction mutation`, "
            "`bit-identical`, `defeats the refactor`, `opts out via`)."
        ),
        historical_prs=[1230],
        fn=detect_ap03,
    ),
    Detector(
        pattern_id="AP-04",
        description=(
            "`FIXME(staged-rollout)` marker present; amplified when the PR "
            "body marks `[x]` on an AC the FIXME admits is broken."
        ),
        historical_prs=[1240],
        fn=detect_ap04,
    ),
    Detector(
        pattern_id="AP-05",
        description=(
            "`_eager_forward` body is a single `return torch.<fn>(...)`; "
            "trivial torch-passthrough disguised as op implementation."
        ),
        historical_prs=[1229],
        fn=detect_ap05,
    ),
    Detector(
        pattern_id="AP-06",
        description=(
            "Class attribute `_<NAME>_fn = staticmethod(torch.<fn>)` "
            "promotes per-op torch fallback to shared base-class machinery."
        ),
        historical_prs=[1222],
        fn=detect_ap06,
    ),
    Detector(
        pattern_id="AP-07",
        description=(
            "PR body `[x]` claims artifact added (file/test/bench), but the "
            "artifact path is absent from `git diff --name-only`."
        ),
        historical_prs=[1235, 1240],
        fn=detect_ap07,
    ),
    Detector(
        pattern_id="AP-08",
        description=(
            "New ctor kwarg has manifest-aligned name AND a legacy alias / "
            "sentinel default ships in the same PR."
        ),
        historical_prs=[1211, 1240],
        fn=detect_ap08,
    ),
    Detector(
        pattern_id="AP-09",
        description=(
            "Test asserts `op(...)` matches `torch.<fn>(...)` AND op's "
            "`_eager_forward` returns the same `torch.<fn>` — the parity "
            "test is vacuous. Surface proxies: `_torch_fallback_fn`, "
            "`ref = torch.<fn>(...)`, `_SPEC_ONLY_OPS` allowlist."
        ),
        historical_prs=[1222, 1229, 1235],
        fn=detect_ap09,
    ),
]


def run_detectors(diff_text: str, pr_body: Optional[str]) -> List[Dict]:
    """Run every detector on the given diff + body. Returns a list of hits."""
    files = parse_unified_diff(diff_text)
    ctx = DetectorContext(files=files, pr_body=pr_body)
    hits: List[Dict] = []
    for det in DETECTORS:
        locs = det.fn(ctx)
        if locs:
            hits.append(
                {
                    "pattern_id": det.pattern_id,
                    "match_count": len(locs),
                    "match_locations": locs,
                    "historical_PRs": det.historical_prs,
                }
            )
    return hits


# ---------------------------------------------------------------------------
# CLI surfaces.
# ---------------------------------------------------------------------------


def _fetch_pr_diff(pr: int) -> str:
    res = subprocess.run(
        ["gh", "pr", "diff", str(pr), "--repo", "tile-ai/TileOPs"],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise SystemExit(f"gh pr diff failed: {res.stderr.strip()}")
    return res.stdout


def _fetch_pr_body(pr: int) -> Optional[str]:
    res = subprocess.run(
        ["gh", "pr", "view", str(pr), "--repo", "tile-ai/TileOPs", "--json", "body"],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return None
    try:
        return json.loads(res.stdout).get("body")
    except json.JSONDecodeError:
        return None


def _scan_local_tree(root: Path) -> str:
    """Build a synthetic unified diff treating every tracked file as 'added'.

    Used by ``--scan <path>`` to feed the detector framework. Honors only
    Python files under the given root.
    """
    chunks: List[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        chunks.append(f"diff --git a/{rel} b/{rel}\n")
        chunks.append("new file mode 100644\n")
        chunks.append(f"--- /dev/null\n+++ b/{rel}\n")
        chunks.append(f"@@ -0,0 +1,{len(text.splitlines())} @@\n")
        chunks.extend(f"+{line}\n" for line in text.splitlines())
    return "".join(chunks)


def cmd_list() -> None:
    for det in DETECTORS:
        print(f"{det.pattern_id}\t{det.description}")


def cmd_replay_pr(pr: int) -> None:
    diff_text = _fetch_pr_diff(pr)
    body = _fetch_pr_body(pr)
    hits = run_detectors(diff_text, body)
    print(json.dumps({"pr": pr, "hits": hits}, indent=2))


def cmd_scan(path: str) -> None:
    root = Path(path)
    if not root.exists():
        raise SystemExit(f"path does not exist: {path}")
    diff_text = _scan_local_tree(root)
    hits = run_detectors(diff_text, pr_body=None)
    print(json.dumps({"scan": str(root), "hits": hits}, indent=2))


def cmd_diff(diff_path: str) -> None:
    diff_text = Path(diff_path).read_text(encoding="utf-8")
    hits = run_detectors(diff_text, pr_body=None)
    print(json.dumps({"diff": diff_path, "hits": hits}, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pure-program TileOPs anti-pattern detector.",
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--list", action="store_true", help="List detector IDs.")
    grp.add_argument("--replay-pr", type=int, metavar="N", help="Replay detectors against PR #N.")
    grp.add_argument("--scan", metavar="PATH", help="Scan a local file tree.")
    grp.add_argument("--diff", metavar="FILE", help="Scan a unified-diff file.")
    args = parser.parse_args(argv)

    if args.list:
        cmd_list()
    elif args.replay_pr is not None:
        cmd_replay_pr(args.replay_pr)
    elif args.scan is not None:
        cmd_scan(args.scan)
    elif args.diff is not None:
        cmd_diff(args.diff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
