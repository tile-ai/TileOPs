# Contributing to TileOPs

## Setup

Activate a virtual environment, then:

```bash
pip install -e '.[dev]' -c constraints.txt
pre-commit install
```

[docs/development.md](docs/development.md) covers the rest.

## Where the rules live

TileOPs is design-first: [`docs/design/`](docs/design/) and
[`src/tileops/manifest/`](src/tileops/manifest/) are the spec, and code conforms to them rather
than the reverse. Before changing an area, read its document.

| Changing         | Read                                                                                                                                                     |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| an op or kernel  | [ops-design.md](docs/design/ops-design.md), [op-slot-rules.md](docs/design/op-slot-rules.md), [.claude/rules/code-style.md](.claude/rules/code-style.md) |
| a manifest entry | [manifest.md](docs/design/manifest.md), [.claude/domain-rules/manifest-spec.md](.claude/domain-rules/manifest-spec.md)                                   |
| a test           | [testing.md](docs/design/testing.md), [trust-model.md](docs/design/trust-model.md)                                                                       |
| a benchmark      | [testing.md § Benchmarks](docs/design/testing.md#benchmarks), [.claude/domain-rules/benchmark.md](.claude/domain-rules/benchmark.md)                     |
| a design doc     | [.claude/domain-rules/design-docs.md](.claude/domain-rules/design-docs.md)                                                                               |

A rule that can be checked mechanically is a `pre-commit` hook rather than a line in a document.
Run `pre-commit run --all-files` before pushing; CI runs the same set.

Gitleaks runs among those hooks. Do not skip it: `SKIP=gitleaks` needs an explicit reason and a
reviewer who accepts it. Secrets reach a workflow through `env:` and `${{ secrets.* }}`, never
inline in a `run:` command where they land in the log.

## Naming

`.claude/conventions/types.sh` is the single source of truth for types, branch names, and labels.
CI sources it, so a title that passes the pattern below cannot fail `validate-pr-title`.

|                   | Form                                                                |
| ----------------- | ------------------------------------------------------------------- |
| commit / PR title | `[Type] description`, or `[Type][foundry][Scope] description`       |
| branch            | `<type>/<area>/<slug>`, all lowercase — `perf/norm/rms-norm-sol`    |
| issue title       | `[TYPE][COMPONENT] short description in lowercase`, ≤ 80 characters |

Check a title before opening the PR:

```bash
source .claude/conventions/types.sh
[[ "$TITLE" =~ $COMMIT_MSG_PATTERN ]] || echo "title does not match: $TITLE"
```

Conventional Commits (`feat(scope): …`) is not used.

## Pull requests

[.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) is the body shape. Keep it to
what changed and how to verify it — one line per bullet, no recounting of the development process.

- A PR touching `tests/` reports `python scripts/test_node_delta.py --base upstream/main`. Growth
  on an existing file needs a one-line justification per new case: which code path, dtype, feature,
  or regression it guards. Above 10% growth the burden is on the author, and reviewer silence is
  not approval.
- A PR touching a kernel or op reports benchmark numbers in the format
  [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) shows.
- Pass the PR body as `--body-file`, never as an inline JSON string — the latter lands literal
  `\n` in the rendered body.

## Reviewing

- Read every changed file in full; the diff alone lacks the surrounding context.
- Anchor each finding to a `file:line`, a manifest entry path, a field name, or a parametrize axis,
  and state a decidable claim about it. "Looks reasonable" and "may want to verify" decide nothing.
- A test's expected value traces to PyTorch, NumPy, a closed form, or IEEE-754 — never to a literal
  someone wrote down.
- Never approve the removal of the last test on an output-distinguishing input: a tile boundary, a
  vectorization alignment, a degenerate dimension, or a dispatch branch with observable behavior.
- Re-run the test node delta yourself against a fresh `upstream/main` rather than trusting the
  reported number. Above 25% growth on existing files, a case you cannot classify — or one carrying
  an unresolved objection — means the PR is not clean.
- `status: implemented` in the manifest is a claim that the code conforms to the entry. Check it
  field by field against `__init__` and `forward` rather than taking it on faith, and treat a
  missing `source.kernel_map` as blocking even though the validator only warns. Grep the op's test
  file: a `pytest.skip`, an `xfail`, or an assertion weakened while the entry was `spec-only` means
  the flip is not earned, and any `FIXME(staged-rollout)` tied to that op goes with it.
