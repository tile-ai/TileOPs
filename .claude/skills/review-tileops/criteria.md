# 1. Review checklists to load

Always load `.claude/review-checklists/pre-review.md` first — it carries the meta-rules every domain checklist depends on (open-set, concrete-and-decidable, reviewer-restraint).

Then apply domain checklists at `.claude/review-checklists/`. Load by **PR title prefix**; a PR may match multiple rows — load all.

| Trigger                                                                                            | Load                      |
| -------------------------------------------------------------------------------------------------- | ------------------------- |
| Title `[Feat]` or `[Enhancement]`                                                                  | `feature.md`              |
| Title `[Refactor]`                                                                                 | `refactor.md`             |
| Title `[Maintain]`, `[Refactor][Manifest]`, or diff flips a `status:` field in `tileops/manifest/` | `manifest.md`             |
| Title `[Doc]` or `[Design]`                                                                        | `doc.md`                  |
| Diff touches any file under `tests/`                                                               | `testing.md` (additional) |

Unmatched prefixes (`[Fix]`, `[BugFix]`, `[Perf]`, `[Bench]`, `[CI]`, `[Chore]`, `[Style]`) fall through to §2 against the diff.

**Approval gate.** Before submitting `APPROVE` on any PR whose diff touches `tests/`, load `.claude/review-checklists/approval-gate.md` and run every check. If any check fails, downgrade the event to `REQUEST_CHANGES` and post the cited node IDs / fixes inline.

**Design docs are on-demand, not eager.** Open one only when a checklist item names it AND the diff makes that item ambiguous.

| Domain                | Design doc                                                 |
| --------------------- | ---------------------------------------------------------- |
| Op / kernel structure | `docs/design/ops-design.md`, `docs/design/architecture.md` |
| Tests                 | `docs/design/testing.md`                                   |
| Manifest spec         | `docs/design/manifest.md`                                  |
| Trust boundaries      | `docs/design/trust-model.md`                               |

Read every changed source file in full — the de-prioritization is design docs only.

# 2. Review priority (descending)

1. **Design consistency** — does the PR follow established patterns, or introduce a new pattern where an existing one applies?
1. **Interface conformance** — Op signatures vs manifest spec, class hierarchy vs `ops-design.md`.
1. **Correctness** — logic errors, edge cases, missing dtype/shape handling.
1. **Scope discipline** — unrelated changes outside the PR's declared type/scope.

Ignore style/formatting (pre-commit handles it).

# 3. Extra-prompt override

If the caller provides extra guidance as the second positional arg, it **overrides** these defaults for this review. Apply it explicitly.

# 4. Submit

```bash
gh api repos/tile-ai/TileOPs/pulls/<N>/reviews \
  -f event="<EVENT>" \
  -f body="<SUMMARY>" \
  -f 'comments=[{"path":"<file>","line":<line>,"body":"<comment>"}, ...]'
```

`<EVENT>`: `REQUEST_CHANGES` if any blocking issue, `APPROVE` if clean, `COMMENT` for non-blocking questions only. Before emitting `APPROVE`, run the §1 approval gate when applicable.

# 5. Inline format

```
<what is wrong and why> → <what to change>
```

One comment per issue. Name the function, variable, or pattern. The reader is an agent that executes fixes literally.

# 6. Summary format (markdown)

The summary is for what does **not** fit in an inline comment. Per-file issues belong inline (§5), not in a summary list. Omit empty sections.

```
### Overall
<one line: top risk + next step. No event echo, no item list.>

### Cross-cutting concerns
- <pattern spanning multiple files — only what an inline comment can't carry>
```

Hard rules:

- Do NOT restate inline comments. If a finding is already inline, it must not appear in the summary.
- Per-file/per-line items go in the `comments=[...]` array of §4, never in the summary body.
- Clean PR: one line, `Clean — no issues.`

# 7. Hard rules

- Read every changed file in full — diff alone lacks context.
- Do not comment outside the PR diff.
- Do not invent issues on a clean PR.
- No eager design-doc reads — only on §1 escalation.
- All review text in English.
- All `gh` calls run with `GH_CONFIG_DIR` already exported by SKILL.md Step 0; never override it. Repo is fixed at `tile-ai/TileOPs`.
