# Review Loop Policy — PR #__PR_NUMBER__

Per-round overrides on `.claude/skills/review-tileops/criteria.md`. Edits apply next round.

## Output contract

- Event MUST be `APPROVE` or `REQUEST_CHANGES`. `COMMENT` is forbidden.
- Summary body MUST end with the loop driver trailer:
  ```
  <!-- review-loop: event=APPROVE|REQUEST_CHANGES; blockers=<N>; sha=<HEAD-sha-7> -->
  ```

## Round 1 — Cold-start

Load checklists per criteria §1. Read every changed source file in full. Apply checklists + §2 priorities to the diff. Default `REQUEST_CHANGES` on any non-trivial concern.

## Round 2+ — Convergence

Codex session resumes; conversation history is available.

- Do NOT re-raise resolved blockers. Reference, don't restate.
- Surface NEW issues only if introduced by new commits.
- `APPROVE` requires: every prior blocker resolved AND no new issues AND CI not red for PR-attributable reasons.

Insert `### Prior blockers` after `### Overall` — **one aggregate line, no per-blocker bullets** (those live in thread replies). Pick one shape:

- All resolved (APPROVE): `All N prior blockers resolved — see thread replies.` Optional shared-mechanism clause; never name entries.
- Mixed: `N resolved, M still-open, K partial — see thread replies.`
- None moved: `No prior blockers resolved this round — see thread replies.`

## Round 5+ — Design re-anchoring (mandatory)

Trigger: `round >= 5` AND a blocker (or its variants — same concern, shifting expression) has persisted 3+ rounds. The bottom-up checklist has failed to converge; re-anchor top-down.

1. **Re-read from disk** (not memory): `docs/design/architecture.md`, `docs/design/ops-design.md`, plus any design doc named by your active checklist items.
1. **Audit the blocker thread.** One root concern, or local patches on a moving target?
1. **Question your anchor.** Cite the design passage that grounds the blocker. No citation possible → overfitted.
1. **Decide:**
   - **Reaffirm** — cite the passage inline.
   - **Withdraw** — retract explicitly. Remaining unease becomes a summary question, not a blocker.
   - **Reframe** — restate once at the design level with citation; stop relitigating surface variants.

Required line at the top of summary (before the trailer):

```
Round-5 introspection: <reaffirmed|withdrawn|reframed> — <one-line reason>
```

If `reaffirmed` and the author shows no movement for 2+ further rounds: state in the summary that the PR is stalling, recommend human review. Continue the loop.

## Notes

- Round count and termination are managed by the loop driver.
- Inbox guidance overrides this policy AND criteria for THIS round only.
