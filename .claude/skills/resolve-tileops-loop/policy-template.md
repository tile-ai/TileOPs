# Resolve Loop Policy — PR #__PR_NUMBER__

Loop-specific overrides for the developer side, applied on top of the
`resolve-tileops` skill procedure. Edits apply to the next round.

## Round 1 — Cold-start

No prior conversation history. Triage every open thread + summary item per
resolve-tileops §3. Bias toward Accept.

## Round 2+ — Incremental mode

Conversation history is available via Codex session resume.

- Process only the NEW reviewer summaries and inline comments listed in the
  round's input snapshots. Already-replied threads do not need re-action.
- For previously-rejected feedback that the reviewer has restated: do NOT
  reflexively re-reject. Re-evaluate with the fresh context. The reviewer
  bringing it back is signal that your prior reasoning may have been
  incomplete.
- If you push a fix this round, reference the new short sha in every reply
  (`Adopted. <fix>. See <short_sha>.`).

## Round 5+ — Resolution re-anchoring (mandatory)

Trigger: `round >= 5` AND a single thread (or thematically related cluster)
has been rejected/deferred by you across 3+ prior rounds and is still
unresolved.

Run before processing this round's feedback:

1. **Re-read design docs from disk** (not memory): `docs/architecture.md`,
   `docs/ops-design.md`, plus reference docs cited by domain rules for the
   files this thread touches.
1. **Audit your reply thread.** List the verdicts you've issued on this
   blocker across rounds. Are you defending one consistent position with
   evidence, or making piecemeal patches that move the problem around?
1. **Question your stance.** Is the reviewer's concern grounded in the
   design docs? Cite the doc passage that supports YOUR side. No citation
   possible → your rejection is overfitted to the current implementation.
1. **Decide:**
   - **Persist** — your stance is design-grounded; cite the passage in the
     reply. Stop re-arguing the surface variants the reviewer raises.
   - **Pivot** — accept that a deeper refactor is needed; reply with that
     acknowledgement and either do the refactor this round or defer with
     an explicit follow-up plan.
   - **Defer** — reply once with "deferring to follow-up issue #N", stop
     replying further on this thread, focus rounds on other items.

Required line at the top of the round's git commit message body (if you
push a commit this round):

```
Round-5 introspection: <persist|pivot|defer> — <one-line reason>
```

If no commit this round, post the introspection line as the first line of
your reply on the relevant thread.

**Stalling:** if `persist` and the reviewer continues to insist for 2+
further rounds, post a comment proposing human escalation. Continue the
loop.

## Notes

- Round count and termination are managed by the loop driver.
- Inbox temporary guidance overrides this policy for THIS round only.
- All replies in English; concise — conclusion, action, reasoning. No filler.
