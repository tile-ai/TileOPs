## Triage each unresolved thread and summary item

For each inline thread's root comment **and** each actionable item from review summaries, classify:

| Verdict    | Criteria                                                          | Action                                |
| ---------- | ----------------------------------------------------------------- | ------------------------------------- |
| **Accept** | Feedback is correct and fix belongs in this PR's scope            | Fix the code                          |
| **Reject** | Feedback is incorrect, irrelevant, or based on a misunderstanding | Reply with reasoning                  |
| **Defer**  | Feedback is valid but fix would significantly expand PR scope     | Acknowledge, explain why out of scope |

**Bias toward Accept.** Only defer when the fix would:

- Touch files/modules unrelated to the PR's stated purpose.
- Require a design decision not yet made.
- Be large enough to warrant its own review cycle.

Evaluate all feedback on merit, regardless of whether the reviewer is human or bot.

## Apply fixes

For each accepted comment:

1. Read the relevant file and understand the context.
1. Make the minimal fix.

After all fixes:

1. Run pre-commit / linters.
1. Commit using `[Chore][<scope>] address review feedback` where `<scope>` is extracted from the PR title's `[Type][Scope]` pattern. If the PR title has no scope, omit it: `[Chore] address review feedback`.
1. Push to the PR branch (use the correct remote for cross-fork PRs).

## Reply and resolve threads; optional top-level note

### Inline threads (always)

For each thread, reply **inline** then resolve:

```bash
# Reply
gh api "repos/$REPO/pulls/$PR/comments/<root_comment_id>/replies" \
  -f body="<reply>"

# Resolve thread
gh api graphql -f query='
  mutation($id:ID!) {
    resolveReviewThread(input:{threadId:$id}) {
      thread { isResolved }
    }
  }' -f id="<thread_node_id>"
```

Reply formats and hard rules: `.claude/skills/resolve-tileops/criteria.md`.

### Top-level PR comment (optional)

Skip if every item is covered by an inline reply. Only post when there is something inline can't carry: cross-cutting context, a defer rationale that spans the PR, or a one-line pointer to the fix commit.

```bash
gh api "repos/$REPO/issues/$PR/comments" -f body="<top-level note>"
```

Format and hard rules: `.claude/skills/resolve-tileops/criteria.md` §Top-level note.

**Deduplicate**: if multiple reviewers (or summary + inline) raise the same point, fix once, reference the same commit in each reply.

Resolve every thread that was replied to, regardless of verdict.
