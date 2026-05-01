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

For each accepted comment, read the relevant file, understand the context, and make the **minimal** fix. Commit message: `[Chore][<scope>] address review feedback` where `<scope>` comes from the PR title's `[Type][Scope]` pattern (omit `[<scope>]` if the title has none).

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

### Top-level PR comment (optional)

Skip if every item is covered by an inline reply. Only post when there is something inline can't carry: cross-cutting context, a defer rationale that spans the PR, or a one-line pointer to the fix commit.

```bash
gh api "repos/$REPO/issues/$PR/comments" -f body="<top-level note>"
```

**Deduplicate**: if multiple reviewers (or summary + inline) raise the same point, fix once, reference the same commit in each reply.

Resolve every thread that was replied to, regardless of verdict.
