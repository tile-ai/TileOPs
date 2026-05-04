---
name: follow-up
description: Introspect a development session and generate follow-up issues for deferred work, discovered problems, and coverage gaps. Max 3 issues per invocation.
---

## Args

| Argument       | Description                                                                               |
| -------------- | ----------------------------------------------------------------------------------------- |
| `<PR_NUMBER>`  | Required. TileOPs PR number.                                                              |
| `--nightshift` | Skip the interactive presentation; auto-accept all candidates; inject `nightshift` label. |

## Contract

Input: PR ref + conversation (if available). Output: ≤3 follow-up issues, in-scope fixes committed, out-of-scope suggestions printed to stdout. **Never edit the PR body** — the review skill owns it.

## Modes

- **Session-rich**: introspection is primary signal; PR supplements.
- **Session-poor**: PR diff + human reviewer comments only.

## Steps

### 1. Resolve PR

```bash
NIGHTSHIFT=0
for arg in "$@"; do
  case "$arg" in
    --nightshift) NIGHTSHIFT=1 ;;
    -*) echo "Unknown flag: $arg" >&2; exit 1 ;;
    *) PR_NUMBER="${PR_NUMBER:-$arg}" ;;
  esac
done
[[ -z "${PR_NUMBER:-}" ]] && { echo "Usage: /follow-up <PR_NUMBER>" >&2; exit 1; }

gh pr view "$PR_NUMBER" --json number,title,url,body
OWNER_REPO=$(gh repo view --json nameWithOwner -q '.nameWithOwner')
```

PR not found → terminate.

### 2. Collect

| Source            | How                                                   |
| ----------------- | ----------------------------------------------------- |
| Diff              | `gh pr diff "$PR_NUMBER"`                             |
| Session           | Scan for deferrals, workarounds, blocked items        |
| In-code markers   | Grep changed files for `TODO`, `FIXME`, `HACK`, `XXX` |
| Reviewer comments | Both endpoints below, filtered to non-author non-bot  |

```bash
export PR_AUTHOR=$(gh pr view "$PR_NUMBER" --json author -q '.author.login')
FILTER='[.[] | select(.user.login != env.PR_AUTHOR and .user.type != "Bot"
        and (.user.login | test("copilot|gemini|github-actions"; "i") | not))]'
gh api "repos/$OWNER_REPO/pulls/$PR_NUMBER/comments"  --paginate --jq "$FILTER"
gh api "repos/$OWNER_REPO/issues/$PR_NUMBER/comments" --paginate --jq "$FILTER"
```

### 3. Classify

**Issue-worthy** (→ follow-up issue):

| Category         | Signal                                                 |
| ---------------- | ------------------------------------------------------ |
| Scope deferral   | "not in this PR", explicit defer                       |
| Fragile coupling | Workarounds, monkey-patches                            |
| Coverage gap     | Untested cases, missing edge cases, skipped benchmarks |
| Consistency gap  | Doc drift; same problem in other modules               |

**Suggestion** (no issue):

| Tier         | Signal                                          | Action |
| ------------ | ----------------------------------------------- | ------ |
| In-scope     | Touches only this PR's files; small, mechanical | Step 7 |
| Out-of-scope | Outside diff, expands behavior, or subjective   | Step 8 |

Uncertain → out-of-scope. Nothing in either bucket → step 8 (empty report). Do not manufacture follow-ups.

### 4. Merge → max 3 issues

Same module / root cause → merge. Different + independent → keep separate.

### 5. Present

`--nightshift`: skip; auto-accept; → step 6.

Default — render in dependency order, wait on `Actions:`:

```
Follow-up candidates from PR #<N>: <title>
1. [TYPE][SCOPE] <title>          Category: <…> | <1–2 sentences>
2. [TYPE][SCOPE] <title>          ← parallel with #1
3. [TYPE][SCOPE] <title>          ← depends on #1

Execution order: {#1, #2} → #3

In-scope fixes:
  - <file:line> — <nit>
Out-of-scope suggestions:
  - <nit>

Actions: confirm all / drop by N / edit / move <item> to out-of-scope
```

### 6. Create — delegate to `creating-issue`

For each confirmed item: write a draft to a tmpfile, then invoke `foundry:creating-issue --from-draft <tmpfile>`. **Never `gh issue create` directly** — that bypasses the 5-section HARD GATE which `foundry:pipeline` Phase A re-validates downstream.

Draft body must conform to the canonical template at `foundry/skills/creating-issue/SKILL.md` Step 4 — `creating-issue` is the single owner; do not duplicate or paraphrase its section names here.

Frontmatter:

```yaml
---
type: <FEAT|BUG|PERF|REFACTOR|DOCS|TEST>
component: <affected module>
labels: [follow-up]   # add `nightshift` only when --nightshift was passed
target_repo: <OWNER_REPO>
---
```

Labels (the `nightshift` label is human-curated — fail fast, do not auto-create):

```bash
gh label list --search follow-up --json name --jq '.[].name' | grep -qx follow-up \
  || gh label create follow-up --color c5def5 --description "Generated from dev session introspection"

if [[ "$NIGHTSHIFT" == "1" ]]; then
  gh label list --search nightshift --json name --jq '.[].name' | grep -qx nightshift \
    || { echo "nightshift label missing in $OWNER_REPO" >&2; exit 1; }
fi
```

### 7. Apply in-scope fixes

For each: verify file is in `gh pr diff --name-only "$PR_NUMBER"` (else demote); Edit; run a fast check (`pre-commit run --files <paths>` or module-scoped unit tests). Commit the batch:

```
[Chore] apply in-scope follow-up suggestions from PR #$PR_NUMBER review

- <file:line> — <what changed>
```

Fix fails its check or adds unrelated diff → `git restore`, demote. Never force-push or amend.

### 8. Report

Stdout only. Omit empty sections.

```
PR #<PR_NUMBER> follow-up complete.

Issues created:
- #<N> — <summary>

Applied (commit <sha>):
- <file:line> — <summary>

Out-of-scope suggestions:
- <nit>

Execution order: {#A, #B} → #C
```

Nothing in any bucket: `PR #<PR_NUMBER>: no follow-up issues or suggestions.`
