---
name: create-follow-up-issue
description: Introspect a development session and generate follow-up issues for deferred work, discovered problems, and coverage gaps. Max 3 issues per invocation.
---

## Arguments

| Argument   | Required | Description                                              |
| ---------- | -------- | -------------------------------------------------------- |
| `<PR_URL>` | No       | GitHub PR URL. If omitted, inferred from current branch. |

## Contract

- **Input**: PR reference + conversation history (if available)
- **Output**: up to 3 follow-up issues + in-scope suggestions committed into the PR + remaining out-of-scope suggestions listed in PR body
- **Termination**: PR body updated — with issues, applied-fix commit, out-of-scope suggestions, or explicit "no follow-up" declaration

## Modes

- **Session-rich**: conversation history available. Introspection is primary signal; PR supplements.
- **Session-poor**: no session context. Use PR diff + human reviewer comments.

## Steps

### 1. GATE

Resolve PR. Try explicit URL first, else current branch:

```bash
gh pr view <PR_URL_or_empty> --json number,title,url,body,baseRefName
OWNER_REPO=$(gh repo view --json nameWithOwner -q '.nameWithOwner')
```

Extract: `PR_NUMBER`, `PR_TITLE`, `PR_URL`, `PR_BODY`, `BASE_BRANCH`, `OWNER_REPO`.

**Fail** → terminate: `No PR found. Provide a PR URL (/create-follow-up-issue <url>), or run on a branch with an associated PR.`

### 2. COLLECT (parallel)

| Source                  | How                                                                    |
| ----------------------- | ---------------------------------------------------------------------- |
| Code diff               | `git diff $BASE_BRANCH...HEAD`                                         |
| Session history         | Scan conversation for deferrals, workarounds, surprises, blocked items |
| In-code markers         | Grep changed files for `TODO`, `FIXME`, `HACK`, `XXX`                  |
| Human reviewer comments | `gh api` — see below                                                   |

**Reviewer comment extraction** (excludes PR author and bots):

```bash
PR_AUTHOR=$(gh pr view $PR_NUMBER --json author -q '.author.login')

# Inline review comments
gh api repos/$OWNER_REPO/pulls/$PR_NUMBER/comments --paginate \
  --jq '[.[] | select(.user.login != env.PR_AUTHOR and .user.type != "Bot"
         and (.user.login | test("copilot|gemini|github-actions"; "i") | not))]'

# General PR comments
gh api repos/$OWNER_REPO/issues/$PR_NUMBER/comments --paginate \
  --jq '[.[] | select(.user.login != env.PR_AUTHOR and .user.type != "Bot"
         and (.user.login | test("copilot|gemini|github-actions"; "i") | not))]'
```

### 3. CLASSIFY

**Issue-worthy** (→ follow-up issue):

| Category             | Signal                                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Scope deferral**   | "not in this PR", "follow-up needed", explicitly deferred                                                          |
| **Fragile coupling** | Workarounds, monkey-patches, breakable assumptions                                                                 |
| **Coverage gap**     | Untested cases, missing edge cases, skipped benchmarks                                                             |
| **Consistency gap**  | *Doc drift*: implementation changed, docs/manifest not updated. *Spec drift*: same problem exists in other modules |

**Suggestion** (→ no issue). Split by scope:

| Sub-tier         | Signal                                                                                                                                                                                                              | Disposition                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| **In-scope**     | Touches only files already in this PR's diff; fix is small, mechanical, low-risk (style, naming, formatting, local refactor, obvious typo); does not change observable behavior beyond what this PR already changes | Apply directly in this PR (step 6.5) |
| **Out-of-scope** | Touches files outside this PR's diff, OR would expand PR's behavioral surface, OR is judgment-dependent (API naming debate, design tradeoff)                                                                        | List as text in PR body (step 7)     |

When uncertain, classify as out-of-scope — do not silently enlarge the PR's diff.

**Termination gate:** Zero issue-worthy items AND zero in-scope suggestions → skip to step 7. Do not manufacture follow-ups.

### 4. MERGE

Reduce to **max 3 issues**:

- Same module + multiple problems → merge
- Same root cause + multiple locations → merge
- Different modules, independent → keep separate
- Still >3 after merging → force-merge smallest into most related candidate

### 5. PRESENT

Show candidates in **dependency order** (prerequisites first). Wait for user confirmation.

```
Follow-up candidates from PR #<number>: <title>
──────────────────────────────────────────────────
1. [TYPE][SCOPE] <title>
   Category: <category> | Summary: <1-2 sentences>

2. [TYPE][SCOPE] <title>          ← parallel with #1
   Category: <category> | Summary: <1-2 sentences>

3. [TYPE][SCOPE] <title>          ← depends on #1
   Category: <category> | Summary: <1-2 sentences>

Execution order: {#1, #2} → #3

In-scope fixes (apply directly in this PR):
  - <file:line> — <nit>

Out-of-scope suggestions (PR body text only):
  - <nit>

Actions: confirm all / drop by number / edit / move <item> to out-of-scope
```

### 6. CREATE

Ensure label exists:

```bash
gh label list --search "follow-up" --json name --jq '.[].name' | grep -qx "follow-up" \
  || gh label create "follow-up" --description "Generated from dev session introspection" --color "c5def5"
```

For each confirmed item, invoke `foundry:creating-issue` with `--from-draft <tmpfile>`:

```markdown
---
type: <FEAT|BUG|PERF|REFACTOR|DOCS|TEST>
component: <affected module>
labels:
  - follow-up
target_repo: <OWNER_REPO>
---

# Description
## Symptom / Motivation
Discovered during PR #<PR_NUMBER> (<PR_TITLE>).
<what was observed, why it matters>

## Root Cause Analysis
<why not addressed in source PR — scope, complexity, risk>

## Related Files
- <paths from diff or session>

# Goal
<one sentence>

# Plan
<!-- type: proposal -->
1. <step>
2. <step>

# Constraints
- Must not regress PR #<PR_NUMBER>

# Acceptance Criteria
- [ ] AC-1: Modified files pass unit tests
```

### 6.5. APPLY IN-SCOPE FIXES

For each confirmed in-scope suggestion:

1. Verify file is in PR diff (`git diff --name-only $BASE_BRANCH...HEAD`). If not, demote to out-of-scope and skip.
1. Apply edit with the Edit tool.
1. Run the most relevant fast check for the file type (e.g., `pre-commit run --files <paths>`, or unit tests for the touched module). Not the full suite.

Commit the batch as a single commit on the current branch:

```bash
git add <files>
git commit -m "[Chore] apply in-scope follow-up suggestions from PR #$PR_NUMBER review

- <one line per applied fix: file:line — what changed>
"
git push
```

Constraints:

- If a fix fails its check or adds unrelated diff noise → `git restore` that file and demote to out-of-scope before committing the rest.
- Never force-push. Never amend existing commits.

Record `APPLIED_FIXES` (file:line + one-line summary per fix) for step 7.

### 7. UPDATE PR BODY

Update existing PR body to match the final implementation (fix summary/description if the plan changed during development). Then append `## Follow-up`.

```bash
CURRENT_BODY=$(gh pr view $PR_NUMBER --json body -q '.body')
gh pr edit $PR_NUMBER --body "$CURRENT_BODY

$FOLLOWUP_SECTION"
```

Omit empty sections entirely — do not write "none".

**Section blocks (compose as needed):**

```markdown
## Follow-up

Issues:
- #<N> — <one-line summary>
- #<N> — <one-line summary> (depends on #<N>)

Applied in this PR:
- <file:line> — <one-line summary>

Out-of-scope suggestions:
- <nit>
```

**Clean close** (nothing in any bucket):

```markdown
## Follow-up

No follow-up issues or suggestions.
```

## Output

```
PR #<number> body updated.
  Issues: #<N>, #<N> (or "none")
  Applied in-PR: <count> (or "none") — commit <sha>
  Out-of-scope suggestions: <count> (or "none")
  Execution order: {#1, #2} → #3
```
