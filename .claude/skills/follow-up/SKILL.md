---
name: follow-up
description: Introspect a development session and generate follow-up issues for deferred work, discovered problems, and coverage gaps. Max 3 issues per invocation.
---

## Arguments

| Argument       | Required | Description                                                                                                                                                 |
| -------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `<PR_NUMBER>`  | Yes      | TileOPs PR number (e.g. `1131`).                                                                                                                            |
| `--nightshift` | No       | Boolean flag. Skip the interactive presentation gate (auto-accept all candidates) and inject the `nightshift` label into every created issue's frontmatter. |

## Contract

- **Input**: PR reference + conversation history (if available)
- **Output**: up to 3 follow-up issues + in-scope suggestions committed into the PR + remaining out-of-scope suggestions printed to stdout for the developer to fold into the PR body at the reviewer's approval gate
- **Termination**: issues created (if any) + applied-fix commit pushed (if any) + a stdout report listing what was created/applied/deferred. **Never edit the PR body** — the review skill owns body updates.

## Modes

- **Session-rich**: conversation history available. Introspection is primary signal; PR supplements.
- **Session-poor**: no session context. Use PR diff + human reviewer comments.

## Steps

### 1. GATE

`<PR_NUMBER>` is required. If missing, terminate immediately: `Missing PR number. Usage: /follow-up <PR_NUMBER>`.

Resolve PR:

```bash
gh pr view <PR_NUMBER> --json number,title,url,body
OWNER_REPO=$(gh repo view --json nameWithOwner -q '.nameWithOwner')
```

Extract: `PR_NUMBER`, `PR_TITLE`, `PR_URL`, `PR_BODY`, `OWNER_REPO`.

**Fail** (PR not found) → terminate: `PR #<PR_NUMBER> not found in $OWNER_REPO.`

### 2. COLLECT (parallel)

| Source                  | How                                                                    |
| ----------------------- | ---------------------------------------------------------------------- |
| Code diff               | `gh pr diff "$PR_NUMBER"`                                              |
| Session history         | Scan conversation for deferrals, workarounds, surprises, blocked items |
| In-code markers         | Grep changed files for `TODO`, `FIXME`, `HACK`, `XXX`                  |
| Human reviewer comments | `gh api` — see below                                                   |

**Reviewer comment extraction** (excludes PR author and bots). Apply the same filter to both endpoints — inline review comments and general PR comments:

```bash
export PR_AUTHOR=$(gh pr view $PR_NUMBER --json author -q '.author.login')
FILTER='[.[] | select(.user.login != env.PR_AUTHOR and .user.type != "Bot"
        and (.user.login | test("copilot|gemini|github-actions"; "i") | not))]'

gh api "repos/$OWNER_REPO/pulls/$PR_NUMBER/comments"  --paginate --jq "$FILTER"
gh api "repos/$OWNER_REPO/issues/$PR_NUMBER/comments" --paginate --jq "$FILTER"
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

| Sub-tier         | Signal                                                                                                                          | Disposition                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **In-scope**     | Touches only files in this PR's diff; fix is small, mechanical, low-risk; does not expand the PR's observable behavior          | Apply directly in this PR (step 7) |
| **Out-of-scope** | Touches files outside this PR's diff, OR expands the PR's behavioral surface, OR is judgment-dependent (subjective design call) | Print in stdout report (step 8)    |

When uncertain, classify as out-of-scope — do not silently enlarge the PR's diff.

**Termination gate:** Zero issue-worthy items AND zero in-scope suggestions → skip to step 8 (print empty report). Do not manufacture follow-ups.

### 4. MERGE

Reduce to **max 3 issues**:

- Same module + multiple problems → merge
- Same root cause + multiple locations → merge
- Different modules, independent → keep separate
- Still >3 after merging → force-merge smallest into most related candidate

### 5. PRESENT

**Default mode** (no `--nightshift`): show candidates in **dependency order** (prerequisites first). Wait for user confirmation via the `Actions:` line below.

**`--nightshift` mode**: skip the candidate presentation and the `Actions: confirm all / drop by number / edit / move <item> to out-of-scope` interaction entirely. Treat every candidate (and every in-scope suggestion) as confirmed — equivalent to the user typing `confirm all` — and proceed straight to Step 6. The auto-accept branch is conditional on the flag; default-mode behavior is unchanged when the flag is absent.

#### Default mode (no `--nightshift`)

The presentation template below applies only to default mode. Render it to stdout and wait on the `Actions:` line.

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

Out-of-scope suggestions (deferred — printed in step 8, not committed):
  - <nit>

Actions: confirm all / drop by number / edit / move <item> to out-of-scope
```

#### `--nightshift` mode (auto-accept)

Do not render the template above. Auto-accept all candidates and proceed directly to Step 6.

### 6. CREATE

Ensure label exists:

```bash
gh label list --search "follow-up" --json name --jq '.[].name' | grep -qx "follow-up" \
  || gh label create "follow-up" --description "Generated from dev session introspection" --color "c5def5"
```

**Nightshift label guard** — only when `--nightshift` was passed. The `nightshift` label is human-curated in the canonical setup (specific color and description); do not auto-create it with arbitrary metadata. Instead, fail fast with a clear error if it is missing:

```bash
# Only when --nightshift was passed:
gh label list --search "nightshift" --json name --jq '.[].name' | grep -qx "nightshift" \
  || { echo "nightshift label missing in $OWNER_REPO — see foundry nightshift docs" >&2; exit 1; }
```

If `--nightshift` was passed and the `nightshift` label does not exist, terminate with: `nightshift label missing in $OWNER_REPO — see foundry nightshift docs`. Do not create the label automatically.

For each confirmed item, invoke `foundry:creating-issue` with `--from-draft <tmpfile>`.

Pick exactly one of the two frontmatter + body templates below based on whether `--nightshift` was passed. The body sections after the frontmatter are byte-identical between the two; only the `labels:` block differs.

**Default invocation** (no `--nightshift`) — emits only the `follow-up` label. Use this template verbatim:

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

**With `--nightshift`** — emits both `follow-up` and `nightshift` labels. Use this template verbatim:

```markdown
---
type: <FEAT|BUG|PERF|REFACTOR|DOCS|TEST>
component: <affected module>
labels:
  - follow-up
  - nightshift
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

### 7. APPLY IN-SCOPE FIXES

For each confirmed in-scope suggestion:

1. Verify the file is in the PR diff with `gh pr diff --name-only "$PR_NUMBER"`. If not, demote to out-of-scope and skip.
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

Record `APPLIED_FIXES` (file:line + one-line summary per fix) for step 8.

### 8. REPORT

**Do not edit the PR body.** The review skill owns body updates at its approval gate; this skill prints a stdout report only.

Omit empty sections entirely — do not write "none" inside a section, just drop the heading.

```
PR #<PR_NUMBER> follow-up complete.

Issues created:
- #<N> — <one-line summary>
- #<N> — <one-line summary> (depends on #<N>)

Applied in this PR (commit <sha>):
- <file:line> — <one-line summary>

Out-of-scope suggestions (fold into PR body at the reviewer's approval gate):
- <nit>

Execution order: {#1, #2} → #3
```

**Clean close** (nothing in any bucket):

```
PR #<PR_NUMBER>: no follow-up issues or suggestions.
```
