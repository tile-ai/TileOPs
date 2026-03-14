---
name: creating-pull-request
description: Create a PR with correct title, body, and labels following TileOPs conventions. Use when you have a pushed branch ready for PR creation.
---

## Current state

- Repo: !`gh repo view --json nameWithOwner -q '.nameWithOwner'`
- Branch: !`git branch --show-current`
- Commits on branch: !`git log --oneline main..HEAD 2>/dev/null || echo "(no commits vs main)"`
- Remote: !`git remote -v | head -2`

## Task

\$ARGUMENTS

## Reference

Read `docs/CONTRIBUTING.md` before proceeding for PR title conventions.

For PR body format, see [template.md](template.md).

**Source of truth** for type lists, regex patterns, and label mappings: `.claude/conventions/types.sh`.

## Steps

Execute these steps in order. **Do NOT skip any HARD GATE.**

### Step 1: Determine owner/repo

```bash
gh repo view --json nameWithOwner -q '.nameWithOwner'
```

Split into `{owner}` and `{repo}`.

### Step 2: Structural compliance check (new ops only)

If this PR introduces or modifies a kernel/op, read `docs/kernel-op-conventions.md` and verify the code against **every** item below. Record each as `PASS` or `FAIL (reason)`.

**T.prim_func (§1)**

- `_<op_name>_kernel(static_params) -> Callable` closure exists
- `@tilelang.jit(out_idx=[...])` wraps the config-parameterised inner function
- `with T.Kernel()` is inside `@T.macro`, not directly in `@T.prim_func`
- No Python builtins (`float()`, `math.cos()`) applied to TileLang IR nodes
- Tile-ops (`T.clear`, `T.copy`, `T.gemm`) are at `T.Kernel` scope, not inside `T.Parallel`

**Kernel class (§2)**

- `Kernel.forward` takes only GPU tensor args — no format conversion, batching, or dtype casting
- `Kernel.forward` body is `return self.kernel(config...)(tensors)` and nothing else
- `default_config` and `autotune_configs` properties are defined
- `supported_archs` class attribute is set
- `accum_dtype` is hardcoded in the kernel — not a property, config key, or parameter
- `@torch.library.custom_op` + `.register_fake` wrapper exists

**Op class (§3)**

- `Op.forward` owns all pre/post-processing (format conversion, dtype cast, batching, reshaping)
- `Op.forward` delegates GPU computation to `self.kernel(...)`
- `accum_dtype` is not stored on `Op` and not a parameter of `Op.__init__`
- `kernel_map` is the last `__init__` parameter; `dispatch_kernel(kernel_map)` called before kernel instantiation
- `__init__.py` exports are synchronized (`__all__` + explicit re-exports)

**Delivery completeness**

- Unit tests in `tests/ops/` with reference comparison (FP16 atol=1e-3, BF16 atol=1.6e-2)
- Benchmark class in `benchmarks/`
- Dtype support matrix documented in PR body

**HARD GATE:** If any item is `FAIL`, fix the code before proceeding to Step 3. Include the full check results in the PR body under `## Structural Compliance`.

If this PR does **not** touch kernel/op code, skip this step entirely.

### Step 3: Create the PR

Use `gh pr create`. **Do NOT use the GitHub MCP tool** — `gh` CLI avoids the `\n` pitfall.

```bash
gh pr create --base main --head "$(git branch --show-current)" \
  --draft \
  --title "[Type] Description" \
  --body "$(cat <<'PREOF'
<fill in from template.md>
PREOF
)"
```

**Draft-first workflow:** Always create PRs as draft. Copilot reviews drafts automatically; Gemini must be triggered separately (see `lifecycle-pull-request` Phase 2b). Once CI passes and both bot reviews are addressed, the `lifecycle-pull-request` skill marks the PR ready, triggering human reviewer notifications.

PR title rules:

- **Must** use bracket format: `[Type] Description` or `[Type][Scope] Description`
- Types: `[Feat]`, `[BugFix]`, `[Fix]`, `[Refactor]`, `[Enhancement]`, `[Doc]`, `[Chore]`, `[Bench]`, `[CI]`, `[Test]`, `[Perf]`
- Keep under ~80 chars
- **Do NOT** put issue references in the title (e.g. `(#123)`). Link issues in the PR body with `Closes #123` instead.

PR body section rules:

- `## Summary` — always required
- `## Test plan` — always required
- `## Benchmark` — required when PR involves performance changes
- `## Regression` — recommended when PR is bugfix or refactor
- `## Additional context` — optional
- **Delete** inapplicable optional sections entirely. Never leave empty headers.

### Step 4: Add labels (MANDATORY)

**At least one label is required.** Select based on PR type:

| PR Type              | Label             |
| -------------------- | ----------------- |
| `[Feat]`             | `feature`         |
| `[BugFix]` / `[Fix]` | `fix`             |
| `[Enhancement]`      | `enhancement`     |
| `[Refactor]`         | `refactor`        |
| `[Doc]`              | `docs`            |
| `[Chore]`            | `chore`           |
| `[Bench]`            | `bench`           |
| `[CI]`               | `ci`              |
| `[Test]`             | `test`            |
| `[Perf]`             | `perf`            |
| Breaking change      | `breaking change` |

Authorship label (**required**):

- `all-ai-powered` — always add this label (you are an AI creating this PR)

```bash
gh pr edit <PR_NUMBER> --add-label "<label1>" --add-label "<label2>"
```

### Step 5: HARD GATE — Validate PR

```bash
.claude/skills/creating-pull-request/scripts/validate.sh <owner/repo> <PR_NUMBER>
```

**If exit code != 0: STOP.** Fix the reported issues:

- Wrong title → `gh pr edit <N> --title "[Type] New title"`
- Missing body section → `gh pr edit <N> --body "$(cat <<'EOF' ... EOF)"`
- Missing labels → `gh pr edit <N> --add-label "<label>"`

Re-run the gate until it passes.

## Responding to reviewer feedback

When addressing reviewer comments (from Copilot, Gemini, or humans):

- **Always reply inline** on the specific comment thread. Never post a summary comment on the PR conversation.
- Use the review API to create inline comments on the exact lines being discussed.
- For each comment: acknowledge, state what was fixed, and reference the commit SHA.

## Return format

Report exactly:

- `PR_NUMBER: <number>`
- `PR_URL: <url>`
- `BRANCH: <branch-name>`
- `SUMMARY: <one-line summary>`
