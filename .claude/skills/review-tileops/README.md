# review-tileops setup

This skill reviews PRs as a GitHub identity distinct from the PR author. The identity is per-developer machine config — it is not in the repo.

## One-time setup

1. Have a second GitHub account (separate from the one you author PRs with).

1. Pick any directory path you like, then authenticate `gh` against it as that account:

   ```bash
   GH_CONFIG_DIR=/path/to/your/reviewer-config gh auth login --hostname github.com
   ```

1. Export the same path in your shell rc:

   ```bash
   export TILEOPS_REVIEW_GH_CONFIG_DIR=/path/to/your/reviewer-config
   ```

## Before reviewing each PR

Run once per PR (the loop driver does this automatically; for a manual single-shot review, run it yourself):

```bash
bash .claude/skills/review-tileops/preflight.sh <PR_NUMBER>
```

It verifies the env var, the `hosts.yml`, and that the reviewer login is distinct from the PR author. The skill itself does no preflight — it assumes this has passed.

## If preflight errors

Each error message names the next command to run. Follow it.
